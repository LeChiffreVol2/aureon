"""Manual matched HTTP/FIFO service; one explicitly selected model backend."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('pytorch', 'vllm'), default='pytorch')
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--io-mode', choices=('staged', 'pinned'), default='staged')
    parser.add_argument('--thp-policy', choices=('inherit', 'disable'),
                        help='Process-local THP policy; PyTorch defaults to disable, vLLM to inherit')
    args = parser.parse_args(argv)
    if args.io_mode=='pinned' and args.backend!='pytorch':
        raise ValueError('Pinned HTTP buffers currently require the PyTorch backend')
    if sys.platform != 'linux' or not 1024 <= args.port <= 65535:
        raise ValueError('Use the isolated WSL runtime and unprivileged loopback port')
    if os.environ.get('VLLM_API_KEY'):
        raise ValueError('Local service does not bypass an API key configuration')
    folder = Path(args.output).resolve()
    if folder.parent != ROOT/'runs/services' or folder.exists():
        raise ValueError('Require a fresh evidence directory under runs/services')
    folder.mkdir()
    # Apply before importing the model runtime or allocating its buffers. This
    # changes only this explicitly launched service, never the WSL sysfs policy.
    from aureon.process_memory import configure_process_memory
    memory_policy = configure_process_memory(
        args.thp_policy or ('disable' if args.backend == 'pytorch' else 'inherit'))
    from aureon.runtime import verify_bundle
    bundle = verify_bundle(ROOT/'models/bundles/kao-terramind-b')
    env = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
        VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1', VLLM_FLOAT32_MATMUL_PRECISION='highest',
        NVIDIA_TF32_OVERRIDE='0', OMP_NUM_THREADS='1', VLLM_ENABLE_V1_MULTIPROCESSING='0',
        AUREON_CUDA_GRAPH='1', AUREON_PACKED_TILES='1', AUREON_SHM_ENGINE='1',
        AUREON_SHARED_MEMORY_API='1', AUREON_RESIDENT_SLOTS='1', AUREON_PHASE5_DIAGNOSTICS='0',
        AUREON_ENGINE_DIAGNOSTICS='0', AUREON_FORWARD_DIAGNOSTICS='0',
        AUREON_REGISTERED_SLOTS='1', AUREON_FINITE_SCAN='1', AUREON_FINITE_SIMD='0',
        AUREON_CPU_ACK='0', AUREON_CPU_CONTROL='0',
        AUREON_VERIFIED_BUNDLE=bundle['bundle_dir'], AUREON_SERVICE_RUN=str(folder))
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location('identity', ROOT/'maintenance/serve-wsl-model.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sources = sorted(set(list((ROOT/'aureon').glob('*.py')) + [ROOT/'maintenance/serve-queued-wsl.py',
        ROOT/'aureon/native/finite_fp32.c', ROOT/'maintenance/build-finite.py',
        ROOT/'build/native/finite-v2/manifest.json', ROOT/'build/native/finite-v2/finite_fp32.so']))
    hashes = {}
    for path in sources:
        name = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        target = folder/'source-snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    write(folder/'launch.json', {'pid': os.getpid(), 'backend': args.backend,
        'io_mode': args.io_mode,
        'process_memory_policy': memory_policy,
        'process_identity': helper._proc_identity(os.getpid()), 'environment': env,
        'linux_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'source_sha256': hashes, 'bundle': bundle, 'port': args.port, 'host': '127.0.0.1'})
    stopping = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *unused: stopping.set())
    adapter = server = thread = journal = observer = None
    error = None
    try:
        from aureon.queued_backend import QueuedBackend
        from aureon.queued_http import QueuedServer
        from aureon.serving_telemetry import Journal, ResourceObserver
        if args.io_mode=='pinned':
            from aureon.pinned_queue import PinnedQueuedBackend
            adapter = PinnedQueuedBackend(bundle, folder)
        else:
            adapter = QueuedBackend(args.backend, bundle, folder)
        journal = Journal(folder/'events.jsonl')
        server = QueuedServer(('127.0.0.1', args.port), event_sink=journal,
                              jobs=adapter.create_jobs(8) if args.io_mode=='pinned' else None)
        observer = ResourceObserver(adapter, server, folder/'resources.jsonl')
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .1})
        thread.start()
        write(folder/'ready.json', adapter.metadata())
        print(json.dumps({'ready': True, 'backend': args.backend, 'pid': os.getpid()}), flush=True)
        while not stopping.is_set():
            server.execute_one(adapter)
            if observer.sample():
                journal.flush()
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        if server is not None:
            if thread is not None and thread.is_alive():
                server.stop_accepting()
            # Reading handlers see closing before enqueue; already admitted
            # requests keep their slots until execution completes.
            while not server.jobs.empty():
                server.execute_one(adapter)
            server.close_connections()
            if thread is not None:
                thread.join()
            if observer is not None:
                observer.sample(force=True)
            write(folder/'requests.json', {'count': server.record_count,
                'records': list(server.records), 'rejected': server.rejected,
                'failed': server.failed, 'free_slots': server.free.qsize(),
                'retained_capacity':4096, 'full_execution_journal':'events.jsonl'})
        if journal is not None:
            journal.close()
        if observer is not None:
            observer.close()
        if adapter is not None:
            write(folder/'backend-final.json', adapter.metadata())
            adapter.close()
        write(folder/'exit.json', {'error': error, 'pid': os.getpid(),
            'source_unchanged': all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest() == v for k,v in hashes.items()),
            'slots_absent': all(not (Path('/dev/shm')/v['name']).exists() for v in adapter.slots.identities.values()) if adapter else None,
            'ended_unix': time.time()})


if __name__ == '__main__':
    main()
