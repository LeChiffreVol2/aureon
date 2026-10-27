"""Manually start one local KAO-TerraMind B resident worker; no acquisition."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import signal
import sys
import threading
from time import perf_counter

from .api import LocalJobService, create_server
from .console import load_console_assets


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-model', action='store_true')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--model',choices=('kao-terramind-a','kao-terramind-b','kao-terramind-c'),default='kao-terramind-b',
                        help='Explicit model identity; bundle must match; qualifications are model-specific')
    parser.add_argument('--streaming',action='store_true',help='Opt in to bounded window/stripe scene execution')
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8031)
    parser.add_argument('--queue-capacity', type=int, default=2)
    parser.add_argument('--max-jobs', type=int, default=None)
    parser.add_argument('--state-dir', type=Path,
                        help='Opt in to durable jobs in an exclusive local Linux filesystem directory; reuse on restart')
    parser.add_argument('--payload-ttl-seconds', type=float, default=3600)
    parser.add_argument('--key-ttl-seconds', type=float, default=86400)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--min-free-bytes', type=int, default=2 * 2**30)
    parser.add_argument('--hourly-rate', type=float, default=1.0)
    parser.add_argument('--currency', default='THB')
    parser.add_argument('--max-runtime-seconds', type=float, default=0,
                        help='Optional finite manual session limit; zero means stop by signal or STOP file')
    args = parser.parse_args(argv)
    if not args.execute_model or sys.platform != 'linux':
        parser.error('Pass --execute-model in the existing isolated Linux runtime')
    if not 0 <= args.port <= 65535 or not 0 <= args.max_runtime_seconds <= 86400:
        parser.error('Invalid port or finite runtime limit')
    output, bundle = args.output.resolve(), args.bundle.resolve()
    args.max_jobs = args.max_jobs if args.max_jobs is not None else (100000 if args.state_dir else 32)
    if output.exists() or not output.parent.is_dir():
        parser.error('Output must be new and its parent must already exist')
    scenes = {}
    for path in args.manifest:
        path = path.resolve(strict=True)
        scene = json.loads(path.read_text(encoding='utf-8'))['scene_id']
        if not isinstance(scene, str) or scene in scenes:
            parser.error('Manifests must supply unique scene IDs')
        scenes[scene] = path
    if args.streaming:
        # Validate every registered scene on CPU before constructing the sole
        # model owner. A malformed later scene must not wait until its GPU job.
        from ..prepare import check_scene
        try:
            for path in scenes.values():
                check_scene(path)
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            parser.error(f'Scene input preflight failed: {error}')
    # Snapshot only the explicitly served assets before constructing the worker.
    console_assets = load_console_assets(Path(__file__).resolve().parents[1]/'static')
    durable_args = None
    if args.state_dir:
        from importlib import metadata
        from .durable import _plain_path
        from .worker import source_hashes
        state_dir = _plain_path(args.state_dir)
        # WSL /mnt/c is not the persistence target. No network filesystem or
        # tmpfs database: locks, fsync and rename must refer to durable local disk.
        mounts = []
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left, right = line.split(' - ', 1)
            mount = Path(left.split()[4].replace('\\040', ' ').replace('\\134', '\\'))
            if state_dir.is_relative_to(mount):
                mounts.append((len(mount.parts), right.split()[0]))
        if not mounts or max(mounts)[1] not in ('ext4', 'xfs', 'btrfs'):
            parser.error('Durable state requires a local ext4/xfs/btrfs filesystem (WSL Linux disk, not /mnt/c)')
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        contract = {'bundle_receipt_sha256': digest(bundle/'bundle_receipt.json'),
            'manifests_sha256': {name: digest(path) for name, path in scenes.items()},
            'source_sha256': source_hashes(), 'precision': 'highest FP32; TF32 off',
            'model_id':args.model,'execution_mode':'streaming' if args.streaming else 'materialized',
            'packages': {name: metadata.version(name) for name in ('torch','terratorch','numpy','rasterio')},
            'cost_scenario': {'hourly_rate': args.hourly_rate, 'currency': args.currency}}
        durable_args = dict(state_dir=state_dir, contract=contract,
            queue_capacity=args.queue_capacity, max_jobs=args.max_jobs, max_attempts=args.max_attempts,
            payload_ttl_seconds=args.payload_ttl_seconds, key_ttl_seconds=args.key_ttl_seconds,
            min_free_bytes=args.min_free_bytes)
    # Importing this module never constructs a worker. Only the owner factory
    # below loads the model, after the explicit execution check above.
    def factory():
        from .worker import ResidentSceneWorker
        return ResidentSceneWorker(bundle, scenes, output/'worker', execute_model=True,
                                   hourly_rate=args.hourly_rate, currency=args.currency,
                                   streaming=args.streaming,model_id=args.model)

    began = perf_counter()
    stop = threading.Event()
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: stop.set())
    service = server = None
    exit_code, reason = 0, 'manual_stop'
    try:
        if durable_args:
            from .durable import DurableJobService
            service = DurableJobService(output, factory, scenes, **durable_args)
        else:
            service = LocalJobService(output, factory, scenes, args.queue_capacity, args.max_jobs)
        asset_snapshot = output/'console-assets'
        asset_snapshot.mkdir(exist_ok=False)
        for name, body, _mime, _digest in console_assets.files:
            with (asset_snapshot/name).open('xb') as stream:
                stream.write(body)
        token = secrets.token_urlsafe(32)
        token_path = output/'token.txt'
        descriptor = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w', encoding='ascii') as stream:
            stream.write(token+'\n')
        server = create_server(service, token, args.port, console_assets=console_assets)
        server.timeout = 0.2
        launch = {'schema':'aureon.local_scene_api_launch.v1', 'pid':os.getpid(),
            'started_utc':datetime.now(timezone.utc).isoformat(),
            'url':f'http://127.0.0.1:{server.server_port}', 'token_file':str(token_path),
            'stop_file':str(output/'STOP'), 'registered_scenes':list(scenes),
            'queue_capacity':args.queue_capacity, 'max_jobs':args.max_jobs,
            'state_directory':str(durable_args['state_dir']) if durable_args else None,
            'max_runtime_seconds':args.max_runtime_seconds,
            'console_url':f'http://127.0.0.1:{server.server_port}/console',
            'console_assets':console_assets.manifest(),
            'console_assets_directory':'console-assets',
            'cost_scenario':{'hourly_rate':args.hourly_rate, 'currency':args.currency},
            'public_deployment':False}
        _write(output/'server-launch.json', launch)
        print(json.dumps({'status':'initializing', 'url':launch['url'],
                          'token_file':str(token_path), 'stop_file':launch['stop_file']}), flush=True)
        ready_recorded = False
        while not stop.is_set() and not (output/'STOP').exists():
            snapshot = service.snapshot()
            if snapshot['state'] == 'ready' and not ready_recorded:
                ready = {'status':'ready', 'seconds_since_launch':perf_counter()-began,
                         'worker':snapshot['worker']}
                _write(output/'server-ready.json', ready)
                print(json.dumps(ready), flush=True)
                ready_recorded = True
            if snapshot['state'] in ('failed', 'unavailable', 'closed'):
                exit_code, reason = 1, 'worker_unavailable'
                break
            if args.max_runtime_seconds and perf_counter()-began >= args.max_runtime_seconds:
                reason = 'explicit_finite_session_limit'
                break
            server.handle_request()
    except BaseException as error:
        exit_code, reason = 1, type(error).__name__
        print(json.dumps({'status':'failed', 'error_type':type(error).__name__}), file=sys.stderr, flush=True)
        raise
    finally:
        if server is not None:
            server.server_close()
        if service is not None:
            service.shutdown()
            snapshot = service.snapshot()
            shutdown_path = output/'worker'/'shutdown.json'
            cleanup = json.loads(shutdown_path.read_text(encoding='utf-8')) if shutdown_path.is_file() else None
            if snapshot.get('error') or not cleanup or not cleanup.get('cleanup_passed'):
                exit_code = 1
            receipt = {'schema':'aureon.local_scene_api_exit.v1', 'status':'closed',
                'reason':reason, 'exit_code':exit_code, 'service':snapshot,
                'worker_cleanup':cleanup, 'observed_service_seconds':perf_counter()-began,
                'scope':'Controlled service interval including initialization, idle, jobs and shutdown; excludes pre-timer CLI imports/argument checks/manifest parsing/console asset snapshot, final exit-receipt serialization, data acquisition and client post-exit audits'}
            _write(output/'server-exit.json', receipt)
            print(json.dumps({'status':'closed', 'exit_code':exit_code,
                              'receipt':str(output/'server-exit.json')}), flush=True)
            if durable_args:
                service.close_storage()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
