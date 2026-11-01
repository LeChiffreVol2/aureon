"""One separate process per registered GPU, sharing the replica v2 authority."""
from __future__ import annotations
from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING, cast
from .contracts import ArtifactFiles, AttemptLease, PathInput, SafeError
if TYPE_CHECKING:
    from .backend import Backend
    from .replica_control import ReplicaControl
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import threading
import traceback

from ..cloud.control import CONTRACT, require, encoded, write_file
from ..scene_api.durable import _plain_path
from .ownership import exclusive, safe_error
from .timing import elapsed
from .worker import scratch_bytes


def work(control: ReplicaControl, worker: str, folder: PathInput, *, config: PathInput | None = None,
         fake: bool = False, execute_model: bool = False, max_seconds: int = 300, max_jobs: int = 100,
         stop: threading.Event | None = None) -> dict[str, object]:
    require(fake==(control.execution=='fake') and (fake or execute_model is True),'execution_mode_mismatch')
    require(type(max_seconds) is int and 1<=max_seconds<=86400 and type(max_jobs) is int and 1<=max_jobs<=10000,'invalid_run_limits')
    spec=control.spec(worker); folder=_plain_path(folder).absolute()
    require(not folder.exists() and folder.parent.is_dir(),'fresh_work_required')
    require(folder!=control.root and not folder.is_relative_to(control.root) and not control.root.is_relative_to(folder),'work_overlaps_state')
    from .storage import require_local
    require_local(folder.parent); require_local(control.root)
    stop=stop or threading.Event(); pulse_stop=threading.Event(); lease_lock=threading.Lock()
    current: AttemptLease | None
    error: object
    cleanup: Mapping[str, object] | None
    lost: list[SafeError]
    current=None; engine=None; generation=None; error=None; cleanup=None; completed=0; lost=[]
    started=elapsed(); pulse_thread=None; device_owner=None
    with ExitStack() as stack:
        stack.enter_context(exclusive(control.root,'replica-'+worker+'.lock'))
        if not fake:
            from .devices import device_lease,device_binding,attest_device
            device_owner=stack.enter_context(device_lease(spec['device']))
            stack.enter_context(device_binding(spec['device']))
            attest_device(spec['device'])
            import torch
            free,total=torch.cuda.mem_get_info()
            require(spec['vram_bytes']<=total and free>=spec['resident_vram_bytes']+spec['transient_vram_bytes'], 'device_capacity_unavailable',429)
        folder.mkdir(mode=0o700)
        try:
            generation=control.register(worker,folder,config=config)
            write_file(folder/'ownership.json',encoded({'worker':worker,'generation':generation,'state':str(control.root),'pid':os.getpid(),'device':spec['device']}))
            def pulse() -> None:
                while not pulse_stop.wait(5):
                    try:
                        with lease_lock:
                            control.pulse(worker,generation)
                            if current is not None: control.heartbeat(current)
                    except BaseException as exc:
                        lost.append(safe_error(exc,'replica_heartbeat_failed')); stop.set(); return
            pulse_thread=threading.Thread(target=pulse,daemon=True); pulse_thread.start()
            if not fake:
                from .backend import Backend
                engine=Backend(control,folder,config=config,execute_model=True,device=spec['device'])
            while not stop.is_set() and elapsed()-started<max_seconds and completed<max_jobs:
                if control.meta('drain',False): break
                require(scratch_bytes(folder)+CONTRACT['max_result_bytes']<=spec['scratch_bytes'],'worker_scratch_capacity',507)
                require(shutil.disk_usage(folder).free>=CONTRACT['min_free_bytes']+CONTRACT['max_result_bytes'],'physical_disk_floor',507)
                with lease_lock: current=control.claim_replica(worker,generation)
                if current is None:
                    stop.wait(.1); continue
                start=elapsed()
                files=None if fake else cast(Backend,engine).run(current)
                require(not lost,'replica_lease_lost',409)
                if not fake:
                    # Cached allocations/RSS remain resident when request credits
                    # release. An underestimated budget must fail closed.
                    require(max(torch.cuda.memory_allocated(),torch.cuda.memory_reserved())<=spec['resident_vram_bytes'],
                        'resident_vram_budget_exceeded',507)
                    require(max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())<=spec['resident_vram_bytes']+spec['transient_vram_bytes'],
                        'peak_vram_budget_exceeded',507)
                    rss=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024
                    require(rss<=spec['resident_host_bytes'],'resident_host_budget_exceeded',507)
                with lease_lock:
                    if fake: control.publish_fake(current)
                    else: control.publish_engine(current,cast(ArtifactFiles,files))
                    finished=current; current=None
                completed+=1
                with (folder/'jobs.jsonl').open('a') as stream:
                    stream.write(json.dumps({'job':finished['job_id'],'attempt':finished['attempt'],
                        'generation':generation,'device':spec['device'],'execution_and_publication_seconds':elapsed()-start})+'\n')
            require(not lost,'worker_background_failed',500)
        except BaseException as exc:
            error=safe_error(exc); traceback.clear_frames(cast(TracebackType,exc.__traceback__))
            if current is not None:
                try: control.fail(current)
                except Exception: pass
        finally:
            pulse_stop.set()
            if pulse_thread is not None: pulse_thread.join()
            if fake: cleanup={'passed':True,'fake':True,'model_executed':False}
            elif engine is not None:
                try: cleanup=engine.close()
                except BaseException as exc: error={'execution':error,'cleanup':safe_error(exc)}
            elif device_owner is not None:
                # Backend construction may fail after CUDA context creation.
                import torch
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                cleanup={'passed':torch.cuda.memory_allocated()==torch.cuda.memory_reserved()==0,
                    'cuda_allocated_bytes':torch.cuda.memory_allocated(),'cuda_reserved_bytes':torch.cuda.memory_reserved()}
            clean=bool(cleanup and cleanup.get('passed') and (fake or cleanup.get('cuda_allocated_bytes')==cleanup.get('cuda_reserved_bytes')==0))
            if generation is not None and clean:
                try: control.close_worker(worker,generation,cast(Mapping[str,object],cleanup))
                except BaseException as exc: error={'execution':error,'closure':safe_error(exc)}; clean=False
            if not clean:
                if device_owner is not None: device_owner.quarantine()
                error=error or {'code':'cleanup_not_verified'}
            result: dict[str,object]={'schema':'aureon.replica_worker.v2','worker':worker,'generation':generation,
                'device':spec['device'],'passed':error is None,'completed':completed,'error':error,
                'cleanup':cleanup,'fake':fake,'elapsed_seconds':elapsed()-started,
                'scratch_retained':True,'GPU_qualification':False}
            write_file(folder/'worker-result.json',encoded(result))
    return result
