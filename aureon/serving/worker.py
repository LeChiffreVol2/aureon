"""One owned GPU worker; durable queues and fencing remain in cloud.Control."""
from __future__ import annotations
from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Unpack, cast
from .contracts import (AttemptLease, BackendFactory, BackendOptions, PathInput, SafeError,
                        WorkerStatus, WorkerStatusOptions)
if TYPE_CHECKING:
    from .backend import Backend
    from .control import Control
    from .devices import DeviceLease
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
import uuid
from ..cloud.control import require,CONTRACT,encoded,write_file
from .ownership import exclusive,process_identity,safe_error
from ..scene_api.durable import _plain_path
from .timing import elapsed,CLOCK

def scratch_bytes(root: PathInput) -> int:
    total=0
    for p in Path(root).rglob('*'):
        require(not p.is_symlink(),'scratch_symlink',500)
        if p.is_file():total+=p.stat().st_size
    return total

def work(control: Control, folder: PathInput, *, execute_model: bool = False, fake: bool = False,
         bundle: PathInput | None = None, config: PathInput | None = None, max_seconds: float = 300, max_jobs: int = 100,
         stop: threading.Event | None = None, backend_factory: BackendFactory | None = None, device: str | None = None) -> dict[str, object]:
    require(fake==(control.execution=='fake') and (fake or execute_model is True),'execution_mode_mismatch')
    if not fake and control.selected['task']=='generation':
        from ..terramind_native.offline import inspect_config
        require(config is not None,'generation_config_required')
        require(inspect_config(Path(cast(PathInput,config)))['model_id']==control.selected['model_id'],
                'model_config_mismatch')
    require(device is None or (not fake and control.selected['task'] in {'generation','generation_scene','features'}),
            'scoped_device_requires_resident_generation')
    require(1<=max_seconds<=86400 and 1<=max_jobs<=10000,'invalid_run_limits')
    folder=_plain_path(folder).absolute();require(not folder.exists() and folder.parent.is_dir(),'fresh_work_required')
    require(folder!=control.root and not folder.is_relative_to(control.root) and not control.root.is_relative_to(folder),'work_overlaps_state')
    if not fake:
        require(sys.platform=='linux' and not str(folder).startswith('/mnt/'),'linux_native_work_required')
        from .storage import require_local
        require_local(folder.parent);require_local(control.root)
    engine: Backend | None
    current: AttemptLease | None
    error: object
    cleanup: Mapping[str, object] | None
    stop=stop or threading.Event();started=elapsed();engine=None;current=None;completed=0;error=None;cleanup=None
    owner_id=uuid.uuid4().hex
    def status(**values: Unpack[WorkerStatusOptions]) -> WorkerStatus:return control.worker_status(owner_id,**values)
    from contextlib import ExitStack
    with ExitStack() as ownership:
        ownership.enter_context(exclusive(control.root))
        device_owner: Mapping[str,object]={'scope':'fake','uuid':None}
        if not fake:
            from .devices import device_lease,device_binding
            device_owner=ownership.enter_context(device_lease(device))
            ownership.enter_context(device_binding(device))
        folder.mkdir(mode=0o700)
        owner={'schema':'aureon.scratch_owner.v1','id':owner_id,'state_instance':control.meta('instance_id'),'path':str(folder),'process':process_identity()}
        write_file(folder/'ownership.json',encoded(owner))
        with control.tx() as db:db.execute('INSERT INTO service_scratch VALUES (?,?,?,"active",NULL)',(owner['id'],str(folder),encoded(owner).decode()))
        control.start_worker(owner_id,pid=os.getpid(),model_loaded=False,jobs_completed=0,device=device_owner,
                             execution_stage='idle',preparation_state='idle',elapsed_clock=CLOCK)
        pulse_stop=threading.Event();lost: list[SafeError]=[];lease_lock=threading.Lock()
        def pulse() -> None:
            while not pulse_stop.wait(5):
                try:
                    with lease_lock:
                        if current is not None:control.heartbeat(current)
                    status()
                except BaseException as exc:lost.append(safe_error(exc,'heartbeat_failed'));stop.set();return
        pulse_thread=threading.Thread(target=pulse,daemon=True);pulse_thread.start()
        prepare_stop=threading.Event();prepare_thread=None;preparations={'completed':0,'seconds':0.0}
        control._preparation_stop=prepare_stop
        def prepare() -> None:
            try:
                while not prepare_stop.is_set() and not stop.is_set():
                    # Clear before probing durable state so a concurrent commit
                    # cannot be lost between the empty probe and the wait.
                    control._preparation_ready.clear()
                    if control.meta('drain',False):break
                    if not control._has_preparation_work():
                        control._preparation_ready.wait(.1)
                        continue
                    status(preparation_state='checking')
                    tick=elapsed();prepared=control.prepare_one()
                    preparations['seconds']+=elapsed()-tick
                    if prepared:preparations['completed']+=1
                    status(preparation_state='idle')
                    if not prepared:control._preparation_ready.wait(.1)
            except BaseException as exc:
                lost.append(safe_error(exc,'preparation_failed'));stop.set()
        try:
            if not fake:
                from .backend import Backend
                options: BackendOptions={'bundle':bundle,'config':config,'execute_model':execute_model}
                if device is not None:options['device']=device
                engine=(backend_factory or Backend)(control,folder,**options)
                if 'workflow' in control.selected:engine.cancel_event=stop
            prepare_thread=threading.Thread(target=prepare,name='aureon-input-preparation')
            prepare_thread.start()
            status(state='available')
            while not stop.is_set() and elapsed()-started<max_seconds and completed<max_jobs:
                control._job_ready.clear()
                if control.meta('drain',False):break
                limits=control.selected.get('workflow',{}).get('limits',{})
                reserve=2*limits.get('max_output_bytes',1024**3)
                require(scratch_bytes(folder)+reserve<=limits.get('max_scratch_bytes',4*1024**3),'scratch_reserve_exhausted',507)
                require(shutil.disk_usage(folder).free>=CONTRACT['min_free_bytes']+reserve,'scratch_disk_floor',507)
                with lease_lock:current=control.claim(expected_owner=owner_id)
                if current:
                    status(state='running',execution_stage='execute',active_job=current['job_id'])
                    tick=elapsed()
                    if not fake:
                        files=cast(Backend,engine).run(current)
                        require(not lost,'worker_lease_lost',409)
                    execution_seconds=elapsed()-tick
                    status(execution_stage='publish')
                    publish_started=elapsed()
                    with lease_lock:
                        if fake:control.publish_fake(current)
                        else:control.publish_engine(current,files)
                        finished_lease=current;current=None
                    with (folder/'jobs.jsonl').open('a') as f:f.write(json.dumps({'job_id':finished_lease['job_id'],'attempt':finished_lease['attempt'],
                      'execution_seconds':execution_seconds,'publication_seconds':elapsed()-publish_started,
                      'execution_and_publication_seconds':elapsed()-tick,'elapsed_clock':CLOCK,
                      'scratch_bytes':scratch_bytes(folder)})+'\n')
                    completed+=1;status(state='available',execution_stage='idle',active_job=None,jobs_completed=completed,model_loaded=not fake and cast(Backend,engine).session is not None)
                else:control._job_ready.wait(.1)
            require(not lost,'worker_background_failed',500)
        except BaseException as exc:
            error=safe_error(exc);traceback.clear_frames(cast(TracebackType,sys.exc_info()[2]))
            if current:
                try:control.fail(current)
                except Exception:pass
        finally:
            # Terminal workers close admission before releasing their ownership.
            # Read-only HTTP delivery can remain alive for a bounded drain grace.
            control.set_meta('drain',True)
            prepare_stop.set()
            # Never release state/device ownership with a live parser thread.
            # Parser cancellation kills only its owned child; supervisor bounds stalls.
            if prepare_thread is not None:prepare_thread.join()
            control._preparation_stop=None
            pulse_stop.set();pulse_thread.join()
            if lost and error is None:error={'code':'worker_background_failed','causes':list(lost)}
            if engine is not None:
                try:cleanup=engine.close()
                except BaseException as exc:error={'execution':error,'cleanup':safe_error(exc,'cleanup_failed')}
            if not fake and (cleanup is None or not cleanup.get('passed')
                             or cleanup.get('cuda_allocated_bytes')!=0 or cleanup.get('cuda_reserved_bytes')!=0):
                cast(DeviceLease,device_owner).quarantine()
                if error is None:error={'code':'cleanup_not_verified'}
            status(terminal=True,state='failed' if error else 'closed',model_loaded=False,device=device_owner,
                   execution_stage='closed',preparation_state='closed')
            result={'schema':'aureon.async_worker.v1','completed':completed,'error':error,'cleanup':cleanup,'elapsed_seconds':elapsed()-started,
              'elapsed_clock':CLOCK,'device':device_owner,'worker_id':owner_id,'preparation':preparations,
              'stages':{'preparation_workers':1,'gpu_workers':1,'publication_workers':1,'publication_overlap':False},
              'scope':'Worker startup through owned close; HTTP/process lifecycle is separate','passed':error is None,'fake':fake}
            write_file(folder/'worker-result.json',encoded(result))
            with control.tx() as db:db.execute('UPDATE service_scratch SET state=?,receipt=? WHERE id=?',('closed' if result['passed'] else 'failed',encoded(result).decode(),owner['id']))
    return result
