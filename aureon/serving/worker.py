"""One owned GPU worker; durable queues and fencing remain in cloud.Control."""
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

def scratch_bytes(root):
    total=0
    for p in Path(root).rglob('*'):
        require(not p.is_symlink(),'scratch_symlink',500)
        if p.is_file():total+=p.stat().st_size
    return total

def work(control,folder,*,execute_model=False,fake=False,bundle=None,config=None,max_seconds=300,max_jobs=100,
         stop=None,backend_factory=None):
    require(fake==(control.execution=='fake') and (fake or execute_model is True),'execution_mode_mismatch')
    folder=_plain_path(folder).absolute();require(not folder.exists() and folder.parent.is_dir(),'fresh_work_required')
    require(folder!=control.root and not folder.is_relative_to(control.root) and not control.root.is_relative_to(folder),'work_overlaps_state')
    if not fake:
        require(sys.platform=='linux' and not str(folder).startswith('/mnt/'),'linux_native_work_required')
        from .storage import require_local
        require_local(folder.parent);require_local(control.root)
    stop=stop or threading.Event();started=time.monotonic();engine=None;current=None;completed=0;error=None;cleanup=None
    state={'state':'starting','pid':os.getpid(),'model_loaded':False,'heartbeat':control.clock(),'jobs_completed':0}
    def status(**values):
        state.update(values,heartbeat=control.clock());control.set_meta('worker',dict(state))
    from contextlib import ExitStack
    with ExitStack() as ownership:
        ownership.enter_context(exclusive(control.root))
        if not fake:
            # Conservative per-service-user host lock: all AUREON GPUs serialize.
            # Separate OS users/containers need an operator-shared lock mount.
            gpu_parent=Path(os.environ.get('AUREON_GPU_LOCK_DIR','/tmp'))
            ownership.enter_context(exclusive(gpu_parent,'aureon-gpu-'+str(os.getuid())+'.lock'))
        folder.mkdir(mode=0o700)
        owner={'schema':'aureon.scratch_owner.v1','id':uuid.uuid4().hex,'state_instance':control.meta('instance_id'),'path':str(folder),'process':process_identity()}
        write_file(folder/'ownership.json',encoded(owner))
        with control.tx() as db:db.execute('INSERT INTO service_scratch VALUES (?,?,?,"active",NULL)',(owner['id'],str(folder),encoded(owner).decode()))
        pulse_stop=threading.Event();lost=[];lease_lock=threading.Lock()
        def pulse():
            while not pulse_stop.wait(5):
                try:
                    with lease_lock:
                        if current is not None:control.heartbeat(current)
                    status()
                except BaseException as exc:lost.append(safe_error(exc,'heartbeat_failed'));stop.set();return
        pulse_thread=threading.Thread(target=pulse,daemon=True);pulse_thread.start()
        try:
            if not fake:
                from .backend import Backend
                engine=(backend_factory or Backend)(control,folder,bundle=bundle,config=config,execute_model=execute_model)
            status(state='available')
            while not stop.is_set() and time.monotonic()-started<max_seconds and completed<max_jobs:
                if control.meta('drain',False):break
                prepared=control.prepare_one()
                require(scratch_bytes(folder)+2*1024**3<=4*1024**3,'scratch_reserve_exhausted',507)
                require(shutil.disk_usage(folder).free>=CONTRACT['min_free_bytes']+2*1024**3,'scratch_disk_floor',507)
                with lease_lock:current=control.claim()
                if current:
                    status(state='running')
                    tick=time.monotonic()
                    if not fake:
                        files=engine.run(current)
                        require(not lost,'worker_lease_lost',409)
                    with lease_lock:
                        if fake:control.publish_fake(current)
                        else:control.publish_engine(current,files)
                        finished_lease=current;current=None
                    with (folder/'jobs.jsonl').open('a') as f:f.write(json.dumps({'job_id':finished_lease['job_id'],'attempt':finished_lease['attempt'],
                      'execution_and_publication_seconds':time.monotonic()-tick,'scratch_bytes':scratch_bytes(folder)})+'\n')
                    completed+=1;status(state='available',jobs_completed=completed,model_loaded=not fake and engine.session is not None)
                elif not prepared:stop.wait(.1)
        except BaseException as exc:
            error=safe_error(exc);traceback.clear_frames(sys.exc_info()[2])
            if current:
                try:control.fail(current)
                except Exception:pass
        finally:
            # Terminal workers close admission before releasing their ownership.
            # Read-only HTTP delivery can remain alive for a bounded drain grace.
            control.set_meta('drain',True)
            pulse_stop.set();pulse_thread.join(6)
            if engine is not None:
                try:cleanup=engine.close()
                except BaseException as exc:error={'execution':error,'cleanup':safe_error(exc,'cleanup_failed')}
            status(state='failed' if error else 'closed',model_loaded=False)
            result={'schema':'aureon.async_worker.v1','completed':completed,'error':error,'cleanup':cleanup,'elapsed_seconds':time.monotonic()-started,
              'scope':'Worker startup through owned close; HTTP/process lifecycle is separate','passed':error is None,'fake':fake}
            write_file(folder/'worker-result.json',encoded(result))
            with control.tx() as db:db.execute('UPDATE service_scratch SET state=?,receipt=? WHERE id=?',('closed' if result['passed'] else 'failed',encoded(result).decode(),owner['id']))
    return result
