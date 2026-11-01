"""Finite Linux process supervisor with explicit capped restart and fresh scratch."""
from __future__ import annotations
from collections.abc import Sequence
from typing import cast
from .contracts import PathInput, ProcessIdentity, SupervisorAttempt, Termination
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from ..cloud.control import require,encoded,write_file,CONTRACT
from ..scene_api.durable import _plain_path
from .ownership import exclusive,process_identity,alive,safe_error
from .timing import elapsed as elapsed_now,CLOCK

def members(identity: ProcessIdentity | None) -> list[ProcessIdentity]:
    # A child can exit before /proc attestation. Keep this an error, never infer
    # ownership from its recycled PID or send signals without start-time proof.
    if identity is None:
        raise TypeError('Process identity is unavailable')
    start_ticks=identity['start_ticks']
    if start_ticks is None:
        raise TypeError('Linux process start ticks are required')
    rows=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:
            values=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(values[2])==identity['pid'] and int(values[3])==identity['pid'] and int(values[19])>=int(start_ticks):
                row=process_identity(int(path.name))
                if row and row['state']!='Z':rows.append(row)
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return rows

def terminate_owned(identity: ProcessIdentity | None, grace: float) -> Termination:
    rows=members(identity)
    for row in rows:
        if alive(row):
            try:os.kill(row['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=elapsed_now()+grace
    while elapsed_now()<deadline and members(identity):time.sleep(.05)
    killed=members(identity)
    for row in killed:
        if alive(row):
            try:os.kill(row['pid'],signal.SIGKILL)
            except ProcessLookupError:pass
    return {'term_targets':rows,'kill_targets':killed,'forced':bool(killed)}

def bytes_used(root: PathInput) -> int:
    size=0;count=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:_plain_path(Path(directory)/name)
        for name in files:
            p=Path(directory)/name
            try:size+=p.stat().st_size;count+=1
            except FileNotFoundError:pass
            require(count<=10000,'supervisor_file_limit',507)
    return size

def supervise(state: PathInput, work: PathInput, *, fake: bool = False, execute_model: bool = False,
              bundle: PathInput | None = None, config: PathInput | None = None, max_seconds: float = 3600, max_jobs: int = 100,
              max_restarts: int = 0, backoff: float = 2, stop_grace: float = 30, drain_seconds: float = 30,
              port: int = 8042, host: str = '127.0.0.1', allowed_hosts: Sequence[str] | None = None,
              resume: bool = False, ready_file: PathInput | None = None, device: str | None = None) -> dict[str, object]:
    require(sys.platform=='linux','linux_supervisor_required')
    require(type(max_restarts) is int and 0<=max_restarts<=2 and 1<=backoff<=30 and 1<=stop_grace<=300,'invalid_restart_policy')
    require(1<=max_seconds<=86400 and 1<=max_jobs<=10000 and 0<=drain_seconds<=300,'invalid_run_limits')
    require(fake != execute_model,'explicit_execution_mode')
    state=_plain_path(state).resolve(strict=True);work=_plain_path(work)
    require(not work.exists() and work.parent.is_dir() and not work.is_relative_to(state) and not state.is_relative_to(work),'fresh_external_supervisor_work')
    from .storage import require_local
    if not fake:require_local(state);require_local(work.parent)
    from .control import Control
    start=elapsed_now();stop=False;records: list[SupervisorAttempt]=[];forced=False;completed=0;failed_jobs=0
    def request_stop(*_: object) -> None:
        nonlocal stop
        stop=True
    prior={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    result: dict[str,object]={'schema':'aureon.service_supervision.v1','passed':False,'attempts':records,'error':None,'model_execute_authorized':execute_model,
      'elapsed_clock':CLOCK,'requested_device':device,
      'limits':{'seconds':max_seconds,'jobs':max_jobs,'restarts':max_restarts,'backoff_seconds':backoff,'shutdown_grace_seconds':stop_grace,'scratch_bytes':4*1024**3}}
    try:
        with exclusive(state,'supervisor.lock'):
            work.mkdir(mode=0o700);write_file(work/'supervisor-owner.json',encoded({'id':uuid.uuid4().hex,'process':process_identity(),'state':str(state),'root':str(work)}))
            control=Control(state)
            try:
                baseline_usage=control.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0]
                baseline_failures=control.db.execute('SELECT COUNT(*) FROM jobs WHERE state="failed"').fetchone()[0]
            finally:control.close()
            for index in range(max_restarts+1):
                remaining=max_seconds-(elapsed_now()-start)
                if stop or remaining<1 or completed>=max_jobs:break
                require(bytes_used(work)+2*1024**3<=4*1024**3,'supervisor_scratch_budget',507)
                control=Control(state)
                try:
                    if index and control.meta('shutdown_requested',False):break
                finally:control.close()
                child_work=work/('worker-'+str(index));log=work/('worker-'+str(index)+'.log')
                args=['serve','--state',str(state),'--work',str(child_work),'--max-seconds',str(max(1,int(remaining))),
                      '--max-jobs',str(max_jobs-completed),'--drain-seconds',str(drain_seconds),'--port',str(port),'--host',host,
                      '--fake' if fake else '--execute-model']
                if index or resume:args+=['--resume']
                if bundle:args+=['--bundle',str(bundle)]
                if config:args+=['--config',str(config)]
                if device:args+=['--device',device]
                target_ready=Path(ready_file) if ready_file else None
                if target_ready and index:target_ready=target_ready.with_name(target_ready.stem+'-restart-'+str(index)+target_ready.suffix)
                if target_ready:args+=['--ready-file',str(target_ready)]
                for value in allowed_hosts or []:args+=['--allowed-host',value]
                code="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('aureon.serving',run_name='__main__')"
                command=[sys.executable,'-I','-B','-c',code,str(Path(__file__).resolve().parents[2]),*args]
                tick=elapsed_now();record: SupervisorAttempt={'index':index,'fresh_work':str(child_work),'started_seconds':tick-start,'termination':None,'ready_file':None if target_ready is None else str(target_ready)}
                with log.open('xb') as output:
                    child=subprocess.Popen(command,cwd=work,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                    identity=process_identity(child.pid);record['process']=identity
                    try:
                        while child.poll() is None:
                            size=bytes_used(work)
                            if stop or elapsed_now()-start>=max_seconds or size>4*1024**3 or log.stat().st_size>16*1024**2:
                                record['termination']=terminate_owned(identity,stop_grace);forced|=cast(Termination,record['termination'])['forced'];break
                            time.sleep(.1)
                        child.wait(timeout=5)
                    finally:
                        if child.poll() is None:record['termination']=terminate_owned(identity,0);forced=True;child.wait(timeout=5)
                        if members(identity):record['descendant_cleanup']=terminate_owned(identity,0);forced=True
                cast(dict[str,object],record).update(exit_code=child.returncode,elapsed_seconds=elapsed_now()-tick,scratch_bytes=bytes_used(work))
                if (child_work/'worker-result.json').is_file():
                    receipt=json.loads((child_work/'worker-result.json').read_bytes());record['worker_result']=receipt
                control=Control(state)
                try:
                    completed=control.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0]-baseline_usage
                    failed_jobs=control.db.execute('SELECT COUNT(*) FROM jobs WHERE state="failed"').fetchone()[0]-baseline_failures
                finally:control.close()
                records.append(record)
                write_file(work/('attempt-'+str(index)+'.json'),encoded(record))
                if child.returncode==0 or stop or record['termination'] or index==max_restarts:break
                until=min(start+max_seconds,elapsed_now()+backoff*(index+1))
                while not stop and elapsed_now()<until:time.sleep(.1)
            result['passed']=bool(records) and records[-1]['exit_code']==0 and not forced and failed_jobs==0
    except Exception as exc:result['error']=safe_error(exc,'supervisor_failed')
    finally:
        for sig,handler in prior.items():signal.signal(sig,handler)
        elapsed=elapsed_now()-start
        result.update(completed_durable_jobs=completed,failed_jobs_delta=failed_jobs,forced_termination=forced,elapsed_seconds=elapsed,interrupted=stop,cost_THB_at_1_per_hour=elapsed/3600)
        if work.exists():write_file(work/'supervisor-result.json',encoded(result))
    return result

def main(argv: Sequence[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['state','work']:p.add_argument('--'+key,type=Path,required=True)
    for key in ['bundle','config','ready-file']:p.add_argument('--'+key,type=Path)
    p.add_argument('--device',help='Full GPU UUID for one resident generation worker process')
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--fake',action='store_true');group.add_argument('--execute-model',action='store_true')
    for key,default in [('max-seconds',3600),('max-jobs',100),('max-restarts',0),('backoff',2),('stop-grace',30),('drain-seconds',30),('port',8042)]:p.add_argument('--'+key,type=int,default=default)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--allowed-host',dest='allowed_hosts',action='append');p.add_argument('--resume',action='store_true')
    result=supervise(**vars(p.parse_args(argv)));print(json.dumps(result));return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
