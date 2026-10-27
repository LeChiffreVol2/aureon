"""Self-hosted async Engine service; one configured model/environment per process."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from .control import Control,initialize
from ..cloud.control import write_file,encoded
from .ownership import exclusive,safe_error

def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0]=='supervise':
        from .supervisor import main as supervise_main
        return supervise_main(argv[1:])
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init');init.add_argument('--state',type=Path,required=True);init.add_argument('--credentials',type=Path,required=True)
    init.add_argument('--tenant',action='append',required=True)
    for name in ('model','environment','profile'):init.add_argument('--'+name,required=True)
    init.add_argument('--denoiser',choices=['native','graph'],default='graph');init.add_argument('--fake',action='store_true')
    init.add_argument('--qualification',type=Path);init.add_argument('--tenant-storage-reservation-bytes',type=int)
    for command in ('serve','worker'):
        run=sub.add_parser(command);run.add_argument('--state',type=Path,required=True);run.add_argument('--work',type=Path,required=True)
        modes=run.add_mutually_exclusive_group(required=True);modes.add_argument('--fake',action='store_true');modes.add_argument('--execute-model',action='store_true')
        run.add_argument('--bundle',type=Path);run.add_argument('--config',type=Path);run.add_argument('--resume',action='store_true',help='Explicitly reopen admission after a previous drain')
        run.add_argument('--max-seconds',type=int,default=3600);run.add_argument('--max-jobs',type=int,default=100)
        if command=='serve':
            run.add_argument('--port',type=int,default=8042);run.add_argument('--host',default='127.0.0.1');run.add_argument('--allowed-host',action='append')
            run.add_argument('--ready-file',type=Path)
            run.add_argument('--drain-seconds',type=int,default=30,help='Read-only download grace after worker termination')
    expire=sub.add_parser('expire');expire.add_argument('--state',type=Path,required=True)
    backup=sub.add_parser('backup');backup.add_argument('--state',type=Path,required=True);backup.add_argument('--output',type=Path,required=True);backup.add_argument('--credentials',type=Path,required=True)
    restore=sub.add_parser('restore');restore.add_argument('--backup',type=Path,required=True);restore.add_argument('--state',type=Path,required=True);restore.add_argument('--manifest-sha256',required=True)
    reclaim=sub.add_parser('reclaim');reclaim.add_argument('--state',type=Path,required=True);reclaim.add_argument('--work',type=Path,required=True)
    for name in ('key-create','key-revoke'):
        key=sub.add_parser(name);key.add_argument('--state',type=Path,required=True);key.add_argument('--tenant',required=True)
        if name=='key-create':key.add_argument('--output',type=Path,required=True)
        else:key.add_argument('--fingerprint',required=True)
    args=p.parse_args(argv)
    if args.command=='init':
        if args.credentials.exists() or not args.credentials.parent.is_dir():p.error('Fresh credentials path with existing parent required')
        keys=initialize(args.state,args.tenant,model=args.model,environment=args.environment,profile=args.profile,fake=args.fake,denoiser=args.denoiser,
          qualification=json.loads(args.qualification.read_bytes()) if args.qualification else None,tenant_storage_reservation_bytes=args.tenant_storage_reservation_bytes)
        from .security import write_secret
        write_secret(args.credentials,keys)
        print(json.dumps({'state':str(args.state),'credentials_file':str(args.credentials),'model_loaded':False}));return 0
    if args.command in ('backup','restore'):
        from . import operations
        result=operations.backup(args.state,args.output,args.credentials) if args.command=='backup' else operations.restore(args.backup,args.state,args.manifest_sha256)
        print(json.dumps(result));return 0
    with exclusive(args.state,'service.lock'):
        return _execute(args,p)

def _execute(args,p):
    control=Control(args.state)
    if args.command in ('expire','reclaim','key-create','key-revoke'):
        try:
            if args.command=='expire':result={'removed':control.expire()}
            elif args.command=='reclaim':
                from .operations import reclaim
                result=reclaim(control,args.work)
            elif args.command=='key-revoke':
                from .security import revoke_key
                result=revoke_key(control,args.tenant,args.fingerprint)
            else:
                from .security import create_key
                if args.output.exists() or not args.output.parent.is_dir():p.error('Fresh private key output required')
                from .security import write_secret
                result=create_key(control,args.tenant);write_secret(args.output,result)
                result={k:v for k,v in result.items() if k!='key'};result['key_file']=str(args.output)
            print(json.dumps(result));return 0
        finally:control.close()
    if not 1<=args.max_seconds<=86400 or not 1<=args.max_jobs<=10000:p.error('Use finite limits: seconds1..86400, jobs1..10000')
    if args.resume:control.set_meta('drain',False);control.set_meta('shutdown_requested',False)
    from .worker import work
    stop=threading.Event();results=[];server=None;thread=None;errors=[]
    def worker():
        try:results.append(work(control,args.work,execute_model=args.execute_model,fake=args.fake,bundle=args.bundle,config=args.config,
          max_seconds=args.max_seconds,max_jobs=args.max_jobs,stop=stop))
        except BaseException as exc:errors.append(safe_error(exc));stop.set()
        finally:control.set_meta('drain',True);stop.set()
    def drain(*_):control.set_meta('drain',True);control.set_meta('shutdown_requested',True);stop.set()
    old={s:signal.signal(s,drain) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        if args.command=='worker':worker()
        else:
            if not 0<=args.drain_seconds<=300:p.error('--drain-seconds must be0..300')
            from ..cloud.http import Server
            server=Server(control,args.port,host=args.host,allowed_hosts=args.allowed_host)
            http=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.1});http.start()
            thread=threading.Thread(target=worker,name='aureon-service-worker');thread.start()
            address={'url':f'http://127.0.0.1:{server.server_port}','pid':os.getpid(),'model_id':control.selected['model_id'],
              'readiness':'GET /v1/ready; binding is not model readiness'}
            if args.ready_file:write_file(args.ready_file,encoded(address))
            print(json.dumps(address),flush=True);deadline=time.monotonic()+args.max_seconds
            try:
                while time.monotonic()<deadline and not stop.wait(.1):
                    if control.meta('drain',False):break
            finally:
                stop.set();thread.join()
                grace=time.monotonic()+args.drain_seconds
                while time.monotonic()<grace and not control.meta('shutdown_requested',False):time.sleep(.1)
                server.shutdown();http.join();server.server_close()
        result={'passed':not errors and bool(results) and all(r['passed'] for r in results),'worker_results':results,'errors':errors}
        print(json.dumps(result),flush=True);return 0 if result['passed'] else 1
    finally:
        for s,handler in old.items():signal.signal(s,handler)
        control.close()
if __name__=='__main__':raise SystemExit(main())
