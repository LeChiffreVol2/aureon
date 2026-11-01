"""Explicit v2 local replica commands. Existing one-worker commands stay unchanged."""
from __future__ import annotations
from collections.abc import Sequence
from aureon.serving.contracts import PathInput
import argparse
import json
from pathlib import Path
import signal
import threading

from .replica_control import ReplicaControl, initialize
from .ownership import exclusive
from .timing import elapsed
from ..cloud.control import encoded,write_file


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init'); init.add_argument('--state',type=Path,required=True)
    init.add_argument('--resources',type=Path,required=True); init.add_argument('--credentials',type=Path,required=True)
    init.add_argument('--tenant',action='append',required=True); init.add_argument('--config',type=Path)
    init.add_argument('--limits',type=Path,help='Explicit finite capacity JSON; creates new replica v3 state. Omit for legacy v2 defaults.')
    init.add_argument('--fake',action='store_true')
    serve=sub.add_parser('serve'); serve.add_argument('--state',type=Path,required=True)
    serve.add_argument('--port',type=int,default=8042); serve.add_argument('--host',default='127.0.0.1')
    serve.add_argument('--allowed-host',action='append'); serve.add_argument('--ready-file',type=Path)
    serve.add_argument('--max-seconds',type=int,default=3600); serve.add_argument('--resume',action='store_true')
    worker=sub.add_parser('worker'); worker.add_argument('--state',type=Path,required=True)
    worker.add_argument('--worker',required=True); worker.add_argument('--work',type=Path,required=True)
    worker.add_argument('--config',type=Path); worker.add_argument('--max-seconds',type=int,default=300)
    worker.add_argument('--max-jobs',type=int,default=100)
    mode=worker.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fake',action='store_true'); mode.add_argument('--execute-model',action='store_true')
    recover=sub.add_parser('recover'); recover.add_argument('--state',type=Path,required=True)
    recover.add_argument('--worker',required=True); recover.add_argument('--generation',required=True)
    reclaim=sub.add_parser('reclaim-scratch',help='Offline explicit retirement of one closed registered generation; no automatic cleanup.')
    reclaim.add_argument('--state',type=Path,required=True);reclaim.add_argument('--worker',required=True)
    reclaim.add_argument('--generation',required=True);reclaim.add_argument('--work-root',type=Path,required=True)
    backup=sub.add_parser('backup',help='Offline same-host replica snapshot; drains and preserves source.')
    backup.add_argument('--state',type=Path,required=True);backup.add_argument('--destination',type=Path,required=True)
    backup.add_argument('--credentials',type=Path,required=True)
    restore=sub.add_parser('restore',help='Verified same-host handover; retires original authority before publication.')
    restore.add_argument('--source',type=Path,required=True);restore.add_argument('--destination',type=Path,required=True)
    restore.add_argument('--manifest-sha256',required=True)
    for name in ('status','expire'):
        command=sub.add_parser(name); command.add_argument('--state',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='init':
        if args.credentials.exists() or not args.credentials.parent.is_dir(): parser.error('Fresh credentials path required')
        keys=initialize(args.state,args.tenant,resources=json.loads(args.resources.read_bytes()),config=args.config,fake=args.fake,
            limits=None if args.limits is None else json.loads(args.limits.read_bytes()))
        from .security import write_secret
        write_secret(args.credentials,keys)
        print(json.dumps({'state':str(args.state),'credentials':str(args.credentials),'schema':json.loads((args.state/'contract.json').read_bytes())['schema'],'model_executed':False})); return 0
    if args.command in ('reclaim-scratch','backup','restore'):
        from . import replica_operations
        if args.command=='reclaim-scratch':
            receipt=replica_operations.reclaim(args.state,args.worker,args.generation,work_root=args.work_root)
        elif args.command=='backup':receipt=replica_operations.backup(args.state,args.destination,args.credentials)
        else:receipt=replica_operations.restore(args.source,args.destination,args.manifest_sha256)
        print(json.dumps(receipt));return 0
    control=ReplicaControl(args.state); stop=threading.Event()
    if args.command in ('worker','serve') and not 1<=args.max_seconds<=86400: parser.error('Finite seconds1..86400 required')
    def signal_stop(*_: object) -> None: stop.set()
    old={s:signal.signal(s,signal_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        if args.command=='worker':
            from .replica_worker import work
            value=work(control,args.worker,args.work,config=args.config,fake=args.fake,execute_model=args.execute_model,
                max_seconds=args.max_seconds,max_jobs=args.max_jobs,stop=stop)
            print(json.dumps(value)); return 0 if value['passed'] else 1
        if args.command=='recover':
            print(json.dumps({'recovered':control.recover(args.worker,args.generation)})); return 0
        if args.command=='status': print(json.dumps(control.worker_snapshot())); return 0
        if args.command=='expire':
            with exclusive(args.state,'retention.lock'): print(json.dumps({'removed':control.expire()}))
            return 0
        # One parser/controller process; workers never duplicate its 4GiB budget.
        with exclusive(args.state,'service.lock'):
            control._require_active_authority()
            if args.resume: control.set_meta('drain',False); control.set_meta('shutdown_requested',False)
            from ..cloud.http import Server
            server=Server(control,args.port,host=args.host,allowed_hosts=args.allowed_host)
            thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.1}); thread.start()
            control._preparation_stop=stop
            address={'url':f'http://127.0.0.1:{server.server_port}','schema':control.contract['schema'],'worker_launch':'separate explicit replicas worker process per configured GPU'}
            if args.ready_file: write_file(args.ready_file,encoded(address))
            print(json.dumps(address),flush=True); deadline=elapsed()+args.max_seconds
            try:
                while not stop.is_set() and elapsed()<deadline and not control.meta('drain',False):
                    if not control.prepare_one(): stop.wait(.1)
            finally:
                control.set_meta('drain',True); stop.set()
                server.shutdown(); thread.join(); server.server_close()
        return 0
    finally:
        for sig,handler in old.items(): signal.signal(sig,handler)
        control.close()


if __name__=='__main__': raise SystemExit(main())
