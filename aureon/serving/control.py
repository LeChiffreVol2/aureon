"""Single-host service routing over the existing durable/tenant control substrate."""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
import sqlite3
import threading
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack, cast, overload
from ..cloud.contracts import ControlContract, Qualification, ServiceRoute, UploadBody, UploadFile
from .contracts import (ArtifactFiles, AttemptLease, BaseLease, BaseRequest, PathInput,
                        Readiness, ServiceMetrics, ServiceRequest, WorkerStatus, WorkerUpdate)
if TYPE_CHECKING:
    from ..cloud.http import Handler
    from .workflow_config import WorkflowBinding

class ControlOptions(TypedDict, total=False):
    clock: Callable[[], float]
    qualification_verifier: Callable[[Qualification], Qualification] | None
from copy import deepcopy
import json
import os
from pathlib import Path
import secrets
import time
from ..cloud.control import (Control as BaseControl, initialize as base_initialize,
    CONTRACT, require, encoded, digest, file_hash, write_file, tenant_storage_limit)
from ..scene_api.durable import _plain_path, _sync_dir
from . import inputs
from . import security

GENERATION='terramind-v1-base'
from ..terramind_native.families import GENERATION_MODELS

def route(model: str, environment: str, profile: str, *, fake: bool = False,
          denoiser: Literal['native', 'graph'] | None = None, qualification: Qualification | None = None,
          workflow: WorkflowBinding | None = None) -> ServiceRoute:
    from ..api import inspect_model
    spec=inspect_model(model);task: Literal['generation','scene']='generation' if model in GENERATION_MODELS else 'scene'
    if workflow is not None:
        from .workflow_config import PROFILES
        require(workflow['model_id']==model and profile==PROFILES[workflow['task']] and environment=='generation'
                and qualification is None and denoiser is None, 'workflow_route_mismatch')
        return {'model_id':model,'task':workflow['task'],'environment':environment,'profile_id':profile,
                'reuse':None,'denoiser':None,'execution':'fake' if fake else 'engine',
                'qualification_sha256':None,'workflow':deepcopy(workflow)}
    require(model in GENERATION_MODELS or model.startswith('kao-terramind-'), 'unsupported_route')
    if denoiser is None:
        denoiser='graph' if model==GENERATION else 'native'
    if task=='generation':
        require(environment=='generation' and profile=='fp32-generation-v1' and denoiser in ('native','graph') and qualification is None,'unsupported_route')
    else:
        require(environment in spec['full_reference_environments'],'unsupported_environment')
        require(profile!='fp32-full-reference-v1' or qualification is None,'full_reference_has_no_qualification')
        if profile!='fp32-full-reference-v1':
            require(model=='kao-terramind-b' and environment=='scene' and qualification is not None,'qualification_required')
            from ..cloud.profiles import verify_qualification
            verified=verify_qualification(cast(Qualification,qualification));require(verified['profile_id']==profile,'qualification_profile')
    return {'model_id':model,'task':task,'environment':environment,'profile_id':profile,'reuse':('auto' if model==GENERATION and denoiser=='graph' else 'off') if task=='generation' else None,
      'denoiser':denoiser if task=='generation' else None,'execution':'fake' if fake else 'engine',
      'qualification_sha256':None if qualification is None else digest(encoded(qualification))}

def initialize(root: PathInput, tenants: Sequence[str], *, model: str, environment: str, profile: str,
               fake: bool = False, denoiser: Literal['native', 'graph'] | None = None, qualification: Qualification | None = None,
               tenant_storage_reservation_bytes: int | None = None, tenant_capacity: int = 15,
               workflow_config: PathInput | None = None) -> dict[str, str]:
    require(type(tenant_capacity) is int and 1<=tenant_capacity<2**31,'invalid_tenant_capacity')
    require('operator' not in tenants and len(tenants)<=tenant_capacity,'reserved_operator_tenant')
    from .workflow_config import load, binding
    configured=None if workflow_config is None else load(workflow_config)
    selected=route(model,environment,profile,fake=fake,denoiser=denoiser,qualification=qualification,
                   workflow=None if configured is None else binding(configured))
    keys=base_initialize(root,[*tenants,'operator'],tenant_storage_reservation_bytes=tenant_storage_reservation_bytes,tenant_capacity=tenant_capacity+1)
    root=Path(root);contract=json.loads((root/'contract.json').read_bytes())
    contract.update(schema='aureon.async_service.v1',model=model,profile=profile,execution=selected['execution'],route=selected)
    if configured is not None:
        contract.update(max_artifact_bytes=configured['limits']['max_artifact_bytes'],max_result_bytes=configured['limits']['max_output_bytes'],
                        request_deadline_seconds=configured['limits']['request_deadline_seconds'])
        write_file(root/'workflow-config.json',encoded(configured))
    # Initialization has no listeners/consumers yet; publish the new immutable contract.
    write_file(root/'contract.new',encoded(contract));os.replace(root/'contract.new',root/'contract.json');_sync_dir(root)
    if qualification is not None:write_file(root/'engine-qualification.json',encoded(qualification))
    import sqlite3
    db=sqlite3.connect(root/'control.sqlite');db.executescript('''
      CREATE TABLE service_requests(job TEXT PRIMARY KEY, body TEXT NOT NULL);
      CREATE TABLE service_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      INSERT INTO service_meta VALUES ('drain','false');
    ''');db.close()
    return keys

class Control(BaseControl):
    _rate: dict[str, tuple[float, float]]
    _preparation_stop: threading.Event | None
    _expected_worker_owner: str | None

    def __init__(self, *args: Unpack[tuple[PathInput]], **kwargs: Unpack[ControlOptions]) -> None:
        super().__init__(*args,**kwargs)
        self._rate={}
        # Hints for the embedded HTTP/worker pair. Durable queues remain the
        # authority; independent processes still use the bounded polling path.
        self._preparation_ready=threading.Event()
        self._job_ready=threading.Event()
        security.schema(self)

    def _expected_contract(self, verifier: Callable[[Qualification], Qualification] | None) -> ControlContract:
        current=self.contract;r: ServiceRoute=current.get('route',cast(ServiceRoute,{}))
        qualification=None
        if r.get('qualification_sha256'):
            qualification=json.loads((self.root/'engine-qualification.json').read_bytes())
        workflow=None
        if 'workflow' in r:
            from .workflow_config import load, binding
            workflow=binding(load(self.root/'workflow-config.json'))
        selected=route(cast(str,r.get('model_id')),cast(str,r.get('environment')),cast(str,r.get('profile_id')),fake=r.get('execution')=='fake',
                       denoiser=(r.get('denoiser') or 'native') if workflow is None else r.get('denoiser'),qualification=qualification,workflow=workflow)
        expected: ControlContract={**CONTRACT,'schema':'aureon.async_service.v1','model':selected['model_id'],'profile':selected['profile_id'],
          'execution':selected['execution'],'route':selected,'max_tenant_storage_reservation_bytes':self._tenant_storage_limit(current.get('max_tenant_storage_reservation_bytes'))}
        if workflow is not None:
            expected['max_artifact_bytes']=workflow['limits']['max_artifact_bytes'];expected['max_result_bytes']=workflow['limits']['max_output_bytes']
            expected['request_deadline_seconds']=workflow['limits']['request_deadline_seconds']
        return expected

    def _tenant_storage_limit(self, value: object) -> int:return tenant_storage_limit(value)

    def tenant_capacity(self) -> int:return 15

    @property
    def selected(self) -> ServiceRoute:return self.contract['route']

    @overload
    def meta(self, key: Literal['worker'], default: WorkerStatus) -> WorkerStatus: ...
    @overload
    def meta(self, key: Literal['drain', 'shutdown_requested'], default: bool = False) -> bool: ...
    @overload
    def meta(self, key: Literal['instance_id', 'replica_fair_cursor'], default: str = '') -> str: ...
    @overload
    def meta(self, key: str, default: object = None) -> object: ...
    def meta(self, key: str, default: object = None) -> object:
        with self.lock:row=self.db.execute('SELECT value FROM service_meta WHERE key=?',(key,)).fetchone()
        return default if row is None else json.loads(row['value'])

    def set_meta(self, key: str, value: object) -> None:
        with self.tx() as db:db.execute('INSERT INTO service_meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,encoded(value).decode()))

    def start_worker(self, owner: str, **values: Unpack[WorkerUpdate]) -> None:
        """Caller must already hold worker.lock; one owner per durable state."""
        require(isinstance(owner,str) and len(owner)==32,'invalid_worker_owner')
        value={**values,'owner_id':owner,'heartbeat':self.clock(),'state':'starting'}
        self.set_meta('worker',value)

    def worker_status(self, owner: str, *, terminal: bool = False, **values: Unpack[WorkerUpdate]) -> WorkerStatus:
        """Fence delayed heartbeat/status updates from a previously closed owner."""
        require('owner_id' not in values and 'heartbeat' not in values,'immutable_worker_identity')
        with self.tx() as db:
            row=db.execute('SELECT value FROM service_meta WHERE key="worker"').fetchone()
            current={} if row is None else json.loads(row[0])
            require(current.get('owner_id')==owner and current.get('state') not in ('closed','failed'),
                    'stale_worker_owner',409)
            if terminal:
                require(values.get('state') in ('closed','failed'),'terminal_worker_state_required')
                db.execute('UPDATE service_meta SET value="true" WHERE key="drain"')
            else:require(values.get('state') not in ('closed','failed'),'terminal_worker_update_required')
            current.update(values,heartbeat=self.clock())
            db.execute('UPDATE service_meta SET value=? WHERE key="worker"',(encoded(current).decode(),))
        return cast(WorkerStatus,current)

    def worker_snapshot(self) -> dict[str, object]:
        """Operator-only placement input; does not claim or route a job."""
        with self.lock:
            worker=self.meta('worker',{})
            counts={r['state']:r['n'] for r in self.db.execute(
                'SELECT state,COUNT(*) AS n FROM jobs WHERE state IN ("queued","running") GROUP BY state')}
            ready=(not self.meta('drain',False) and worker.get('state') in ('available','running')
                   and worker.get('heartbeat',0)+20>self.clock())
        from .timing import elapsed,CLOCK
        return {'schema':'aureon.worker_snapshot.v1','worker_id':self.meta('instance_id'),'generation':worker.get('owner_id'),
                'state_instance':self.meta('instance_id'),'route':dict(self.selected),'available':ready,
                'device':worker.get('device'),'model_loaded':worker.get('model_loaded',False),
                'queued':counts.get('queued',0),'running':counts.get('running',0),
                'admission_slots':max(0,CONTRACT['max_active']-sum(counts.values())) if ready else 0,
                'serial_gpu':True,'placement_only':True,'distributed_dispatch':False,
                'stages':{'preparation':worker.get('preparation_state'),'execution':worker.get('execution_stage')},
                'heartbeat':worker.get('heartbeat'),'measured_at':elapsed(),'elapsed_clock':CLOCK,
                'resource_estimates':None,'execution_key':None}

    def _admit(self, db: sqlite3.Connection) -> None:
        require(db.execute('SELECT value FROM service_meta WHERE key="drain"').fetchone()[0]=='false','service_draining',503)

    def _can_claim(self, db: sqlite3.Connection) -> bool:
        owner=getattr(self,'_expected_worker_owner',None)
        if owner is not None:
            row=db.execute('SELECT value FROM service_meta WHERE key="worker"').fetchone()
            worker={} if row is None else json.loads(row[0])
            require(worker.get('owner_id')==owner and worker.get('state') in ('available','running')
                    and worker.get('heartbeat',0)+20>self.clock(),'stale_worker_owner',409)
        return cast(bool,db.execute('SELECT value FROM service_meta WHERE key="drain"').fetchone()[0]=='false')

    def create_input(self, tenant: str, body: UploadBody) -> dict[str, object]:
        require(tenant!='operator','operator_not_customer',403)
        return super().create_input(tenant,body)

    def complete_input(self, tenant: str, identifier: str) -> dict[str, object]:
        result=super().complete_input(tenant,identifier)
        if result['state']=='verifying':self._preparation_ready.set()
        return result

    def submit(self, tenant: str, body: BaseRequest, key: str) -> dict[str, object]:
        result=super().submit(tenant,body,key)
        self._job_ready.set()
        return result

    def _has_preparation_work(self) -> bool:
        """Read-only idle probe; claim, expiry and validation stay transactional."""
        now=self.clock()
        with self.lock:
            row=self.db.execute('SELECT 1 FROM inputs WHERE (state="verifying" AND expires>?) '
                'OR (state="validating" AND lease<=?) LIMIT 1',(now,now)).fetchone()
        return row is not None

    def _validate_input(self, body: UploadBody) -> list[UploadFile]:
        files=inputs.validate(body,self.selected['task'])
        if 'workflow' in self.selected:
            require(sum(f['bytes'] for f in files)<=self.selected['workflow']['limits']['max_input_bytes'], 'workflow_input_budget',413)
        return files
    def _inspect_uploaded_tiff(self, path: Path) -> None:
        pass  # Magic/hash checked in the parent; GDAL runs only in the bounded child.

    def _prepare_input(self, body: UploadBody, folder: Path, token: str, identifier: str) -> Path:
        from .parser import run
        if 'workflow' in self.selected:
            return run(body,folder,token,self.selected['task'],stop=getattr(self,'_preparation_stop',None),workflow=self.selected['workflow'])
        return run(body,folder,token,self.selected['task'],stop=getattr(self,'_preparation_stop',None))

    def request_admission(self, tenant: str) -> None:security.admit_request(self,tenant)

    def expire(self) -> list[dict[str, str]]:
        from .operations import expire
        return expire(self)

    def _validate_submission(self, body: BaseRequest) -> None:
        require(isinstance(body,dict) and set(body)=={'input_id','model_id','profile_id','task','parameters'})
        request=cast(ServiceRequest,body)  # Required key set was checked above.
        require(all(cast(Mapping[str, object],request)[k]==cast(Mapping[str, object],self.selected)[k] for k in ('model_id','profile_id','task')),'profile_not_available')
        if self.selected['task'] in ('generation','generation_scene','features'):
            p=request['parameters'];require(isinstance(p,dict) and set(p)=={'seed'} and type(p['seed']) is int and 0<=p['seed']<2**32,'unsupported_generation_recipe')
        else:require(request['parameters']=={},'unsupported_parameters')

    def _submitted(self, db: sqlite3.Connection, identifier: str, body: BaseRequest) -> None:
        if 'workflow' in self.selected:
            limits=self.selected['workflow']['limits']
            tenant=db.execute('SELECT tenant FROM jobs WHERE id=?',(identifier,)).fetchone()[0]
            for scope,args,limit in [('AND j.tenant=?',(tenant,),limits['max_tenant_output_bytes']),('',(),limits['max_global_output_bytes'])]:
                retained=db.execute('SELECT COUNT(*) FROM jobs j WHERE j.state!="cancelled" '+scope+
                    ' AND NOT EXISTS(SELECT 1 FROM service_gc g WHERE g.kind="results" AND g.id=j.id AND g.state="removed")',args).fetchone()[0]
                require(retained*limits['max_output_bytes']<=limit,'workflow_output_reservation',429)
        db.execute('INSERT INTO service_requests VALUES (?,?)',(identifier,encoded(body).decode()))
        security.reserve(self,db,identifier)

    def _terminal(self, db: sqlite3.Connection, identifier: str, state: str) -> None:security.settle(db,identifier,state)

    def job(self, tenant: str, identifier: str) -> dict[str, object]:
        value=super().job(tenant,identifier);value.update(model_id=self.selected['model_id'],task=self.selected['task'])
        value['validation']={'status':'not_evaluated','reason':'fake_executor' if self.execution=='fake' else 'inspect_native_job_receipt',
                             'qualification_sha256':self.selected['qualification_sha256']}
        return value

    def claim(self, *, expected_owner: str | None = None) -> AttemptLease | None:
        # The expected generation is checked inside the existing claim transaction.
        # No separate router queue, reservation table or second attempt token.
        with self.lock:
            self._expected_worker_owner=expected_owner
            try:lease=super().claim()
            finally:self._expected_worker_owner=None
            if lease:
                row=self.db.execute('SELECT body FROM service_requests WHERE job=?',(lease['job_id'],)).fetchone()
                # This table is written by _submitted after ServiceRequest validation.
                full_lease=cast(AttemptLease,lease)
                full_lease['request']=json.loads(row['body'])
                if expected_owner is not None:lease['worker_owner']=expected_owner
            return cast(AttemptLease | None,lease)

    def _fenced(self, db: sqlite3.Connection, lease: BaseLease) -> sqlite3.Row:
        row=super()._fenced(db,lease)
        if 'worker_owner' in lease:
            record=db.execute('SELECT value FROM service_meta WHERE key="worker"').fetchone()
            worker={} if record is None else json.loads(record[0])
            require(worker.get('owner_id')==lease['worker_owner'] and worker.get('state') not in ('closed','failed'),
                    'stale_worker_owner',409)
        return row

    def capabilities(self) -> dict[str, object]:
        return {**self.selected,'max_running':1,'max_active':CONTRACT['max_active'],'max_attempts':CONTRACT['max_attempts'],
          'lease_seconds':CONTRACT['lease_seconds'],'input':'prepared manifest plus bound local files; no crop or URL fetching',
          'cancellation':{'allowed_states':['queued'],'interrupts_gpu':False},
          'stages':{'preparation_workers':1,'gpu_workers':1,'publication_workers':1,
                    'publication_overlap':False,'max_publication_payload_bytes':self.contract['max_result_bytes']},
          'per_job_numerical_status':'full_reference_internal' if self.execution=='engine' and self.profile=='fp32-full-reference-v1' else 'not_evaluated',
          'hosted_qualification':False,'scientific_quality':'not_evaluated',
          'security':deepcopy(security.POLICY),'parser':'separate CPU process; Linux CPU/address-space/file/fd limits; external OS sandbox still required for hostile public inputs'}

    def readiness(self) -> Readiness:
        health=self.health();worker=self.meta('worker',{});drain=self.meta('drain',False)
        live=worker.get('heartbeat',0)+20>self.clock() and worker.get('state') in ('available','running')
        ready=health['control_plane']['ready'] and live and not drain
        return {'schema':'aureon.async_readiness.v1','ready':ready,'draining':drain,'worker':worker,
          'model_loaded':worker.get('model_loaded',False),'lazy_model_start':True,'capabilities':self.capabilities()}

    def http_route(self, handler: Handler, tenant: str, method: str, parts: list[str], query: dict[str, list[str]]) -> tuple[int, object, bool] | None:
        if ('workflow' in self.selected and method=='GET' and len(parts)==5 and parts[:2]==['v1','jobs'] and parts[3]=='artifacts'):
            from .streaming import download
            return 200,download(self,tenant,parts[2],parts[4],query.get('ticket',[''])[0]),True
        if parts==['v1','metrics'] and method=='GET':
            require(tenant=='operator','operator_required',403);require(not query,'invalid_query')
            return 200,self.metrics(),False
        if parts==['v1','worker'] and method=='GET':
            require(tenant=='operator','operator_required',403);require(not query,'invalid_query')
            return 200,self.worker_snapshot(),False
        if parts==['v1','keys'] and method in ('GET','POST'):
            require(not query,'invalid_query')
            if method=='GET':return 200,security.key_list(self,tenant),False
            require(handler.json()=={},'empty_object_required')
            return 201,security.create_key(self,tenant),False
        if parts==['v1','keys','revoke'] and method=='POST':
            body=handler.json();require(not query and isinstance(body,dict) and set(body)=={'fingerprint'},'invalid_key_request')
            return 200,security.revoke_key(self,tenant,cast(dict[str,str],body)['fingerprint']),False
        if parts==['v1','usage'] and method=='GET':
            require(not query,'invalid_query');return 200,security.shadow_statement(self,tenant),False
        if parts==['v1','models'] and method=='GET':
            require(not query,'invalid_query')
            # Replica contracts include host/device identifiers and installed
            # asset bindings. Customers need public request limits, not those
            # operator diagnostics. Keep new private extension fields opt-in.
            public_keys=set(CONTRACT)|{'model','route','request_deadline_seconds'}
            limits={k:deepcopy(v) for k,v in self.contract.items() if tenant=='operator' or k in public_keys}
            return 200,{'schema':'aureon.async_models.v1','models':[self.capabilities()],'limits':limits},False
        if parts==['v1','ready'] and method=='GET':
            require(not query,'invalid_query');value=self.readiness()
            if tenant!='operator':value['worker']=cast(WorkerStatus,{k:v for k,v in value['worker'].items() if k in ('state','model_loaded')})
            return (200 if value['ready'] else 503),value,False
        if parts==['v1','drain'] and method=='POST':
            require(tenant=='operator','operator_required',403);require(not query and handler.json()=={},'empty_object_required')
            self.set_meta('drain',True);self.set_meta('shutdown_requested',True)
            return 202,{'draining':True,'running_job':'finish then close','queued_jobs':'retained for explicit restart'},False

        return None

    def metrics(self) -> ServiceMetrics:
        """Operator aggregate view; direct Python callers own authorization."""
        from .metrics import snapshot
        return snapshot(self)

    def publish_engine(self, lease: AttemptLease, files: ArtifactFiles) -> dict[str, object]:
        """Copy durable outputs before atomic fenced publication; duplicates never bill twice."""
        if 'workflow' in self.selected:
            from .streaming import publish
            return publish(self,lease,files)
        require(self.execution=='engine','engine_not_allowed',409)
        expected={'job.json','native-job.json','provenance.json'}|({'class_map.tif','valid_mask.tif'} if self.selected['task']=='scene' else {'synthetic-s1grd.tif','raw.npy','tokens.npy'})
        require(set(files)==expected,'incomplete_artifacts',409)
        # Reject stale work and over-budget results before allocating payloads.
        with self.tx() as db:self._fenced(db,lease)
        sizes={name:_plain_path(path).stat().st_size for name,path in files.items()}
        require(all(size<=CONTRACT['max_artifact_bytes'] for size in sizes.values()),'artifact_too_large',413)
        require(sum(sizes.values())<=CONTRACT['max_result_bytes'],'result_too_large',413)
        payload={};total=0
        for name,path in files.items():
            with _plain_path(path).open('rb') as stream:payload[name]=stream.read(sizes[name]+1)
            require(len(payload[name])==sizes[name],'artifact_size_changed',409)
            total+=len(payload[name]);require(total<=CONTRACT['max_result_bytes'],'result_too_large',413)
        import shutil
        require(shutil.disk_usage(self.root).free>=CONTRACT['min_free_bytes']+sum(map(len,payload.values())),'result_storage_capacity',507)
        receipt=json.loads(payload['job.json'])
        require(receipt.get('schema')=='aureon.async_result.v1' and receipt.get('job_id')==lease['job_id'] and receipt.get('attempt')==lease['attempt']
          and receipt.get('model_id')==self.selected['model_id'] and receipt.get('task')==self.selected['task'] and receipt.get('profile_id')==self.profile
          and receipt.get('input_manifest_sha256')==file_hash(Path(lease['manifest'])),'result_binding_mismatch',409)
        require(set(receipt['artifacts'])==expected-{'job.json'},'result_artifact_binding',409)
        for name,item in receipt['artifacts'].items():require(item['sha256']==digest(payload[name]) and item['bytes']==len(payload[name]),'artifact_integrity',409)
        with self.tx() as db:
            row=self._fenced(db,lease);folder=self._folder('results',row['id']);folder.mkdir(exist_ok=True)
            attempt=folder/('attempt-'+str(lease['attempt']));attempt.mkdir(exist_ok=True);record={}
            for name,data in payload.items():
                path=_plain_path(attempt/name)
                if path.exists():require(path.read_bytes()==data,'artifact_conflict',409)
                else:write_file(path,data)
                record[name]={'path':str(path.relative_to(self.root)),'bytes':len(data),'sha256':digest(data)}
            now=self.clock()
            db.execute('UPDATE jobs SET state="completed",result=?,expires=?,lease=NULL WHERE id=?',(encoded(record).decode(),now+CONTRACT['output_ttl_seconds'],row['id']))
            db.execute('UPDATE attempts SET state="completed",finished=? WHERE job=? AND number=?',(now,row['id'],lease['attempt']))
            db.execute('UPDATE outbox SET state="finished" WHERE job=?',(row['id'],))
            db.execute('INSERT INTO usage VALUES (?,?,1,?,0)',(row['id'],row['tenant'],now))
            self._terminal(db,row['id'],'completed')
        return {'job_id':row['id'],'state':'completed','model_executed':True}
