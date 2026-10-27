"""Single-host service routing over the existing durable/tenant control substrate."""
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

def route(model,environment,profile,*,fake=False,denoiser='graph',qualification=None):
    from ..api import inspect_model
    spec=inspect_model(model);task='generation' if model==GENERATION else 'scene'
    if task=='generation':
        require(environment=='generation' and profile=='fp32-generation-v1' and denoiser in ('native','graph') and qualification is None,'unsupported_route')
    else:
        require(environment in spec['full_reference_environments'],'unsupported_environment')
        require(profile!='fp32-full-reference-v1' or qualification is None,'full_reference_has_no_qualification')
        if profile!='fp32-full-reference-v1':
            require(model=='kao-terramind-b' and environment=='scene' and qualification is not None,'qualification_required')
            from ..cloud.profiles import verify_qualification
            verified=verify_qualification(qualification);require(verified['profile_id']==profile,'qualification_profile')
    return {'model_id':model,'task':task,'environment':environment,'profile_id':profile,'reuse':'auto' if task=='generation' else None,
      'denoiser':denoiser if task=='generation' else None,'execution':'fake' if fake else 'engine',
      'qualification_sha256':None if qualification is None else digest(encoded(qualification))}

def initialize(root,tenants,*,model,environment,profile,fake=False,denoiser='graph',qualification=None,tenant_storage_reservation_bytes=None):
    require('operator' not in tenants and len(tenants)<=15,'reserved_operator_tenant')
    selected=route(model,environment,profile,fake=fake,denoiser=denoiser,qualification=qualification)
    keys=base_initialize(root,[*tenants,'operator'],tenant_storage_reservation_bytes=tenant_storage_reservation_bytes)
    root=Path(root);contract=json.loads((root/'contract.json').read_bytes())
    contract.update(schema='aureon.async_service.v1',model=model,profile=profile,execution=selected['execution'],route=selected)
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
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._rate={}
        security.schema(self)

    def _expected_contract(self,verifier):
        current=self.contract;r=current.get('route',{})
        qualification=None
        if r.get('qualification_sha256'):
            qualification=json.loads((self.root/'engine-qualification.json').read_bytes())
        selected=route(r.get('model_id'),r.get('environment'),r.get('profile_id'),fake=r.get('execution')=='fake',
                       denoiser=r.get('denoiser') or 'native',qualification=qualification)
        expected={**CONTRACT,'schema':'aureon.async_service.v1','model':selected['model_id'],'profile':selected['profile_id'],
          'execution':selected['execution'],'route':selected,'max_tenant_storage_reservation_bytes':tenant_storage_limit(current.get('max_tenant_storage_reservation_bytes'))}
        return expected

    @property
    def selected(self):return self.contract['route']

    def meta(self,key,default=None):
        with self.lock:row=self.db.execute('SELECT value FROM service_meta WHERE key=?',(key,)).fetchone()
        return default if row is None else json.loads(row['value'])

    def set_meta(self,key,value):
        with self.tx() as db:db.execute('INSERT INTO service_meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,encoded(value).decode()))

    def _admit(self,db):
        require(db.execute('SELECT value FROM service_meta WHERE key="drain"').fetchone()[0]=='false','service_draining',503)

    def _can_claim(self,db):
        return db.execute('SELECT value FROM service_meta WHERE key="drain"').fetchone()[0]=='false'

    def create_input(self,tenant,body):
        require(tenant!='operator','operator_not_customer',403)
        return super().create_input(tenant,body)

    def _validate_input(self,body):return inputs.validate(body,self.selected['task'])
    def _inspect_uploaded_tiff(self,path):
        pass  # Magic/hash checked in the parent; GDAL runs only in the bounded child.

    def _prepare_input(self,body,folder,token,identifier):
        from .parser import run
        return run(body,folder,token,self.selected['task'])

    def request_admission(self,tenant):security.admit_request(self,tenant)

    def expire(self):
        from .operations import expire
        return expire(self)

    def _validate_submission(self,body):
        require(isinstance(body,dict) and set(body)=={'input_id','model_id','profile_id','task','parameters'})
        require(all(body[k]==self.selected[k] for k in ('model_id','profile_id','task')),'profile_not_available')
        if self.selected['task']=='generation':
            p=body['parameters'];require(isinstance(p,dict) and set(p)=={'seed'} and type(p['seed']) is int and 0<=p['seed']<2**32,'unsupported_generation_recipe')
        else:require(body['parameters']=={},'unsupported_parameters')

    def _submitted(self,db,identifier,body):
        db.execute('INSERT INTO service_requests VALUES (?,?)',(identifier,encoded(body).decode()))
        security.reserve(self,db,identifier)

    def _terminal(self,db,identifier,state):security.settle(db,identifier,state)

    def job(self,tenant,identifier):
        value=super().job(tenant,identifier);value.update(model_id=self.selected['model_id'],task=self.selected['task'])
        value['validation']={'status':'not_evaluated','reason':'fake_executor' if self.execution=='fake' else 'inspect_native_job_receipt',
                             'qualification_sha256':self.selected['qualification_sha256']}
        return value

    def claim(self):
        if self.meta('drain',False):return None
        lease=super().claim()
        if lease:
            with self.lock:row=self.db.execute('SELECT body FROM service_requests WHERE job=?',(lease['job_id'],)).fetchone()
            lease['request']=json.loads(row['body'])
        return lease

    def capabilities(self):
        return {**self.selected,'max_running':1,'max_active':CONTRACT['max_active'],'max_attempts':CONTRACT['max_attempts'],
          'lease_seconds':CONTRACT['lease_seconds'],'input':'prepared manifest plus bound local files; no crop or URL fetching',
          'cancellation':{'allowed_states':['queued'],'interrupts_gpu':False},
          'per_job_numerical_status':'full_reference_internal' if self.execution=='engine' and self.profile=='fp32-full-reference-v1' else 'not_evaluated',
          'hosted_qualification':False,'scientific_quality':'not_evaluated',
          'security':security.POLICY,'parser':'separate CPU process; Linux CPU/address-space/file/fd limits; external OS sandbox still required for hostile public inputs'}

    def readiness(self):
        health=self.health();worker=self.meta('worker',{});drain=self.meta('drain',False)
        live=worker.get('heartbeat',0)+20>self.clock() and worker.get('state') in ('available','running')
        ready=health['control_plane']['ready'] and live and not drain
        return {'schema':'aureon.async_readiness.v1','ready':ready,'draining':drain,'worker':worker,
          'model_loaded':worker.get('model_loaded',False),'lazy_model_start':True,'capabilities':self.capabilities()}

    def http_route(self,handler,tenant,method,parts,query):
        if parts==['v1','keys'] and method in ('GET','POST'):
            require(not query,'invalid_query')
            if method=='GET':return 200,security.key_list(self,tenant),False
            require(handler.json()=={},'empty_object_required')
            return 201,security.create_key(self,tenant),False
        if parts==['v1','keys','revoke'] and method=='POST':
            body=handler.json();require(not query and isinstance(body,dict) and set(body)=={'fingerprint'},'invalid_key_request')
            return 200,security.revoke_key(self,tenant,body['fingerprint']),False
        if parts==['v1','usage'] and method=='GET':
            require(not query,'invalid_query');return 200,security.shadow_statement(self,tenant),False
        if parts==['v1','models'] and method=='GET':
            return 200,{'schema':'aureon.async_models.v1','models':[self.capabilities()],'limits':self.contract},False
        if parts==['v1','ready'] and method=='GET':
            require(not query,'invalid_query');value=self.readiness()
            if tenant!='operator':value['worker']={k:v for k,v in value['worker'].items() if k in ('state','model_loaded')}
            return (200 if value['ready'] else 503),value,False
        if parts==['v1','drain'] and method=='POST':
            require(tenant=='operator','operator_required',403);require(not query and handler.json()=={},'empty_object_required')
            self.set_meta('drain',True);self.set_meta('shutdown_requested',True)
            return 202,{'draining':True,'running_job':'finish then close','queued_jobs':'retained for explicit restart'},False

    def publish_engine(self,lease,files):
        """Copy durable outputs before atomic fenced publication; duplicates never bill twice."""
        require(self.execution=='engine','engine_not_allowed',409)
        expected={'job.json','native-job.json','provenance.json'}|({'class_map.tif','valid_mask.tif'} if self.selected['task']=='scene' else {'synthetic-s1grd.tif','raw.npy','tokens.npy'})
        require(set(files)==expected,'incomplete_artifacts',409)
        payload={}
        for name,path in files.items():
            with _plain_path(path).open('rb') as stream:payload[name]=stream.read(CONTRACT['max_artifact_bytes']+1)
            require(len(payload[name])<=CONTRACT['max_artifact_bytes'],'artifact_too_large',413)
        require(sum(map(len,payload.values()))<=CONTRACT['max_result_bytes'],'result_too_large',413)
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
