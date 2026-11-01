"""Bounded workflow transport and resident wiring. Simulated model outputs only."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from aureon.client import APIError
from aureon.cloud.control import Error, encoded, file_hash, write_file
from aureon.cloud.http import Server
from aureon.serving.client import Client
from aureon.serving.control import Control, initialize
from aureon.serving.workflow_config import normalize, binding, artifacts
from aureon.serving.verified_files import VerifiedFiles


def configuration(root, task='features', **limits):
    return {'schema':'aureon.terramind_service_workflow.v1',
        'model_id':'terramind-v1-base-backbone' if task=='features' else 'terramind-v1-base',
        'task':task,'model_file':str(root/'model.safetensors'),
        'tokenizer_files':{} if task=='features' else {'S1GRD':str(root/'s1.safetensors')},
        'options':{},'limits':limits}


class ServiceWorkflowTests(unittest.TestCase):
    def setUp(self):
        parent=Path(os.environ.get('AUREON_SERVICE_TEST_ROOT',tempfile.gettempdir()))
        self.root=parent/('workflow-'+uuid.uuid4().hex);self.root.mkdir(parents=True)
        self.now=2000000000.;self.controls=[]

    def tearDown(self):
        for control in self.controls:control.close()

    def create(self, task='features', **limits):
        config=self.root/('config-'+uuid.uuid4().hex+'.json');config.write_bytes(encoded(configuration(self.root,task,**limits)))
        state=self.root/('state-'+uuid.uuid4().hex)
        keys=initialize(state,['alpha','beta'],model='terramind-v1-base-backbone' if task=='features' else 'terramind-v1-base',
            environment='generation',profile='fp32-features-v1' if task=='features' else 'fp32-generation-scene-v1',workflow_config=config)
        control=Control(state,clock=lambda:self.now);self.controls.append(control)
        return control,keys,config

    def ready(self, control):
        identifier=uuid.uuid4().hex;folder=control.root/'inputs'/identifier;folder.mkdir()
        path=folder/'manifest.json';path.write_bytes(encoded({'fixture':'simulated model output','model_executed':False}))
        with control.tx() as db:
            db.execute('INSERT INTO inputs(id,tenant,state,declaration,bytes,created,expires,manifest) VALUES (?,"alpha","ready","{}",123,?,?,?)',
                (identifier,self.now,self.now+3600,str(path.relative_to(control.root))))
        body={'input_id':identifier,'model_id':control.selected['model_id'],'profile_id':control.profile,
              'task':control.selected['task'],'parameters':{'seed':17}}
        return body

    def files(self,control,lease,size=0):
        folder=self.root/('outputs-'+uuid.uuid4().hex);folder.mkdir();files={}
        primary='features.npz' if control.selected['task']=='features' else 'synthetic-s1grd.tif'
        for name in artifacts(control.selected['task'])-{'job.json'}:
            path=folder/name
            if name==primary and size:
                with path.open('xb') as stream:
                    block=b'x'*1024**2
                    for _ in range(size//len(block)):stream.write(block)
                    stream.write(block[:size%len(block)])
            else:path.write_bytes(encoded({'simulated_output':True,'model_executed':False}))
            files[name]=path
        receipt={'schema':'aureon.async_result.v1','job_id':lease['job_id'],'attempt':lease['attempt'],
            **control.selected,'input_manifest_sha256':file_hash(Path(lease['manifest'])),
            'workflow_config_sha256':control.selected['workflow']['config_sha256'],'parameters':lease['request']['parameters'],
            'artifacts':{name:{'original_path':str(path),'bytes':path.stat().st_size,'sha256':file_hash(path)} for name,path in files.items()},
            'test_fixture':{'simulated':True,'model_executed':False}}
        path=folder/'job.json';path.write_bytes(encoded(receipt));files['job.json']=path
        return files

    @contextmanager
    def http(self,control,keys):
        server=Server(control);thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01});thread.start()
        url='http://127.0.0.1:'+str(server.server_port)
        try:yield Client(url,token=keys['alpha']),Client(url,token=keys['beta']),Client(url,token=keys['operator'])
        finally:server.shutdown();thread.join(5);server.server_close();self.assertFalse(thread.is_alive())

    def test_locked_config_pins_paths_and_discovery_is_public(self):
        control,keys,config=self.create()
        with self.http(control,keys) as (client,_,operator):
            public=client.json('GET','/v1/models')
            self.assertNotIn(str(self.root),json.dumps(public))
            self.assertEqual(public['models'][0]['task'],'features')
            self.assertIn('asset_pins',public['limits']['route']['workflow'])
            with self.assertRaises(APIError) as exc:client.json('GET','/v1/metrics')
            self.assertEqual(exc.exception.status,403)
            self.assertEqual(operator.json('GET','/v1/metrics')['committed_usage_rows'],0)
        stored=control.root/'workflow-config.json';value=json.loads(stored.read_bytes());value['options']['modalities']=['S1GRD']
        stored.write_bytes(encoded(value))
        with self.assertRaises((Error,ValueError)):Control(control.root)

    def test_reject_incompatible_profiles_assets_and_unbounded_options(self):
        for change in ({'task':'features','model_id':'terramind-v1-base'}, {'tokenizer_files':{'S1GRD':'x'}},
                       {'options':{'arbitrary':True}}, {'limits':{'max_output_bytes':8*1024**3}},
                       {'limits':{'request_deadline_seconds':601}}, {'task':[]}):
            value=configuration(self.root);value.update(change)
            with self.assertRaises((Error,ValueError)):normalize(value,self.root)
        scene=normalize(configuration(self.root,'generation_scene'),self.root)
        self.assertEqual(set(binding(scene)['asset_pins']['tokenizers']),{'S1GRD'})

    def test_deadline_is_finite_operator_bound_and_legacy_unchanged(self):
        control,keys,_=self.create(request_deadline_seconds=240)
        self.assertEqual(control.contract['request_deadline_seconds'],240)
        timer=threading.Timer;observed=[]
        def capture(seconds,callback):observed.append(seconds);return timer(seconds,callback)
        with self.http(control,keys) as (client,_,_),patch('aureon.cloud.http.threading.Timer',side_effect=capture):
            self.assertEqual(client.json('GET','/v1/models')['limits']['request_deadline_seconds'],240)
        self.assertEqual(observed,[240])
        legacy=self.root/'legacy';initialize(legacy,['alpha'],model='terramind-v1-base',environment='generation',profile='fp32-generation-v1',fake=True)
        old=Control(legacy);self.controls.append(old)
        self.assertEqual(old.contract.get('request_deadline_seconds',45),45)

    def test_atomic_quota_cancel_replay_recovery_and_once_only_usage(self):
        control,_,_=self.create(max_output_bytes=1024**2,max_artifact_bytes=1024**2,max_tenant_output_bytes=1024**2,max_global_output_bytes=2*1024**2)
        body=self.ready(control);job=control.submit('alpha',body,'same-request')
        self.assertEqual(control.submit('alpha',body,'same-request')['id'],job['id'])
        with self.assertRaises(Error) as exc:control.submit('alpha',body,'blocked-quota')
        self.assertEqual(exc.exception.code,'workflow_output_reservation')
        control.cancel('alpha',job['id']);replacement=control.submit('alpha',body,'replacement')
        old=control.claim();self.now+=61;fresh=control.claim();self.assertEqual(fresh['attempt'],2)
        with self.assertRaises(Error):control.publish_engine(old,self.files(control,old))
        control.publish_engine(fresh,self.files(control,fresh))
        with self.assertRaises(Error):control.publish_engine(fresh,self.files(control,fresh))
        self.assertEqual(control.metrics()['committed_usage_rows'],1)
        self.assertEqual(control.metrics()['usage']['settled']['jobs'],1)
        self.now+=3601
        with self.assertRaises(Error):control.submit('alpha',self.ready(control),'still-retained')
        self.assertTrue((control.root/'results'/replacement['id']).exists())

    def test_failed_output_credit_releases_only_after_successful_expiry_GC(self):
        control,_,_=self.create(max_output_bytes=1024**2,max_artifact_bytes=1024**2,max_tenant_output_bytes=1024**2,max_global_output_bytes=2*1024**2)
        body=self.ready(control);job=control.submit('alpha',body,'failed-job');control.fail(control.claim())
        retained=control.root/'results'/job['id'];retained.mkdir();(retained/'partial.bin').write_bytes(b'owned disposable failed fixture')
        self.now+=3601;next_body=self.ready(control)
        with self.assertRaises(Error):control.submit('alpha',next_body,'before-gc')
        with patch('aureon.serving.operations.shutil.rmtree',side_effect=OSError('simulated deletion failure')):
            with self.assertRaises(OSError):control.expire()
        self.assertTrue(retained.exists())
        with self.assertRaises(Error):control.submit('alpha',next_body,'pending-gc')
        control.expire();self.assertFalse(retained.exists())
        result=control.submit('alpha',next_body,'after-gc');self.assertEqual(result['state'],'queued')
        self.assertEqual(control.metrics()['committed_usage_rows'],0)

    def test_both_artifact_contracts_and_http_isolation(self):
        for task in ('features','generation_scene'):
            with self.subTest(task=task):
                control,keys,_=self.create(task);body=self.ready(control)
                with self.http(control,keys) as (client,other,_):
                    job=client.submit_model(body['input_id'],model=body['model_id'],profile=body['profile_id'],task=task,key='job-'+task,seed=17)
                    lease=control.claim();control.publish_engine(lease,self.files(control,lease))
                    self.assertEqual(client.json('GET','/v1/jobs/'+job['id'])['state'],'completed')
                    with self.assertRaises(APIError):other.json('GET','/v1/jobs/'+job['id']+'/artifacts')
                    result=client.download_result(job['id'],self.root/('download-'+task),expected_manifest_sha256=file_hash(Path(lease['manifest'])),model=body['model_id'],task=task)
                    self.assertEqual(set(result['artifacts']),artifacts(task));self.assertTrue(result['hashes_verified'])
                    self.assertEqual(result['numerical_comparison'],'not_evaluated_by_transport')

    def test_over_64mib_http_delivery_uses_bounded_reads(self):
        control,keys,_=self.create();body=self.ready(control);job=control.submit('alpha',body,'large-stream')
        lease=control.claim();files=self.files(control,lease,65*1024**2+7)
        control.publish_engine(lease,files)
        original=Client._read
        def bounded(client,response,limit):
            self.assertLessEqual(limit,4*1024**2,'Large artifact used byte-buffer download path')
            return original(response,limit)
        with self.http(control,keys) as (client,other,_),patch.object(Client,'_read',bounded):
            listing=client.json('GET','/v1/jobs/'+job['id']+'/artifacts')
            with self.assertRaises(APIError):
                with other._response('GET',listing['artifacts']['features.npz']['url']):pass
            result=client.download_result(job['id'],self.root/'large-download',expected_manifest_sha256=file_hash(Path(lease['manifest'])),model=body['model_id'],task='features')
        self.assertEqual(result['artifacts']['features.npz']['bytes'],65*1024**2+7)
        self.assertEqual(result['artifacts']['features.npz']['sha256'],file_hash(files['features.npz']))

    def test_stream_persistence_rejects_truncation_extra_and_digest(self):
        for payload,digest in ((b'ab','0'*64),(b'abcd',hashlib.sha256(b'abc').hexdigest()),(b'abc','0'*64)):
            store=VerifiedFiles(self.root/uuid.uuid4().hex,max_total_bytes=3)
            with self.assertRaises(ValueError):store.save_stream('item',io.BytesIO(payload),expected={'bytes':3,'sha256':digest})
            self.assertTrue(store.failed)

    def test_stream_reader_never_requests_unbounded_allocation(self):
        class Bounded(io.BytesIO):
            def read(self,size=-1):
                if not 0<size<=1024**2:raise AssertionError('unbounded read')
                return super().read(size)
        data=b'a'*(2*1024**2+9);store=VerifiedFiles(self.root/'bounded',max_total_bytes=len(data))
        result=store.save_stream('item',Bounded(data),expected={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        self.assertFalse(store.data);self.assertEqual(result['bytes'],len(data))

    def test_HTTP_upload_preparation_submit_complete_download(self):
        # Transport is real HTTP. This stdlib suite explicitly substitutes only
        # the optional NumPy parser/model, tested with real payloads separately.
        control,keys,_=self.create();source=self.root/'source';source.mkdir()
        payload=source/'inputs.npz';payload.write_bytes(b'simulated-npz-fixture')
        manifest=source/'manifest.json';manifest.write_bytes(encoded({'schema':'aureon.terramind_features_input.v1',
            'inputs':{'path':'inputs.npz','bytes':payload.stat().st_size,'sha256':file_hash(payload)}}))
        def parser(body,folder,token,task,**kwargs):
            prepared=folder/('prepared-'+token);prepared.mkdir()
            for name,uploaded in body['spec']['members'].items():
                (prepared/name).write_bytes((folder/'source'/uploaded).read_bytes())
            self.assertEqual(kwargs['workflow'],control.selected['workflow'])
            return prepared/'manifest.json'
        with self.http(control,keys) as (client,_,_),patch('aureon.serving.parser.run',side_effect=parser):
            upload=client.upload_prepared(manifest,task='features')
            self.assertTrue(control.prepare_one())
            prepared=client.wait_input(upload['input_id']);self.assertEqual(prepared['state'],'ready')
            job=client.submit_model(upload['input_id'],model=control.selected['model_id'],profile=control.profile,task='features',key='uploaded-job')
            control.publish_engine(lease:=control.claim(),self.files(control,lease))
            self.assertEqual(client.wait_job(job['id'])['state'],'completed')
            result=client.download_result(job['id'],self.root/'upload-download',expected_manifest_sha256=upload['manifest_sha256'],model=control.selected['model_id'],task='features')
            self.assertTrue(result['hashes_verified']);self.assertEqual(control.metrics()['committed_usage_rows'],1)

    def test_selected_scene_backend_borrows_one_best_resident(self):
        from aureon.serving.backend import Backend
        for task in ('generation_scene',):
            with self.subTest(task=task):
                control,_,config=self.create(task);work=self.root/('work-'+task);work.mkdir()
                native_calls=[];instances=[]
                class Engine:
                    def __init__(inner,*args,**kwargs):
                        inner.work=Path(kwargs.get('work',work/'session'));inner.closed=0;inner.state='ready';instances.append((inner,args,kwargs))
                    def close(inner):inner.closed+=1;inner.state='closed';return {'passed':True}
                    def metadata(inner):return {'model_load_count':1}
                def scene(engine,request,**kwargs):
                    native_calls.append((engine,request,kwargs));target=Path(request.output);target.mkdir()
                    for name in ('synthetic-s1grd.tif','provenance.json','tiles.jsonl','preflight.json','completion.json'):
                        (target/name).write_bytes(encoded({'simulated':True,'model_executed':False}))
                    return {}
                with patch('aureon.serving.backend.installed_binding',return_value={'version':'0.1.0','root':str(self.root),'modules':{}}), \
                     patch('aureon.api.profiles.require_profile'), \
                     patch('aureon.api.best.BestTerraMindSession',Engine), \
                     patch('aureon.api.selected_scene.native_config',return_value={'fixture':True}) as admission, \
                     patch('aureon.api.best_workflows.run_best_scene',side_effect=scene), \
                     patch('aureon.api.generation_engine.GenerationEngine',side_effect=AssertionError('alternate executor used')):
                    backend=Backend(control,work,config=config,execute_model=True)
                    body=self.ready(control)
                    for number in range(2):
                        control.submit('alpha',body,'resident-'+str(number));lease=control.claim()
                        Path(lease['manifest']).write_bytes(encoded({'image':{'path':'image.tif'},'mask':{'path':'mask.tif'},'inputs':{'path':'inputs.npz'}}))
                        files=backend.run(lease);self.assertEqual(set(files),artifacts(task));control.publish_engine(lease,files)
                    self.assertEqual(len(instances),1);self.assertIs(native_calls[0][0],native_calls[1][0]);self.assertEqual(instances[0][0].closed,0)
                    self.assertEqual(admission.call_count,1)
                    self.assertTrue(all(call[1].seed==17 and call[2]['execute_model'] for call in native_calls))
                    self.assertEqual(backend.loads,1);self.assertTrue(backend.close()['passed']);self.assertEqual(instances[0][0].closed,1)

    def test_selected_backend_rejects_unselected_executor_without_loading(self):
        from aureon.serving.backend import Backend
        control,_,config=self.create('features');work=self.root/'unselected';work.mkdir()
        with patch('aureon.serving.backend.installed_binding',return_value={'version':'0.1.0','root':str(self.root),'modules':{}}), \
             patch('aureon.api.profiles.require_profile'), \
             patch('aureon.api.terramind_encoder.TerraMindEncoder',side_effect=AssertionError('unselected executor loaded')):
            backend=Backend(control,work,config=config,execute_model=True)
            control.submit('alpha',self.ready(control),'unselected');lease=control.claim()
            with self.assertRaisesRegex(ValueError,'does not enable another model executor'):
                backend.run(lease)
            self.assertIsNone(backend.session);self.assertEqual(backend.loads,0)
            self.assertEqual(control.metrics()['committed_usage_rows'],0)
            self.assertTrue(backend.close()['passed'])

    def test_encoder_owner_cleanup_releases_cache_and_preserves_primary_failure(self):
        from aureon.serving.backend import Backend
        for broken in (False,True):
            with self.subTest(broken=broken):
                work=self.root/uuid.uuid4().hex;work.mkdir();events=[]
                class Encoder:
                    state='ready'
                    def close(inner):
                        events.append('encoder.close')
                        if broken:raise RuntimeError('private-primary-detail')
                        inner.state='closed'
                def sync():
                    events.append('synchronize')
                    if broken:raise RuntimeError('private-secondary-detail')
                torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True,synchronize=sync,
                    empty_cache=lambda:events.append('empty_cache'),memory_allocated=lambda:0,memory_reserved=lambda:0),
                    _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda:events.append('clear_cublas')))
                backend=Backend.__new__(Backend);backend.owner=threading.get_ident();backend._closed=False;backend._close_result=None
                backend.session=Encoder();backend.last_cleanup=None;backend.workflow={'task':'features'};backend.work=work;backend.calls=2;backend.loads=1
                backend.binding={'version':'0.1.0','root':str(self.root),'modules':{}}
                with patch.dict('sys.modules',{'torch':torch}),patch('aureon.serving.backend.installed_binding',return_value=backend.binding), \
                     patch('aureon.serving.backend.gc.collect',side_effect=lambda:events.append('gc')):
                    if broken:
                        with self.assertRaises(Error):backend.close()
                        receipt=json.loads((work/'cleanup.json').read_bytes())
                        self.assertEqual(receipt['close_error']['code'],'engine_close_failed')
                        self.assertNotIn('private-',json.dumps(receipt))
                    else:
                        receipt=backend.close();self.assertTrue(receipt['passed']);self.assertEqual(receipt['engine_cleanup']['encoder_state'],'closed')
                        self.assertEqual(events,['encoder.close','gc','synchronize','clear_cublas','empty_cache'])


if __name__=='__main__':unittest.main()
