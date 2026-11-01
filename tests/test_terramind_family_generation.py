"""Generation family contracts using CPU fixtures; no foundation weights or CUDA."""
from contextlib import closing, nullcontext
from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from aureon.api import GenerationEngine, GenerationInput
from aureon.api import profiles
from aureon.api.generation_bridges import generation_capabilities, generation_compatibility, GenerationRecipe
from aureon.api.generation_parallel import qualification, open_lane_session
from aureon.terramind_native import loader, offline
from aureon.terramind_native.execution import plan_generation_execution
from aureon.terramind_native.families import GENERATION_MODELS, get_terramind_spec
from aureon.terramind_native.common import INPUT_BANDS
from aureon.terramind_native.session import TerraMindSession


RECIPE = dict(token_steps=8, diffusion_steps=10, temperature=1., top_p=.8, top_k=0, seed=17)


class FamilyGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp=self.enterContext(tempfile.TemporaryDirectory(prefix='aureon-family-'))
        self.root=Path(self.temp)
        for name in ('model','tokenizer'):(self.root/name).write_bytes(b'CPU fixture, not weights')

    def config(self, model_id='terramind-v1-base', filename='config.json'):
        with patch.object(profiles,'require_profile'), patch.object(offline,'verify_file'), \
                patch.object(offline.metadata,'version',side_effect=profiles.PROFILES['generation'].__getitem__):
            offline.create_config(model=self.root/'model',tokenizer=self.root/'tokenizer',
                                  output=self.root/filename,model_id=model_id)
        return self.root/filename

    def verify_fixture(self, path):
        # Only artifact IO/dependency presence is doubled; real config/pin checks run.
        with patch.object(loader,'verify_file',side_effect=lambda r,b:Path(b)/r['path']), \
                patch.object(loader.metadata,'version',side_effect=profiles.PROFILES['generation'].__getitem__), \
                patch.object(loader.metadata,'distribution'):
            return loader.verify_config(path)

    def data(self):
        return GenerationInput(np.ones((12,224,224),np.float32),bands=INPUT_BANDS,
                               units='surface_reflectance_x10000',valid_mask=np.ones((224,224),bool))

    def test_all_sizes_bind_distinct_checkpoints_and_shared_tokenizer(self):
        hashes=set();tokenizers=set()
        for model_id in GENERATION_MODELS:
            with self.subTest(model_id=model_id):
                path=self.config(model_id,model_id+'.json');value,_,_=self.verify_fixture(path)
                self.assertEqual(loader.config_model_id(value),model_id)
                self.assertEqual(value['model']['sha256'],get_terramind_spec(model_id).checkpoint.sha256)
                hashes.add(value['model']['sha256']);tokenizers.add(value['tokenizer']['sha256'])
                with patch.object(profiles,'require_profile'), patch.object(loader,'verify_config',return_value=self.verify_fixture(path)):
                    self.assertEqual(offline.inspect_config(path)['model_id'],model_id)
        self.assertEqual(len(hashes),4);self.assertEqual(len(tokenizers),1)

    def test_legacy_base_config_content_identity_is_preserved(self):
        explicit=self.config(filename='explicit.json')
        with patch.object(profiles,'require_profile'), patch.object(offline,'verify_file'), \
                patch.object(offline.metadata,'version',side_effect=profiles.PROFILES['generation'].__getitem__):
            offline.create_config(model=self.root/'model',tokenizer=self.root/'tokenizer',output=self.root/'legacy.json')
        self.assertEqual(explicit.read_bytes(),(self.root/'legacy.json').read_bytes())
        value,_,_=self.verify_fixture(explicit)
        self.assertNotIn('model_id',value);self.assertEqual(value['model']['sha256'],loader.MODEL_SHA)
        self.assertEqual(offline.generation_assets(),offline.ASSETS)

    def test_wrong_size_or_mutated_pin_rejected_before_loading(self):
        path=self.config('terramind-v1-tiny');original=json.loads(path.read_text())
        mutations=[dict(original,model_id='terramind-v1-large'),dict(original,model_id='terramind-v1-tiny-backbone')]
        absent=deepcopy(original);del absent['model_id'];mutations.append(absent)
        for name in ('model','tokenizer'):
            for key,value in (('sha256','0'*64),('revision','wrong'),('bytes',1)):
                bad=deepcopy(original);bad[name][key]=value;mutations.append(bad)
        for i,bad in enumerate(mutations):
            changed=self.root/f'bad-{i}.json';changed.write_text(json.dumps(bad))
            with self.subTest(i=i), self.assertRaises(ValueError):self.verify_fixture(changed)

    def test_arbitrary_local_files_never_pass_real_hash_validation(self):
        for model_id in GENERATION_MODELS:
            with patch.object(profiles,'require_profile'), \
                    patch.object(offline.metadata,'version',side_effect=profiles.PROFILES['generation'].__getitem__):
                with self.assertRaises(ValueError):
                    offline.create_config(model=self.root/'model',tokenizer=self.root/'tokenizer',
                                          output=self.root/(model_id+'.json'),model_id=model_id)
            self.assertFalse((self.root/(model_id+'.json')).exists())

    def test_session_mismatch_rejected_without_work_or_runtime(self):
        path=self.config('terramind-v1-small')
        with patch.object(TerraMindSession,'_runtime_factory',side_effect=AssertionError('no model')):
            with self.assertRaisesRegex(ValueError,'differs'):
                TerraMindSession(path,self.root/'bad-work',execute_model=True,model_id='terramind-v1-base')
            self.assertFalse((self.root/'bad-work').exists())
            with TerraMindSession(path,self.root/'work',execute_model=True,model_id='terramind-v1-small') as session:
                self.assertEqual(session.metadata()['model_id'],'terramind-v1-small')
                self.assertEqual(json.loads((session.work/'session.json').read_text())['model_id'],'terramind-v1-small')
                self.assertEqual(session.metadata()['model_load_count'],0)

    def test_capability_compatibility_and_profiles_are_size_bound(self):
        keys=set();data=self.data()
        for model_id in GENERATION_MODELS:
            value=generation_compatibility(data,model_id=model_id)
            self.assertEqual(value['key']['model_id'],model_id)
            self.assertEqual(value['key']['model_sha256'],get_terramind_spec(model_id).checkpoint.sha256)
            keys.add(value['sha256']);self.assertEqual(generation_capabilities(model_id=model_id)['model_id'],model_id)
            plan=plan_generation_execution(recipe=RECIPE,model_id=model_id)
            self.assertEqual(plan['model_id'],model_id)
            self.assertEqual(plan['planned'],dict(reuse='auto',denoiser='graph') if model_id.endswith('-base') else dict(reuse='off',denoiser='native'))
            if not model_id.endswith('-base'):self.assertIn('model_size_not_gpu_qualified',plan['reasons'])
            self.assertEqual(plan_generation_execution(recipe=RECIPE,model_id=model_id,execution_profile='reference')['planned'],dict(reuse='off',denoiser='native'))
        self.assertEqual(len(keys),4)
        with self.assertRaises(ValueError):generation_capabilities(model_id='terramind-v1-base-tim')

    def test_unqualified_sizes_do_not_inherit_parallel_evidence(self):
        for model_id in GENERATION_MODELS:
            with GenerationEngine(model_id,model_file='unused',tokenizer_file='unused',work=self.root/model_id,
                                  execute_model=True,parallelism='auto') as engine:
                cap=engine.capabilities();self.assertEqual(cap['model_id'],model_id)
                if model_id.endswith('-base'):self.assertTrue(qualification(model_id))
                else:
                    self.assertEqual(qualification(model_id),{})
                    self.assertEqual(cap['parallel']['planned'],1)
                    self.assertEqual(cap['parallel']['qualification_profiles'],[])
                    self.assertFalse(cap['batching']['step_interleaving'])
            if not model_id.endswith('-base'):
                with self.assertRaisesRegex(ValueError,'only for terramind-v1-base'):
                    GenerationEngine(model_id,model_file='unused',tokenizer_file='unused',work=self.root/'unused',
                                     execute_model=True,parallelism=2,allow_unqualified_parallelism=True)
                with patch.object(profiles,'require_profile'),patch.object(offline,'inspect_config',return_value={'model_id':model_id}):
                    with self.assertRaisesRegex(ValueError,'base generation topology'):open_lane_session(config='unused',work='unused')

    def test_facade_forwards_model_to_config_and_session(self):
        for model_id in GENERATION_MODELS:
            fake=MagicMock();fake.close.return_value={'passed':True};fake.generate.return_value={'provenance':{}}
            with patch.object(offline,'create_config') as created,patch('aureon.api.generation_engine.open_model_session',return_value=fake) as opened:
                with GenerationEngine(model_id,model_file='m',tokenizer_file='t',work=self.root/model_id,execute_model=True) as engine:
                    engine.generate(self.data())
                    self.assertEqual(created.call_args.kwargs['model_id'],model_id)
                    self.assertEqual(opened.call_args.args[0],model_id)

    def test_route_preserves_all_generation_ids_and_rejects_encoder_modes(self):
        from aureon.serving.control import route
        from aureon.cloud.control import Error
        with patch('aureon.api.inspect_model',return_value={'full_reference_environments':['scene']}):
            for model_id in GENERATION_MODELS:
                result=route(model_id,'generation','fp32-generation-v1')
                self.assertEqual((result['model_id'],result['task']),(model_id,'generation'))
                self.assertEqual((result['reuse'],result['denoiser']),('auto','graph') if model_id.endswith('-base') else ('off','native'))
                explicit=route(model_id,'generation','fp32-generation-v1',denoiser='graph')
                self.assertEqual(explicit['denoiser'],'graph');self.assertIsNone(explicit['qualification_sha256'])
            for model_id in ('terramind-v1-tiny-backbone','terramind-v1-large-tim'):
                with self.assertRaises(Error):route(model_id,'generation','fp32-generation-v1')

    def test_http_worker_mismatch_fails_before_device_or_work(self):
        from aureon.serving.worker import work
        from aureon.cloud.control import Error
        control=SimpleNamespace(execution='engine',selected={'task':'generation','model_id':'terramind-v1-tiny'})
        with patch.object(offline,'inspect_config',return_value={'model_id':'terramind-v1-base'}), \
                patch('aureon.serving.devices.device_lease',side_effect=AssertionError('no device')):
            with self.assertRaises(Error):work(control,self.root/'worker',config='unused',execute_model=True)
        self.assertFalse((self.root/'worker').exists())

    def test_http_discovery_keeps_size_and_default_profile_after_reopen(self):
        import http.client
        import threading
        from aureon.serving.control import Control, initialize
        from aureon.cloud.http import Server
        for model_id in GENERATION_MODELS:
            state=self.root/model_id
            keys=initialize(state,['alpha'],model=model_id,environment='generation',profile='fp32-generation-v1',fake=True)
            with closing(Control(state)) as control:
                original=control.capabilities()
            with closing(Control(state)) as control:
                self.assertEqual(control.capabilities(),original)
                server=Server(control);thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01})
                thread.start();connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                try:
                    connection.request('GET','/v1/models',headers={'Authorization':'Bearer '+keys['alpha']})
                    response=connection.getresponse();value=json.loads(response.read());self.assertEqual(response.status,200)
                    self.assertEqual(value['models'][0]['model_id'],model_id)
                    self.assertEqual(value['models'][0]['model_id'],model_id)
                    self.assertFalse(value['models'][0]['hosted_qualification'])
                    self.assertEqual(value['models'][0]['per_job_numerical_status'],'not_evaluated')
                    self.assertEqual(value['models'][0]['denoiser'],'graph' if model_id.endswith('-base') else 'native')
                finally:
                    connection.close();server.shutdown();server.server_close();thread.join(5)
                self.assertFalse(thread.is_alive());self.assertEqual(server.socket.fileno(),-1)

    def test_loader_dispatches_exact_upstream_factory_without_pretrained(self):
        registry=ModuleType('terramind_register');registry.hf_hub_download=MagicMock(side_effect=AssertionError('download'))
        tokenizers=ModuleType('tokenizer_register');tokenizers.hf_hub_download=MagicMock(side_effect=AssertionError('download'))
        modules={name:ModuleType(name) for name in ('terratorch','terratorch.models','terratorch.models.backbones',
                 'terratorch.models.backbones.terramind','terratorch.models.backbones.terramind.model',
                 'terratorch.models.backbones.terramind.tokenizer')}
        modules['terratorch.models.backbones.terramind.model'].terramind_register=registry
        modules['terratorch.models.backbones.terramind.tokenizer'].tokenizer_register=tokenizers
        torch=ModuleType('torch');torch.float32='float32';dtype=['original']
        torch.cuda=SimpleNamespace(is_initialized=lambda:False)
        torch.get_default_dtype=lambda:dtype[0]
        torch.set_default_dtype=lambda value:dtype.__setitem__(0,value)
        modules['torch']=torch
        for model_id in GENERATION_MODELS:
            model=MagicMock();core=SimpleNamespace(encoder_embeddings={'untok_sen2l2a@224':object()},decoder_embeddings={'tok_sen1grd@224':object()})
            model.sampler=SimpleNamespace(model=core);model.encoder_embeddings=core.encoder_embeddings;model.decoder_embeddings=core.decoder_embeddings
            model.tokenizer={'tok_sen1grd@224':object()};model.state_dict.return_value={}
            factory=MagicMock(return_value=model);setattr(registry,get_terramind_spec(model_id).upstream_name,factory)
            config={'model_id':model_id,'model':{'sha256':'fixture'},'tokenizer':{'sha256':'fixture'}}
            with patch.dict(sys.modules,modules),patch.object(loader,'verify_config',return_value=(config,{'model':'m','tokenizer':'t'},RECIPE)), \
                    patch.object(loader,'strict_load',return_value={'matched_all_active_tensor_values':True}) as strict, \
                    patch.object(loader,'file_hash',return_value='fixture'),patch('aureon.terramind_native.rng.rng_capsule',return_value=nullcontext({})):
                result=loader.load_native('fixture')
            self.assertIs(result[0],model);self.assertEqual(result[1]['model_id'],model_id)
            self.assertEqual(factory.call_count,1);self.assertIs(factory.call_args.kwargs['pretrained'],False)
            self.assertEqual(factory.call_args.kwargs['modalities'],['S2L2A'])
            self.assertEqual(strict.call_count,2);self.assertEqual(dtype,['original'])

    def test_cold_generation_import_preserves_rng_on_success_and_failure(self):
        import builtins
        import random
        import numpy as np
        import torch
        original_import = builtins.__import__
        registry = SimpleNamespace(hf_hub_download=lambda:None)
        tokenizers = SimpleNamespace(hf_hub_download=lambda:None)
        model=MagicMock();core=SimpleNamespace(encoder_embeddings={'untok_sen2l2a@224':object()},decoder_embeddings={'tok_sen1grd@224':object()})
        model.sampler=SimpleNamespace(model=core);model.encoder_embeddings=core.encoder_embeddings;model.decoder_embeddings=core.decoder_embeddings
        model.tokenizer={'tok_sen1grd@224':object()};model.state_dict.return_value={}
        registry.terramind_v1_tiny=MagicMock()
        registry.terramind_v1_tiny_generate=MagicMock(return_value=model)
        config={'model_id':'terramind-v1-tiny','model':{'sha256':'fixture'},'tokenizer':{'sha256':'fixture'}}
        for fail in (False,True):
            with self.subTest(import_failure=fail):
                before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
                def importing(name,globals=None,locals=None,fromlist=(),level=0):
                    if name in ('terratorch.models.backbones.terramind.model','terratorch.models.backbones.terramind.tokenizer'):
                        random.random();np.random.random();torch.rand(1)
                        if fail:raise RuntimeError('cold generation import failure')
                        return SimpleNamespace(terramind_register=registry,tokenizer_register=tokenizers)
                    return original_import(name,globals,locals,fromlist,level)
                with patch.object(loader,'verify_config',return_value=(config,{'model':'m','tokenizer':'t'},RECIPE)), \
                        patch.object(loader,'strict_load',return_value={}),patch.object(loader,'file_hash',return_value='fixture'), \
                        patch.object(builtins,'__import__',side_effect=importing):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'cold generation import failure'):loader.load_native('fixture')
                    else:loader.load_native('fixture')
                self.assertEqual(random.getstate(),before[0])
                after=np.random.get_state();self.assertEqual(after[0],before[1][0]);self.assertTrue(np.array_equal(after[1],before[1][1]))
                self.assertEqual(after[2:],before[1][2:]);self.assertTrue(torch.equal(torch.get_rng_state(),before[2]))

    def test_strict_tensor_selection_rejects_missing_shape_dtype_and_active_extras(self):
        class Tensor:
            def __init__(self,shape=(2,2),dtype='float32'):self.shape=shape;self.dtype=dtype
        expected={'encoder_embeddings.s2.weight':Tensor()}
        good={'encoder_embeddings.s2.weight':Tensor(),'decoder_embeddings.unused.weight':Tensor()}
        selected,unused=loader.select_state(expected,good,allow_inactive_embeddings=True)
        self.assertEqual(set(selected),set(expected));self.assertEqual(unused,['decoder_embeddings.unused.weight'])
        bad=[{}, {'encoder_embeddings.s2.weight':Tensor((3,2))},
             {'encoder_embeddings.s2.weight':Tensor(dtype='float16')},
             {**good,'encoder_embeddings.s2.bias':Tensor()}, {**good,'unrelated.weight':Tensor()}]
        for state in bad:
            with self.assertRaises(ValueError):loader.select_state(expected,state,allow_inactive_embeddings=True)

    def test_strict_local_tensor_load_has_no_missing_weight_fallback(self):
        import torch
        module=torch.nn.Linear(2,3)
        state={name:torch.full_like(value,0.25) for name,value in module.state_dict().items()}
        path=self.root/'small-fixture.pt';torch.save(state,path)
        receipt=loader.strict_load(module,path)
        self.assertTrue(receipt['matched_all_active_tensor_values'])
        self.assertEqual(receipt['tensor_count'],2)
        for name,value in module.state_dict().items():self.assertTrue(torch.equal(value,state[name]))
        torch.save({'weight':state['weight']},path)
        with self.assertRaisesRegex(ValueError,'Missing active checkpoint tensors'):loader.strict_load(module,path)
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_result_and_provenance_keep_size_identity_with_cpu_double(self):
        # Load the sibling fixture from the selected source archive, not an
        # unrelated installed package named "tests" or the development checkout.
        spec=importlib.util.spec_from_file_location('terramind_family_session_fixture',Path(__file__).with_name('test_terramind_session.py'))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        fixtures=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        fixture=fixtures.SessionTests();fixture.setUp()
        try:
            for model_id in GENERATION_MODELS:
                fixture.config['model_id']=model_id
                fixture.config_path.write_text(json.dumps(fixture.config))
                session=fixture.session(model_id=model_id,execution_profile='reference')
                result=session.generate(self.data(),job_id='family',attempt_id='1')
                self.assertEqual(result['model_id'],model_id)
                self.assertEqual(result['provenance']['model_id'],model_id)
                self.assertEqual(result['provenance']['session']['model_id'],model_id)
                self.assertEqual(result['execution_selection']['model_id'],model_id)
                self.assertTrue(session.close()['passed'])
        finally:fixture.tearDown()


if __name__=='__main__':unittest.main()
