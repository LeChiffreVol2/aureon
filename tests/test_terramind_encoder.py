"""Synthetic CPU encoder contracts; no foundation weights or neural model forward."""
import ast
from dataclasses import replace
from functools import partial
import hashlib
import gc
import importlib.metadata
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
from types import ModuleType
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from aureon.api import terramind_encoder as api
from aureon.terramind_native import encoder_loader as loading
from aureon.terramind_native.families import AssetPin, get_terramind_spec


class EncoderBootstrapTests(unittest.TestCase):
    def test_retained_optional_import_error_cannot_hold_encoder_or_outputs(self):
        retained=ModuleType('synthetic_optional_import');seen=[];refs=[]
        original_import=api.importlib.import_module
        def imports(name,package=None):
            if name not in api._ENCODER_IMPORTS:return original_import(name,package)
            if not seen:
                exec("try:\n raise ImportError('optional fixture')\nexcept ImportError as error:\n import_error=error\n",retained.__dict__)
                random.random();np.random.random();torch.rand(1)
                torch.set_float32_matmul_precision('high');torch.set_default_dtype(torch.float64)
            seen.append((name,threading.get_ident()))
            module=ModuleType(name);sys.modules[name]=module
            return module
        class Constant:
            training=True
            def modules(self):return [self]
            def eval(self):self.training=False;return self
            def __call__(self,inputs):return [torch.ones((1,196,768),dtype=torch.float32) for _ in range(12)]
        def load(*args,**kwargs):
            for name in api._ENCODER_IMPORTS:api.importlib.import_module(name)
            return loading.LoadedEncoder(Constant(),{})
        original_own=api._own_features
        def own(*args,**kwargs):
            values=original_own(*args,**kwargs)
            refs.extend((weakref.ref(tensor),weakref.ref(tensor.untyped_storage())) for tensor in values)
            return values
        before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),
                torch.get_default_dtype(),torch.get_float32_matmul_precision())
        owner=threading.get_ident()
        with patch.dict(sys.modules,{name:None for name in api._ENCODER_IMPORTS}), \
                patch.object(api.importlib,'import_module',new=imports), \
                patch.object(api,'load_encoder',new=load),patch.object(api,'_own_features',new=own):
            engine=api.TerraMindEncoder('terramind-v1-base-backbone',model_file='absent',modalities=['S2L2A'],execute_model=True)
            runtime_ref=weakref.ref(engine)
            for _ in range(2):
                result=engine.predict({'S2L2A':torch.ones((1,12,224,224),dtype=torch.float32)},seed=17)
                result=None
                gc.collect()
                self.assertFalse(any(tensor() is not None or storage() is not None for tensor,storage in refs))
            engine.close();engine=None;gc.collect()
            self.assertIsNone(runtime_ref())
            error=retained.import_error
            names=[];files=[];frame=error.__traceback__.tb_frame
            while frame is not None:
                names.append(frame.f_code.co_name);files.append(frame.f_code.co_filename);frame=frame.f_back
            self.assertIn('_encoder_import_target',names)
            self.assertNotIn('predict',names)
            self.assertFalse(any(name=='run' and file.endswith('model_pool.py') for name,file in zip(names,files)))
            self.assertTrue(all(thread!=owner for _,thread in seen[:2]))
            self.assertEqual(len(seen),4)  # Two guarded cold imports + two warm loader imports.
        self.assertEqual(random.getstate(),before[0])
        after=np.random.get_state()
        self.assertEqual(after[0],before[1][0]);self.assertTrue(np.array_equal(after[1],before[1][1]))
        self.assertEqual(after[2:],before[1][2:])
        self.assertTrue(torch.equal(torch.get_rng_state(),before[2]))
        self.assertEqual(torch.get_default_dtype(),before[3]);self.assertEqual(torch.get_float32_matmul_precision(),before[4])

    def test_fatal_import_restores_policy_and_disables_failed_encoder(self):
        before=(random.getstate(),torch.get_rng_state().clone(),torch.get_default_dtype(),torch.get_float32_matmul_precision())
        def fail(name,package=None):
            random.random();torch.rand(1);torch.set_default_dtype(torch.float64)
            torch.set_float32_matmul_precision('high')
            raise ImportError('fatal import fixture')
        with patch.dict(sys.modules,{name:None for name in api._ENCODER_IMPORTS}),patch.object(api.importlib,'import_module',new=fail):
            engine=api.TerraMindEncoder('terramind-v1-base-backbone',model_file='absent',modalities=['S2L2A'],execute_model=True)
            with self.assertRaisesRegex(ImportError,'fatal import fixture'):
                engine.predict({'S2L2A':torch.ones((1,12,224,224),dtype=torch.float32)})
            self.assertEqual(engine.state,'failed');engine.close()
        self.assertEqual(random.getstate(),before[0]);self.assertTrue(torch.equal(torch.get_rng_state(),before[1]))
        self.assertEqual(torch.get_default_dtype(),before[2]);self.assertEqual(torch.get_float32_matmul_precision(),before[3])


class FakeEncoder(nn.Module):
    def __init__(self, modalities=('S2L2A',), targets=(), merge='mean', fail=False):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.tokenizer = nn.ModuleDict({loading.TOKEN_KEYS[m]: nn.Linear(1, 1) for m in set(targets) | (set(modalities) & {'LULC', 'NDVI'})})
        if targets:
            child = nn.Module()
            child.encoder = self.encoder
            child.decoder = nn.Linear(2, 2)
            child.mask_token = nn.Parameter(torch.zeros(1, 1, 2))
            self.sampler = nn.Module()
            self.sampler.model = child
        self.merge_method = merge
        self.fail = fail
        self.calls = 0
        self.observed = None

    def forward(self, inputs):
        self.calls += 1
        self.observed = (self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                         torch.is_autocast_enabled('cpu'), torch.get_float32_matmul_precision())
        tensor = next(iter(inputs.values()))
        tensor.add_(1)
        random.random(); np.random.random()
        value = torch.rand(tensor.shape[0], 196, 2)
        names = list(inputs)
        inputs.clear()
        if self.fail:
            raise RuntimeError('synthetic forward failure')
        return [{name: value.clone() for name in names}] if self.merge_method == 'dict' else [value]


class EncoderTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ.get('AUREON_ENCODER_TEST_ROOT', tempfile.gettempdir()))
        parent.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix='encoder-', dir=parent))
        # These fixtures replace loading with a synthetic model. Real import
        # isolation is exercised separately below without foundation weights.
        bootstrap=patch.object(api,'_prepare_encoder_modules',return_value=None)
        bootstrap.start();self.addCleanup(bootstrap.stop)

    def inputs(self):
        return {'S2L2A': torch.ones(1, 12, 224, 224)}

    def facade(self, **kwargs):
        return api.TerraMindEncoder('terramind-v1-base-backbone', model_file='missing',
                                  modalities=['S2L2A'], execute_model=True, **kwargs)

    def asset(self, name, state):
        path = self.work / name
        torch.save(state, path)
        data = path.read_bytes()
        return path, AssetPin('synthetic/unit-tests', 'fixture', name, hashlib.sha256(data).hexdigest(), len(data))

    def test_import_constructor_and_close_do_not_import_torch_or_touch_assets(self):
        code = '''import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','terratorch','rasterio'): raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from aureon.api.terramind_encoder import TerraMindEncoder
with TerraMindEncoder('terramind-v1-tiny-backbone',model_file='absent',modalities=['S2L2A'],execute_model=True) as e:
 assert e.state=='unloaded'
assert e.state=='closed'
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_execution_optin_is_required(self):
        with self.assertRaises(api.ExecutionRequired):
            api.TerraMindEncoder('terramind-v1-base-backbone', model_file='absent', modalities=['S2L2A'])

    def test_all_eight_ids_and_exact_tokenizer_requirements(self):
        for size in ('tiny', 'small', 'base', 'large'):
            for mode in ('backbone', 'tim'):
                with self.subTest(size=size, mode=mode):
                    e = api.TerraMindEncoder(f'terramind-v1-{size}-{mode}', model_file='absent',
                        modalities=['S2L2A'], tokenizer_files={'LULC': 'absent'} if mode == 'tim' else {}, execute_model=True)
                    self.assertEqual(e.state, 'unloaded')
                    e.close()
        for mapping in ({}, {'S1GRD': 'extra'}, {'LULC': 'ok', 'DEM': 'extra'}):
            with self.assertRaises(ValueError):
                api.TerraMindEncoder('terramind-v1-tiny-tim', model_file='absent', modalities=['S2L2A'],
                    tokenizer_files=mapping, execute_model=True)

    def test_validation_before_loading(self):
        with patch.object(api, 'load_encoder') as loader:
            e = self.facade()
            for data in ({}, {'S2L2A': torch.ones(1, 3, 224, 224)},
                         {'S2L2A': torch.ones(1, 12, 224, 224, dtype=torch.float64)},
                         {'S2L2A': torch.full((1, 12, 224, 224), float('nan'))}):
                with self.assertRaises(ValueError):
                    e.predict(data)
            for seed in (-1, True, 2**63):
                with self.assertRaises(ValueError):
                    e.predict(self.inputs(), seed=seed)
            loader.assert_not_called()
            self.assertEqual(e.state, 'unloaded')

    def test_invalid_modalities_modes_and_options_fail_early(self):
        base = dict(model_file='absent', modalities=['S2L2A'], execute_model=True)
        cases = [dict(modalities=['unknown']), dict(modalities='S2L2A'), dict(modalities=['DEM', 'DEM']),
                 dict(tim_decoding_steps=0), dict(tim_temperature=float('nan')), dict(tim_top_p=2),
                 dict(tim_top_k=True), dict(merge_method='sum'), dict(device='mps'), dict(tim_modalities=['DEM'])]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                api.TerraMindEncoder('terramind-v1-base-backbone', **(base | kwargs))
        with self.assertRaises(ValueError):
            api.TerraMindEncoder('terramind-v1-base', **base)
        with self.assertRaises(ValueError):
            api.TerraMindEncoder('terramind-v1-base-tim', **base, tim_modalities=['S2L2A'], tokenizer_files={'S2L2A':'x'})

    def test_resident_predict_replay_and_owned_input_output(self):
        fake = FakeEncoder().eval()
        data = self.inputs(); original = data['S2L2A'].clone()
        with patch.object(api, 'load_encoder', return_value=loading.LoadedEncoder(fake, {'nested': {'value': 1}})) as load:
            with self.facade() as e:
                first = e.predict(data, seed=17)
                second = e.predict(data, seed=17)
                third = e.predict(data, seed=18)
                self.assertTrue(torch.equal(first['features'][0], second['features'][0]))
                self.assertFalse(torch.equal(first['features'][0], third['features'][0]))
                self.assertTrue(torch.equal(data['S2L2A'], original))
                first['features'][0].zero_()
                self.assertFalse(torch.equal(first['features'][0], second['features'][0]))
                first['provenance']['nested']['value'] = 2
                self.assertEqual(second['provenance']['nested']['value'], 1)
                load.assert_called_once()
                self.assertEqual(fake.observed, (False, False, True, False, 'highest'))
                self.assertTrue(second['provenance']['rng']['restored'])
            self.assertEqual(e.state, 'closed'); e.close()
            with self.assertRaises(RuntimeError): e.predict(data)

    def test_rng_precision_modes_restore_after_success_and_failure(self):
        for fail in (False, True):
            fake = FakeEncoder(fail=fail).train()
            fake.encoder.eval()
            modes = [m.training for m in fake.modules()]
            torch_state = torch.get_rng_state().clone(); py = random.getstate(); np_state = np.random.get_state()
            old_precision = torch.get_float32_matmul_precision()
            try:
                torch.set_float32_matmul_precision('medium')
                with patch.object(api, 'load_encoder', return_value=loading.LoadedEncoder(fake, {})):
                    e = self.facade()
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, 'synthetic'): e.predict(self.inputs())
                        else:
                            e.predict(self.inputs())
                        self.assertTrue(torch.is_autocast_enabled('cpu'))
                    self.assertEqual(torch.get_float32_matmul_precision(), 'medium')
                    self.assertEqual([m.training for m in fake.modules()], modes)
                    self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
                    self.assertEqual(random.getstate(), py)
                    self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
                    if fail:
                        self.assertEqual(e.state, 'failed'); self.assertIsNone(e._loaded)
                        with self.assertRaises(RuntimeError): e.predict(self.inputs())
            finally:
                torch.set_float32_matmul_precision(old_precision)

    def test_merge_preserves_native_structure(self):
        for merge in ('mean', 'max', 'concat', 'dict', None):
            with patch.object(api, 'load_encoder', return_value=loading.LoadedEncoder(FakeEncoder(merge=merge), {})):
                result = self.facade(merge_method=merge).predict(self.inputs())
                self.assertIsInstance(result['features'][0], dict if merge == 'dict' else torch.Tensor)
                self.assertEqual(result['provenance']['merge_method'], merge)

    def test_busy_predict_and_close_are_rejected(self):
        e = self.facade(); e._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'busy'): e.predict(self.inputs())
            with self.assertRaisesRegex(RuntimeError, 'busy'): e.close()
        finally: e._lock.release()

    def test_native_bad_output_poison_session(self):
        fake = FakeEncoder()
        with patch.object(fake, 'forward', return_value=[torch.ones(1, 2)]), patch.object(api, 'load_encoder', return_value=loading.LoadedEncoder(fake, {})):
            e = self.facade()
            with self.assertRaisesRegex(ValueError, 'native FP32'): e.predict(self.inputs())
            self.assertEqual(e.state, 'failed')

    def test_retained_failure_traceback_does_not_retain_owned_model(self):
        refs=[]
        def construct(*args,**kwargs):
            model=FakeEncoder(fail=True)
            refs.append(weakref.ref(model))
            return loading.LoadedEncoder(model,{})
        caught=None
        with patch.object(api,'load_encoder',side_effect=construct):
            e=self.facade()
            try: e.predict(self.inputs())
            except RuntimeError as error: caught=error
        gc.collect()
        self.assertIsNotNone(caught)
        self.assertIsNone(refs[0]())
        self.assertIsNone(e._loaded)

    def test_full_band_lulc_and_batch_validation(self):
        e = api.TerraMindEncoder('terramind-v1-base-tim',model_file='x',modalities=['S2L2A','LULC'],
            tokenizer_files={'LULC':'x'},execute_model=True)
        good = self.inputs() | {'LULC': torch.zeros(1,224,224,dtype=torch.int64)}
        copied = e._inputs(good, torch.device('cpu')); self.assertEqual(copied['LULC'].dtype, torch.int64)
        for invalid in [torch.zeros(1,1,224,224,dtype=torch.int64),torch.zeros(2,224,224,dtype=torch.int64),torch.full((1,224,224),10,dtype=torch.int64)]:
            with self.assertRaises(ValueError): e._inputs(good | {'LULC':invalid},torch.device('cpu'))

    def flat_state(self, model):
        return {k.removeprefix('sampler.model.'):v.clone() for k,v in model.state_dict().items() if not k.startswith('tokenizer.')}

    def test_state_mapping_loads_both_tim_branches_and_tokenizers(self):
        model = FakeEncoder(targets=('LULC',))
        checkpoint = self.flat_state(model)
        checkpoint['encoder.weight'].fill_(7)
        tokens = {name:module.state_dict() for name,module in model.tokenizer.items()}
        receipt = loading.bind_encoder_state(model,checkpoint,tokens)
        self.assertTrue(receipt['all_active_tensors_matched'])
        self.assertEqual(receipt['random_fallback_tensors'],0)
        self.assertTrue(torch.equal(model.sampler.model.encoder.weight, checkpoint['encoder.weight']))

    def test_missing_shape_dtype_nonfinite_and_unexpected_rejected(self):
        model = FakeEncoder(); state = self.flat_state(model)
        bad = [dict(state, **{'encoder.weight':torch.zeros(3,2)}), dict(state, **{'encoder.weight':state['encoder.weight'].double()}),
               dict(state, **{'encoder.weight':torch.full((2,2),float('nan'))}), dict(state, surprise=torch.zeros(1))]
        missing=dict(state); missing.pop('encoder.weight'); bad.append(missing)
        for value in bad:
            with self.assertRaises(ValueError): loading.bind_encoder_state(model,value,{})

    def test_backbone_full_checkpoint_decoder_extras_are_recorded(self):
        model = FakeEncoder(); state = self.flat_state(model)
        state.update({'decoder.0.weight':torch.ones(2,2),'decoder_proj_context.weight':torch.ones(2,2),
                      'decoder_norm.weight':torch.ones(2),'mask_token':torch.zeros(1,1,2),
                      'encoder_embeddings.unused.weight':torch.ones(2)})
        receipt=loading.bind_encoder_state(model,state,{})
        self.assertEqual(len(receipt['unused_inactive_checkpoint_keys']),5)

    def test_missing_or_extra_tokenizer_state_rejected(self):
        model=FakeEncoder(targets=('LULC',)); state=self.flat_state(model)
        with self.assertRaisesRegex(ValueError,'Missing active'): loading.bind_encoder_state(model,state,{})
        token=model.tokenizer['tok_lulc@224'].state_dict(); token['extra']=torch.ones(1)
        with self.assertRaisesRegex(ValueError,'Unexpected or unused'): loading.bind_encoder_state(model,state,{'tok_lulc@224':token})

    def constructors(self, builder):
        path=Path(importlib.metadata.distribution('terratorch').locate_file('terratorch/models/backbones/terramind/model/terramind_register.py'))
        tree=ast.parse(path.read_text(encoding='utf-8'))
        namespace={'partial':partial,'LayerNorm':nn.LayerNorm,'nn':nn,'PRETRAINED_BANDS':{},'tokenizer_dict':{'v1':{}},
                   'build_terrammind_vit':builder,'build_terrammind_tim':builder,'hf_hub_download':lambda **kw:None}
        names={f'terramind_v1_{size}{suffix}' for size in ('tiny','small','base','large') for suffix in ('','_tim')}
        nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
        self.assertEqual(len(nodes),8)
        for node in nodes: node.decorator_list=[]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
        return SimpleNamespace(**{name:namespace[name] for name in names},hf_hub_download=namespace['hf_hub_download'])

    def test_actual_upstream_factory_routes_all_sizes_through_strict_local_loader(self):
        calls=[]
        def build(**kwargs):
            calls.append(kwargs)
            self.assertIs(kwargs['pretrained'],False)
            return FakeEncoder(kwargs['modalities'],kwargs.get('tim_modalities',()),kwargs['merge_method'])
        registry=self.constructors(build)
        for size in ('tiny','small','base','large'):
            for mode in ('backbone','tim'):
                with self.subTest(size=size,mode=mode):
                    spec,opts=loading.validate_options(f'terramind-v1-{size}-{mode}',['S2L2A'],None,'concat',1,1.0,0.8,0,True)
                    fake=FakeEncoder(targets=opts.tim_modalities)
                    path,pin=self.asset(f'{size}-{mode}.pt',self.flat_state(fake))
                    files={}; pins={}
                    for name in loading.required_tokenizers(opts):
                        files[name],pins[name]=self.asset(f'{size}-{mode}-{name}.pt',fake.tokenizer[loading.TOKEN_KEYS[name]].state_dict())
                    before=torch.get_rng_state().clone()
                    with patch.object(loading.importlib,'import_module',side_effect=[registry,SimpleNamespace(hf_hub_download=lambda:None)]), patch.object(loading,'TOKENIZER_ASSETS',pins):
                        loaded=loading.load_encoder(replace(spec,checkpoint=pin),opts,path,files,torch.device('cpu'))
                    self.assertEqual(loaded.model.merge_method,'concat')
                    self.assertFalse(loaded.model.training)
                    self.assertTrue(torch.equal(torch.get_rng_state(),before))
                    self.assertTrue(loaded.provenance['tensor_binding']['all_active_tensors_matched'])
                    self.assertEqual(calls[-1]['dim'],{'tiny':192,'small':384,'base':768,'large':1024}[size])
                    self.assertEqual(calls[-1]['encoder_depth'],24 if size=='large' else 12)
                    if mode=='tim': self.assertEqual(calls[-1]['decoder_depth'],{'tiny':4,'small':6,'base':12,'large':24}[size])

    def test_asset_mismatch_fails_before_constructor(self):
        spec,opts=loading.validate_options('terramind-v1-base-backbone',['S2L2A'],None,'mean',1,1.,.8,0,True)
        path,pin=self.asset('wrong.pt',{'x':torch.ones(1)})
        with patch.object(loading.importlib,'import_module') as imports:
            with self.assertRaisesRegex(ValueError,'pinned'): loading.load_encoder(spec,opts,path,{},torch.device('cpu'))
            imports.assert_not_called()

    def test_cold_upstream_import_restores_rng_and_precision_on_success_and_failure(self):
        # TerraTorch's transitive clay_v15 import sets global matmul to medium.
        # Import-time effects belong inside the same ownership boundary as loading.
        spec, opts = loading.validate_options('terramind-v1-tiny-backbone', ['S2L2A'], None, 'mean', 1, 1., .8, 0, True)
        path, pin = self.asset('cold-import.pt', self.flat_state(FakeEncoder()))
        old_precision = torch.get_float32_matmul_precision()
        try:
            for fail in (False, True):
                with self.subTest(import_failure=fail):
                    torch.set_float32_matmul_precision('highest')
                    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
                    def builder(**kwargs):
                        self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
                        return FakeEncoder()
                    registry = SimpleNamespace(terramind_v1_tiny=builder, hf_hub_download=lambda: None)
                    def importing(name):
                        torch.set_float32_matmul_precision('medium')
                        random.random(); np.random.random(); torch.rand(1)
                        if fail: raise RuntimeError('synthetic cold import failure')
                        return registry if name.endswith('terramind_register') else SimpleNamespace(hf_hub_download=lambda: None)
                    with patch.object(loading.importlib, 'import_module', side_effect=importing):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, 'cold import failure'):
                                loading.load_encoder(replace(spec, checkpoint=pin), opts, path, {}, torch.device('cpu'))
                        else:
                            loading.load_encoder(replace(spec, checkpoint=pin), opts, path, {}, torch.device('cpu'))
                    self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
                    self.assertEqual(random.getstate(), before[0])
                    after = np.random.get_state()
                    self.assertEqual(after[0], before[1][0]); self.assertTrue(np.array_equal(after[1], before[1][1]))
                    self.assertEqual(after[2:], before[1][2:]); self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        finally:
            torch.set_float32_matmul_precision(old_precision)

    def test_download_attempt_is_blocked_and_policies_restore(self):
        spec,opts=loading.validate_options('terramind-v1-base-backbone',['S2L2A'],None,'mean',1,1.,.8,0,True)
        path,pin=self.asset('model.pt',self.flat_state(FakeEncoder()))
        reg=SimpleNamespace(hf_hub_download=lambda:None)
        reg.terramind_v1_base=lambda **kw:reg.hf_hub_download()
        before=torch.get_rng_state().clone(); dtype=torch.get_default_dtype()
        with patch.object(loading.importlib,'import_module',side_effect=[reg,SimpleNamespace(hf_hub_download=lambda:None)]):
            with self.assertRaisesRegex(RuntimeError,'downloads forbidden'):
                loading.load_encoder(replace(spec,checkpoint=pin),opts,path,{},torch.device('cpu'))
        self.assertTrue(torch.equal(torch.get_rng_state(),before)); self.assertEqual(torch.get_default_dtype(),dtype)

    def test_default_dtype_restored_when_construction_raises(self):
        spec,opts=loading.validate_options('terramind-v1-small-backbone',['S2L2A'],None,'mean',1,1.,.8,0,True)
        path,pin=self.asset('model.pt',self.flat_state(FakeEncoder()))
        reg=SimpleNamespace(hf_hub_download=lambda:None)
        def constructor(**kwargs):
            self.assertEqual(torch.get_default_dtype(),torch.float32)
            raise RuntimeError('constructor failed')
        reg.terramind_v1_small=constructor
        old=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with patch.object(loading.importlib,'import_module',side_effect=[reg,SimpleNamespace(hf_hub_download=lambda:None)]):
                with self.assertRaisesRegex(RuntimeError,'constructor failed'):
                    loading.load_encoder(replace(spec,checkpoint=pin),opts,path,{},torch.device('cpu'))
            self.assertEqual(torch.get_default_dtype(),torch.float64)
        finally: torch.set_default_dtype(old)


if __name__ == '__main__':
    unittest.main()
