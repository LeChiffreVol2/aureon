import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import unittest

import torch
from torch import nn
from aureon.api import ModelBridge, BridgeCancelled, PreprocessingRecipe


def forward(model, inputs):
    return model(inputs['image'])


def output(value):
    return {'logits': value}


class Stateful(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('count', torch.zeros(1))
        self.calls = 0

    def forward(self, value):
        self.count.add_(1); self.calls += 1
        value.add_(1)
        random.random(); torch.rand(1)
        return value + self.count


class BridgeTests(unittest.TestCase):
    def test_checkpoint_round_trip_and_identity(self):
        model = nn.Linear(3,2).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'weights.pt'; torch.save(model.state_dict(), path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with ModelBridge.from_checkpoint({'in':3,'out':2}, path, expected_sha256=digest,
                    factory=lambda c: nn.Linear(c['in'],c['out']), forward=forward, output=output) as bridge:
                x = torch.randn(4,3)
                self.assertTrue(torch.equal(bridge.predict({'image':x})['logits'],model(x)))
                record = bridge.metadata()['checkpoint']
                self.assertEqual(set(record['tensor_identity']), {'weight','bias'})
                self.assertTrue(record['strict'])
                self.assertEqual(bridge.metadata()['ownership'],'engine_owned_snapshot')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                ModelBridge.from_checkpoint({},path,expected_sha256='0'*64,factory=lambda c:model,
                                            forward=forward,output=output)

    def test_checkpoint_rejects_keys_dtype_and_shape(self):
        model = nn.Linear(3,2).eval()
        cases=[{}, {'weight':torch.ones(2,3,dtype=torch.float64),'bias':torch.zeros(2)},
               {'weight':torch.ones(3,3),'bias':torch.zeros(2)}]
        with tempfile.TemporaryDirectory() as folder:
            for state in cases:
                path=Path(folder)/'weights.pt';torch.save(state,path)
                with self.assertRaises(ValueError):
                    ModelBridge.from_checkpoint({},path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        factory=lambda c:nn.Linear(3,2),forward=forward,output=output)

    def test_snapshot_ownership_state_and_rng(self):
        model=Stateful().eval();x=torch.zeros(2,3)
        rng=torch.get_rng_state().clone();py=random.getstate()
        bridge=ModelBridge(model,forward=forward,output=output)
        first=bridge.predict({'image':x})['logits'];first.zero_()
        second=bridge.predict({'image':x})['logits']
        self.assertTrue(torch.equal(second,torch.full_like(x,2)))
        self.assertEqual(model.calls,0);self.assertEqual(model.count.item(),0)
        self.assertTrue(torch.equal(x,torch.zeros_like(x)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(py,random.getstate())
        bridge.close();self.assertFalse(model.training)
        with self.assertRaises(RuntimeError):bridge.predict({'image':x})

    def test_failure_restores_flags_rng_and_model(self):
        model=Stateful().eval()
        def failing(m,inputs):
            m(inputs['image']);torch.set_default_dtype(torch.float64)
            raise RuntimeError('forward failed')
        bridge=ModelBridge(model,forward=failing,output=output)
        rng=torch.get_rng_state().clone();dtype=torch.get_default_dtype()
        with self.assertRaisesRegex(RuntimeError,'forward failed'):
            bridge.predict({'image':torch.zeros(1,3)})
        self.assertEqual(bridge.state,'failed');self.assertEqual(torch.get_default_dtype(),dtype)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(model.count.item(),0)

    def test_eval_dtype_and_output_contract(self):
        with self.assertRaises(ValueError):ModelBridge(nn.Linear(2,2),forward=forward,output=output)
        with self.assertRaises(ValueError):ModelBridge(nn.Linear(2,2).double().eval(),forward=forward,output=output)
        bridge=ModelBridge(nn.Identity().eval(),forward=forward,output=lambda v:{'bad':v.double()})
        with self.assertRaises(ValueError):bridge.predict({'image':torch.ones(1,2)})

    def test_concurrency_close_and_cancellation(self):
        entered=threading.Event();release=threading.Event();cancel=threading.Event();errors=[]
        def slow(m,inputs):
            entered.set();release.wait(5);return m(inputs['image'])
        bridge=ModelBridge(nn.Identity().eval(),forward=slow,output=output)
        def run():
            try:bridge.predict({'image':torch.ones(1,2)},cancel=cancel)
            except BaseException as e:errors.append(e)
        worker=threading.Thread(target=run);worker.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaisesRegex(RuntimeError,'busy'):bridge.predict({'image':torch.ones(1,2)})
            with self.assertRaisesRegex(RuntimeError,'busy'):bridge.close()
            cancel.set()
        finally:release.set();worker.join(6)
        self.assertFalse(worker.is_alive());self.assertIsInstance(errors[0],BridgeCancelled)
        self.assertEqual(bridge.state,'ready')
        with self.assertRaises(BridgeCancelled):bridge.predict({'image':torch.ones(1,2)},cancel=cancel)
        bridge.close()

    def test_preprocessing_order_math_and_ownership(self):
        recipe=PreprocessingRecipe(('red','nir'),'dn','standardized',scale=.1,mean=(1.,2.),std=(2.,4.))
        x=torch.arange(8,dtype=torch.float32).reshape(1,2,2,2);before=x.clone()
        prepared=recipe.apply(x,bands=('nir','red'),units='dn')
        expected=(x[:,[1,0]]*.1-torch.tensor([1.,2.]).reshape(1,2,1,1))/torch.tensor([2.,4.]).reshape(1,2,1,1)
        self.assertTrue(torch.equal(prepared.tensor,expected));prepared.tensor.zero_()
        self.assertTrue(torch.equal(x,before))
        with self.assertRaisesRegex(ValueError,'double'):recipe.apply(prepared,bands=prepared.bands,units=prepared.units)
        with self.assertRaises(ValueError):recipe.apply(x,bands=('nir','red'),units='unknown')
        with self.assertRaises(ValueError):recipe.apply(x,bands=('red','red'),units='dn')

    def test_native_preprocessing_boundaries(self):
        bands=('B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12')
        recipe=PreprocessingRecipe(bands,'surface_reflectance_x10000','surface_reflectance_x10000',workflow='generation_s2l2a')
        x=torch.ones(1,12,2,2)
        self.assertTrue(torch.equal(x,recipe.apply(x,bands=bands,units=recipe.input_units).tensor))
        with self.assertRaises(ValueError):PreprocessingRecipe(bands[:3],'raw','standardized',workflow='encoder_s2l2a')
        with self.assertRaises(ValueError):PreprocessingRecipe(bands,'raw','standardized',workflow='encoder_s2l2a')
        with self.assertRaises(ValueError):PreprocessingRecipe(bands,'raw','standardized',workflow='generation_s2l2a')

    def test_cold_import_has_no_optional_runtime(self):
        public=Path(__file__).resolve().parents[1]
        code='import sys;sys.path.insert(0,'+repr(str(public))+');import aureon.api;assert not set(sys.modules)&{"torch","numpy","terratorch"}'
        subprocess.run([sys.executable,'-I','-B','-c',code],check=True,capture_output=True)

    def test_rng_contention_is_retryable_without_rebuilding(self):
        from aureon.terramind_native.rng import rng_capsule
        bridge = ModelBridge(nn.Identity().eval(), forward=forward, output=output)
        with rng_capsule(9):
            with self.assertRaisesRegex(RuntimeError, 'RNG capsule'):
                bridge.predict({'image': torch.ones(1, 2)})
        self.assertEqual(bridge.state, 'ready')
        self.assertTrue(torch.equal(bridge.predict({'image': torch.ones(1, 2)})['logits'], torch.ones(1, 2)))
        bridge.close()

    def test_invalid_seed_is_retryable(self):
        bridge = ModelBridge(nn.Identity().eval(), forward=forward, output=output)
        for seed in (-1, True, 2**63):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                bridge.predict({'image': torch.ones(1, 2)}, seed=seed)
            self.assertEqual(bridge.state, 'ready')
        bridge.close()

    def test_constructor_rejects_noncallable_adapters(self):
        for adapters in ({'forward': None, 'output': output}, {'forward': forward, 'output': None}):
            with self.subTest(adapters=adapters), self.assertRaises(TypeError):
                ModelBridge(nn.Identity().eval(), **adapters)

    def test_metadata_is_json_serializable_and_detached(self):
        bridge = ModelBridge(nn.Identity().eval(), forward=forward, output=output)
        metadata = json.loads(json.dumps(bridge.metadata()))
        self.assertEqual(metadata['capabilities']['precision'], 'highest_fp32')
        metadata['capabilities']['precision'] = 'changed'
        self.assertEqual(json.loads(json.dumps(bridge.metadata()))['capabilities']['precision'], 'highest_fp32')
        bridge.close()

    def test_preprocessing_inside_inference_mode_returns_ordinary_tensor(self):
        recipe = PreprocessingRecipe(('red',), 'dn', 'dn')
        with torch.inference_mode():
            x = torch.ones(1, 1, 2, 2)
            prepared = recipe.apply(x, bands=('red',), units='dn')
        self.assertFalse(torch.is_inference(prepared.tensor))
        prepared.tensor.add_(1)
        self.assertTrue(torch.equal(x, torch.ones(1, 1, 2, 2)))

    def test_identity_preprocessing_preserves_fp32_bytes(self):
        recipe = PreprocessingRecipe(('red',), 'dn', 'dn')
        x = torch.tensor([-0.0, 0.0, -1.0, 1.0]).reshape(1, 1, 2, 2)
        prepared = recipe.apply(x, bands=('red',), units='dn')
        self.assertTrue(torch.equal(x.view(torch.uint8), prepared.tensor.view(torch.uint8)))

    def test_preprocessing_rejects_empty_images_and_invalid_units(self):
        recipe = PreprocessingRecipe(('red',), 'dn', 'dn')
        for shape in ((0, 1, 2, 2), (1, 1, 0, 2), (1, 1, 2, 0)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                recipe.apply(torch.empty(shape), bands=('red',), units='dn')
        for units in (42, True, '   '):
            with self.subTest(units=units), self.assertRaises(ValueError):
                PreprocessingRecipe(('red',), units, 'dn')

    def test_preprocessing_uses_cpu_statistics_under_another_default_device(self):
        recipe = PreprocessingRecipe(('red',), 'dn', 'standardized', mean=(1.,), std=(2.,))
        x = torch.full((1, 1, 2, 2), 3., device='cpu')
        with torch.device('meta'):
            prepared = recipe.apply(x, bands=('red',), units='dn')
        self.assertEqual(prepared.tensor.device.type, 'cpu')
        self.assertTrue(torch.equal(prepared.tensor, torch.ones_like(x)))

    def test_bad_input_contract_is_retryable_and_bad_output_invalidates(self):
        bridge = ModelBridge(nn.Identity().eval(), forward=forward, output=output)
        for inputs in ([], {'': torch.ones(1)}, {'x': torch.ones(1, dtype=torch.float64)}):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                bridge.predict(inputs)
            self.assertEqual(bridge.state, 'ready')
        bridge.close()
        bridge = ModelBridge(nn.Identity().eval(), forward=forward, output=lambda value: [value])
        with self.assertRaisesRegex(ValueError, 'Output adapter'):
            bridge.predict({'image': torch.ones(1)})
        self.assertEqual(bridge.state, 'failed')
        bridge.close()

    def test_standardized_values_cannot_be_relabelled_as_raw_generation(self):
        bands = ('B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12')
        with self.assertRaises(ValueError):
            PreprocessingRecipe(bands, 'standardized', 'surface_reflectance_x10000',
                                workflow='generation_s2l2a')

    def test_model_and_checkpoint_reject_nonfinite_or_complex_tensors(self):
        complex_model = nn.Module().eval()
        complex_model.register_buffer('invalid', torch.zeros(1, dtype=torch.complex64))
        with self.assertRaises(ValueError):
            ModelBridge(complex_model, forward=forward, output=output)
        model = nn.Linear(2, 2).eval()
        with torch.no_grad():
            model.weight.fill_(float('inf'))
        with self.assertRaises(ValueError):
            ModelBridge(model, forward=forward, output=output)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'nonfinite.pt'
            torch.save(model.state_dict(), path)
            with self.assertRaises(ValueError):
                ModelBridge.from_checkpoint({}, path,
                    expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    factory=lambda c: nn.Linear(2, 2), forward=forward, output=output)


if __name__=='__main__':unittest.main()
