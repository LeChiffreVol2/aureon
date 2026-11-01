"""CPU tests of explicit precision control; CUDA discovery/TLS is simulated.

No GPU tensors, foundation models, assets, or native-precision qualification.
"""
from contextlib import ExitStack, contextmanager, nullcontext
import os
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from aureon.api import precision as policies
from aureon.api import terramind_encoder as api
from aureon.terramind_native.encoder_loader import LoadedEncoder
from aureon.terramind_native.rng import rng_capsule


@contextmanager
def cuda_fixture(capability=(8, 6), bf16=True, available=True):
    """Patch queries only; CUDA autocast TLS is real, CUDA kernels never execute."""
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, NVIDIA_TF32_OVERRIDE='1'))
        stack.enter_context(patch.object(torch.version, 'cuda', '12.6'))
        stack.enter_context(patch.object(torch.version, 'hip', None))
        stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=available))
        stack.enter_context(patch.object(torch.cuda, 'current_device', return_value=0))
        stack.enter_context(patch.object(torch.cuda, 'device_count', return_value=1))
        capability_query=stack.enter_context(patch.object(torch.cuda,'get_device_capability',return_value=capability))
        bf16_query=stack.enter_context(patch.object(torch.cuda,'is_bf16_supported',return_value=bf16))
        stack.enter_context(patch.object(torch.cuda,'device',side_effect=lambda *a,**k:nullcontext()))
        yield capability_query,bf16_query


@contextmanager
def two_cuda_devices():
    """Ambient device 0 lacks native BF16; requested device 1 supports it."""
    state = {'current': 0, 'bf16_queries': []}

    @contextmanager
    def select(device):
        index = device if isinstance(device, int) else torch.device(device).index
        assert index in (0, 1)
        previous = state['current']
        state['current'] = index
        try:
            yield
        finally:
            state['current'] = previous

    def bf16(including_emulation=True):
        state['bf16_queries'].append((state['current'], including_emulation))
        return state['current'] == 1

    with cuda_fixture(), \
            patch.object(torch.cuda, 'current_device', side_effect=lambda: state['current']), \
            patch.object(torch.cuda, 'device_count', return_value=2), \
            patch.object(torch.cuda, 'get_device_capability', side_effect=lambda i: (7, 0) if i == 0 else (8, 6)), \
            patch.object(torch.cuda, 'is_bf16_supported', side_effect=bf16), \
            patch.object(torch.cuda, 'device', side_effect=select):
        yield state


def reduction_state():
    matmul = torch.backends.cuda.matmul
    return (
        (matmul.allow_fp16_reduced_precision_reduction,
         matmul.allow_fp16_reduced_precision_reduction_split_k),
        (matmul.allow_bf16_reduced_precision_reduction,
         matmul.allow_bf16_reduced_precision_reduction_split_k),
        torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        getattr(matmul, 'allow_fp16_accumulation', None),
    )


@contextmanager
def caller_reductions(fp16=(False, False), bf16=(True, True)):
    matmul = torch.backends.cuda.matmul
    saved = reduction_state()
    try:
        matmul.allow_fp16_reduced_precision_reduction = fp16
        matmul.allow_bf16_reduced_precision_reduction = bf16
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        if saved[3] is not None:
            matmul.allow_fp16_accumulation = True
        yield reduction_state()
    finally:
        matmul.allow_fp16_reduced_precision_reduction = saved[0]
        matmul.allow_bf16_reduced_precision_reduction = saved[1]
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(saved[2])
        if saved[3] is not None:
            matmul.allow_fp16_accumulation = saved[3]


class SyntheticBackbone(nn.Module):
    def __init__(self, dtype=torch.float32, fail=False, nonfinite=False):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(2,2))
        self.dtype=dtype;self.fail=fail;self.nonfinite=nonfinite;self.seen=None

    def forward(self, inputs):
        self.seen=(torch.is_autocast_enabled('cuda'),torch.get_autocast_dtype('cuda'),
                   self.training,torch.is_grad_enabled(),self.weight.dtype,inputs['S2L2A'].dtype)
        value=inputs['S2L2A']
        value.add_(3)
        random.random();np.random.random();torch.rand(1)
        if self.fail: raise RuntimeError('synthetic forward failed')
        feature=(value.mean()*self.weight).unsqueeze(0).to(self.dtype)
        if self.nonfinite: feature[0,0,0]=float('inf')
        inputs.clear()
        return [feature]


class EncoderPrecisionTests(unittest.TestCase):
    def engine(self, policy='highest_fp32', **kwargs):
        return api.TerraMindEncoder('terramind-v1-tiny-backbone',model_file='absent',modalities=['S2L2A'],
            execute_model=True,precision=policy,**kwargs)

    def test_explicit_optin_and_cpu_and_mode_rejections_are_lazy(self):
        with patch.object(api,'load_encoder') as load,patch.object(torch.cuda,'is_available') as gpu:
            for policy in ('tf32','amp_bf16','amp_fp16'):
                with self.assertRaisesRegex(ValueError,'allow_experimental_precision'):
                    self.engine(policy,device='cuda')
                with self.assertRaisesRegex(ValueError,'CUDA'):
                    self.engine(policy,allow_experimental_precision=True)
                with self.assertRaisesRegex(ValueError,'Backbone'):
                    api.TerraMindEncoder('terramind-v1-tiny-tim',model_file='absent',modalities=['S2L2A'],
                        tokenizer_files={'LULC':'absent'},device='cuda',execute_model=True,
                        precision=policy,allow_experimental_precision=True)
                with self.assertRaisesRegex(ValueError,'Backbone'):
                    policies.validate_precision(policy,mode='generation',device='cuda',allow_experimental_precision=True)
            load.assert_not_called();gpu.assert_not_called()

    def test_bad_policy_and_non_bool_optin(self):
        with self.assertRaises(ValueError):self.engine('automatic')
        with self.assertRaises(ValueError):self.engine(allow_experimental_precision=1)

    def test_preflight_native_bf16_disables_emulation_and_selects_device(self):
        with cuda_fixture() as (cc,bf):
            receipt=policies.precision_preflight('amp_bf16',torch.device('cuda:0'))
            self.assertEqual(receipt['compute_capability'],[8,6]);cc.assert_called_once_with(0)
            bf.assert_called_once_with(including_emulation=False)

    def test_preflight_rejects_cpu_unavailable_architecture_emulation_and_rocm(self):
        with self.assertRaises(ValueError):policies.precision_preflight('tf32',torch.device('cpu'))
        with cuda_fixture(available=False),self.assertRaisesRegex(ValueError,'native NVIDIA'):
            policies.precision_preflight('tf32',torch.device('cuda:0'))
        for policy,cc in [('tf32',(7,5)),('amp_bf16',(7,5)),('amp_fp16',(6,1))]:
            with cuda_fixture(capability=cc),self.assertRaisesRegex(ValueError,'compute capability'):
                policies.precision_preflight(policy,torch.device('cuda:0'))
        with cuda_fixture(bf16=False),self.assertRaisesRegex(ValueError,'emulation'):
            policies.precision_preflight('amp_bf16',torch.device('cuda:0'))
        with cuda_fixture(),patch.object(torch.version,'hip','6.0'),self.assertRaises(ValueError):
            policies.precision_preflight('tf32',torch.device('cuda:0'))
        with cuda_fixture(),self.assertRaises(ValueError):
            policies.precision_preflight('tf32',torch.device('cuda:1'))

    def test_default_cpu_policy_does_not_probe_cuda(self):
        with patch.object(torch.cuda,'is_available',side_effect=AssertionError('probe')):
            with rng_capsule(17),policies.compute_precision_scope('highest_fp32',torch.device('cpu')) as record:
                self.assertFalse(torch.is_autocast_enabled('cpu'))
                self.assertEqual(record['matmul_precision'],'highest')

    def test_tf32_environment_override_rejects_before_cuda_probe(self):
        with patch.dict(os.environ, NVIDIA_TF32_OVERRIDE='0'), \
                patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA probe')):
            with self.assertRaisesRegex(ValueError, 'NVIDIA_TF32_OVERRIDE=0'):
                policies.precision_preflight('tf32', torch.device('cuda:0'))
            with rng_capsule(17), policies.compute_precision_scope('highest_fp32', torch.device('cpu')):
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        with cuda_fixture(), patch.dict(os.environ, NVIDIA_TF32_OVERRIDE='0'):
            for policy in ('amp_bf16', 'amp_fp16'):
                self.assertEqual(policies.precision_preflight(policy, torch.device('cuda:0'))['policy'], policy)

    def test_experimental_scope_uses_target_device_and_restores_on_forward_failure(self):
        original_autocast = torch.autocast
        for policy in ('tf32', 'amp_bf16', 'amp_fp16'):
            for fail in (False, True):
                with self.subTest(policy=policy, fail=fail), two_cuda_devices() as state:
                    def construct(*args, **kwargs):
                        self.assertEqual(state['current'], 1)
                        return original_autocast(*args, **kwargs)
                    with patch.object(torch, 'autocast', side_effect=construct):
                        try:
                            with rng_capsule(17), policies.compute_precision_scope(policy, torch.device('cuda:1')) as record:
                                self.assertEqual(state['current'], 1)
                                self.assertEqual(record['device'], 'cuda:1')
                                self.assertEqual(torch.is_autocast_enabled('cuda'), policy.startswith('amp_'))
                                if fail:
                                    raise RuntimeError('target forward failed')
                        except RuntimeError as error:
                            self.assertTrue(fail)
                            self.assertEqual(str(error), 'target forward failed')
                    self.assertEqual(state['current'], 0)
                    if policy == 'amp_bf16':
                        self.assertIn((1, False), state['bf16_queries'])
                        self.assertIn((1, True), state['bf16_queries'])
                        self.assertTrue(all(index == 1 for index, _ in state['bf16_queries']))

    def test_target_device_restores_when_preflight_or_autocast_construction_fails(self):
        for stage in ('preflight', 'autocast'):
            with self.subTest(stage=stage), two_cuda_devices() as state, caller_reductions() as reductions:
                precision = torch.get_float32_matmul_precision()
                cache = torch.is_autocast_cache_enabled()
                def fail_construction(*args, **kwargs):
                    self.assertEqual(state['current'], 1)
                    raise RuntimeError('autocast construction failed')
                failure = (patch.object(torch.cuda, 'is_bf16_supported', return_value=False)
                           if stage == 'preflight' else patch.object(torch, 'autocast', side_effect=fail_construction))
                with failure, self.assertRaisesRegex((ValueError, RuntimeError), 'emulation|construction failed'):
                    with rng_capsule(17), policies.compute_precision_scope('amp_bf16', torch.device('cuda:1')):
                        self.fail('An unsupported precision scope reached forward')
                self.assertEqual(state['current'], 0)
                self.assertEqual(reduction_state(), reductions)
                self.assertEqual(torch.get_float32_matmul_precision(), precision)
                self.assertEqual(torch.is_autocast_cache_enabled(), cache)

    def test_amp_owns_reductions_split_k_accumulation_and_cache_and_restores_exactly(self):
        for policy in ('amp_bf16', 'amp_fp16'):
            for fp16, bf16 in (((False, False), (False, True)),
                              ((False, True), (False, False)), ((True, True), (True, True))):
                for fail in (False, True):
                    with self.subTest(policy=policy, fp16=fp16, bf16=bf16, fail=fail), \
                            cuda_fixture(), caller_reductions(fp16, bf16) as before, \
                            torch.autocast('cuda', dtype=torch.float16, cache_enabled=True):
                        try:
                            with rng_capsule(17), policies.compute_precision_scope(policy, torch.device('cuda:0')) as record:
                                self.assertEqual(reduction_state(), ((False, True), (False, True), False,
                                                                    False if before[3] is not None else None))
                                self.assertFalse(torch.is_autocast_cache_enabled())
                                self.assertFalse(record['autocast_cache_enabled'])
                                for key in ('cuda_matmul_allow_fp16_reduced_precision_reduction',
                                            'cuda_matmul_allow_bf16_reduced_precision_reduction',
                                            'math_sdp_allow_fp16_bf16_reduction'):
                                    self.assertIs(record[key], False)
                                for key in ('cuda_matmul_allow_fp16_reduced_precision_reduction_split_k',
                                            'cuda_matmul_allow_bf16_reduced_precision_reduction_split_k'):
                                    self.assertIs(record[key], True)
                                self.assertIs(record['cuda_matmul_allow_fp16_accumulation'],
                                              False if before[3] is not None else None)
                                if fail:
                                    raise RuntimeError('reduction scope failed')
                        except RuntimeError as error:
                            self.assertTrue(fail)
                            self.assertEqual(str(error), 'reduction scope failed')
                        self.assertEqual(reduction_state(), before)
                        self.assertTrue(torch.is_autocast_cache_enabled())
                        self.assertTrue(torch.is_autocast_enabled('cuda'))
                        self.assertEqual(torch.get_autocast_dtype('cuda'), torch.float16)

    def test_fp32_policies_retain_callers_reduction_and_cache_settings(self):
        for policy in ('highest_fp32', 'tf32'):
            with self.subTest(policy=policy), cuda_fixture(), caller_reductions() as before, \
                    torch.autocast('cuda', dtype=torch.float16, cache_enabled=True):
                with rng_capsule(17), policies.compute_precision_scope(policy, torch.device('cuda:0')) as record:
                    self.assertEqual(reduction_state(), before)
                    self.assertTrue(torch.is_autocast_cache_enabled())
                    self.assertNotIn('math_sdp_allow_fp16_bf16_reduction', record)
                self.assertEqual(reduction_state(), before)
        with patch.object(torch.cuda, 'device', side_effect=AssertionError('Default changed device')):
            with rng_capsule(17), policies.compute_precision_scope('highest_fp32', torch.device('cuda:0')):
                pass

    def test_optional_fp16_accumulation_unavailable_is_recorded(self):
        matmul_type = type(torch.backends.cuda.matmul)
        original = matmul_type.__getattr__
        def without_accumulation(instance, name):
            if name == 'allow_fp16_accumulation':
                raise AttributeError(name)
            return original(instance, name)
        with patch.object(matmul_type, '__getattr__', without_accumulation):
            before = reduction_state()
            with policies._amp_reduction_policy() as record:
                self.assertIsNone(record['cuda_matmul_allow_fp16_accumulation'])
            self.assertEqual(reduction_state(), before)

    def test_unapplied_reduction_or_cache_policy_fails_and_restores(self):
        original_autocast = torch.autocast
        def ignore_cache(*args, **kwargs):
            kwargs['cache_enabled'] = True
            return original_autocast(*args, **kwargs)
        for setting in ('reduction', 'cache'):
            with self.subTest(setting=setting), cuda_fixture(), caller_reductions() as before:
                ignored = (patch.object(torch.backends.cuda, 'allow_fp16_bf16_reduction_math_sdp')
                           if setting == 'reduction' else patch.object(torch, 'autocast', side_effect=ignore_cache))
                with ignored, self.assertRaisesRegex(RuntimeError, 'policy was not applied'):
                    with rng_capsule(17), policies.compute_precision_scope('amp_fp16', torch.device('cuda:0')):
                        self.fail('An unapplied setting reached forward')
                self.assertEqual(reduction_state(), before)

    def test_context_restores_precision_dtype_both_autocast_states_and_rng_on_error(self):
        saved=(torch.get_default_dtype(),torch.get_float32_matmul_precision(),torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)
        try:
            for policy in ('highest_fp32','tf32','amp_bf16','amp_fp16'):
                for fail in (False,True):
                    with self.subTest(policy=policy,fail=fail),cuda_fixture():
                        torch.set_default_dtype(torch.float64);torch.set_float32_matmul_precision('medium')
                        torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
                        rng=torch.get_rng_state().clone();py=random.getstate();npstate=np.random.get_state()
                        with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False),torch.autocast('cpu',dtype=torch.bfloat16):
                            try:
                                with rng_capsule(7),policies.compute_precision_scope(policy,torch.device('cuda:0')) as record:
                                    self.assertEqual(torch.get_default_dtype(),torch.float32)
                                    self.assertEqual(record['applied_policy'],policy)
                                    self.assertEqual(torch.is_autocast_enabled('cuda'),policy.startswith('amp_'))
                                    self.assertTrue(torch.is_autocast_enabled('cpu'))
                                    self.assertTrue(torch.is_inference_mode_enabled())
                                    torch.rand(1);random.random();np.random.random()
                                    if fail:raise RuntimeError('scope failure')
                            except RuntimeError as error:
                                self.assertTrue(fail);self.assertEqual(str(error),'scope failure')
                            self.assertTrue(torch.is_autocast_enabled('cuda'));self.assertEqual(torch.get_autocast_dtype('cuda'),torch.float16)
                            self.assertFalse(torch.is_autocast_cache_enabled());self.assertTrue(torch.is_autocast_enabled('cpu'))
                        self.assertEqual(torch.get_default_dtype(),torch.float64)
                        self.assertEqual(torch.get_float32_matmul_precision(),'medium')
                        self.assertTrue(torch.backends.cudnn.allow_tf32);self.assertTrue(torch.backends.cudnn.benchmark)
                        self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(random.getstate(),py)
                        self.assertTrue(np.array_equal(np.random.get_state()[1],npstate[1]))
        finally:
            torch.set_default_dtype(saved[0]);torch.set_float32_matmul_precision(saved[1]);torch.backends.cudnn.allow_tf32=saved[2];torch.backends.cudnn.benchmark=saved[3]

    def test_silent_autocast_disable_is_error_and_restores_context(self):
        before=torch.get_float32_matmul_precision()
        with cuda_fixture(),patch.object(torch,'autocast',return_value=nullcontext()),self.assertRaisesRegex(RuntimeError,'not applied'):
            with rng_capsule(17),policies.compute_precision_scope('amp_bf16',torch.device('cuda:0')):pass
        self.assertEqual(torch.get_float32_matmul_precision(),before)

    def test_global_scope_cannot_overlap_another_aureon_rng_owner(self):
        with rng_capsule(1):
            with self.assertRaisesRegex(RuntimeError,'already in use'):
                with rng_capsule(2),policies.compute_precision_scope('highest_fp32',torch.device('cpu')):pass

    @contextmanager
    def resident_fixture(self,policy,fake):
        device='cpu' if policy=='highest_fp32' else 'cuda:0'
        engine=self.engine(policy,device=device,allow_experimental_precision=policy!='highest_fp32')
        # Only CPU tensors and CPU RNG snapshots. CUDA discovery/TLS are simulated.
        original_inputs=engine._inputs
        with cuda_fixture(),patch.object(api,'_prepare_encoder_modules'),patch.object(api,'load_encoder',return_value=LoadedEncoder(fake,{'precision':'float32/highest'})) as load, \
                patch.object(engine,'_inputs',side_effect=lambda values,device:original_inputs(values,torch.device('cpu'))), \
                patch.object(api,'rng_capsule',side_effect=lambda seed,devices: rng_capsule(seed)):
            yield engine,load
        engine.close()

    def test_forward_policy_reaches_runtime_and_returns_owned_fp32_with_provenance(self):
        for policy,dtype in [('highest_fp32',torch.float32),('tf32',torch.float32),('amp_bf16',torch.bfloat16),('amp_fp16',torch.float16)]:
            with self.subTest(policy=policy):
                fake=SyntheticBackbone(dtype=dtype).eval();before=fake.weight.detach().clone()
                inputs={'S2L2A':torch.ones(1,12,224,224)};input_copy=inputs['S2L2A'].clone()
                with self.resident_fixture(policy,fake) as (engine,load):
                    result=engine.predict(inputs);again=engine.predict(inputs)
                    self.assertEqual(result['features'][0].dtype,torch.float32)
                    self.assertTrue(torch.equal(result['features'][0],again['features'][0]))
                    result['features'][0].zero_();self.assertFalse(torch.equal(result['features'][0],again['features'][0]))
                    self.assertTrue(torch.equal(inputs['S2L2A'],input_copy));self.assertTrue(torch.equal(fake.weight,before))
                    record=result['provenance']['compute_precision']
                    self.assertEqual(record['applied_policy'],policy)
                    self.assertEqual(record['observed_native_output_dtypes'],[str(dtype)])
                    self.assertEqual(record['public_output_dtype'],'torch.float32')
                    self.assertEqual(fake.seen[0],policy.startswith('amp_'))
                    self.assertEqual(fake.seen[2:],(False,False,torch.float32,torch.float32))
                    self.assertEqual(engine.configuration()['precision'],policy);load.assert_called_once()

    def test_preflight_failure_does_not_load_or_copy_inputs(self):
        engine=self.engine('amp_bf16',device='cuda:0',allow_experimental_precision=True)
        with cuda_fixture(capability=(7,5)),patch.object(api,'load_encoder') as load,patch.object(engine,'_inputs') as inputs:
            with self.assertRaises(ValueError):engine.predict({})
            load.assert_not_called();inputs.assert_not_called();self.assertEqual(engine.state,'unloaded')

    def test_bad_native_dtype_and_nonfinite_fail_closed_restore_and_release(self):
        for policy,dtype,nonfinite in [('amp_bf16',torch.float16,False),('amp_fp16',torch.bfloat16,False),
                                      ('tf32',torch.float16,False),('amp_bf16',torch.float64,False),
                                      ('amp_fp16',torch.float16,True),('amp_bf16',torch.bfloat16,True)]:
            with self.subTest(policy=policy,dtype=dtype,nonfinite=nonfinite):
                fake=SyntheticBackbone(dtype=dtype,nonfinite=nonfinite).train()
                with self.resident_fixture(policy,fake) as (engine,_):
                    before=torch.get_float32_matmul_precision();rng=torch.get_rng_state().clone()
                    with self.assertRaises(ValueError):engine.predict({'S2L2A':torch.ones(1,12,224,224)})
                    self.assertEqual(engine.state,'failed');self.assertIsNone(engine._loaded);self.assertTrue(fake.training)
                    self.assertEqual(torch.get_float32_matmul_precision(),before);self.assertTrue(torch.equal(torch.get_rng_state(),rng))

    def test_forward_exception_restores_all_policies(self):
        with self.resident_fixture('amp_bf16',SyntheticBackbone(fail=True)) as (engine,_):
            precision=torch.get_float32_matmul_precision();rng=torch.get_rng_state().clone()
            with self.assertRaisesRegex(RuntimeError,'forward failed'):engine.predict({'S2L2A':torch.ones(1,12,224,224)})
            self.assertEqual(torch.get_float32_matmul_precision(),precision);self.assertTrue(torch.equal(torch.get_rng_state(),rng))
            self.assertFalse(torch.is_autocast_enabled('cuda'))

    def test_mixed_native_fp32_and_target_amp_dtypes_and_dict_are_recorded(self):
        for policy,dtype in [('amp_bf16',torch.bfloat16),('amp_fp16',torch.float16)]:
            seen=set();native=[{'S2L2A':torch.ones(1,2,3,dtype=dtype),'DEM':torch.ones(1,2,3)}]
            output=api._own_features(native,merge_method='dict',precision=policy,native_dtypes=seen)
            self.assertEqual(seen,{str(dtype),'torch.float32'})
            self.assertTrue(all(t.dtype==torch.float32 for t in output[0].values()))


if __name__=='__main__':unittest.main()
