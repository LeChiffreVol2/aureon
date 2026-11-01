"""COP-GEN contract/ownership/COG tests with CPU fixtures; no model assets loaded."""
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
import weakref
from unittest.mock import Mock, patch

import numpy as np

from aureon.api.eo import ConditioningSpec, EOGrid, EORequest, GeneratedTarget, SamplingRecipe, UnsupportedRoute
from aureon.copgen_native import CopgenInput, CopgenEngine, export_copgen, plan_copgen
from aureon.copgen_native.contracts import BANDS, ADAPTER
from aureon.copgen_native import engine as implementation
from aureon.copgen_native.profiles import profile_requirements, validate_profile


def request():
    grid=EOGrid('EPSG:32647',(10.,0.,632420.,0.,-10.,1583670.),(192,192))
    return EORequest('copgen-base',
        (ConditioningSpec('S2L2A',BANDS,'surface_reflectance_x10000','none','explicit_all_valid',grid,'real-crop'),),
        (GeneratedTarget('S1RTC',('VV','VH'),'linear_backscatter',grid),),
        SamplingRecipe(ADAPTER,17,(('sample_steps',50),)),'reference')


class FakeRuntime:
    instances=[]
    def __init__(self,*args,**kwargs):
        self.identity={'fixture':True};self.calls=[];self.closed=False;self.fail=False
        self.profile=kwargs.get('dependency_profile')
        self.instances.append(self)
    def generate(self,image,plan):
        self.calls.append((image.copy(),plan))
        if self.fail:raise RuntimeError('injected native failure')
        image[:]=0
        return {'array':np.ones((2,192,192),np.float32), 'decoded':np.zeros((2,192,192),np.float32),
                'latent':np.zeros((8,24,24),np.float32),'rng':{'restored':True},'precision':{'restored':True},'native_trace':{}}
    def close(self):
        self.closed=True
        return {'allocated_bytes':0,'reserved_bytes':0}


class Contracts(unittest.TestCase):
    def data(self,**kwargs):
        values={'valid_mask':np.ones((192,192),bool),'bands':BANDS,'units':'surface_reflectance_x10000',
                'grid':request().conditioning[0].grid,'source_id':'real-crop'}
        values.update(kwargs)
        return CopgenInput(np.ones((4,192,192),np.float32),**values)
    def test_shared_request_route_is_explicit_and_rejects_s1grd(self):
        req=request();plan=plan_copgen(req)
        self.assertEqual(plan.seed,17);self.assertEqual(plan.sample_steps,50)
        for bad in (replace(req,model_id='terramind-v1-base'),replace(req,execution_profile='recommended'),
                    replace(req,targets=(replace(req.targets[0],modality='S1GRD'),)),
                    replace(req,targets=(replace(req.targets[0],units='dB'),)),
                    replace(req,sampling=replace(req.sampling,parameters=(('sample_steps',10),))),
                    replace(req,conditioning=(replace(req.conditioning[0],bands=tuple(reversed(BANDS))),))):
            with self.assertRaises(UnsupportedRoute):plan_copgen(bad)
    def test_input_owns_bytes_and_mask_is_not_inferred(self):
        data=self.data();first=data.identity
        image,mask=data.snapshot();image[:]=100;mask[:]=False
        self.assertEqual(first,data.identity);self.assertEqual(data.snapshot()[0].mean(),1)
        with self.assertRaises(ValueError):self.data(valid_mask=np.zeros((192,192),bool))
        with self.assertRaises(ValueError):self.data(bands=tuple(reversed(BANDS)))
        with self.assertRaises(ValueError):self.data().validate_plan(plan_copgen(replace(request(),conditioning=(replace(request().conditioning[0],source_id='other'),))))
    def test_no_implicit_cast_or_grid_conversion(self):
        for image in (np.ones((4,192,192),np.float64),np.ones((4,224,224),np.float32),np.full((4,192,192),np.nan,np.float32)):
            with self.assertRaises(ValueError):CopgenInput(image,valid_mask=np.ones((192,192),bool),bands=BANDS,
                units='surface_reflectance_x10000',grid=request().conditioning[0].grid,source_id='x')
    def test_execution_optin_and_rejections_precede_runtime(self):
        with self.assertRaises(ValueError):CopgenEngine(source='missing',assets='missing')
        engine=CopgenEngine(source='missing',assets='missing',execute_model=True)
        try:
            with patch.object(implementation,'NativeRuntime',side_effect=AssertionError('must not load')):
                with self.assertRaises(UnsupportedRoute):engine.generate(replace(request(),model_id='unknown'),self.data())
        finally:engine.close()
    def test_cpu_runtime_forwarding_repeat_and_close(self):
        data=self.data()
        with patch.object(implementation,'NativeRuntime',FakeRuntime):
            engine=CopgenEngine(source='missing',assets='missing',execute_model=True)
            try:
                first=engine.generate(request(),data);second=engine.generate(request(),data)
                self.assertEqual(len(engine._runtime.calls),2)
                self.assertEqual(engine._runtime.calls[-1][0].mean(),1)
                self.assertEqual(first['provenance']['seed'],17)
                self.assertFalse(second['provenance']['sensor_observation'])
            finally:engine.close()
            with self.assertRaises(RuntimeError):engine.generate(request(),data)
    def test_failure_is_terminal_and_releases_process_owner(self):
        with patch.object(implementation,'NativeRuntime',FakeRuntime):
            engine=CopgenEngine(source='missing',assets='missing',execute_model=True);engine.load()
            runtime=engine._runtime;runtime.fail=True
            with self.assertRaisesRegex(RuntimeError,'native failure'):engine.generate(request(),self.data())
            self.assertTrue(runtime.closed)
            with self.assertRaises(RuntimeError):engine.generate(request(),self.data())
            engine.close()
            other=CopgenEngine(source='missing',assets='missing',execute_model=True)
            try:other.load()
            finally:other.close()
    def test_exclusive_process_owner(self):
        with patch.object(implementation,'NativeRuntime',FakeRuntime):
            engine=CopgenEngine(source='missing',assets='missing',execute_model=True)
            other=CopgenEngine(source='missing',assets='missing',execute_model=True)
            try:
                engine.load()
                with self.assertRaisesRegex(RuntimeError,'owns'):other.load()
            finally:engine.close();other.close()
    def test_cog_exact_roundtrip_and_mutated_output_rejection(self):
        with patch.object(implementation,'NativeRuntime',FakeRuntime):
            engine=CopgenEngine(source='missing',assets='missing',execute_model=True)
            try:result=engine.generate(request(),self.data())
            finally:engine.close()
        with tempfile.TemporaryDirectory() as tmp:
            receipt=export_copgen(result,Path(tmp)/'output')
            self.assertTrue(receipt['decoded_values_exact']);self.assertEqual(receipt['units'],'linear_backscatter')
            with self.assertRaises(FileExistsError):export_copgen(result,Path(tmp)/'output')
            result['array'][0,0,0]+=1
            with self.assertRaisesRegex(ValueError,'changed'):export_copgen(result,Path(tmp)/'changed')
            self.assertFalse((Path(tmp)/'changed').exists())
    def test_fp32_scope_disables_only_local_autocast_and_restores_on_error(self):
        import torch
        from aureon.copgen_native.runtime import fp32_scope
        native=types.ModuleType('native_fixture');native.torch=torch
        ae=types.ModuleType('libs.autoencoder');ae.torch=torch
        before=(torch.get_float32_matmul_precision(),torch.get_num_threads(),torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled())
        original=torch.autocast
        with patch.dict('sys.modules',{'libs.autoencoder':ae}):
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                with fp32_scope(native) as receipt:
                    with native.torch.autocast('cpu'):
                        self.assertFalse(torch.is_autocast_enabled('cpu'))
                    self.assertIs(torch.autocast,original)
                    raise RuntimeError('fixture')
        self.assertIs(native.torch,torch);self.assertIs(ae.torch,torch);self.assertTrue(receipt['restored'])
        self.assertEqual(before,(torch.get_float32_matmul_precision(),torch.get_num_threads(),torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled()))
    def test_invalid_crs_and_shared_source_label_fail_before_execution(self):
        bad=EOGrid('EPSG:9999999',request().conditioning[0].grid.transform,(192,192))
        with self.assertRaisesRegex(ValueError,'CRS'):self.data(grid=bad)
        for source in ('https://host/private?token=value','bad\nlabel','x'*257,''):
            with self.assertRaises(ValueError):self.data(source_id=source)
    def test_primary_generate_failure_survives_cleanup_failure_and_owner_releases(self):
        with patch.object(implementation,'NativeRuntime',FakeRuntime):
            engine=CopgenEngine(source='missing',assets='missing',execute_model=True);engine.load()
            runtime=engine._runtime;runtime.fail=True
            with patch.object(runtime,'close',side_effect=RuntimeError('cleanup failure')):
                with self.assertRaisesRegex(RuntimeError,'injected native failure') as error:engine.generate(request(),self.data())
            self.assertIn('cleanup failure',str(error.exception.__notes__))
            self.assertIsNone(engine._runtime)
            self.assertTrue(engine.close()['cleanup_failed'])
            other=CopgenEngine(source='missing',assets='missing',execute_model=True)
            try:other.load()
            finally:other.close()
    def test_explicit_close_failure_is_terminal_but_does_not_strand_owner(self):
        with patch.object(implementation,'NativeRuntime',FakeRuntime):
            engine=CopgenEngine(source='missing',assets='missing',execute_model=True);engine.load()
            with patch.object(engine._runtime,'close',side_effect=RuntimeError('close failed')):
                with self.assertRaisesRegex(RuntimeError,'close failed'):engine.close()
            self.assertTrue(engine.close()['cleanup_failed'])
            with self.assertRaisesRegex(RuntimeError,'terminal'):engine.load()
            with CopgenEngine(source='missing',assets='missing',execute_model=True) as other:other.load()
    def test_context_exit_keeps_body_failure_and_reports_cleanup_failure(self):
        with patch.object(implementation,'NativeRuntime',FakeRuntime):
            with self.assertRaisesRegex(ValueError,'body failed') as error:
                with CopgenEngine(source='missing',assets='missing',execute_model=True) as engine:
                    engine.load()
                    engine._runtime.close=lambda:(_ for _ in ()).throw(RuntimeError('cleanup failed'))
                    raise ValueError('body failed')
            self.assertIn('cleanup failed',str(error.exception.__notes__))
            self.assertTrue(engine.close()['cleanup_failed'])
    def test_partial_native_constructor_and_temporary_allocations_are_released(self):
        from aureon.copgen_native.runtime import NativeRuntime
        refs={}
        class Allocation:pass
        class PartialModel:
            def __init__(self):
                self.nnet=Allocation();refs['model']=weakref.ref(self);refs['nnet']=weakref.ref(self.nnet)
                temporary=Allocation();refs['temporary']=weakref.ref(temporary)
                raise ValueError('partial native construction')
        def initialize(runtime,*args):
            runtime._owns_cuda=True  # Inject failure after successful admission.
            runtime._construct_model(PartialModel)
        with patch.object(NativeRuntime,'_initialize',initialize):
            try:
                NativeRuntime(Path('missing'),Path('missing'))
            except ValueError as error:
                self.assertEqual(str(error),'partial native construction')
                self.assertIsNotNone(error.__traceback__)
                # Retain the real exception traceback while proving its unwound
                # constructor frames no longer retain partial/temporary owners.
                self.assertTrue(all(ref() is None for ref in refs.values()))
            else:self.fail('Partial construction should fail')
    def test_partial_cleanup_failure_preserves_primary_and_releases_engine_owner(self):
        from aureon.copgen_native.runtime import NativeRuntime
        def initialize(runtime,*args):
            runtime._owns_cuda=True
            raise ValueError('construct failed')
        with patch.object(NativeRuntime,'_initialize',initialize), \
             patch.object(NativeRuntime,'close',side_effect=RuntimeError('partial close failed')):
            engine=CopgenEngine(source='missing',assets='missing',execute_model=True)
            with self.assertRaisesRegex(ValueError,'construct failed') as error:engine.load()
            self.assertIn('partial close failed',str(error.exception.__notes__))
            self.assertTrue(engine.close()['cleanup_unknown'])
        with patch.object(implementation,'NativeRuntime',FakeRuntime):
            with CopgenEngine(source='missing',assets='missing',execute_model=True) as other:other.load()
    def foreign_torch(self,version='2.6.0+cu124',available=True,devices=1,allocated=0,reserved=0):
        torch=types.ModuleType('torch');torch.__version__=version
        torch.version=types.SimpleNamespace(cuda='12.4')
        torch.cuda=types.SimpleNamespace(is_available=Mock(return_value=available),device_count=Mock(return_value=devices),
            memory_allocated=Mock(return_value=allocated),memory_reserved=Mock(return_value=reserved),
            is_initialized=Mock(return_value=True),synchronize=Mock(),empty_cache=Mock())
        torch._C=types.SimpleNamespace(_cuda_clearCublasWorkspaces=Mock())
        return torch
    def test_rejected_profile_device_or_foreign_allocator_never_runs_global_cleanup(self):
        from aureon.copgen_native import runtime as module
        cases=({'version':'2.7.0','allocated':4096},{'available':False,'allocated':4096},
               {'devices':2,'allocated':4096},{'allocated':4096},{'reserved':4096})
        for options in cases:
            with self.subTest(options=options):
                torch=self.foreign_torch(**options)
                owner=module.NativeRuntime.__new__(module.NativeRuntime)
                with patch.dict('sys.modules',{'torch':torch}),patch.object(module,'verify_assets') as verify, \
                     patch('aureon.copgen_native.profiles.importlib.metadata.version',side_effect=profile_requirements('torch260').__getitem__), \
                     patch.object(module.gc,'collect') as collect:
                    with self.assertRaisesRegex(RuntimeError,'Pinned profile|empty allocator'):
                        owner.__init__(Path('missing'),Path('missing'))
                    counts=(torch.cuda.memory_allocated.call_count,torch.cuda.memory_reserved.call_count)
                    self.assertFalse(owner._owns_cuda)
                    self.assertEqual(owner.close(),{'cuda_owned':False,'cleanup_performed':False,'cuda_memory':'not_measured'})
                    self.assertEqual(counts,(torch.cuda.memory_allocated.call_count,torch.cuda.memory_reserved.call_count))
                    verify.assert_not_called();collect.assert_not_called()
                    torch.cuda.is_initialized.assert_not_called()
                    torch.cuda.synchronize.assert_not_called();torch.cuda.empty_cache.assert_not_called()
                    torch._C._cuda_clearCublasWorkspaces.assert_not_called()
    def test_admitted_partial_failure_cleans_once_then_relinquishes_cuda(self):
        from aureon.copgen_native import runtime as module
        torch=self.foreign_torch();owner=module.NativeRuntime.__new__(module.NativeRuntime)
        def fail_after_admission(*args):
            self.assertTrue(owner._owns_cuda)
            raise ValueError('injected asset failure after admission')
        with patch.dict('sys.modules',{'torch':torch}),patch.object(module,'verify_assets',side_effect=fail_after_admission), \
             patch('aureon.copgen_native.profiles.importlib.metadata.version',side_effect=profile_requirements('torch260').__getitem__):
            with self.assertRaisesRegex(ValueError,'after admission') as error:
                owner.__init__(Path('missing'),Path('missing'))
            self.assertIn('allocated_bytes',str(error.exception.__notes__))
            self.assertFalse(owner._owns_cuda)
            torch.cuda.synchronize.assert_called_once();torch.cuda.empty_cache.assert_called_once()
            torch._C._cuda_clearCublasWorkspaces.assert_called_once()
            self.assertEqual(owner.close()['cuda_memory'],'not_measured')
            torch.cuda.synchronize.assert_called_once();torch.cuda.empty_cache.assert_called_once()

    def test_profile_is_explicit_and_forwarded_without_auto_detection(self):
        with self.assertRaises(ValueError):CopgenEngine(source='x',assets='x',execute_model=True,dependency_profile='auto')
        with patch.object(implementation,'NativeRuntime',FakeRuntime):
            for profile in ('torch260','torch210'):
                with CopgenEngine(source='x',assets='x',execute_model=True,dependency_profile=profile) as engine:
                    engine.load();self.assertEqual(engine._runtime.profile,profile)
            with CopgenEngine(source='x',assets='x',execute_model=True) as engine:
                engine.load();self.assertEqual(engine._runtime.profile,'torch260')

    def test_exact_dependency_profiles_and_builds(self):
        for profile,version,cuda in (('torch260','2.6.0+cu124','12.4'),('torch210','2.10.0+cu128','12.8')):
            requirements=profile_requirements(profile)
            with patch('aureon.copgen_native.profiles.importlib.metadata.version',side_effect=requirements.__getitem__):
                self.assertTrue(validate_profile(profile,torch_version=version,cuda_build=cuda)['exact_direct_tuple'])
                with self.assertRaisesRegex(RuntimeError,'Torch build'):
                    validate_profile(profile,torch_version=version,cuda_build='unknown')
                for name in requirements:
                    wrong=dict(requirements);wrong[name]='unqualified'
                    with patch('aureon.copgen_native.profiles.importlib.metadata.version',side_effect=wrong.__getitem__):
                        with self.assertRaisesRegex(RuntimeError,'mismatch'):
                            validate_profile(profile,torch_version=version,cuda_build=cuda)
            requirements['numpy']='mutated'
            self.assertNotEqual(profile_requirements(profile)['numpy'],'mutated')

    def test_missing_dependency_rejected_before_cuda_or_assets(self):
        from aureon.copgen_native import runtime as module
        import importlib.metadata
        torch=self.foreign_torch()
        def version(name):
            if name=='lmdb':raise importlib.metadata.PackageNotFoundError(name)
            return profile_requirements('torch260')[name]
        with patch.dict('sys.modules',{'torch':torch}),patch.object(module,'verify_assets') as verify, \
             patch('aureon.copgen_native.profiles.importlib.metadata.version',side_effect=version):
            with self.assertRaisesRegex(RuntimeError,'missing'):
                module.NativeRuntime(Path('missing'),Path('missing'))
            torch.cuda.is_available.assert_not_called();torch.cuda.memory_allocated.assert_not_called()
            torch.cuda.synchronize.assert_not_called();torch.cuda.empty_cache.assert_not_called();verify.assert_not_called()
            torch._C._cuda_clearCublasWorkspaces.assert_not_called()


if __name__=='__main__':unittest.main()
