"""Finite generic facade/collector planning and execution with CPU doubles."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from aureon.api import GenerationEngine, GenerationInput, GenerationRecipe
from aureon.api.generation_parallel import LaneSession, PairAdmissionRejected, QUALIFICATION
from aureon.api.parallel_policy import ParallelPolicy, GIB, LANE_RESERVATION
from aureon.api.generation_queue import INPUT_BYTES, OUTPUT_BYTES
from aureon.terramind_native.common import INPUT_BANDS
from aureon.terramind_native.session import SessionFailed


def data(value):
    return GenerationInput(np.full((12,224,224),value,np.float32),bands=INPUT_BANDS,
        units='surface_reflectance_x10000',valid_mask=np.ones((224,224),bool))


class FakeSession:
    def __init__(self):self.groups=[];self.singles=[];self.failure=None
    def generate(self,value,**options):
        self.singles.append(options['recipe']['seed']);return self.result(value,options['recipe']['seed'])
    def result(self,value,seed):
        image,mask,_,_=value.snapshot()
        return dict(array=image[:2].copy(),valid_mask=mask.copy(),seed=seed)
    def generate_group(self,jobs):
        self.groups.append([j.recipe.seed for j in jobs])
        if self.failure:raise self.failure
        return [self.result(j.input_data,j.recipe.seed) for j in jobs]
    def metadata(self):return {'test_double':True}
    def close(self):return {'passed':True}


class GenericFacade(unittest.TestCase):
    def setUp(self):
        self.temp=self.enterContext(tempfile.TemporaryDirectory());self.count=0;self.fake=FakeSession()
        self.enterContext(patch('aureon.terramind_native.offline.create_config'))
        self.opening=self.enterContext(patch('aureon.api.generation_parallel.open_lane_session',return_value=self.fake))
        self.enterContext(patch('aureon.api.generation_engine.open_model_session',return_value=self.fake))
    def engine(self,n,credits=None,**kwargs):
        self.count+=1;credits=credits if credits is not None else n
        value=GenerationEngine(model_file='unused',tokenizer_file='unused',work=Path(self.temp)/str(self.count),
            execute_model=True,parallelism=n,max_pending=credits,max_results=credits,
            parallel_workspace_bytes=8*GIB,parallel_wait_ms=250,**kwargs)
        self.addCleanup(value.close,cancel_pending=True,timeout=5)
        return value
    def test_public_execution_N1_N2_N3_N5_and_result_order(self):
        for n in (1,2,3,5):
            with self.subTest(n=n):
                engine=self.engine(n,allow_unqualified_parallelism=True)
                handles=[engine.submit(data(i),seed=17+i) for i in range(n)]
                results=[h.result(5) for h in handles]
                self.assertEqual([r['seed'] for r in results],list(range(17,17+n)))
                self.assertTrue(all(r['parallel_selection']['actual']==n for r in results))
                self.assertTrue(all(r['parallel_selection']['neural_batch_size']==1 for r in results))
                if n>1:
                    self.assertEqual(self.fake.groups[-1],list(range(17,17+n)))
                    results[0]['array'].fill(99);self.assertTrue((results[-1]['array']==n-1).all())
    def test_large_requested_count_is_bounded_before_any_group_allocation(self):
        engine=self.engine(10**12,credits=3,allow_unqualified_parallelism=True)
        plan=engine.capabilities()['parallel']
        self.assertEqual(plan['requested'],10**12);self.assertEqual(plan['planned'],3)
        self.assertIn('input_result_credits',plan['fallback_reason'])
        self.assertFalse(Path(engine.work).exists())
        handles=[engine.submit(data(i),seed=i) for i in range(3)]
        self.assertEqual([h.result(5)['parallel_selection']['actual'] for h in handles],[3]*3)
        self.assertEqual([len(g) for g in self.fake.groups],[3])
    def test_omitted_credits_follow_N_and_workspace_without_allocation(self):
        for n,workspace,expected in ((1,2*GIB,2),(3,2*GIB,3),(8,4*GIB,8),(10**12,2*GIB,4)):
            self.count+=1
            engine=GenerationEngine(model_file='unused',tokenizer_file='unused',work=Path(self.temp)/str(self.count),
                execute_model=True,parallelism=n,parallel_workspace_bytes=workspace)
            self.addCleanup(engine.close)
            self.assertEqual(engine.capabilities()['credits']['max_pending'],expected)
            self.assertEqual(engine.capabilities()['credits']['max_results'],expected)
            self.assertFalse(engine.work.exists())
    def test_singleton_and_fallback_provenance_record_identical_selection(self):
        engine=self.engine(3)
        direct=engine.generate(data(1))
        self.assertEqual(direct['parallel_selection'],direct['provenance']['parallel_selection'])
        self.fake.failure=PairAdmissionRejected('fixture fallback')
        results=[h.result(5) for h in [engine.submit(data(i),seed=i) for i in range(3)]]
        for r in results:self.assertEqual(r['parallel_selection'],r['provenance']['parallel_selection'])
    def test_partial_group_timeout_and_mixed_sizes_preserve_FIFO(self):
        engine=self.engine(3,allow_unqualified_parallelism=True)
        for count in (3,2,3):
            handles=[engine.submit(data(i),seed=i) for i in range(count)]
            self.assertEqual([h.result(5)['seed'] for h in handles],list(range(count)))
        self.assertEqual([len(g) for g in self.fake.groups],[3,2,3])
    def test_no_retry_after_partial_group_failure(self):
        engine=self.engine(5,allow_unqualified_parallelism=True);self.fake.failure=SessionFailed('after lane 3')
        handles=[engine.submit(data(i),seed=i) for i in range(5)]
        for h in handles:
            with self.assertRaises(SessionFailed):h.result(5)
        self.assertEqual(len(self.fake.groups),1);self.assertEqual(self.fake.singles,[])
    def test_rejection_policy_error_never_executes_fallback(self):
        engine=self.engine(3,parallel_fallback='error');self.fake.failure=PairAdmissionRejected('unqualified N3')
        handles=[engine.submit(data(i),seed=i) for i in range(3)]
        for h in handles:
            with self.assertRaisesRegex(ValueError,'before execution'):h.result(5)
        self.assertEqual(self.fake.singles,[])
    def test_serial_fallback_has_explicit_requested_admitted_actual_reason(self):
        engine=self.engine(3);self.fake.failure=PairAdmissionRejected('unqualified N3')
        results=[h.result(5) for h in [engine.submit(data(i),seed=i) for i in range(3)]]
        for r in results:
            s=r['parallel_selection'];self.assertEqual((s['requested'],s['planned'],s['admitted'],s['actual']),(3,3,1,1))
            self.assertEqual(s['fallback_reason'],'unqualified N3')
    def test_invalid_parallelism_and_lazy_unused_construction(self):
        for n in (0,-1,True,1.5,'largest'):
            with self.assertRaises((TypeError,ValueError)):self.engine(n,credits=2)
        value=self.engine(5,credits=2)
        self.assertFalse(value.work.exists());self.assertEqual(self.opening.call_count,0)


class GenericAdmission(unittest.TestCase):
    def session(self,requested=3,allow=False,hardware='other GPU',workspace=4*GIB,free=8*GIB,cached=0):
        s=object.__new__(LaneSession);s.policy=ParallelPolicy(requested,allow,'serial',workspace,3*GIB)
        s.plan=s.policy.plan(dict(max_pending=8,max_results=8,max_input_bytes=8*INPUT_BYTES,max_output_bytes=8*OUTPUT_BYTES),QUALIFICATION)
        cuda=SimpleNamespace(get_device_name=lambda i:hardware,memory_allocated=lambda:2*GIB,
                             memory_reserved=lambda:2*GIB,mem_get_info=lambda:(free,8*GIB),
                             get_device_capability=lambda i:(8,6),CUDAGraph=object,device_count=lambda:1)
        s._runtime=SimpleNamespace(torch=SimpleNamespace(cuda=cuda),parallel_lanes=SimpleNamespace(graphs=[None]*cached) if cached else None)
        return s
    def recipes(self,n):return [GenerationRecipe(seed=17+i).native() for i in range(n)]
    def test_auto_capability_admits_without_gpu_name_or_historical_count(self):
        for hardware,count in (('other GPU',2),(QUALIFICATION['hardware'],3)):
            s=self.session('auto',True,hardware)
            with patch('aureon.api.group_generation.NativeGroupSession._group_admission',return_value={}):
                s._group_admission(self.recipes(count))
    def test_N3_N5_capability_admission_still_reports_unqualified(self):
        for count in (3,5):
            s=self.session(count,False)
            with patch('aureon.api.group_generation.NativeGroupSession._group_admission',return_value={}):
                s._group_admission(self.recipes(count))
            s=self.session(count,True)
            with patch('aureon.api.group_generation.NativeGroupSession._group_admission',return_value={}):
                s._group_admission(self.recipes(count))
            self.assertEqual(s.last_selection['admitted'],count)
            self.assertEqual(s.last_selection['qualification']['status'],'executable_not_qualified')
    def test_N2_uses_prior_scope_evidence_not_generic_GPU_claim(self):
        s=self.session('auto',False,QUALIFICATION['hardware'])
        with patch('aureon.api.group_generation.NativeGroupSession._group_admission',return_value={}):s._group_admission(self.recipes(2))
        self.assertEqual(s.last_selection['qualification']['status'],'executable_not_qualified')
    def test_workspace_residency_free_memory_fail_before_preparation(self):
        cases=[self.session(3,True,workspace=LANE_RESERVATION),self.session(3,True,free=1)]
        extra=self.session(3,True);extra._runtime.parallel_residency_bytes=4*GIB;cases.append(extra)
        for s in cases:
            with self.assertRaises(PairAdmissionRejected):s._group_admission(self.recipes(3))
    def test_cached_capacity_not_reallocated_for_3_2_3(self):
        s=self.session(3,True,cached=3)
        with patch('aureon.api.group_generation.NativeGroupSession._group_admission',return_value={}):
            for n in (3,2,3):
                s._group_admission(self.recipes(n))
                self.assertEqual(s.last_selection['retained_lane_capacity'],3)
                self.assertEqual(s.last_selection['new_lane_reservation_bytes'],0)


if __name__=='__main__':unittest.main()
