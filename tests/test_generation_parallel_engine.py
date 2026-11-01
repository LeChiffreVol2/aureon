"""Public facade/collector contracts with CPU-owned results and no model calls."""
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from aureon.api import GenerationEngine, GenerationInput, GenerationRecipe
from aureon.api.generation_parallel import PairAdmissionRejected, open_lane_session, LaneSession, QUALIFICATION
from aureon.terramind_native.denoiser_batch import BatchIneligible
from aureon.terramind_native.session import SessionFailed
from aureon.terramind_native.common import INPUT_BANDS


def data(value=1):
    return GenerationInput(np.full((12,224,224),value,np.float32),bands=INPUT_BANDS,
                           units='surface_reflectance_x10000',valid_mask=np.ones((224,224),bool))


class FakeSession:
    def __init__(self):
        self.singles=[];self.pairs=[];self.error=None;self.closed=0
        self.entered=threading.Event();self.release=threading.Event();self.release.set()
    def result(self,value,seed):
        image,mask,_,_=value.snapshot()
        return {'array':image[:2].copy(),'valid_mask':mask.copy(),'seed':seed}
    def generate(self,value,**options):
        self.singles.append(options)
        return self.result(value,options['recipe']['seed'])
    def generate_pair(self,jobs):
        self.pairs.append(jobs);self.entered.set()
        if not self.release.wait(5):raise SessionFailed('test deadline')
        if self.error:raise self.error
        return [self.result(j.input_data,j.recipe.seed) for j in jobs]
    generate_group=generate_pair
    def metadata(self):return {'test_double':True}
    def close(self):self.closed+=1;return {'passed':True}


class ParallelEngineContracts(unittest.TestCase):
    def setUp(self):
        self.temp=self.enterContext(tempfile.TemporaryDirectory())
        self.root=Path(self.temp);self.count=0;self.fake=FakeSession()
        self.enterContext(patch('aureon.terramind_native.offline.create_config'))
        self.opener=self.enterContext(patch('aureon.api.generation_parallel.open_lane_session',return_value=self.fake))
        self.enterContext(patch('aureon.api.generation_parallel.QUALIFICATION',{'scope':'CPU test double only'}))
    def engine(self,**kwargs):
        self.count+=1
        value=GenerationEngine(model_file='model',tokenizer_file='tokenizer',work=self.root/str(self.count),
                               execute_model=True,parallel='two-lane',pair_wait_ms=kwargs.pop('pair_wait_ms',1000),**kwargs)
        def close():
            self.fake.release.set();value.close(cancel_pending=True,timeout=5)
        self.addCleanup(close)
        return value
    def pair(self,engine):
        return [engine.submit(data(i),seed=seed,job_id=str(seed)) for i,seed in ((1,17),(2,101))]
    def test_pair_dispatch_preserves_order_and_independent_result_ownership(self):
        engine=self.engine();handles=self.pair(engine)
        results=[h.result(5) for h in handles]
        self.assertEqual([r['seed'] for r in results],[17,101])
        self.assertEqual(len(self.fake.pairs),1);self.assertEqual(self.fake.singles,[])
        self.assertEqual(self.opener.call_count,1)
        self.assertTrue(all(j.execution_profile is None for j in self.fake.pairs[0]))
        results[0]['array'].fill(99);self.assertTrue((results[1]['array']==2).all())
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)
        self.assertEqual(results[1]['parallel_selection']['neural_batch_size'],1)
    def test_singleton_timeout_uses_original_single_entry(self):
        engine=self.engine(pair_wait_ms=0)
        self.assertEqual(engine.submit(data(),seed=17).result(5)['seed'],17)
        self.assertEqual(len(self.fake.singles),1);self.assertEqual(self.fake.pairs,[])
        self.assertEqual(self.fake.singles[0]['execution_profile'],'recommended')
    def test_reference_and_unsupported_recipe_use_B1_without_pair(self):
        engine=self.engine()
        a=engine.submit(data(),execution_profile='reference')
        b=engine.submit(data(),recipe=GenerationRecipe(token_steps=1))
        a.result(5);b.result(5)
        self.assertEqual(self.fake.pairs,[])
        self.assertEqual([x['execution_profile'] for x in self.fake.singles],['reference','recommended'])
    def test_pre_execution_rejection_falls_back_once_per_job(self):
        engine=self.engine();self.fake.error=PairAdmissionRejected('unsupported device')
        results=[h.result(5) for h in self.pair(engine)]
        self.assertEqual(len(self.fake.pairs),1);self.assertEqual(len(self.fake.singles),2)
        self.assertTrue(all(r['parallel_selection']['before_pair_execution'] for r in results))
        self.assertEqual([r['seed'] for r in results],[17,101])
    def test_failure_after_pair_entry_never_retries_serially(self):
        engine=self.engine();self.fake.error=SessionFailed('decoder failed')
        for handle in self.pair(engine):
            with self.assertRaises(SessionFailed):handle.result(5)
        self.assertEqual(self.fake.singles,[])
        self.assertEqual(engine.metadata()['state'],'failed')
    def test_generic_ineligible_error_is_not_a_fallback_permission(self):
        engine=self.engine();self.fake.error=BatchIneligible('not an admission receipt')
        for handle in self.pair(engine):
            with self.assertRaises(SessionFailed):handle.result(5)
        self.assertEqual(self.fake.singles,[])
    def test_running_discard_does_not_cancel_neighbor_or_reexecute(self):
        engine=self.engine();self.fake.release.clear()
        a,b=self.pair(engine);self.assertTrue(self.fake.entered.wait(5))
        a.discard();self.fake.release.set()
        self.assertEqual(b.result(5)['seed'],101)
        self.assertEqual(len(self.fake.pairs),1);self.assertEqual(self.fake.singles,[])
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)
    def test_direct_single_before_queue_reuses_one_session(self):
        engine=self.engine();engine.generate(data(),seed=9)
        for handle in self.pair(engine):handle.result(5)
        self.assertEqual(self.opener.call_count,1);self.assertEqual(len(self.fake.singles),1)
        self.assertEqual(len(self.fake.pairs),1)
    def test_lazy_stream_routes_pair_and_final_singleton(self):
        engine=self.engine(pair_wait_ms=50)
        with engine.generate_many([data(1),data(2),data(3)]) as stream:
            results=list(stream)
        self.assertEqual([float(r['array'][0,0,0]) for r in results],[1,2,3])
        self.assertEqual(len(self.fake.pairs),1);self.assertEqual(len(self.fake.singles),1)
    def test_capabilities_expose_logical_pair_not_fused_batch_or_speed_claim(self):
        value=self.engine().capabilities()
        self.assertEqual(value['batching']['max_execution_batch_size'],1)
        self.assertFalse(value['batching']['fused_batching'])
        self.assertFalse(value['parallel']['performance_claim'])
    def test_auto_without_evidence_uses_resource_credits_without_loading(self):
        from aureon.api.parallel_policy import ParallelPolicy
        with patch('aureon.api.generation_parallel.QUALIFICATION',None):
            engine=self.engine()
            plan=ParallelPolicy('auto').plan(engine._limits,None)
            self.assertEqual(plan['planned'],2)
            self.assertIsNone(plan['fallback_reason'])
    def test_unsupported_capability_is_pre_execution_fallback(self):
        session=object.__new__(LaneSession)
        from aureon.api.parallel_policy import ParallelPolicy, GIB
        session.policy=ParallelPolicy(2)
        cuda=SimpleNamespace(mem_get_info=lambda:(4*GIB,8*GIB),
            get_device_capability=lambda i:(8,6),device_count=lambda:1)
        session._runtime=SimpleNamespace(torch=SimpleNamespace(cuda=cuda))
        recipes=[GenerationRecipe(seed=seed).native() for seed in (17,101)]
        with self.assertRaisesRegex(PairAdmissionRejected,'serial_reference'):
            session._group_admission(recipes)


if __name__=='__main__':unittest.main()
