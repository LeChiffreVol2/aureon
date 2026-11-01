"""Real CPU arrays + deterministic serial-worker doubles. No model/GPU execution."""
from concurrent.futures import CancelledError, TimeoutError
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from aureon.api import (GenerationEngine, GenerationInput, GenerationRecipe, GenerationJob, TorchGeoAdapter,
                        generation_capabilities, generation_compatibility, GenerationBusy as SessionBusy,
                        GenerationClosed as SessionClosed, GenerationFailed as SessionFailed)
from aureon.api.generation_queue import INPUT_BYTES, OUTPUT_BYTES
from aureon.terramind_native.common import INPUT_BANDS


def data(value=1):
    return GenerationInput(np.full((12,224,224), value, np.float32), bands=INPUT_BANDS,
                           units="surface_reflectance_x10000", valid_mask=np.ones((224,224), bool))


class FakeSession:
    def __init__(self):
        self.calls=[]; self.threads=[]; self.active=0; self.peak=0; self.closes=0
        self.entered=threading.Event(); self.release=threading.Event(); self.release.set()
        self.failure=None

    def generate(self, value, **options):
        self.active+=1; self.peak=max(self.peak,self.active)
        self.calls.append(options); self.threads.append(threading.get_ident())
        self.entered.set()
        try:
            if not self.release.wait(5): raise SessionFailed("fake worker timed out")
            if self.failure: raise self.failure
            image,mask,metadata,identity=value.snapshot()
            return {"array":image[:2].copy(),"valid_mask":mask,
                    "provenance":{"seed":options['recipe']['seed'],"source_id":metadata['source_id']},
                    "execution_selection":{"requested":options['execution_profile']}}
        finally: self.active-=1

    def metadata(self): return {"calls":len(self.calls)}
    def close(self): self.closes+=1; return {"passed":True,"model_load_count":1}


class ConcurrentGeneration(unittest.TestCase):
    def setUp(self):
        self.fake=FakeSession()
        self.enterContext(patch('aureon.terramind_native.offline.create_config'))
        self.opening=self.enterContext(patch('aureon.api.generation_engine.open_model_session',return_value=self.fake))
        self.root=Path(tempfile.mkdtemp(prefix='concurrent-engine-'))
        self.count=0

    def engine(self, **kwargs):
        self.count+=1
        engine=GenerationEngine(model_file='model',tokenizer_file='tokenizer',work=self.root/str(self.count),
                                execute_model=True,**kwargs)
        def close():
            self.fake.release.set()
            engine.close(cancel_pending=True,timeout=5)
        self.addCleanup(close)
        return engine

    def test_fifo_one_residency_one_dispatcher_and_owned_results(self):
        engine=self.engine(max_pending=3,max_results=3)
        self.fake.release.clear()
        first=engine.submit(data(3),seed=19)
        self.assertTrue(self.fake.entered.wait(2))
        second=engine.submit(data(4),recipe=GenerationRecipe(seed=20),execution_profile='reference')
        third=engine.submit(data(5),seed=21)
        self.fake.release.set()
        results=[h.result(3) for h in (first,second,third)]
        self.assertEqual([x['provenance']['seed'] for x in results],[19,20,21])
        self.assertEqual([x['recipe']['seed'] for x in self.fake.calls],[19,20,21])
        self.assertEqual(self.fake.peak,1)
        self.assertEqual(len(set(self.fake.threads)),1)
        self.assertEqual(self.opening.call_count,1)
        results[0]['array'].fill(99)
        self.assertTrue((results[1]['array']==4).all())
        with self.assertRaises(RuntimeError): first.result()
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)
        engine.close(); engine.close()
        self.assertEqual(self.fake.closes,1)

    def test_pending_result_and_byte_credit_boundaries(self):
        engine=self.engine(max_pending=3,max_results=3,max_input_bytes=INPUT_BYTES,max_output_bytes=OUTPUT_BYTES)
        self.fake.release.clear()
        handle=engine.submit(data())
        self.assertTrue(self.fake.entered.wait(2))
        with self.assertRaises(SessionBusy): engine.submit(data())
        with self.assertRaises(TimeoutError): handle.result(0)
        self.fake.release.set()
        engine.close()  # Completes without needing the consumer to free result credit.
        state=engine.metadata()['queue']
        self.assertEqual(state['pending'],0)
        self.assertEqual(state['reserved_results'],1)
        self.assertEqual(state['output_bytes_reserved'],OUTPUT_BYTES)
        self.assertEqual(state['high_water_pending'],1)
        handle.result(1)
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)

    def test_completed_unconsumed_results_block_new_admission(self):
        engine=self.engine(max_pending=1,max_results=1)
        handle=engine.submit(data())
        # Wait deterministically for completion without consuming its result.
        condition=handle._owner._condition
        with condition:
            self.assertTrue(condition.wait_for(lambda:handle._state=='completed',timeout=2))
        with self.assertRaises(SessionBusy): engine.submit(data())
        handle.discard()
        with self.assertRaises(RuntimeError): handle.result()
        engine.submit(data()).result(2)

    def test_cancel_only_queued_and_running_discard_returns_credit_after_finish(self):
        engine=self.engine()
        self.fake.release.clear()
        first=engine.submit(data())
        self.assertTrue(self.fake.entered.wait(2))
        second=engine.submit(data())
        self.assertFalse(first.cancel())
        self.assertTrue(second.cancel())
        with self.assertRaises(CancelledError): second.result()
        first.discard()
        self.assertEqual(engine.metadata()['queue']['reserved_results'],1)
        self.fake.release.set(); engine.close()
        self.assertEqual(len(self.fake.calls),1)
        self.assertEqual(first.state,'discarded')
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)

    def test_terminal_failure_fails_waiters_without_model_retry(self):
        engine=self.engine()
        self.fake.release.clear(); self.fake.failure=SessionFailed('injected failure')
        first=engine.submit(data())
        self.assertTrue(self.fake.entered.wait(2))
        second=engine.submit(data())
        self.fake.release.set()
        with self.assertRaisesRegex(SessionFailed,'injected'): first.result(2)
        with self.assertRaisesRegex(SessionFailed,'not executed'): second.result(2)
        with self.assertRaises(SessionClosed): engine.submit(data())
        engine.close()
        self.assertEqual(len(self.fake.calls),1)
        self.assertEqual(engine.metadata()['queue']['pending'],0)
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)
        self.assertEqual(engine.metadata()['state'],'failed')

    def test_admission_error_does_not_kill_resident_dispatcher(self):
        engine=self.engine()
        self.fake.failure=FileExistsError('duplicate attempt')
        with self.assertRaises(FileExistsError): engine.submit(data()).result(2)
        self.fake.failure=None
        engine.submit(data()).result(2)
        self.assertFalse(engine.metadata()['queue']['failed'])

    def test_close_timeout_stops_admission_and_queued_cancellation(self):
        engine=self.engine(); self.fake.release.clear()
        first=engine.submit(data())
        self.assertTrue(self.fake.entered.wait(2))
        second=engine.submit(data())
        with self.assertRaises(TimeoutError): engine.close(cancel_pending=True,timeout=0)
        with self.assertRaises(SessionClosed): engine.submit(data())
        with self.assertRaises(CancelledError): second.result()
        self.assertTrue(first.running())
        self.fake.release.set(); engine.close(timeout=3)
        first.result(1)
        self.assertFalse(engine.metadata()['queue']['dispatcher_alive'])

    def test_concurrent_producers_cannot_over_admit(self):
        engine=self.engine(max_pending=3,max_results=3)
        self.fake.release.clear(); value=data()
        barrier=threading.Barrier(9); admitted=[]; rejected=[]
        def producer():
            barrier.wait()
            try: admitted.append(engine.submit(value))
            except SessionBusy: rejected.append(True)
        threads=[threading.Thread(target=producer) for _ in range(8)]
        for thread in threads: thread.start()
        barrier.wait()
        for thread in threads: thread.join(3); self.assertFalse(thread.is_alive())
        self.assertEqual((len(admitted),len(rejected)),(3,5))
        self.fake.release.set()
        for handle in admitted: handle.result(3)
        self.assertEqual(self.fake.peak,1)
        self.assertEqual(engine.metadata()['queue']['high_water_results'],3)

    def test_stream_is_lazy_bounded_and_producer_overlaps_execution(self):
        engine=self.engine(); self.fake.release.clear(); pulled=[]
        def samples():
            for index in range(4):
                pulled.append(index)
                if index==1:
                    self.assertTrue(self.fake.entered.wait(2))
                    self.fake.release.set()
                yield data(index)
        stream=engine.generate_many(samples())
        self.assertEqual(pulled,[]); self.assertFalse(engine.work.exists())
        with stream:
            first=next(stream)
            self.assertEqual(pulled,[0,1])
            self.assertTrue((first['array']==0).all())
            results=list(stream)
        self.assertEqual([int(x['array'][0,0,0]) for x in results],[1,2,3])
        self.assertLessEqual(engine.metadata()['queue']['high_water_results'],2)
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)

    def test_busy_stream_does_not_advance_input_and_early_close_does_not_affect_other_work(self):
        engine=self.engine(); self.fake.release.clear()
        first=engine.submit(data()); self.assertTrue(self.fake.entered.wait(2))
        other=engine.submit(data()); pulled=[]
        def samples():
            pulled.append(1); yield data()
        stream=engine.generate_many(samples())
        with self.assertRaises(SessionBusy): next(stream)
        self.assertEqual(pulled,[])
        stream.close()
        self.fake.release.set()
        first.result(2); other.result(2)
        self.assertEqual(len(self.fake.calls),2)

    def test_stream_adapter_failure_releases_reserved_credit_and_owned_prefetch(self):
        engine=self.engine()
        closed=[]
        def values():
            try:
                yield data(); yield 'invalid'; yield data()
            finally: closed.append(True)
        with self.assertRaises(TypeError): next(engine.generate_many(values()))
        engine.close()
        self.assertEqual(closed,[True])
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)

    def test_stream_close_releases_its_prefetch_and_keeps_caller_results(self):
        engine=self.engine()
        stream=engine.generate_many(data(n) for n in range(20))
        result=next(stream)
        stream.close(); engine.close()
        self.assertTrue((result['array']==0).all())
        self.assertLessEqual(len(self.fake.calls),2)
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)

    def test_close_during_adapter_invalidates_reservation_without_execution(self):
        engine=self.engine()
        def adapter(item):
            engine.close()
            return data()
        with self.assertRaises(SessionClosed): next(engine.generate_many([1],adapter=adapter))
        self.assertEqual(self.opening.call_count,0)
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)

    def test_recipe_forwarding_and_explicit_torchgeo_metadata(self):
        engine=self.engine()
        settings=dict(modalities=['S2L2A'],output_modalities=['S1GRD'],standardize=True,
                      img_size=224,decoding_steps=1,temps=1.,top_p=.8,top_k=0,timesteps=12)
        recipe=GenerationRecipe.from_terratorch(settings,seed=59,model_id='terramind-v1-base',terratorch_version='1.2.13')
        adapter=TorchGeoAdapter(bands=INPUT_BANDS,units='surface_reflectance_x10000',
                               valid_mask=lambda sample:sample['valid'],source_id=lambda sample:sample['id'])
        sample={'image':np.ones((12,224,224),np.float32),'mask':np.full((224,224),9,np.uint8),
                'valid':np.ones((224,224),bool),'bounds':('never','guess'),'id':'sample-9'}
        with engine.generate_many([sample],adapter=adapter,recipe=recipe) as results:
            result=next(results)
        self.assertEqual(result['provenance'],{'seed':59,'source_id':'sample-9',
                         'parallel_selection':result['parallel_selection']})
        self.assertEqual(result['parallel_selection']['actual'],1)
        self.assertEqual(self.fake.calls[0]['recipe'],recipe.native())
        with self.assertRaises(ValueError): engine.submit(data(),recipe=recipe,seed=17)
        with self.assertRaises(ValueError): GenerationRecipe.from_terratorch({**settings,'unknown':1},seed=59,model_id='terramind-v1-base',terratorch_version='1.2.13')
        with self.assertRaises(ValueError): GenerationRecipe.from_terratorch({**settings,'standardize':False},seed=59,model_id='terramind-v1-base',terratorch_version='1.2.13')

    def test_capabilities_are_detached_and_do_not_promote_batching_or_gpu(self):
        engine=self.engine()
        capabilities=engine.capabilities()
        self.assertEqual(capabilities['batching']['max_execution_batch_size'],1)
        self.assertFalse(capabilities['batching']['fused_batching'])
        self.assertFalse(capabilities['batching']['step_interleaving'])
        capabilities['input']['bands'].clear()
        self.assertEqual(generation_capabilities()['input']['bands'],INPUT_BANDS)
        first=generation_compatibility(data(),recipe=GenerationRecipe(seed=1))
        second=generation_compatibility(data(2),recipe=GenerationRecipe(seed=1))
        self.assertEqual(first['sha256'],second['sha256'])
        self.assertNotEqual(first['input_identity'],second['input_identity'])
        self.assertFalse(engine.work.exists())
        self.opening.assert_not_called()

    def test_sync_generate_joins_enabled_queue_and_input_snapshot_is_stable(self):
        engine=self.engine(); self.fake.release.clear()
        array=np.full((12,224,224),7,np.float32)
        value=GenerationInput(array,bands=INPUT_BANDS,units='reflectance_x10000',valid_mask=np.ones((224,224),bool))
        handle=engine.submit(value)
        self.assertTrue(self.fake.entered.wait(2)); array.fill(99)
        results=[]
        thread=threading.Thread(target=lambda:results.append(engine.generate(data(8),seed=81)))
        thread.start()
        condition=handle._owner._condition
        # The commit notification makes the synchronized queue observation deterministic.
        with condition:
            self.assertTrue(condition.wait_for(lambda:len(handle._owner._queue)==1,timeout=2))
        self.fake.release.set(); first=handle.result(2); thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue((first['array']==7).all())
        self.assertEqual(results[0]['provenance']['seed'],81)
        self.assertEqual(self.fake.peak,1)

    def test_close_drains_queued_work_and_stream_can_consume_accepted_results(self):
        engine=self.engine()
        stream=engine.generate_many(data(n) for n in range(5))
        first=next(stream)
        engine.close()
        tail=list(stream)
        self.assertEqual([int(x['array'][0,0,0]) for x in [first]+tail],[0,1])
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)

    def test_unexpected_output_failure_terminates_dispatcher_and_clears_input_credit(self):
        engine=self.engine()
        self.fake.generate=lambda *a,**kw:{'array':np.zeros((20,224,224),np.float32),'valid_mask':np.ones((224,224),bool)}
        with self.assertRaisesRegex(SessionFailed,'credit'): engine.submit(data()).result(2)
        self.assertEqual(engine.metadata()['state'],'failed')
        with self.assertRaises(SessionClosed): engine.submit(data())
        self.assertEqual(engine.metadata()['queue']['pending'],0)

    def test_invalid_limits_recipe_and_timeout_do_not_consume_credits(self):
        for kwargs in ({'max_pending':0},{'max_results':True},{'max_input_bytes':INPUT_BYTES-1},
                       {'max_output_bytes':OUTPUT_BYTES-1}):
            with self.assertRaises(ValueError): self.engine(**kwargs)
        engine=self.engine()
        with self.assertRaises(ValueError): engine.submit(data(),seed=True)
        with self.assertRaises(TypeError): engine.submit(data(),recipe={'seed':17})
        self.assertFalse(engine.work.exists())
        handle=engine.submit(data())
        with self.assertRaises(ValueError): handle.result(float('nan'))
        handle.result(2)
        with self.assertRaises(ValueError): engine.close(timeout=-1)
        self.assertFalse(engine.metadata()['closing'])

    def test_iterator_cleanup_preserves_producer_error_and_releases_credits(self):
        class Broken:
            closed = 0
            def __iter__(self): return self
            def __next__(self): raise ValueError('invalid input item')
            def close(self):
                self.closed += 1
                raise RuntimeError('iterator cleanup failed')
        source = Broken()
        engine = self.engine()
        stream = engine.generate_many(source)
        with self.assertRaisesRegex(ValueError, 'invalid input item') as failure:
            next(stream)
        self.assertIn('iterator cleanup failed', failure.exception.__notes__[0])
        self.assertEqual(source.closed, 1)
        self.assertEqual(engine.metadata()['queue']['reserved_results'], 0)
        self.assertEqual(engine.metadata()['queue']['pending'], 0)
        self.assertEqual(self.fake.calls, [])
        stream.close()
        self.assertEqual(source.closed, 1)

    def test_iterator_cleanup_preserves_context_body_error_but_explicit_close_raises(self):
        class BrokenClose:
            def __iter__(self): return self
            def __next__(self): return data()
            def close(self): raise RuntimeError('iterator cleanup failed')
        engine = self.engine()
        with self.assertRaisesRegex(ValueError, 'consumer failed') as failure:
            with engine.generate_many(BrokenClose(), max_in_flight=1) as stream:
                next(stream)
                raise ValueError('consumer failed')
        self.assertIn('iterator cleanup failed', failure.exception.__notes__[0])
        self.assertEqual(engine.metadata()['queue']['reserved_results'], 0)
        stream = engine.generate_many(BrokenClose(), max_in_flight=1)
        next(stream)
        with self.assertRaisesRegex(RuntimeError, 'iterator cleanup failed'):
            stream.close()
        self.assertEqual(engine.metadata()['queue']['reserved_results'], 0)

    def test_lazy_jobs_keep_per_item_seeds_profiles_and_ids_without_shared_overrides(self):
        engine=self.engine(); seen=[]
        def jobs():
            for index,seed in enumerate((71,13,92)):
                seen.append(seed)
                yield GenerationJob(data(index),recipe=GenerationRecipe(seed=seed),
                    execution_profile='reference',job_id=f'tile-{index}',attempt_id='a')
        stream=engine.generate_many(jobs())
        self.assertEqual(seen,[])
        with stream: results=list(stream)
        self.assertEqual([r['provenance']['seed'] for r in results],[71,13,92])
        self.assertEqual([c['job_id'] for c in self.fake.calls],['tile-0','tile-1','tile-2'])
        self.assertTrue(all(c['attempt_id']=='a' and c['execution_profile']=='reference' for c in self.fake.calls))
        job=GenerationJob(data(),recipe=GenerationRecipe(seed=23),job_id='direct-job')
        self.assertEqual(engine.submit(job).result(2)['provenance']['seed'],23)
        with self.assertRaises(ValueError): engine.submit(job,seed=17)
        with self.assertRaises(ValueError): next(engine.generate_many([job],execution_profile='reference'))
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)


if __name__=='__main__': unittest.main()
