"""CPU collection/cancellation/result-credit contracts; no model execution."""
from concurrent.futures import CancelledError, TimeoutError
import threading
from time import monotonic
import unittest
import numpy as np

from aureon.api.generation_pair_queue import GenerationPairQueue, pair_compatible
from aureon.api.generation_queue import INPUT_BYTES, OUTPUT_BYTES
from aureon.terramind_native.session import SessionBusy, SessionClosed, SessionFailed


def options(seed=17,job_id=None,profile='recommended',steps=10):
    return dict(recipe=dict(token_steps=8,diffusion_steps=steps,temperature=1.,top_p=.8,top_k=0,seed=seed),
                execution_profile=profile,job_id=job_id,attempt_id='1')
def output(value): return {'array':np.full((2,2),value,np.float32),'valid_mask':np.ones((2,2),bool)}


class CollectionTests(unittest.TestCase):
    def setUp(self): self.queues=[];self.calls=[];self.entered=threading.Event();self.release=threading.Event();self.release.set()
    def tearDown(self):
        self.release.set()
        for q in self.queues:q.shutdown(cancel_pending=True,timeout=3)
    def queue(self,**kw):
        def single(data,**opts):self.calls.append(('single',data,opts));return output(data)
        def pair(requests):
            self.calls.append(('pair',requests));self.entered.set()
            if not self.release.wait(3):raise RuntimeError('fixture timeout')
            return [output(data) for data,_ in requests]
        limits=dict(max_pending=4,max_results=4,max_input_bytes=4*INPUT_BYTES,max_output_bytes=4*OUTPUT_BYTES,max_wait_ms=100)
        limits.update(kw)
        q=GenerationPairQueue(single,lambda:False,execute_pair=pair,**limits);self.queues.append(q);return q
    def submit(self,q,data,**kw):
        handle=q.reserve();return q.commit(handle,data,options(**kw))
    def test_pair_order_different_seeds_and_independent_ownership(self):
        q=self.queue();one=self.submit(q,1,seed=17);two=self.submit(q,2,seed=101)
        a,b=one.result(2),two.result(2)
        self.assertEqual([call[0] for call in self.calls],['pair'])
        self.assertEqual([r[1]['recipe']['seed'] for r in self.calls[0][1]],[17,101])
        self.assertEqual(a['array'][0,0],1);self.assertEqual(b['array'][0,0],2)
        self.assertFalse(np.shares_memory(a['array'],b['array']))
        with self.assertRaises(RuntimeError):one.result()
        self.assertEqual(q.metadata()['reserved_results'],0)
    def test_singleton_timeout_uses_single_executor(self):
        q=self.queue(max_wait_ms=20);start=monotonic();handle=self.submit(q,1)
        self.assertEqual(handle.result(2)['array'][0,0],1)
        self.assertEqual([c[0] for c in self.calls],['single'])
        self.assertLess(monotonic()-start,2)
        self.assertEqual(q.metadata()['collection']['timeout_singletons'],1)
    def test_incompatible_neighbors_never_overtake(self):
        q=self.queue();one=self.submit(q,1);two=self.submit(q,2,profile='reference');three=self.submit(q,3)
        [h.result(2) for h in (one,two,three)]
        self.assertEqual([(c[0],c[1]) for c in self.calls],[('single',1),('single',2),('single',3)])
    def test_cancel_during_collection_removes_only_that_request(self):
        q=self.queue(max_wait_ms=1000);one=self.submit(q,1)
        self.assertTrue(one.cancel());self.assertEqual(q.metadata()['pending'],0)
        with self.assertRaises(CancelledError):one.result()
        two=self.submit(q,2);three=self.submit(q,3)
        two.result(2);three.result(2)
        self.assertEqual([r[0] for r in self.calls[0][1]],[2,3])
    def test_running_discard_preserves_other_result_and_releases_each_credit(self):
        self.release.clear();q=self.queue(max_results=2,max_pending=2)
        one=self.submit(q,1);two=self.submit(q,2)
        self.assertTrue(self.entered.wait(2));self.assertFalse(one.cancel());one.discard()
        with self.assertRaises(SessionBusy):q.reserve()
        self.release.set();result=two.result(2)
        self.assertEqual(result['array'][0,0],2)
        self.assertEqual(q.metadata()['reserved_results'],0)
        with self.assertRaises(RuntimeError):one.result()
    def test_result_timeout_does_not_cancel_or_release_credit(self):
        self.release.clear();q=self.queue();one=self.submit(q,1);two=self.submit(q,2)
        self.assertTrue(self.entered.wait(2))
        with self.assertRaises(TimeoutError):one.result(.001)
        self.assertEqual(q.metadata()['reserved_results'],2)
        self.release.set();one.result(2);two.result(2)
    def test_count_or_byte_limits_prevent_useless_wait_for_pair(self):
        q=self.queue(max_input_bytes=INPUT_BYTES,max_wait_ms=1000)
        one=self.submit(q,1);one.result(2)
        self.assertFalse(q.metadata()['collection']['enabled_by_credits'])
        self.assertEqual(self.calls[0][0],'single')
    def test_drain_flushes_singleton_without_waiting_for_pair_timeout(self):
        q=self.queue(max_wait_ms=1000);one=self.submit(q,1)
        start=monotonic();q.shutdown(timeout=2)
        self.assertLess(monotonic()-start,.8);self.assertEqual(one.result()['array'][0,0],1)
        with self.assertRaises(SessionClosed):q.reserve()
    def test_cancel_pending_retires_collector_and_reserved_preparation(self):
        q=self.queue(max_wait_ms=1000);reserved=q.reserve();queued=self.submit(q,1)
        q.shutdown(cancel_pending=True,timeout=2)
        self.assertEqual(self.calls,[])
        self.assertEqual(q.metadata()['reserved_results'],0)
        for h in (reserved,queued):
            with self.assertRaises(CancelledError):h.result()
    def test_pair_failure_fails_waiters_without_serial_retry(self):
        self.release.clear();q=self.queue()
        def fail(requests):
            self.entered.set();self.release.wait(3);raise RuntimeError('fixture model failure')
        q._execute_pair=fail
        one=self.submit(q,1);two=self.submit(q,2);self.assertTrue(self.entered.wait(2));waiting=self.submit(q,3)
        self.release.set()
        for h in (one,two,waiting):
            with self.assertRaises(SessionFailed):h.result(2)
        self.assertEqual(self.calls,[]);self.assertTrue(q.metadata()['failed'])
        self.assertEqual(q.metadata()['pending'],0);self.assertEqual(q.metadata()['reserved_results'],0)
    def test_matching_ids_and_different_recipe_or_reference_never_pair(self):
        self.assertFalse(pair_compatible(options(job_id='same'),options(job_id='same')))
        self.assertFalse(pair_compatible(options(),options(steps=9)))
        self.assertFalse(pair_compatible(options(profile='reference'),options(profile='reference')))
        self.assertTrue(pair_compatible(options(seed=17),options(seed=101)))
    def test_invalid_collection_timeout_rejected(self):
        for value in (-1,1001,float('nan'),True):
            with self.assertRaises(ValueError):self.queue(max_wait_ms=value)


if __name__=='__main__':unittest.main()
