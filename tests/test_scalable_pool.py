"""Heterogeneous policy and actual local spawned-process dispatch with fake neural work."""
from dataclasses import replace
from pathlib import Path
import os,tempfile,time,unittest
from unittest.mock import patch
import numpy as np
from aureon.api.pool_placement import WorkerCapabilities,WorkerOffer,rank_worker_offers
from aureon.api import model_pool as module
from test_model_pool import FakeSession,fake_attestation,ORIGINAL_WORKER,MIB

class SlotFailureSession(FakeSession):
    def run(self,payload,seed):
        if self.slot.name=='fast' and seed==99:os._exit(86)
        return super().run(payload,seed)

def worker(connection,route,slot,work,generation):
    module._NativeSession=SlotFailureSession;module._device_attestation=fake_attestation
    ORIGINAL_WORKER(connection,route,slot,work,generation)

class PlacementTests(unittest.TestCase):
    def test_generic_offers_capacity_caps_affinity_and_drain(self):
        caps=WorkerCapabilities(model_ids=('COP-GEN','TerraMind'),tags=('math_sdpa',))
        offers=[WorkerOffer('small',caps,10,1),WorkerOffer('large',caps,40,4),WorkerOffer('wrong',WorkerCapabilities(model_ids=('other',)),100,10)]
        rank=lambda xs,**kw:rank_worker_offers(xs,model_id='COP-GEN',task='generation',precision='float32',required_bytes=8,required=('math_sdpa',),**kw)
        self.assertEqual([o.worker_id for o in rank(offers)],['large','small'])
        self.assertEqual(rank([replace(offers[1],draining=True),offers[0]])[0].worker_id,'small')
        self.assertEqual(rank([replace(offers[0],warm=True),offers[1]])[0].worker_id,'small')
        self.assertEqual(rank([replace(offers[1],relative_capacity=1),offers[0]],affinity='small')[0].worker_id,'small')
        self.assertEqual(rank([replace(offers[0],available_bytes=1)]),[])
        with self.assertRaises(ValueError):replace(offers[0],relative_capacity=float('nan'))

    def create(self,temp):
        slots=[module.DeviceSlot('slow',16*MIB,'0',relative_capacity=1),module.DeviceSlot('fast',32*MIB,'1',relative_capacity=4)]
        resources=module.PoolResources(MIB,MIB,MIB,MIB,3*MIB,MIB)
        route=module.EncoderRoute('slow_model','terramind-v1-tiny-backbone','missing',('S2L2A',),('slow','fast'),resources)
        pool=module.ModelPool([route],slots,work=Path(temp)/'pool',host_memory_bytes=128*MIB,execute_model=True)
        data=module.EncoderInput({'S2L2A':np.zeros((1,12,224,224),np.float32)})
        return pool,data

    def test_dispatch_prefers_capacity_then_drains_without_preempting(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(module,'_worker_main',worker):
            pool,data=self.create(temp)
            try:
                first=pool.submit('slow_model',data,seed=17)
                deadline=time.monotonic()+15
                while first.state!='running' and time.monotonic()<deadline:time.sleep(.01)
                self.assertEqual(first.state,'running')
                pool.drain_device('fast')
                second=pool.submit('slow_model',data,seed=18)
                self.assertEqual(first.result(15).device_slot,'fast')
                self.assertEqual(second.result(15).device_slot,'slow')
                pool.drain_device('slow')
                with self.assertRaises(module.PoolBusy):pool.submit('slow_model',data)
                pool.resume_device('slow')
                self.assertEqual(pool.submit('slow_model',data,seed=19).result(15).device_slot,'slow')
            finally:receipt=pool.close(cancel_pending=True,timeout=20)
            self.assertTrue(receipt['passed'],receipt)

    def test_one_failed_replica_does_not_poison_same_route_on_healthy_device(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(module,'_worker_main',worker):
            pool,data=self.create(temp)
            try:
                failed=pool.submit('slow_model',data,seed=99)
                with self.assertRaises(module.PoolExecutionError):failed.result(15)
                good=pool.submit('slow_model',data,seed=18).result(15)
                self.assertEqual(good.device_slot,'slow');self.assertTrue(np.all(good.value['features'][0]==18))
                self.assertNotIn('slow_model',pool.metadata()['failed_routes'])
                self.assertEqual(pool.metadata()['devices']['fast']['assignments'],1)
            finally:receipt=pool.close(cancel_pending=True,timeout=20)
            self.assertFalse(receipt['passed']);self.assertIn('fast',receipt['quarantined_slots'])

    def test_capability_mismatch_rejected_without_spawn(self):
        resources=module.PoolResources(MIB,MIB,MIB,MIB,3*MIB,MIB)
        route=module.EncoderRoute('route','terramind-v1-tiny-backbone','missing',('S2L2A',),('slot',),resources,required_capabilities=('required',))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'No compatible slot'):
                module.ModelPool([route],[module.DeviceSlot('slot',16*MIB,'0')],work=Path(temp)/'pool',host_memory_bytes=128*MIB,execute_model=True)

if __name__=='__main__':unittest.main()
