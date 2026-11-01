"""Real scheduler/process tests with simulated GPUs; no GPU speedup claims.

The node names are labels, not remote hosts. Real process overlap and the actual
ModelPool dispatcher are exercised. Device attestation and model calls are the
only substitutes. This cannot qualify networks, P2P/NCCL or physical GPUs.
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from aureon.api import model_pool as poolmod
from aureon.api.routing import ExecutionKey, PlacementRequest, WorkerSnapshot, plan_placement
from test_model_pool import FakeSession, fake_attestation, ORIGINAL_WORKER, MIB


class BarrierSession(FakeSession):
    def run(self, payload, seed):
        if self.route.name.startswith('barrier'):
            (self.work / ('waiting-'+str(seed))).write_text('ready')
            deadline = time.monotonic()+30
            while not (self.work.parent/'release').exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Parent did not release simulation barrier')
                time.sleep(.01)
        return super().run(payload, seed)


def scaling_worker(connection, route, slot, work, generation):
    poolmod._NativeSession = BarrierSession
    poolmod._device_attestation = fake_attestation
    ORIGINAL_WORKER(connection, route, slot, work, generation)


class ScalingTests(unittest.TestCase):
    def test_four_devices_two_residents_each_overlap_and_drain_96_jobs(self):
        # 2 labelled nodes x 2 devices x 2 distinct resident models; arbitrary N.
        names = tuple(f'node{n}-gpu{g}' for n in range(2) for g in range(2))
        slots = [poolmod.DeviceSlot(name, 12*MIB, str(i), max_residents=2) for i,name in enumerate(names)]
        resources = poolmod.PoolResources(2*MIB, MIB, MIB, MIB, 3*MIB, MIB)
        routes = [poolmod.EncoderRoute('barrier_'+name, 'terramind-v1-tiny-backbone',
            'missing-model', ('S2L2A',), names, resources) for name in ('a','b')]
        data = poolmod.EncoderInput({'S2L2A':np.zeros((1,12,224,224),np.float32)})
        with tempfile.TemporaryDirectory(prefix='pool-scaling-') as folder, \
             patch.object(poolmod, '_worker_main', scaling_worker):
            pool = poolmod.ModelPool(routes, slots, work=Path(folder)/'pool',
                host_memory_bytes=512*MIB, max_pending=8, max_results=8, max_pending_per_tenant=4, execute_model=True)
            try:
                handles = [pool.submit(routes[i%2].name, data, seed=i, tenant=str(i%2)) for i in range(8)]
                deadline = time.monotonic()+25
                while len(list(pool.work.glob('*/waiting-*'))) < 8 and time.monotonic() < deadline:
                    time.sleep(.02)
                meta = pool.metadata()
                self.assertEqual(len(list(pool.work.glob('*/waiting-*'))), 8, meta)
                self.assertEqual(sum(r['active'] for r in meta['residents']), 8)
                self.assertTrue(all(d['reserved_bytes'] <= d['budget_bytes'] for d in meta['devices'].values()))
                with self.assertRaises(poolmod.PoolBusy): pool.submit(routes[0].name, data)
                (pool.work/'release').write_text('release')
                first = [h.result(15) for h in handles]
                self.assertEqual(len({r.value['provenance']['pid'] for r in first}),8)
                self.assertEqual({r.device_slot for r in first},set(names))
                for i,result in enumerate(first): self.assertTrue(np.all(result.value['features'][0]==i))
                for wave in range(1,12):
                    jobs=[pool.submit(routes[i%2].name,data,seed=wave*8+i,tenant=str(i%2)) for i in range(8)]
                    for i,handle in enumerate(jobs):
                        result=handle.result(15)
                        self.assertTrue(np.all(result.value['features'][0]==wave*8+i))
                    meta=pool.metadata()
                    self.assertEqual(meta['pending'],0)
                    self.assertEqual(meta['reserved_results'],0)
                    self.assertLessEqual(meta['host_reserved_bytes'],512*MIB)
            finally:
                if pool.work.exists(): (pool.work/'release').touch()
                receipt=pool.close(cancel_pending=True, timeout=20)
            self.assertTrue(receipt['passed'],receipt)
            self.assertEqual(receipt['host_reserved_bytes'],0)
            self.assertEqual(receipt['unverified_live_workers'],[])

    def test_dead_simulated_node_does_not_block_healthy_node_or_replay_work(self):
        slots=[poolmod.DeviceSlot('node-a',8*MIB,'0'),poolmod.DeviceSlot('node-b',8*MIB,'1')]
        resources=poolmod.PoolResources(MIB,MIB,MIB,MIB,3*MIB,MIB)
        routes=[poolmod.EncoderRoute(name,'terramind-v1-tiny-backbone','missing-model',('S2L2A',),
                    (slot.name,),resources) for name,slot in zip(('healthy','crash_b'),slots)]
        data=poolmod.EncoderInput({'S2L2A':np.zeros((1,12,224,224),np.float32)})
        with tempfile.TemporaryDirectory(prefix='pool-node-loss-') as folder, \
             patch.object(poolmod,'_worker_main',scaling_worker):
            pool=poolmod.ModelPool(routes,slots,work=Path(folder)/'pool',host_memory_bytes=256*MIB,execute_model=True)
            try:
                failed=pool.submit('crash_b',data)
                healthy=pool.submit('healthy',data,seed=71)
                with self.assertRaises(poolmod.PoolExecutionError): failed.result(15)
                self.assertTrue(np.all(healthy.result(15).value['features'][0]==71))
                with self.assertRaises(poolmod.PoolExecutionError): pool.submit('crash_b',data)
                self.assertTrue(np.all(pool.submit('healthy',data,seed=72).result(15).value['features'][0]==72))
            finally: receipt=pool.close(cancel_pending=True,timeout=20)
            self.assertFalse(receipt['passed'])
            self.assertEqual(set(receipt['quarantined_slots']),{'node-b'})
            self.assertGreater(receipt['host_reserved_bytes'],0)  # Not falsely reclaimed.
            self.assertEqual(receipt['unverified_live_workers'],[])

    def test_64_logical_devices_reject_stale_model_and_capacity_mismatches(self):
        key=ExecutionKey('model','weights','recipe','generation','S2L2A','S1GRD','highest_fp32')
        request=PlacementRequest(key,2,3,4,5,deadline_seconds=3.)
        workers=[WorkerSnapshot(f'node-{i//8}-gpu-{i%8}','generation-1',f'device-{i}',key,
            True,True,10,10,10,0.,1.,.2,2.,100.) for i in range(64)]
        workers[0]=replace(workers[0],measured_at=80.)
        workers[1]=replace(workers[1],free_vram_bytes=1)
        workers[2]=replace(workers[2],key=replace(key,precision_profile='amp_bf16'))
        workers[3]=replace(workers[3],ready=False)
        result=plan_placement(request,workers,now=101.)
        self.assertEqual(result['eligible_count'],60)
        self.assertEqual(len(result['rejected']),4)
        self.assertFalse(result['reservation_acquired'])
        self.assertFalse(result['dispatched'])


if __name__=='__main__': unittest.main()
