"""Real spawned CPU processes with fake sessions; no weights, CUDA or network."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import os
import gc
import subprocess
import sys
import tempfile
import threading
import queue
import time
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from aureon.api import model_pool as poolmod
from aureon.api.generation_engine import GenerationEngine
from aureon.api.memory_input import GenerationInput
from aureon.api.generation_bridges import GenerationRecipe
from aureon.terramind_native.common import INPUT_BANDS

MIB = 1024**2
ORIGINAL_WORKER = poolmod._worker_main


class FakeSession:
    def __init__(self, route, slot, work):
        self.route, self.slot, self.work = route, slot, work
        work.mkdir(parents=True, exist_ok=False)
        if route.name.startswith('loadfail'):
            raise ValueError('fixture load failed')

    def run(self, payload, seed):
        (self.work / ('start-'+str(seed))).write_text(str(time.monotonic()))
        if self.route.name.startswith('crash'):
            os._exit(86)
        if self.route.name.startswith('fail'):
            raise ValueError('fixture inference failed')
        if self.route.name.startswith('slow'):
            time.sleep(0.35)
        if isinstance(self.route, poolmod.GenerationRoute):
            value = {'array': np.full((2,224,224), seed, np.float32),
                     'valid_mask': np.ones((224,224), bool),
                     'provenance': {'seed': seed, 'pid': os.getpid()}}
        else:
            value = {'features': [np.full((1,4,8), seed, np.float32)],
                     'provenance': {'seed': seed, 'pid': os.getpid()}}
        if self.route.name.startswith('oversize'):
            value['features'] = [np.zeros((1,1024,1024), np.float32)]
        (self.work / ('end-'+str(seed))).write_text(str(time.monotonic()))
        return value

    def close(self):
        (self.work / 'closed').write_text('closed')
        if self.route.name.startswith('nonjsonclose'):
            return {'passed':True,'unsupported':{1,2}}
        if self.route.name.startswith('scalarclose'):
            return ['invalid cleanup receipt']
        failed=self.route.name.startswith('badclose')
        return {'passed': not failed, 'fake_cpu_session': True,
                'owned_session_released':True,'cuda':{'allocated':602112 if failed else 0,'reserved':2097152 if failed else 0}}


def fake_attestation(slot):
    return {'kind': 'cuda' if slot.cuda_visible_device else 'cpu',
            'uuid': ('same-device' if slot.name.startswith('alias') else slot.name) if slot.cuda_visible_device else None,
            'pid': os.getpid(), 'selector': slot.cuda_visible_device}


def fake_worker(connection, route, slot, work, generation):
    poolmod._NativeSession = FakeSession
    poolmod._device_attestation = fake_attestation
    ORIGINAL_WORKER(connection, route, slot, work, generation)


def stalled_worker(connection, route, slot, work, generation):
    # A child which never receives commands. Closing the pool must still honor
    # its timeout and allow callers to inspect/cancel admitted work.
    time.sleep(0.8)
    connection.close()


def ready_but_not_reading_worker(connection, route, slot, work, generation):
    poolmod._send(connection,{'type':'ready','identity':poolmod._route_identity(route),
        'generation':generation,'device':fake_attestation(slot)},poolmod._META_LIMIT)
    # Parent admits a multi-megabyte command, larger than the OS pipe buffer.
    # Its writer pump may block, but the admission/lifecycle lock must not.
    time.sleep(0.8)
    connection.close()


def cpu_route(name='encoder', *, output=MIB):
    return poolmod.EncoderRoute(name, 'terramind-v1-tiny-backbone', 'missing-model',
        ('S2L2A',), ('cpu',), poolmod.PoolResources(0, MIB, 0, MIB, 3*MIB, output))


def generation_route(name='generation', *, devices=('gpu',)):
    return poolmod.GenerationRoute(name, 'terramind-v1-base', 'model', 'tokenizer',
        devices, poolmod.PoolResources(8*MIB, MIB, MIB, MIB, 3*MIB, MIB))


class ModelPoolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='model-pool-')
        self.root = Path(self.directory.name)
        self.patch = patch.object(poolmod, '_worker_main', fake_worker)
        self.patch.start()
        self.pools = []
        self.data = poolmod.EncoderInput({'S2L2A': np.zeros((1,12,224,224), np.float32)})

    def tearDown(self):
        for pool in self.pools:
            pool.close(cancel_pending=True, timeout=15)
        self.patch.stop()
        self.directory.cleanup()

    def pool(self, routes, *, devices=None, **options):
        value = poolmod.ModelPool(routes, devices or [poolmod.DeviceSlot('cpu',0)],
            work=self.root/('pool-'+str(len(self.pools))), host_memory_bytes=256*MIB,
            execute_model=True, **options)
        self.pools.append(value)
        return value

    def wait_running(self, handle):
        end = time.monotonic()+10
        while handle.state == 'queued' and time.monotonic()<end:
            time.sleep(0.01)
        self.assertEqual(handle.state, 'running')

    def manual_resident(self):
        route=cpu_route()
        pool=self.pool([route])
        pool.work.mkdir()
        parent,child=pool._context.Pipe(duplex=True)
        generation='controlled-close'
        slot=poolmod.DeviceSlot('cpu',0)
        process=pool._context.Process(target=fake_worker,args=(child,route,slot,pool.work/generation,generation),daemon=True)
        process.start(); child.close()
        transport=poolmod._Transport(parent,route)
        resident=poolmod._Resident(route,slot,generation,process,parent,transport=transport)
        pool._residents.append(resident)
        pool._host_used+=route.resources.resident_host_bytes
        end=time.monotonic()+10
        while resident.state=='loading' and time.monotonic()<end:
            with pool._condition: pool._poll()
            time.sleep(.005)
        self.assertEqual(resident.state,'idle')
        return pool,resident

    def test_binding_is_detached_and_lazy(self):
        work = self.root/'unused'
        engine = GenerationEngine(model_file='model',tokenizer_file='tokenizer',work=work,execute_model=True)
        value = engine.binding()
        value['model_id'] = 'changed'; value['work'] = 'changed'
        self.assertEqual(engine.binding()['model_id'], 'terramind-v1-base')
        self.assertEqual(engine.binding()['work'],str(work.resolve()))
        self.assertFalse(work.exists())
        self.assertEqual(engine.close()['model_load_count'],0)

    def test_registration_validation_and_import_laziness(self):
        code = "import sys; import aureon.api.model_pool; assert not any(x in sys.modules for x in ('torch','numpy','rasterio','terratorch'))"
        child = subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(child.returncode,0,child.stderr)
        route = cpu_route()
        with self.assertRaises(FrozenInstanceError): route.name='other'
        with self.assertRaises(poolmod.ExecutionRequired):
            poolmod.ModelPool([route],[poolmod.DeviceSlot('cpu',0)],work=self.root/'none',host_memory_bytes=256*MIB)
        with self.assertRaisesRegex(ValueError,'unique'):
            self.pool([route],devices=[poolmod.DeviceSlot('g1',20*MIB,'0'),poolmod.DeviceSlot('g2',20*MIB,'0')])
        with self.assertRaises(ValueError): replace(route, devices=('unknown',)) and self.pool([replace(route,devices=('unknown',))])
        with self.assertRaises(ValueError): self.pool([generation_route()],devices=[poolmod.DeviceSlot('gpu',9*MIB,'0')])
        with self.assertRaises(ValueError): poolmod.DeviceSlot('cpu',1)
        with self.assertRaises(ValueError): replace(route,tokenizer_files=(('LULC','x'),))

    def test_encoder_input_owns_values_and_rejects_bad_semantics(self):
        array=np.ones((1,12,224,224),np.float32)
        data=poolmod.EncoderInput({'S2L2A':array}); identity=data.identity
        array[:]=7
        self.assertEqual(data.identity,identity)
        self.assertTrue(np.frombuffer(data._items[0][3],np.float32).min()==1)
        for bad in (array.astype(np.float16),array[:, :11],np.ma.array(array),np.full_like(array,np.nan)):
            with self.subTest(dtype=bad.dtype):
                with self.assertRaises(ValueError): poolmod.EncoderInput({'S2L2A':bad})
        with self.assertRaises(ValueError): poolmod.EncoderInput({'S2L2A':array},max_bytes=4)
        with self.assertRaises(ValueError): poolmod.EncoderInput({'LULC':np.full((1,224,224),10,np.int64)})

    def test_cpu_two_residents_really_overlap_and_outputs_owned(self):
        pool=self.pool([cpu_route('slow_a'),cpu_route('slow_b')],devices=[poolmod.DeviceSlot('cpu',0,max_residents=2)])
        # Warm both processes first; startup variance cannot fake serialization.
        pool.submit('slow_a',self.data,seed=1).result(15)
        pool.submit('slow_b',self.data,seed=2).result(15)
        a=pool.submit('slow_a',self.data,seed=3,tenant='a')
        b=pool.submit('slow_b',self.data,seed=4,tenant='b')
        ra,rb=a.result(15),b.result(15)
        wa,wb=pool.work/ra.worker_generation,pool.work/rb.worker_generation
        self.assertLess(max(float((wa/'start-3').read_text()),float((wb/'start-4').read_text())),
                        min(float((wa/'end-3').read_text()),float((wb/'end-4').read_text())))
        self.assertNotEqual(ra.value['provenance']['pid'],rb.value['provenance']['pid'])
        self.assertEqual(pool.metadata()['devices']['cpu']['reserved_bytes'],0)
        ra.value['features'][0][:]=99
        self.assertTrue(np.all(rb.value['features'][0]==4))
        report=pool.close(timeout=15)
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['host_reserved_bytes'],0)
        self.assertEqual(pool.close(),report)

    def test_idle_only_lru_no_active_eviction_and_tenant_fairness(self):
        pool=self.pool([cpu_route('slow_a'),cpu_route('encoder_b')])
        first=pool.submit('slow_a',self.data,seed=1,tenant='a')
        self.wait_running(first)
        second=pool.submit('slow_a',self.data,seed=2,tenant='a')
        other=pool.submit('encoder_b',self.data,seed=3,tenant='b')
        r1,r3,r2=first.result(15),other.result(15),second.result(15)
        w1,w3,w2=(pool.work/r.worker_generation for r in (r1,r3,r2))
        self.assertNotEqual(r1.worker_generation,r2.worker_generation)
        self.assertTrue((w1/'closed').exists())
        self.assertLess(float((w1/'end-1').read_text()),float((w3/'start-3').read_text()))
        self.assertLess(float((w3/'end-3').read_text()),float((w2/'start-2').read_text()))

    def test_pending_result_backpressure_and_cancellation(self):
        pool=self.pool([cpu_route('slow_a')],max_pending=2,max_results=2)
        first=pool.submit('slow_a',self.data,seed=1)
        self.wait_running(first)
        queued=pool.submit('slow_a',self.data,seed=2)
        with self.assertRaises(poolmod.PoolBusy): pool.submit('slow_a',self.data)
        self.assertFalse(first.cancel()); self.assertTrue(queued.cancel())
        with self.assertRaises(poolmod.CancelledError): queued.result()
        first.result(15)
        third=pool.submit('slow_a',self.data,seed=3)
        self.wait_running(third); third.discard()
        report=pool.close(timeout=15)
        self.assertEqual(third.state,'discarded')
        self.assertEqual(report['reserved_results'],0)
        self.assertEqual(report['host_reserved_bytes'],0)

    def test_result_credit_until_consumed_and_unknown_route(self):
        pool=self.pool([cpu_route()],max_results=1)
        handle=pool.submit('encoder',self.data)
        end=time.monotonic()+15
        while not handle.done() and time.monotonic()<end: time.sleep(.01)
        self.assertTrue(handle.done())
        with self.assertRaises(poolmod.PoolBusy): pool.submit('encoder',self.data)
        value=handle.result()
        with self.assertRaises(poolmod.PoolError): handle.result()
        with self.assertRaises(ValueError): pool.submit('missing',self.data)
        self.assertEqual(value.execution_key.precision_profile,'highest_fp32')
        pool.submit('encoder',self.data).result(15)

    def test_load_and_forward_failures_are_route_isolated(self):
        for failed in ('loadfail_a','fail_a','oversize_a'):
            with self.subTest(route=failed):
                pool=self.pool([cpu_route(failed),cpu_route('healthy')])
                handle=pool.submit(failed,self.data)
                with self.assertRaises(poolmod.PoolExecutionError): handle.result(15)
                with self.assertRaises(poolmod.PoolExecutionError): pool.submit(failed,self.data)
                pool.submit('healthy',self.data,seed=7).result(15)
                report=pool.close(timeout=15)
                self.assertTrue(report['passed'],report)
                self.assertEqual(report['host_reserved_bytes'],0)

    def test_process_death_and_failed_close_quarantine_capacity(self):
        for failed in ('crash_a','badclose_a'):
            with self.subTest(route=failed):
                pool=self.pool([cpu_route(failed)])
                handle=pool.submit(failed,self.data)
                if failed.startswith('crash'):
                    with self.assertRaises(poolmod.PoolExecutionError): handle.result(15)
                else: handle.result(15)
                report=pool.close(timeout=15)
                self.assertFalse(report['passed'])
                self.assertIn('cpu',report['quarantined_slots'])
                self.assertGreater(report['host_reserved_bytes'],0)
                self.assertEqual(report['unverified_live_workers'],[])
                self.assertEqual(pool.close(),report)

    def test_failed_cleanup_preserves_worker_memory_receipt_and_exit_fence(self):
        pool=self.pool([cpu_route('badclose_a')])
        pool.submit('badclose_a',self.data).result(15)
        report=pool.close(timeout=15)
        failed=report['cleanup'][-1]
        self.assertFalse(report['passed'])
        self.assertEqual(failed['reason'],'Worker reported unsuccessful cleanup')
        self.assertFalse(failed['receipt']['passed'])
        self.assertEqual(failed['receipt']['cuda'],{'allocated':602112,'reserved':2097152})
        self.assertTrue(failed['receipt']['owned_session_released'])
        self.assertEqual(failed['process_exitcode'],0)
        self.assertTrue(failed['reader_done']);self.assertTrue(failed['incoming_drained'])
        self.assertGreater(report['host_reserved_bytes'],0)

    def test_malformed_close_receipts_fail_closed_without_crashing_dispatcher(self):
        for name in ('nonjsonclose','scalarclose'):
            pool=self.pool([cpu_route(name)])
            pool.submit(name,self.data).result(15)
            report=pool.close(timeout=15)
            self.assertFalse(report['passed'])
            failed=report['cleanup'][-1]
            self.assertEqual(failed['process_exitcode'],0)
            self.assertTrue(failed['receipt']['invalid_receipt'])
            self.assertFalse(failed['receipt']['passed'])
            self.assertGreater(report['host_reserved_bytes'],0)
            self.assertNotIn('dispatcher failed',failed['reason'])

    def test_close_timeout_does_not_preempt_or_hold_condition(self):
        pool=self.pool([cpu_route('slow_a')])
        handle=pool.submit('slow_a',self.data)
        self.wait_running(handle)
        start=time.monotonic()
        with self.assertRaises(poolmod.TimeoutError): pool.close(timeout=.01)
        self.assertLess(time.monotonic()-start,.25)
        self.assertFalse(pool.metadata()['accepting'])
        self.assertFalse(handle.cancel())
        handle.result(15)
        self.assertTrue(pool.close(timeout=15)['passed'])

    def test_stalled_process_does_not_block_status_or_close_timeout(self):
        with patch.object(poolmod,'_worker_main',stalled_worker):
            pool=self.pool([cpu_route()]); handle=pool.submit('encoder',self.data)
            start=time.monotonic()
            with self.assertRaises(poolmod.TimeoutError): pool.close(timeout=.03)
            self.assertLess(time.monotonic()-start,.25)
            self.assertTrue(handle.cancel())
            self.assertEqual(pool.metadata()['pending'],0)
            self.assertFalse(pool.close(timeout=15)['passed'])

    def test_blocked_command_pipe_keeps_close_timeout_and_retains_failed_reservations(self):
        with patch.object(poolmod,'_worker_main',ready_but_not_reading_worker):
            pool=self.pool([cpu_route()]); handle=pool.submit('encoder',self.data)
            self.wait_running(handle)
            reserved=pool.metadata()['host_reserved_bytes']
            start=time.monotonic()
            with self.assertRaises(poolmod.TimeoutError): pool.close(timeout=.03)
            self.assertLess(time.monotonic()-start,.25)
            self.assertFalse(handle.cancel())
            with self.assertRaises(poolmod.PoolExecutionError): handle.result(15)
            report=pool.close(timeout=15)
            self.assertFalse(report['passed'])
            self.assertEqual(report['host_reserved_bytes'],reserved)

    def test_close_ack_enqueued_after_empty_snapshot_and_after_real_process_exit(self):
        pool,resident=self.manual_resident()
        transport=resident.transport
        held=threading.Event(); release=threading.Event()
        original=poolmod._Transport._deliver

        def delayed(receiver,value):
            if receiver is transport and isinstance(value,dict) and value.get('type')=='closed':
                held.set()
                if not release.wait(10): raise AssertionError('Test did not release held close acknowledgement')
            original(receiver,value)

        class EmptySnapshotQueue(queue.Queue):
            armed=False
            def get_nowait(self):
                try: return super().get_nowait()
                except queue.Empty:
                    if self.armed:
                        self.armed=False
                        # The poll has observed empty, but its reader completes
                        # before the following process-is-alive check.
                        release.set()
                        if not transport.reader_done.wait(5): raise AssertionError('Reader fence not reached')
                        transport.reader.join(5)
                        if self.empty(): raise AssertionError('Expected buffered final acknowledgement')
                    raise

        messages=EmptySnapshotQueue(2)
        transport.incoming=messages
        with patch.object(poolmod._Transport,'_deliver',delayed):
            try:
                with pool._condition: pool._close_resident(resident)
                self.assertTrue(held.wait(10))
                resident.process.join(10)
                self.assertFalse(resident.process.is_alive())
                self.assertEqual(resident.process.exitcode,0)
                self.assertFalse(transport.reader_done.is_set())
                reserved=pool.metadata()['host_reserved_bytes']
                messages.armed=True
                with pool._condition: pool._poll()
                self.assertEqual(pool.metadata()['quarantined_slots'],{})
                self.assertEqual(pool.metadata()['host_reserved_bytes'],reserved)
                self.assertIn(resident,pool._residents)
                with pool._condition: pool._poll()
                self.assertEqual(pool._residents,[])
                self.assertEqual(pool.metadata()['host_reserved_bytes'],0)
                receipt=pool.close()
                self.assertTrue(receipt['passed'],receipt)
                self.assertTrue(receipt['cleanup'][0]['receipt']['passed'])
            finally:
                release.set()

    def test_dead_process_reader_drain_timeout_quarantines_without_returning_credits(self):
        pool,resident=self.manual_resident()
        transport=resident.transport
        held=threading.Event(); release=threading.Event()
        original=poolmod._Transport._deliver

        def held_reader(receiver,value):
            if receiver is transport and isinstance(value,dict) and value.get('type')=='closed':
                held.set()
                if not release.wait(10): raise AssertionError('Test did not release stalled reader')
            original(receiver,value)

        with patch.object(poolmod._Transport,'_deliver',held_reader),patch.object(poolmod,'_EXIT_DRAIN_SECONDS',.05):
            try:
                with pool._condition: pool._close_resident(resident)
                self.assertTrue(held.wait(10))
                resident.process.join(10)
                self.assertFalse(resident.process.is_alive())
                reserved=pool.metadata()['host_reserved_bytes']
                with pool._condition: pool._poll()
                self.assertEqual(pool.metadata()['quarantined_slots'],{})
                time.sleep(.06)
                with pool._condition: pool._poll()
                self.assertEqual(resident.state,'quarantined')
                self.assertEqual(pool.metadata()['host_reserved_bytes'],reserved)
                self.assertIn('drain timed out',pool.metadata()['quarantined_slots']['cpu'])
            finally:
                release.set()
                self.assertTrue(transport.reader_done.wait(5))
                with pool._condition: pool._poll()
                with pool._condition: pool._poll()
        self.assertFalse(pool.close()['passed'])
        self.assertEqual(pool.metadata()['host_reserved_bytes'],0)

    def test_generation_dispatch_identity_seed_and_device_reservations(self):
        data=GenerationInput(np.ones((12,224,224),np.float32),bands=INPUT_BANDS,
            units='surface_reflectance_x10000',valid_mask=np.ones((224,224),bool))
        route=generation_route()
        pool=self.pool([route],devices=[poolmod.DeviceSlot('gpu',20*MIB,'0')])
        first=pool.submit(route.name,data,seed=19).result(15)
        second=pool.submit(route.name,data,seed=21).result(15)
        self.assertEqual(first.worker_generation,second.worker_generation)
        self.assertNotEqual(first.execution_key.recipe_identity,second.execution_key.recipe_identity)
        self.assertEqual(first.route_identity,second.route_identity)
        self.assertTrue(np.all(first.value['array']==19))
        self.assertEqual(pool.metadata()['devices']['gpu']['reserved_bytes'],8*MIB)
        self.assertTrue(pool.close(timeout=15)['passed'])

    def test_physical_device_alias_rejected_before_second_model_call(self):
        data=GenerationInput(np.ones((12,224,224),np.float32),bands=INPUT_BANDS,
            units='surface_reflectance_x10000',valid_mask=np.ones((224,224),bool))
        a,b=generation_route('a',devices=('alias_a',)),generation_route('b',devices=('alias_b',))
        pool=self.pool([a,b],devices=[poolmod.DeviceSlot('alias_a',20*MIB,'0'),poolmod.DeviceSlot('alias_b',20*MIB,'1')])
        first=pool.submit('a',data).result(15)
        with self.assertRaises(poolmod.PoolExecutionError): pool.submit('b',data).result(15)
        self.assertEqual(pool.submit('a',data).result(15).worker_generation,first.worker_generation)
        report=pool.close(timeout=15)
        self.assertFalse(report['passed'])
        self.assertEqual(set(report['quarantined_slots']),{'alias_b'})
        self.assertEqual(report['host_reserved_bytes'],0)

    def test_host_and_input_byte_admission_and_tenant_result_limits(self):
        route=cpu_route()
        with self.assertRaisesRegex(ValueError,'Host budget'):
            poolmod.ModelPool([route],[poolmod.DeviceSlot('cpu',0)],work=self.root/'too-small',host_memory_bytes=MIB,execute_model=True)
        pool=self.pool([replace(route,resources=replace(route.resources,max_input_bytes=16))])
        with self.assertRaisesRegex(poolmod.PoolBusy,'Input'): pool.submit(route.name,self.data)
        self.assertFalse(pool.work.exists())
        second=self.pool([route],max_results_per_tenant=1)
        first=second.submit(route.name,self.data,tenant='a')
        with self.assertRaises(poolmod.PoolBusy): second.submit(route.name,self.data,tenant='a')
        other=second.submit(route.name,self.data,tenant='b')
        first.result(15); other.result(15)

    def test_minimum_host_budget_reuses_resident_and_switches_after_verified_eviction(self):
        route=cpu_route('one')
        route=replace(route,resources=replace(route.resources,max_input_bytes=self.data.nbytes))
        resources=route.resources
        budget=resources.resident_host_bytes+resources.transient_host_bytes+8*(resources.max_input_bytes+resources.max_output_bytes)+4*MIB
        pool=poolmod.ModelPool([route,replace(route,name='two')],[poolmod.DeviceSlot('cpu',0)],
            work=self.root/'minimum',host_memory_bytes=budget,execute_model=True)
        self.pools.append(pool)
        first=pool.submit('one',self.data).result(15)
        reused=pool.submit('one',self.data).result(15)
        self.assertEqual(first.worker_generation,reused.worker_generation)
        second=pool.submit('two',self.data).result(15)
        self.assertNotEqual(second.worker_generation,first.worker_generation)
        self.assertTrue((pool.work/first.worker_generation/'closed').is_file())
        self.assertLessEqual(pool.metadata()['host_reserved_bytes'],budget)
        self.assertEqual(pool.close(timeout=15)['host_reserved_bytes'],0)

    def test_native_generation_adapter_forwards_exact_binding_recipe(self):
        class Engine:
            def __init__(self, model, **kwargs): self.model,self.kwargs=model,kwargs
            def generate(self,data,**kwargs):
                self.input,self.recipe=data,kwargs['recipe']
                return {'array':np.ones((2,224,224),np.float32),'valid_mask':np.ones((224,224),bool),'provenance':{}}
        route=generation_route()
        data=GenerationInput(np.ones((12,224,224),np.float32),bands=INPUT_BANDS,
            units='surface_reflectance_x10000',valid_mask=np.ones((224,224),bool))
        with patch('aureon.api.generation_engine.GenerationEngine',Engine):
            session=poolmod._NativeSession(route,poolmod.DeviceSlot('gpu',20*MIB,'0'),self.root/'native')
            session.run(poolmod._input_wire(data),123)
            self.assertEqual(session.runtime.input.identity,data.identity)
            self.assertEqual(session.runtime.recipe,replace(route.recipe,seed=123))
            self.assertEqual(session.runtime.kwargs['execution_profile'],'recommended')
            self.assertEqual(session.runtime.kwargs['model_file'],route.model_file)

    def test_native_encoder_adapter_owns_cpu_outputs(self):
        try: import torch
        except ImportError: self.skipTest('Torch runtime not installed; no environment changes')
        class Encoder:
            def __init__(self,model,**kwargs): self.model,self.kwargs=model,kwargs
            def predict(self,tensors,seed):
                self.tensors,self.seed=tensors,seed
                self.output=torch.full((1,4,8),float(seed))
                return {'features':[self.output], 'provenance':{'seed':seed}}
        with patch('aureon.api.terramind_encoder.TerraMindEncoder',Encoder):
            session=poolmod._NativeSession(cpu_route(),poolmod.DeviceSlot('cpu',0),self.root/'encoder')
            result=session.run(poolmod._input_wire(self.data),12)
            session.runtime.output.fill_(100)
            self.assertTrue(np.all(result['features'][0]==12))
            self.assertEqual(session.runtime.kwargs['device'],'cpu')
            self.assertEqual(session.runtime.seed,12)

    def test_native_encoder_ownership_releases_feature_and_position_storage_on_cpu(self):
        import torch
        from aureon.api import terramind_encoder as encoder
        from aureon.terramind_native.encoder_loader import LoadedEncoder
        refs=[]
        def observe(tensor,name):
            refs.append((name,weakref.ref(tensor),weakref.ref(tensor.untyped_storage())))
        class ConstantStandIn:
            training=True
            def __init__(self):
                self.pos=torch.ones((1,196,768),dtype=torch.float32)
                observe(self.pos,'position')
            def modules(self):return [self]
            def eval(self):self.training=False;return self
            def __call__(self,inputs):
                values=[torch.ones((1,196,768),dtype=torch.float32) for _ in range(12)]
                for tensor in values:observe(tensor,'native')
                return values
        def load(*args,**kwargs):return LoadedEncoder(ConstantStandIn(),{})
        original=encoder._own_features
        def own(*args,**kwargs):
            values=original(*args,**kwargs)
            for tensor in values:observe(tensor,'owned')
            return values
        with patch.object(encoder,'load_encoder',new=load),patch.object(encoder,'_own_features',new=own), \
                patch.object(encoder,'_prepare_encoder_modules',return_value=None):
            session=poolmod._NativeSession(cpu_route(output=10*MIB),poolmod.DeviceSlot('cpu',0),self.root/'standin')
            for _ in range(2):
                result=session.run(poolmod._input_wire(self.data),17)
                gc.collect()
                self.assertTrue(all(value.flags.owndata and value.base is None for value in result['features']))
                self.assertEqual([name for name,tensor,storage in refs if tensor() is not None or storage() is not None],['position'])
            self.assertTrue(session.close()['passed'])
            self.assertFalse(any(tensor() is not None or storage() is not None for _,tensor,storage in refs))


if __name__=='__main__': unittest.main()
