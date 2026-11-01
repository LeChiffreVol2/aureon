"""CPU graph/stream scheduling doubles; never claims CUDA execution or overlap."""
from concurrent.futures import CancelledError
from contextlib import contextmanager
from copy import deepcopy
import threading
import unittest
from unittest.mock import patch
import torch

from aureon.terramind_native.denoiser_lanes import TwoLaneGraphs, LaneRuntime, _NativeRuntime
from aureon.terramind_native.graph_runtime import FixedTensorGraph


class LaneScalar:
    """Logical device differs from CPU, so setup/replay signature regressions fail."""
    device='lane-device'
    dtype=torch.int64
    shape=()
    def __init__(self,value):self.value=value
    def clone(self):return LaneScalar(self.value.clone())
    def copy_(self,other):self.value.copy_(other.value)
    def data_ptr(self):return self.value.data_ptr()


class LaneSchedule:
    def __init__(self,steps):self.steps=steps.clone()
    def __getitem__(self,index):return LaneScalar(self.steps[index])


class Backend:
    label='cpu_stream_test_double'
    def __init__(self):self.current_id=0;self.next_id=0;self.events=[]
    def create_stream(self):self.next_id+=1;return self.next_id
    def current(self):return self.current_id
    @contextmanager
    def context(self,stream):
        old=self.current_id;self.current_id=stream
        try:yield
        finally:self.current_id=old
    def wait(self,stream,origin):self.events.append(('wait',stream,origin))
    def record(self,tensor,stream):self.events.append(('record',stream))
    def sync(self,stream):self.events.append(('sync',stream))
    def to_lane_timesteps(self,steps):return LaneSchedule(steps)
    def event(self,stream):self.events.append(('event',stream));return (stream,len(self.events))
    def join(self,origin,event):self.events.append(('join',origin,event))
    def stream_id(self,stream):return stream
    def intervals(self,*args):return [{'test_double':True}]*2


class Binding:
    def __init__(self,model):
        self.shared=model;self.frequency=torch.ones(4);self.embedding=lambda *a:None
    def namespace(self):return self.shared


class Graph:
    def __init__(self,binding,backend,index):
        self.binding=binding;self.backend=backend;self.index=index
        self.started=False;self.active=False;self.closed=False;self.buffers={};self.output=None
        self.signature=None;self.fail_close=False
        self.total=dict(warmup_stage_forwards=0,capture_stage_forwards=0,replay_stage_forwards=0,released=False)
        self.graph=self
    def pool(self):return (123,self.index)
    @contextmanager
    def request(self):
        if self.active or self.closed:raise RuntimeError('bad graph lifecycle')
        self.active=True
        try:yield
        finally:self.active=False
    def call(self,inputs):
        # Exercise the accepted graph runtime's actual signature function.
        signature=FixedTensorGraph._signature(inputs)
        if self.signature is not None and signature!=self.signature:raise ValueError('Graph input signature changed')
        self.backend.events.append(('call',self.backend.current_id,self.index,inputs['timestep'].device))
        if not self.started:
            self.signature=signature;self.started=True
            self.buffers={k:v.clone() for k,v in inputs.items()}
            self.total['warmup_stage_forwards']=self.total['capture_stage_forwards']=1
        else:
            for k,v in inputs.items():self.buffers[k].copy_(v)
        timestep=inputs['timestep'].value if isinstance(inputs['timestep'],LaneScalar) else inputs['timestep']
        value=inputs['image']*.25+inputs['conditioning'].mean()+timestep.float()/100
        if self.output is None:self.output=value.clone()
        else:self.output.copy_(value)
        self.total['replay_stage_forwards']+=1
        return self.output.clone()
    def close(self):
        self.backend.events.append(('close',self.index));self.closed=True
        self.buffers.clear();self.output=None;self.binding.frequency=None;self.total['released']=True
        if self.fail_close:raise RuntimeError('fixture close failure')
        return dict(self.total)


class Scheduler:
    def __init__(self,backend=None,index=0):
        self.timesteps=torch.arange(9,-1,-1,dtype=torch.int64);self.backend=backend;self.index=index
    def step(self,pred,timestep,image,generator=None):
        assert generator is None
        if self.backend:self.backend.events.append(('scheduler',self.backend.current_id,self.index))
        from types import SimpleNamespace
        return SimpleNamespace(prev_sample=image*.75+pred*.25)


def states(backend=None):
    return [dict(image=torch.full((1,2,224,224),float(i)),conditioning=torch.full((1,5,14,14),i/10),
                 scheduler=Scheduler(backend,i)) for i in (1,2)]


class LaneContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.backend=Backend();self.graphs=[]
        self.original=lambda *a:None;self.namespace={'timestep_embedding':self.original}
        def factory(binding):
            graph=Graph(binding,self.backend,len(self.graphs));self.graphs.append(graph);return graph
        self.lanes=TwoLaneGraphs(self.namespace,backend=self.backend,binding_factory=Binding,graph_factory=factory)
    def tearDown(self):
        if not self.lanes.closed:self.lanes.close()
    def test_cold_to_warm_uses_same_logical_device_signature(self):
        first=self.lanes.run(states(self.backend),cancel=lambda:None)
        self.assertEqual(first['serial_setup_output_steps'],1)
        self.assertEqual(first['parallel_submission_steps'],9)
        cold=[g.signature for g in self.graphs]
        for signature in cold:
            self.assertEqual(next(row[3] for row in signature if row[0]=='timestep'),'lane-device')
        second=self.lanes.run(states(self.backend),cancel=lambda:None)
        self.assertEqual([g.signature for g in self.graphs],cold)
        self.assertEqual(second['serial_setup_output_steps'],0)
        self.assertEqual(second['parallel_submission_steps'],10)
        self.assertTrue(all(x['warmup_stage_forwards']==0 and x['capture_stage_forwards']==0
                            and x['replay_stage_forwards']==10 for x in second['lane_counters']))
        self.assertIs(self.namespace['timestep_embedding'],self.original)
    def test_each_iteration_submits_both_B1_calls_before_scheduler_work(self):
        self.lanes.run(states(self.backend),cancel=lambda:None)
        events=[x for x in self.backend.events if x[0] in ('call','scheduler')]
        self.assertEqual(len(events),40)
        for i in range(0,40,4):self.assertEqual([x[0] for x in events[i:i+4]],['call','call','scheduler','scheduler'])
        calls=[e for e in events if e[0]=='call']
        self.assertEqual([e[1] for e in calls[:2]],[0,0])
        self.assertEqual([e[1] for e in calls[2:]],[1,2]*9)
    def test_results_equal_two_serial_B1_cpu_fixtures_and_do_not_alias(self):
        values=states(self.backend);reference=states()
        for state in reference:
            for step in state['scheduler'].timesteps:
                pred=state['image']*.25+state['conditioning'].mean()+step.float()/100
                state['image']=state['scheduler'].step(pred,step,state['image']).prev_sample
        receipt=self.lanes.run(values,cancel=lambda:None)
        for actual,expected in zip(values,reference):self.assertTrue(torch.equal(actual['image'],expected['image']))
        self.assertNotEqual(values[0]['image'].data_ptr(),values[1]['image'].data_ptr())
        self.assertTrue(receipt['ownership']['disjoint_buffers'])
        self.assertTrue(receipt['ownership']['distinct_graph_pools'])
    def test_cancel_drains_both_streams_before_graphs_can_be_closed(self):
        calls=[]
        def cancel():
            calls.append(1)
            if len(calls)==4:raise CancelledError()
        with self.assertRaises(CancelledError):self.lanes.run(states(),cancel=cancel)
        self.assertEqual(self.backend.events[-2:],[('sync',1),('sync',2)])
        self.lanes.close()
        self.assertEqual(self.backend.events[-2:],[('close',0),('close',1)])
    def test_shared_graph_pool_is_rejected_and_namespace_restored(self):
        self.graphs[1].pool=lambda:(123,0)
        with self.assertRaisesRegex(RuntimeError,'must not alias'):self.lanes.run(states(),cancel=lambda:None)
        self.assertIs(self.namespace['timestep_embedding'],self.original)
        self.assertEqual(self.backend.events[-2:],[('sync',1),('sync',2)])
    def test_close_attempts_both_graphs_even_if_one_reset_fails(self):
        self.graphs[0].fail_close=True
        receipt=self.lanes.close()
        self.assertFalse(receipt['released'])
        self.assertEqual(self.backend.events[-2:],[('close',0),('close',1)])
        self.assertTrue(all(g.closed for g in self.graphs))
        repeated=self.lanes.close()
        self.assertFalse(repeated['released'])
        self.assertEqual(repeated['errors'],receipt['errors'])
    def test_cancel_with_failed_close_is_fatal_and_final_receipt_preserves_failure(self):
        runtime=LaneRuntime();runtime.parallel_lanes=self.lanes
        self.graphs[0].fail_close=True
        def cancel():raise CancelledError('fixture cancellation')
        with self.assertRaisesRegex(RuntimeError,'session cannot be reused') as error:
            runtime.lane_stage(states(),cancel=cancel)
        self.assertIsInstance(error.exception.__cause__,CancelledError)
        self.assertIs(runtime.parallel_lanes,self.lanes)
        with patch.object(_NativeRuntime,'close',return_value={'passed':True}):
            receipt=runtime.close()
        self.assertFalse(receipt['passed'])
        self.assertFalse(receipt['parallel_lanes']['released'])
        self.assertFalse(receipt['parallel_cleanup_failure']['released'])
    def test_terminal_close_reports_failed_lane_release(self):
        runtime=LaneRuntime();runtime.parallel_lanes=self.lanes
        self.graphs[0].fail_close=True
        with patch.object(_NativeRuntime,'close',return_value={'passed':True}):
            receipt=runtime.close()
        self.assertFalse(receipt['passed'])
        self.assertFalse(receipt['parallel_lanes']['released'])
    def test_cancel_with_raised_cleanup_retains_lanes_and_is_fatal(self):
        runtime=LaneRuntime();runtime.parallel_lanes=self.lanes
        def cancel():raise CancelledError()
        with patch.object(self.lanes,'close',side_effect=RuntimeError('drain failure')):
            with self.assertRaisesRegex(RuntimeError,'session cannot be reused'):
                runtime.lane_stage(states(),cancel=cancel)
        self.assertIs(runtime.parallel_lanes,self.lanes)
        with patch.object(_NativeRuntime,'close',return_value={'passed':True}):
            receipt=runtime.close()
        self.assertFalse(receipt['passed'])
        self.assertIsNotNone(receipt['parallel_cleanup_failure'])
    def test_wrong_owner_rejects_without_stream_calls(self):
        failures=[];before=list(self.backend.events)
        def other():
            try:self.lanes.run(states(),cancel=lambda:None)
            except RuntimeError as error:failures.append(str(error))
        thread=threading.Thread(target=other);thread.start();thread.join(2)
        self.assertEqual(failures,['Lane owner thread differs'])
        self.assertEqual(before,self.backend.events)


if __name__=='__main__':unittest.main()
