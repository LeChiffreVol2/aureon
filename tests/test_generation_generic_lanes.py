"""Actual generic scheduling with tiny CPU tensor/stream doubles, never CUDA."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
from concurrent.futures import CancelledError
from unittest.mock import patch
import torch
from test_generation_lanes import Backend, Binding, Graph, Scheduler
from aureon.terramind_native.denoiser_lanes import LaneGraphs, LaneRuntime
from aureon.api.group_generation import NativeGroupSession
from aureon.api import GenerationJob, GenerationRecipe
from aureon.api.generation_queue import INPUT_BYTES,OUTPUT_BYTES
from aureon.terramind_native.common import write_json
from test_generation_batch import Runtime, RECIPE, data


def states(n,backend=None):
    return [dict(image=torch.full((1,2,224,224),float(i)),conditioning=torch.full((1,5,14,14),i/10),
                 scheduler=Scheduler(backend,i)) for i in range(n)]


class GenericLaneContracts(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);self.owned=[]
    def make(self,n,cls=LaneGraphs):
        backend=Backend();graphs=[];namespace={'timestep_embedding':lambda *a:None}
        def factory(binding):
            graph=Graph(binding,backend,len(graphs));graphs.append(graph);return graph
        kwargs=dict(backend=backend,binding_factory=Binding,graph_factory=factory)
        lane=cls(namespace,**kwargs) if cls is not LaneGraphs else cls(namespace,max_lanes=n,**kwargs)
        self.owned.append(lane)
        return lane,graphs,backend,namespace
    def tearDown(self):
        for lane in self.owned:lane.close()
    def test_N1_N2_N3_N5_equal_independent_serial_arithmetic(self):
        for n in (1,2,3,5):
            with self.subTest(n=n):
                lane,graphs,backend,namespace=self.make(n)
                actual=states(n,backend);reference=states(n)
                for state in reference:
                    for step in state['scheduler'].timesteps:
                        pred=state['image']*.25+state['conditioning'].mean()+step.float()/100
                        state['image']=state['scheduler'].step(pred,step,state['image']).prev_sample
                receipt=lane.run(actual,cancel=lambda:None)
                self.assertEqual(receipt['logical_group_size'],n)
                self.assertEqual(receipt['neural_batch_size'],1)
                self.assertTrue(all(torch.equal(a['image'],b['image']) for a,b in zip(actual,reference)))
                self.assertTrue(all(g.binding.shared is namespace for g in graphs))
                events=[e[0] for e in backend.events if e[0] in ('call','scheduler')]
                self.assertEqual(events,(['call']*n+['scheduler']*n)*10)
    def test_mixed_3_2_3_retains_pools_and_never_recaptures_warm_lanes(self):
        lane,graphs,_,_=self.make(5)
        a=lane.run(states(3),cancel=lambda:None);pools=[g.pool() for g in graphs]
        for n in (2,3):
            r=lane.run(states(n),cancel=lambda:None)
            self.assertEqual(r['newly_captured_lanes'],0)
            self.assertEqual(r['initialized_lanes'],3)
            self.assertEqual([g.pool() for g in graphs],pools)
            self.assertTrue(all(c['capture_stage_forwards']==0 for c in r['lane_counters']))
    def test_growth_1_3_5_captures_only_added_lanes_with_serial_first_step(self):
        lane,graphs,_,_=self.make(5)
        for n,new in ((1,1),(3,2),(5,2)):
            r=lane.run(states(n),cancel=lambda:None)
            self.assertEqual(r['newly_captured_lanes'],new)
            self.assertEqual(sum(c['capture_stage_forwards'] for c in r['lane_counters']),new)
            self.assertEqual(sum(c['replay_stage_forwards'] for c in r['lane_counters']),n*10)
    def test_disjoint_check_includes_third_and_cached_inactive_lane(self):
        lane,graphs,_,_=self.make(3);lane.run(states(3),cancel=lambda:None)
        graphs[2].pool=graphs[0].pool
        with self.assertRaisesRegex(RuntimeError,'must not alias'):lane.run(states(2),cancel=lambda:None)
    def test_all_scheduler_identities_checked_not_only_first_two(self):
        lane,_,_,_=self.make(5);values=states(5);values[4]['scheduler']=values[0]['scheduler']
        with self.assertRaises(ValueError):lane.run(values,cancel=lambda:None)
    def test_large_capacity_constructs_no_graphs_streams_or_weights_copies(self):
        lane,graphs,backend,_=self.make(10**12)
        self.assertEqual(graphs,[]);self.assertEqual(backend.next_id,0)
        lane.run(states(3),cancel=lambda:None)
        self.assertEqual(len(graphs),3);self.assertEqual(backend.next_id,3)
    def test_partial_setup_failure_closes_every_created_lane_and_no_retry(self):
        lane,graphs,backend,_=self.make(5);runtime=LaneRuntime(5);runtime.parallel_lanes=lane
        original=lane._setup_call;calls=[]
        def fail(graph,inputs):
            calls.append(1)
            if len(calls)==3:raise RuntimeError('third setup fails')
            return original(graph,inputs)
        with patch.object(lane,'_setup_call',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'third setup'):runtime.lane_stage(states(5),cancel=lambda:None)
        self.assertEqual(len(calls),3);self.assertTrue(all(g.closed for g in graphs))
        self.assertIsNone(runtime.parallel_lanes)
        self.assertEqual([x for x in backend.events if x[0]=='close'],[('close',i) for i in range(5)])
    def test_group_cancel_drains_every_lane_and_runtime_retires_all_pools(self):
        lane,graphs,backend,_=self.make(5);runtime=LaneRuntime(5);runtime.parallel_lanes=lane;calls=[]
        def cancel():
            calls.append(1)
            if len(calls)==7:raise CancelledError()
        with self.assertRaises(CancelledError):runtime.lane_stage(states(5),cancel=cancel)
        self.assertTrue(all(g.closed for g in graphs));self.assertIsNone(runtime.parallel_lanes)
        self.assertTrue(set(('sync',i) for i in range(1,6))<=set(backend.events))
    def test_N2_cold_warm_drift_against_byte_preserved_qualified_implementation(self):
        path=Path(__file__).parent/'fixtures/qualified_denoiser_lanes_v1.py'
        spec=importlib.util.spec_from_file_location('aureon.terramind_native._qualified_lane_fixture',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        old,_,_,_=self.make(2,cls=module.TwoLaneGraphs);new,_,_,_=self.make(2)
        for _ in range(2):
            left,right=states(2),states(2)
            old_receipt=old.run(left,cancel=lambda:None);new_receipt=new.run(right,cancel=lambda:None)
            self.assertTrue(all(torch.equal(a['image'],b['image']) for a,b in zip(left,right)))
            for key in ('lane_counters','serial_setup_output_steps','parallel_submission_steps'):
                self.assertEqual(old_receipt[key],new_receipt[key])


class GenericNativeOwnerContracts(unittest.TestCase):
    def setUp(self):
        self.temp=self.enterContext(tempfile.TemporaryDirectory());self.root=Path(self.temp)
        write_json(self.root/'config.json',{'recipe':RECIPE})
        self.runtime=Runtime();self.groups=[]
        backend=Backend();namespace={'timestep_embedding':lambda *a:None};graphs=[]
        def graph_factory(binding):
            graph=Graph(binding,backend,len(graphs));graphs.append(graph);return graph
        self.lanes=None
        def stage(states,cancel):
            if self.lanes is None:
                self.lanes=LaneGraphs(namespace,max_lanes=5,backend=backend,binding_factory=Binding,graph_factory=graph_factory)
            return self.lanes.run(states,cancel=cancel)
        self.runtime.lane_stage=stage
        def prepare_stage(prepares,*,cancel):
            from aureon.terramind_native.denoiser_batch import _unchanged_rng
            prepared=[prepare() for prepare in prepares]
            with _unchanged_rng(torch,()) as rng:receipt=stage(prepared,cancel)
            return prepared,receipt,rng
        self.runtime.prepare_stage=prepare_stage
        self.enterContext(patch.object(NativeGroupSession,'_runtime_factory',return_value=self.runtime))
        self.enterContext(patch.object(NativeGroupSession,'_group_admission',return_value={'test_double':True}))
        self.session=NativeGroupSession(self.root/'config.json',self.root/'work',execute_model=True,
            allow_unqualified_batch=True,max_lanes=5,max_input_bytes=5*INPUT_BYTES,max_output_bytes=5*OUTPUT_BYTES,reuse='off')
        self.addCleanup(self.session.close)
        # The native owner owns the graph lifecycle, including the CPU test double.
        original=self.runtime.close
        self.runtime.close=lambda: (self.lanes.close() if self.lanes else None,original())[1]
    def jobs(self,seeds):
        return [GenerationJob(data(),recipe=GenerationRecipe(seed=seed),job_id='j'+str(i)) for i,seed in enumerate(seeds)]
    def test_native_preparation_N3_17_101_17_preserves_seed_association_and_commit(self):
        import numpy as np
        results=self.session.generate_group(self.jobs([17,101,17]))
        self.assertTrue(np.array_equal(results[0]['array'],results[2]['array']))
        self.assertFalse(np.shares_memory(results[0]['array'],results[2]['array']))
        self.assertEqual([r['provenance']['recipe']['seed'] for r in results],[17,101,17])
        self.assertTrue(all(r['provenance']['rng']['restored'] for r in results))
        self.assertEqual(self.session.metadata()['jobs_completed'],3)
        self.assertEqual(self.session.metadata()['model_load_count'],1)
        self.assertEqual(self.session.metadata()['batch_candidate']['credits']['active_requests'],0)
        self.assertEqual(len({r['group_commit']['sha256'] for r in results}),1)
    def test_precancel_group_consumes_no_model_or_attempts(self):
        import threading
        event=threading.Event();event.set()
        with self.assertRaises(CancelledError):self.session.generate_group(self.jobs([17,101,17]),cancel_event=event)
        self.assertEqual(self.session._attempts,set());self.assertEqual(self.session.metadata()['model_load_count'],0)


if __name__=='__main__':unittest.main()
