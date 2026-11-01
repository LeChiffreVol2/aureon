"""Two request-owned B1 CUDA graphs on separate streams; qualification is external.

Weights are immutable/shared. Shapes, timestep embedding and numerical decoder
binding reuse the accepted B1 graph implementation. Graph pools, inputs, outputs,
frequency buffers, stream dependencies and scheduler state are independent.
"""
from contextlib import ExitStack
from copy import deepcopy
import threading

from .session import _NativeRuntime
from .denoiser_batch import TARGET, BatchIneligible, eligibility, run_pair

PROFILE = 'experimental_s1grd224_fp32_two_b1_graph_lanes_v1'


class CudaLaneBackend:
    label = 'cuda_streams'
    def __init__(self, torch): self.torch = torch
    def create_stream(self): return self.torch.cuda.Stream(device='cuda:0')
    def current(self): return self.torch.cuda.current_stream('cuda:0')
    def context(self, stream): return self.torch.cuda.stream(stream)
    def wait(self, stream, origin): stream.wait_stream(origin)
    def record(self, tensor, stream): tensor.record_stream(stream)
    def sync(self, stream): stream.synchronize()
    def to_lane_timesteps(self, steps): return steps.to(device='cuda:0')
    def event(self, stream):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event
    def join(self, origin, event): origin.wait_event(event)
    def stream_id(self, stream): return int(stream.cuda_stream)
    def intervals(self, origin_event, starts, ends):
        return [{'start_ms': origin_event.elapsed_time(a), 'end_ms': origin_event.elapsed_time(b)}
                for a,b in zip(starts,ends)]


class TwoLaneGraphs:
    """One bounded graph/pool and stream per lane, reused only on the owner thread."""
    def __init__(self, model, *, backend=None, binding_factory=None, graph_factory=None):
        import torch
        from .denoiser_binding import S1GRDDecoderBinding
        from .graph_runtime import FixedTensorGraph
        self.torch = torch
        self.owner_thread = threading.get_ident()
        self.backend = backend if backend is not None else CudaLaneBackend(torch)
        binding_factory = binding_factory or S1GRDDecoderBinding
        graph_factory = graph_factory or FixedTensorGraph
        self.graphs = [graph_factory(binding_factory(model)) for _ in range(2)]
        self.streams = [self.backend.create_stream() for _ in range(2)]
        if len({self.backend.stream_id(s) for s in self.streams}) != 2:
            raise RuntimeError('Parallel lanes require distinct owned streams')
        self.closed = False
        self.close_receipt = None

    def _setup_call(self, graph, inputs):
        # Only setup calls execute Python embedding code. Replays use their own
        # captured buffers; no shared module/global patch remains during overlap.
        from .denoiser_graph import _PATCH_LOCK
        if not _PATCH_LOCK.acquire(blocking=False):
            raise RuntimeError('Decoder embedding binding is already owned')
        namespace = None
        try:
            namespace = graph.binding.namespace()
            original = namespace['timestep_embedding']
            namespace['timestep_embedding'] = graph.binding.embedding
            return graph.call(inputs)
        finally:
            if namespace is not None: namespace['timestep_embedding'] = original
            _PATCH_LOCK.release()

    def _independent_buffers(self):
        addresses = []
        pools = []
        for graph in self.graphs:
            tensors = [*graph.buffers.values(), graph.output, graph.binding.frequency]
            addresses.append({value.data_ptr() for value in tensors if value is not None})
            pools.append(tuple(graph.graph.pool()))
        if addresses[0] & addresses[1] or pools[0] == pools[1]:
            raise RuntimeError('Lane graph storage/pools must not alias')
        return {'disjoint_buffers':True, 'distinct_graph_pools':True,
                'graph_pools':pools, 'owned_buffer_counts':[len(a) for a in addresses]}

    def run(self, states, *, cancel):
        t, backend = self.torch, self.backend
        if threading.get_ident()!=self.owner_thread: raise RuntimeError('Lane owner thread differs')
        if self.closed: raise RuntimeError('Parallel lane graphs are closed')
        if len(states)!=2 or states[0]['scheduler'] is states[1]['scheduler']:
            raise ValueError('Two independently prepared request states are required')
        steps = states[0]['scheduler'].timesteps
        if len(steps)!=10 or not t.equal(steps,states[1]['scheduler'].timesteps):
            raise ValueError('Parallel lanes require identical ten-step schedules')
        before = [deepcopy(g.total) for g in self.graphs]
        cold = [not g.started for g in self.graphs]
        if cold[0] != cold[1]: raise RuntimeError('Partial graph setup cannot be reused')
        origin = backend.current()
        starts = ends = None
        predictions = None
        with ExitStack() as stack:
            for graph in self.graphs: stack.enter_context(graph.request())
            try:
                # Keep the graph signature device-identical from setup onward.
                lane_steps = [backend.to_lane_timesteps(steps) for _ in range(2)]
                first_parallel_step = 0
                if all(cold):
                    # Use the first setup replay's output for the real first step,
                    # rather than adding an unused replay/full generation warmup.
                    predictions = []
                    for graph,state,schedule in zip(self.graphs,states,lane_steps):
                        cancel()
                        predictions.append(self._setup_call(graph,dict(image=state['image'],
                            timestep=schedule[0],conditioning=state['conditioning'])))
                    with t.amp.autocast('cuda',enabled=False):
                        for state,pred in zip(states,predictions):
                            state['image'] = state['scheduler'].step(pred.float(),steps[0],
                                state['image'],generator=None).prev_sample
                    predictions = pred = None
                    first_parallel_step = 1
                ownership = self._independent_buffers()
                # Upload scalar schedules once. Per-replay CPU->GPU scalar copies
                # would introduce host waits and defeat asynchronous submissions.
                origin_event = backend.event(origin)
                for stream,state,schedule,graph in zip(self.streams,states,lane_steps,self.graphs):
                    backend.wait(stream,origin)
                    for tensor in (state['image'],state['conditioning'],schedule,
                                   *graph.buffers.values(),graph.output,graph.binding.frequency):
                        backend.record(tensor,stream)
                starts = [backend.event(stream) for stream in self.streams]
                for index in range(first_parallel_step,len(steps)):
                    cancel()
                    predictions = []
                    # Submit BOTH graph replays before scheduler work can cause a
                    # host wait. Each call still has exactly the B1 arithmetic.
                    for stream,graph,state,schedule in zip(self.streams,self.graphs,states,lane_steps):
                        with backend.context(stream):
                            predictions.append(graph.call(dict(image=state['image'],
                                timestep=schedule[index],conditioning=state['conditioning'])))
                    with t.amp.autocast('cuda',enabled=False):
                        for stream,state,pred in zip(self.streams,states,predictions):
                            with backend.context(stream):
                                state['image'] = state['scheduler'].step(pred.float(),steps[index],
                                    state['image'],generator=None).prev_sample
                    predictions = pred = None
                ends = [backend.event(stream) for stream in self.streams]
                for event in ends: backend.join(origin,event)
            finally:
                # Drain before releasing input states or graph pools, including
                # cancellation and failures. An active GPU kernel is never killed.
                drain_errors = []
                for stream in self.streams:
                    try: backend.sync(stream)
                    except BaseException as exc: drain_errors.append(exc)
                predictions = None
                if drain_errors: raise RuntimeError('Parallel lane stream drain failed') from drain_errors[0]
        intervals = backend.intervals(origin_event,starts,ends)
        delta = [{key:graph.total[key]-old[key] for key in
                  ('warmup_stage_forwards','capture_stage_forwards','replay_stage_forwards')}
                 for graph,old in zip(self.graphs,before)]
        return {'profile_id':PROFILE,'neural_batch_size':1,'logical_pair_size':2,
                'cuda_stream_ids':[backend.stream_id(s) for s in self.streams],
                'backend':backend.label,'lane_counters':delta,'ownership':ownership,
                'serial_setup_output_steps':first_parallel_step,
                'parallel_submission_steps':10-first_parallel_step,
                'scheduler_steps_per_request':10,'extra_full_generations':0,
                'lane_event_intervals_ms':intervals,
                'event_scope':'stream intervals include waits; not proof of kernel overlap or speedup',
                'qualification':'external_real_model_parity_required'}

    def close(self):
        if threading.get_ident()!=self.owner_thread: raise RuntimeError('Lane owner thread differs')
        if self.closed: return {**deepcopy(self.close_receipt),'already_closed':True}
        drain_errors=[]
        for stream in self.streams:
            try:self.backend.sync(stream)
            except BaseException as exc:drain_errors.append(exc)
        if drain_errors:
            # Keep graph storage alive if device work could still own it.
            raise RuntimeError('Cannot release undrained parallel lane graphs') from drain_errors[0]
        receipts=[]
        errors=[]
        try:
            for graph in self.graphs:
                try: receipts.append(graph.close())
                except BaseException as exc:
                    from .harness import capture_failure
                    errors.append(capture_failure(exc))
        finally:
            self.graphs.clear();self.streams.clear();self.closed=True
        self.close_receipt={'released':not errors and all(r['released'] for r in receipts),'lanes':receipts,'errors':errors}
        return deepcopy(self.close_receipt)


class LaneRuntime(_NativeRuntime):
    def __init__(self):
        super().__init__();self.parallel_lanes=None;self.parallel_cleanup_failure=None

    def lane_stage(self, states, *, cancel):
        if self.parallel_lanes is None:self.parallel_lanes=TwoLaneGraphs(self.model)
        try:return self.parallel_lanes.run(states,cancel=cancel)
        except BaseException as original:
            # Cancellation can keep the resident model, but any interrupted
            # lane graph state is retired before another pair is admitted.
            try: receipt=self.parallel_lanes.close()
            except BaseException as cleanup_error:
                from .harness import capture_failure
                self.parallel_cleanup_failure={'released':False,'errors':[capture_failure(cleanup_error)]}
                raise RuntimeError('Parallel lane cleanup failed; session cannot be reused') from original
            if not receipt['released']:
                self.parallel_cleanup_failure=deepcopy(receipt)
                raise RuntimeError('Parallel lane cleanup failed; session cannot be reused') from original
            self.parallel_lanes=None
            raise

    def close(self):
        lane_receipt = None
        lane_error = None
        try:
            if self.parallel_lanes is not None: lane_receipt=self.parallel_lanes.close()
        except BaseException as exc:
            from .harness import capture_failure
            lane_error=capture_failure(exc)
        finally:self.parallel_lanes=None
        result=super().close()
        result['parallel_lanes']=lane_receipt
        result['parallel_cleanup_error']=lane_error
        result['parallel_cleanup_failure']=deepcopy(self.parallel_cleanup_failure)
        result['passed']=result['passed'] and lane_error is None and self.parallel_cleanup_failure is None and (lane_receipt is None or lane_receipt['released'])
        return result


def lane_eligibility(runtime, recipes, **budgets):
    result=eligibility(runtime,recipes,**budgets)
    if recipes[0]['token_steps']!=8:
        raise BatchIneligible('Independent B1 graph lanes are initially restricted to eight token steps')
    result.update(profile_id=PROFILE, batch_size=1, logical_pair_size=2,
                  graph=True, graph_lanes=2, image_shape=[1,2,224,224],conditioning_shape=[1,5,14,14],
                  qualification='unqualified_two_B1_graph_lanes_candidate')
    return result


def run_lane_pair(runtime,*args,**kwargs):
    return run_pair(runtime,*args,stage_runner=runtime.lane_stage,**kwargs)
