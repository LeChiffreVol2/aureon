"""Bounded request-owned B1 CUDA graph lanes; qualification is external.

Weights are immutable/shared. Shapes, timestep embedding and numerical decoder
binding reuse the accepted B1 graph implementation. Graph pools, inputs, outputs,
frequency buffers, stream dependencies and scheduler state are independent.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING, cast
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import ModuleType
from ._types import TerraMindModel, Metadata, TensorMap, PreparedState, Stream, Event, LaneBackend, DecoderBinding, GraphBinding, Graph
if TYPE_CHECKING:
    from torch import Tensor
    from .graph_runtime import FixedTensorGraph
from contextlib import ExitStack
from copy import deepcopy
import threading

from .session import _NativeRuntime
from .selected_core import SelectedPrimitives
from .denoiser_batch import BatchIneligible, native_group_eligibility, run_prepared_group, _unchanged_rng

PROFILE = 'experimental_s1grd224_fp32_b1_graph_lanes_v2'


class CudaLaneBackend:
    label = 'cuda_streams'
    def __init__(self, torch: ModuleType) -> None: self.torch = torch
    def create_stream(self) -> Stream: return cast(Stream, self.torch.cuda.Stream(device='cuda:0'))
    def current(self) -> Stream: return cast(Stream, self.torch.cuda.current_stream('cuda:0'))
    def context(self, stream: Stream) -> AbstractContextManager[None]: return cast("AbstractContextManager[None]", self.torch.cuda.stream(stream))
    def wait(self, stream: Stream, origin: Stream) -> None: stream.wait_stream(origin)
    def record(self, tensor: Tensor, stream: Stream) -> None: tensor.record_stream(cast("Any", stream))
    def sync(self, stream: Stream) -> None: stream.synchronize()
    def to_lane_timesteps(self, steps: Tensor) -> Tensor: return steps.to(device='cuda:0')
    def event(self, stream: Stream) -> Event:
        event: Event = self.torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event
    def join(self, origin: Stream, event: Event) -> None: origin.wait_event(event)
    def stream_id(self, stream: Stream) -> int: return int(stream.cuda_stream)
    def intervals(self, origin_event: Event, starts: list[Event], ends: list[Event]) -> list[dict[str, float]]:
        return [{'start_ms': origin_event.elapsed_time(a), 'end_ms': origin_event.elapsed_time(b)}
                for a,b in zip(starts,ends)]


class LaneGraphs:
    """One bounded graph/pool and stream per lane, reused only on the owner thread."""
    def __init__(self, model: TerraMindModel, *, max_lanes: int, backend: LaneBackend | None=None, binding_factory: Callable[[TerraMindModel], DecoderBinding] | None=None, graph_factory: Callable[[GraphBinding], FixedTensorGraph] | None=None, numerical_core: SelectedPrimitives | None=None) -> None:
        if type(max_lanes) is not int or max_lanes<1:raise ValueError('max_lanes must be a positive integer')
        import torch
        from .denoiser_binding import S1GRDDecoderBinding
        from .graph_runtime import FixedTensorGraph
        self.torch = torch
        self.owner_thread = threading.get_ident()
        self.backend = backend if backend is not None else CudaLaneBackend(torch)
        self.numerical_core = numerical_core
        self.binding_factory = binding_factory or (numerical_core.binding_factory if numerical_core is not None else S1GRDDecoderBinding)
        self.graph_factory = graph_factory or (numerical_core.tensor_graph_factory if numerical_core is not None else FixedTensorGraph)
        self.model: TerraMindModel | None=model;self.max_lanes=max_lanes
        self.graphs: list[FixedTensorGraph]=[];self.streams: list[Stream]=[]
        self.closed = False
        self.failed = False
        self.close_receipt: Metadata | None = None

    def _ensure(self,count: int) -> None:
        if not 1<=count<=self.max_lanes:raise ValueError('Group exceeds admitted lane capacity')
        # Grow only to actual group size. A large requested limit allocates nothing.
        while len(self.graphs)<count:
            graph=self.graph_factory(self.binding_factory(cast(TerraMindModel, self.model)))
            self.graphs.append(graph)
            self.streams.append(self.backend.create_stream())
        if len({self.backend.stream_id(s) for s in self.streams})!=len(self.streams):
            raise RuntimeError('Parallel lanes require distinct owned streams')

    def _setup_call(self, graph: FixedTensorGraph, inputs: TensorMap) -> Tensor:
        # Only setup calls execute Python embedding code. Replays use their own
        # captured buffers; no shared module/global patch remains during overlap.
        from .denoiser_graph import _PATCH_LOCK
        patch_lock = self.numerical_core.patch_lock if self.numerical_core is not None else _PATCH_LOCK
        if not patch_lock.acquire(blocking=False):
            raise RuntimeError('Decoder embedding binding is already owned')
        namespace = None
        try:
            namespace = cast(DecoderBinding, graph.binding).namespace()
            original = namespace['timestep_embedding']
            namespace['timestep_embedding'] = cast(DecoderBinding, graph.binding).embedding
            return graph.call(inputs)
        finally:
            if namespace is not None: namespace['timestep_embedding'] = original
            patch_lock.release()

    def _independent_buffers(self) -> Metadata:
        addresses = []
        pools = []
        for graph in self.graphs:
            tensors = [*graph.buffers.values(), graph.output, cast(DecoderBinding, graph.binding).frequency]
            addresses.append({value.data_ptr() for value in tensors if value is not None})
            pools.append(tuple(cast(Graph, graph.graph).pool()))
        seen: set[int]=set()
        for address in addresses:
            if seen & address:raise RuntimeError('Lane graph storage/pools must not alias')
            seen.update(address)
        if len(set(pools))!=len(pools):raise RuntimeError('Lane graph storage/pools must not alias')
        return {'disjoint_buffers':True, 'distinct_graph_pools':True,
                'graph_pools':pools, 'owned_buffer_counts':[len(a) for a in addresses]}

    def run(self,states: list[PreparedState],*,cancel: Callable[[], None]) -> Metadata:
        if self.failed:raise RuntimeError('Failed lane group must be closed')
        try:return self._run(states,cancel=cancel)
        except BaseException:
            self.failed=True
            raise

    def _run(self, states: list[PreparedState], *, cancel: Callable[[], None]) -> Metadata:
        t, backend = self.torch, self.backend
        if threading.get_ident()!=self.owner_thread: raise RuntimeError('Lane owner thread differs')
        if self.closed: raise RuntimeError('Parallel lane graphs are closed')
        if not states or len(states)>self.max_lanes or len({id(s['scheduler']) for s in states})!=len(states):
            raise ValueError('Independently prepared request states within capacity are required')
        steps = states[0]['scheduler'].timesteps
        if len(steps)!=10 or any(not t.equal(steps,s['scheduler'].timesteps) for s in states):
            raise ValueError('Parallel lanes require identical ten-step schedules')
        self._ensure(len(states))
        graphs=self.graphs[:len(states)];streams=self.streams[:len(states)]
        before = [deepcopy(g.total) for g in graphs]
        cold = [not g.started for g in graphs]
        origin = backend.current()
        starts = ends = None
        predictions: list[Tensor] | None = None
        pred: Tensor | None
        with ExitStack() as stack:
            for graph in graphs: stack.enter_context(graph.request())
            try:
                # Keep the graph signature device-identical from setup onward.
                lane_steps = [backend.to_lane_timesteps(steps) for _ in states]
                first_parallel_step = 0
                if any(cold):
                    # Use the first setup replay's output for the real first step,
                    # rather than adding an unused replay/full generation warmup.
                    predictions = []
                    for graph,state,schedule,is_cold in zip(graphs,states,lane_steps,cold):
                        cancel()
                        inputs=dict(image=state['image'],timestep=schedule[0],conditioning=state['conditioning'])
                        predictions.append(self._setup_call(graph,inputs) if is_cold else graph.call(inputs))
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
                for stream,state,schedule,graph in zip(streams,states,lane_steps,graphs):
                    backend.wait(stream,origin)
                    for tensor in (state['image'],state['conditioning'],schedule,
                                   *graph.buffers.values(),graph.output,cast(DecoderBinding, graph.binding).frequency):
                        backend.record(cast("Tensor", tensor),stream)
                starts = [backend.event(stream) for stream in streams]
                for index in range(first_parallel_step,len(steps)):
                    cancel()
                    predictions = []
                    # Submit all graph replays before scheduler work can cause a
                    # host wait. Each call still has exactly the B1 arithmetic.
                    for stream,graph,state,schedule in zip(streams,graphs,states,lane_steps):
                        with backend.context(stream):
                            predictions.append(graph.call(dict(image=state['image'],
                                timestep=schedule[index],conditioning=state['conditioning'])))
                    with t.amp.autocast('cuda',enabled=False):
                        for stream,state,pred in zip(streams,states,predictions):
                            with backend.context(stream):
                                state['image'] = state['scheduler'].step(pred.float(),steps[index],
                                    state['image'],generator=None).prev_sample
                    predictions = pred = None
                ends = [backend.event(stream) for stream in streams]
                for event in ends: backend.join(origin,event)
            finally:
                # Drain before releasing input states or graph pools, including
                # cancellation and failures. An active GPU kernel is never killed.
                drain_errors = []
                for stream in streams:
                    try: backend.sync(stream)
                    except BaseException as exc: drain_errors.append(exc)
                predictions = None
                if drain_errors: raise RuntimeError('Parallel lane stream drain failed') from drain_errors[0]
        intervals = backend.intervals(origin_event,starts,ends)
        delta = [{key:graph.total[key]-old[key] for key in
                  ('warmup_stage_forwards','capture_stage_forwards','replay_stage_forwards')}
                 for graph,old in zip(graphs,before)]
        return {'profile_id':PROFILE,'neural_batch_size':1,'logical_group_size':len(states),
                'logical_pair_size':len(states),'initialized_lanes':len(self.graphs),
                'newly_captured_lanes':sum(cold),
                'cuda_stream_ids':[backend.stream_id(s) for s in streams],
                'backend':backend.label,'lane_counters':delta,'ownership':ownership,
                'serial_setup_output_steps':first_parallel_step,
                'parallel_submission_steps':10-first_parallel_step,
                'scheduler_steps_per_request':10,'extra_full_generations':0,
                'lane_event_intervals_ms':intervals,
                'event_scope':'stream intervals include waits; not proof of kernel overlap or speedup',
                'qualification':'external_real_model_parity_required'}

    def close(self) -> Metadata:
        if threading.get_ident()!=self.owner_thread: raise RuntimeError('Lane owner thread differs')
        if self.closed: return {**deepcopy(cast(Metadata, self.close_receipt)),'already_closed':True}
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
            self.graphs.clear();self.streams.clear();self.model=None;self.closed=True
        self.close_receipt={'released':not errors and all(r['released'] for r in receipts),'lanes':receipts,'errors':errors}
        return deepcopy(cast(Metadata, self.close_receipt))

    def prepare_pipeline(self, prepares: list[Callable[[], PreparedState]], *, cancel: Callable[[], None]) -> tuple[list[PreparedState], Metadata, Metadata]:
        """Overlap warmed immutable graph replays with serial native preparation.

        Preparation still runs only on the owner thread. A previously captured
        graph invokes no mutable Python sampler/decoder hooks during replay.
        Every request retains its own RNG capsule, scheduler, graph and stream.
        Cold capture is deliberately excluded because capture synchronizes CUDA.
        """
        if threading.get_ident()!=self.owner_thread:raise RuntimeError('Lane owner thread differs')
        if self.closed or self.failed:raise RuntimeError('Lane graphs are unavailable')
        count=len(prepares)
        if not 1<=count<=len(self.graphs) or any(not g.started for g in self.graphs[:count]):
            raise ValueError('Preparation pipeline requires already captured lanes')
        t,backend=self.torch,self.backend
        graphs=self.graphs[:count];streams=self.streams[:count]
        before=[deepcopy(g.total) for g in graphs]
        origin=backend.current();origin_event=backend.event(origin)
        states: list[PreparedState]=[];starts: list[Event]=[];ends: list[Event]=[]
        schedules: list[Tensor]=[];rngs: list[Metadata]=[]
        pred: Tensor | None=None
        try:
            with ExitStack() as stack:
                for graph in graphs:stack.enter_context(graph.request())
                for prepare,graph,stream in zip(prepares,graphs,streams):
                    cancel()
                    # Never run two native preparations concurrently. Earlier
                    # replay work may be in flight on its independent stream.
                    state=prepare();states.append(state)
                    if len({id(s['scheduler']) for s in states})!=len(states):
                        raise ValueError('Preparation pipeline requires independent schedulers')
                    steps=state['scheduler'].timesteps
                    if len(steps)!=10 or not t.equal(steps,states[0]['scheduler'].timesteps):
                        raise ValueError('Preparation pipeline requires identical ten-step schedules')
                    with _unchanged_rng(t,(0,),capsule=self.numerical_core.rng_module if self.numerical_core is not None else None) as rng:
                        schedule=backend.to_lane_timesteps(steps);schedules.append(schedule)
                        backend.wait(stream,origin)
                        for tensor in (state['image'],state['conditioning'],schedule,
                                       *graph.buffers.values(),graph.output,cast(DecoderBinding,graph.binding).frequency):
                            backend.record(cast('Tensor',tensor),stream)
                        starts.append(backend.event(stream))
                        with backend.context(stream),t.amp.autocast('cuda',enabled=False):
                            for index in range(10):
                                cancel()
                                pred=graph.call(dict(image=state['image'],timestep=schedule[index],conditioning=state['conditioning']))
                                state['image']=state['scheduler'].step(pred.float(),steps[index],state['image'],generator=None).prev_sample
                                pred=None
                        ends.append(backend.event(stream))
                    rngs.append(rng)
                for event in ends:backend.join(origin,event)
        except BaseException:
            self.failed=True
            raise
        finally:
            # Drain before releasing any state/schedule on every error path,
            # including an exception preparing a later request.
            errors=[]
            for stream in streams:
                try:backend.sync(stream)
                except BaseException as exc:errors.append(exc)
            pred=None
            if errors:
                self.failed=True
                raise RuntimeError('Preparation pipeline stream drain failed') from errors[0]
        if any(r['before_sha256']!=rngs[0]['before_sha256'] or not r['unchanged'] for r in rngs):
            self.failed=True
            raise RuntimeError('Preparation pipeline did not preserve caller RNG')
        rng={'before_sha256':rngs[0]['before_sha256'],'after_sha256':rngs[-1]['after_sha256'],'unchanged':True}
        delta=[{key:g.total[key]-old[key] for key in ('warmup_stage_forwards','capture_stage_forwards','replay_stage_forwards')} for g,old in zip(graphs,before)]
        stage={'profile_id':PROFILE,'neural_batch_size':1,'logical_group_size':count,'logical_pair_size':count,
               'initialized_lanes':len(self.graphs),'newly_captured_lanes':0,
               'cuda_stream_ids':[backend.stream_id(s) for s in streams],'backend':backend.label,
               'lane_counters':delta,'ownership':self._independent_buffers(),
               'serial_setup_output_steps':0,'parallel_submission_steps':10,
               'scheduler_steps_per_request':10,'extra_full_generations':0,
               'preparation_schedule':'serial_prepare_overlap_prior_lane_replays',
               'lane_event_intervals_ms':backend.intervals(origin_event,starts,ends),
               'event_scope':'stream intervals include waits; not proof of kernel overlap or speedup',
               'qualification':'external_real_model_parity_required'}
        return states,stage,rng


class TwoLaneGraphs(LaneGraphs):
    """Backward-compatible constructor; execution is the generic bounded core."""
    def __init__(self,model: TerraMindModel,**kwargs: Any) -> None:
        super().__init__(model,max_lanes=2,**kwargs)
        self._ensure(2)


class LaneRuntime(_NativeRuntime):
    parallel_residency_bytes: int
    def __init__(self,max_lanes: int=2) -> None:
        super().__init__();self.parallel_lanes: LaneGraphs | None=None;self.parallel_cleanup_failure: Metadata | None=None
        self.max_lanes=max_lanes

    def lane_stage(self, states: list[PreparedState], *, cancel: Callable[[], None]) -> Metadata:
        if self.parallel_lanes is None:self.parallel_lanes=LaneGraphs(cast(TerraMindModel, self.model),max_lanes=self.max_lanes,numerical_core=self.numerical_core)
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

    def prepare_stage(self, prepares: list[Callable[[], PreparedState]], *, cancel: Callable[[], None]) -> tuple[list[PreparedState], Metadata, Metadata]:
        lanes=self.parallel_lanes
        if lanes is not None and len(lanes.graphs)>=len(prepares) and all(g.started for g in lanes.graphs[:len(prepares)]):
            return lanes.prepare_pipeline(prepares,cancel=cancel)
        states=[prepare() for prepare in prepares]
        with _unchanged_rng(cast(ModuleType,self.torch),(0,),capsule=self.numerical_core.rng_module if self.numerical_core is not None else None) as rng:
            stage=self.lane_stage(states,cancel=cancel)
        stage['preparation_schedule']='serial_cold_capture'
        return states,stage,rng

    def close(self) -> Metadata:
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


def lane_eligibility(runtime: _NativeRuntime, recipes: list[Metadata], **budgets: int) -> Metadata:
    result=native_group_eligibility(runtime,recipes,**budgets)
    if recipes[0]['token_steps']!=8:
        raise BatchIneligible('Independent B1 graph lanes are initially restricted to eight token steps')
    result.update(profile_id=PROFILE, batch_size=1, logical_group_size=len(recipes),logical_pair_size=len(recipes),
                  graph=True, graph_lanes=len(recipes), image_shape=[1,2,224,224],conditioning_shape=[1,5,14,14],
                  qualification='generic_lane_runtime_requires_scope_evidence')
    return result


def run_lane_pair(runtime: LaneRuntime,*args: Any,**kwargs: Any) -> list[Metadata]:
    kwargs.pop('max_workspace_bytes',None);kwargs.pop('min_free_cuda_bytes',None)
    return run_prepared_group(runtime,*args,stage_runner=runtime.lane_stage,prepare_runner=runtime.prepare_stage,**kwargs)
