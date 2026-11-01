"""Explicit parallel facade admission; source qualification is recorded separately."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING, cast
from types import ModuleType
from collections.abc import Sequence
from .contracts import ParallelPlan, QueueLimits
from .generation_bridges import GenerationJob
from ..terramind_native._types import Metadata, PathInput
if TYPE_CHECKING:
    from ..terramind_native.denoiser_lanes import LaneRuntime, LaneGraphs
from copy import deepcopy

from .group_generation import NativeGroupSession
from .parallel_policy import ParallelPolicy
from .resource_policy import DeviceCapability, WorkloadMemory
from ..terramind_native.denoiser_batch import BatchIneligible

# Bounded source qualification; the separately built wheel needs its own installed
# acceptance. This does not qualify a different GPU or establish a speed advantage.
QUALIFICATION = {
    'status':'qualified_explicit_opt_in',
    'lanes':2,
    'evidence_id':'gpu-parallel-20261028-r1/lanes-r1',
    'hardware':'NVIDIA GeForce RTX 3070', 'VRAM_GiB':8, 'platform':'Linux/WSL',
    'model':'terramind-v1-base', 'input':'all-valid S2L2A224', 'output':'synthetic S1GRD VV/VH',
    'precision':'highest_fp32', 'token_steps':8, 'diffusion_steps':10,
    'temperature':1.0,'top_p':0.8,'top_k':0,
    'tested_seed_pairs':[[17,101],[101,17]], 'logical_outputs':4,
    'raw_exact':True,'tokens_exact':True,'rng_exact':True,'cleanup_cuda_zero':True,
    'source_profile':'experimental_s1grd224_fp32_two_b1_graph_lanes_v1',
    'source_manifest_sha256':'4bce8b55891fe05ac43bd7a5122407f1f94760ac1ead76529224974442cbba6c',
    'independent_review_checks':418,
    'independent_review_sha256':'879de5b8dac7c5ce971f0c8c62ad111c43c9937355856affa8162af394c607db',
    'GPU_result_sha256':'ce7bd640ec442d011f830adb5cfa8bae0bb2932063c529dee229f0e075d68e40',
    'performance_measured':False,'physical_multi_GPU_measured':False,
    'installed_artifact_acceptance':'separate artifact-bound receipt required',
}


class PairAdmissionRejected(BatchIneligible):
    """A pair was rejected before either attempt or model preparation began."""


def qualification(model_id: str='terramind-v1-base') -> Metadata:
    from ..terramind_native.families import GENERATION_MODELS
    if model_id not in GENERATION_MODELS:
        raise ValueError('Expected TerraMind generation model_id')
    return deepcopy(QUALIFICATION) if model_id == 'terramind-v1-base' else {}


class LaneSession(NativeGroupSession):
    def __init__(self,*args: Any,policy: ParallelPolicy,plan: ParallelPlan,**kwargs: Any) -> None:
        from ..terramind_native.loader import config_model_id
        from ..terramind_native.common import read_json
        if config_model_id(read_json(args[0] if args else kwargs['config'])) != 'terramind-v1-base':
            raise ValueError('LaneSession supports only the base generation topology')
        self.policy=policy;self.plan=deepcopy(plan);self.last_selection: Metadata | None=None
        super().__init__(*args,**kwargs)

    def metadata(self) -> Metadata:
        value=super().metadata()
        value['parallelism']={'plan':deepcopy(self.plan),'last_selection':deepcopy(self.last_selection),
                             'qualification':qualification()}
        return value

    def generate_pair(self,jobs: Sequence[GenerationJob],**kwargs: Any) -> list[Metadata]:
        if not isinstance(jobs,(tuple,list)) or len(jobs)!=2:raise ValueError('Legacy generate_pair requires two jobs')
        return self.generate_group(jobs,**kwargs)

    def _execute_group(self,*args: Any,**kwargs: Any) -> list[Metadata]:
        results=super()._execute_group(*args,**kwargs)
        if cast(ModuleType, cast("LaneRuntime", self._runtime).torch).cuda.memory_reserved()>self.policy.resident_bytes+self.policy.workspace_bytes+self.policy.reserve_bytes:
            raise RuntimeError('Observed CUDA reservation exceeded configured group envelope; no retry')
        selection=deepcopy(cast(Metadata, self.last_selection))
        selection['actual']=len(results)
        selection['group_actual']=len(results)
        self.last_selection=deepcopy(selection)
        for result in results:
            result['parallel_selection']=deepcopy(selection)
            result['provenance'].update(schema='aureon.parallel_generation_job.v1',
                parallel_selection=deepcopy(selection),source_qualification=selection['qualification'],
                numerical_fidelity='Request-local highest FP32; exact parity evidence only for recorded qualification scope')
        return results

    def _group_admission(self, recipes: list[Metadata]) -> Metadata:
        count=len(recipes)
        cuda=cast(ModuleType, cast("LaneRuntime", self._runtime).torch).cuda
        free,total=cuda.mem_get_info()
        capability=DeviceCapability('cuda', int(total), int(free),
            tuple(cuda.get_device_capability(0)), graph=hasattr(cuda, 'CUDAGraph'),
            visible_devices=int(cuda.device_count()))
        rejection=capability.lane_rejection(dtype=self.policy.workload.inputs.dtype)
        if rejection:
            raise PairAdmissionRejected(rejection + '; use the serial reference path')
        estimate=self.policy.estimate
        per_lane=estimate.workspace_bytes
        cached=len(cast("LaneGraphs", cast("LaneRuntime", self._runtime).parallel_lanes).graphs) if cast("LaneRuntime", self._runtime).parallel_lanes is not None else 0
        reserved_lanes=max(cached,count)
        if reserved_lanes*per_lane>self.policy.workspace_bytes:
            raise PairAdmissionRejected('Group plus retained lane pools exceeds workspace reservation budget')
        if count*estimate.pinned_bytes>self.policy.pinned_bytes or count*estimate.output_bytes>self.policy.output_bytes:
            raise PairAdmissionRejected('Group exceeds pinned/output lifecycle budgets')
        if not hasattr(cast("LaneRuntime", self._runtime),'parallel_residency_bytes'):
            cast("LaneRuntime", self._runtime).parallel_residency_bytes=max(cuda.memory_allocated(),cuda.memory_reserved())
        residency=cast("LaneRuntime", self._runtime).parallel_residency_bytes
        if residency>self.policy.resident_bytes:
            raise PairAdmissionRejected('Measured pre-lane residency exceeds configured resident budget')
        needed=max(0,count-cached)*per_lane+self.policy.reserve_bytes
        if free<needed:
            raise PairAdmissionRejected('Insufficient free CUDA memory for new lane reservations and reserve')
        try:
            admission=super()._group_admission(recipes)
        except BatchIneligible as exc:
            raise PairAdmissionRejected(str(exc)) from None
        qualification_record: Metadata={'status':'executable_not_qualified',
            'historical_reference':qualification(),
            'reason':'Capability admission does not bind an installed artifact to historical evidence',
            'performance_measured':False}
        self.last_selection={**deepcopy(self.plan),'admitted':count,'actual':0,'group_actual':0,
            'qualification':qualification_record,'selected':'graph_lanes','requested':self.policy.requested,
            'fallback_reason':self.plan['fallback_reason'],'neural_batch_size':1,
            'retained_lane_capacity':reserved_lanes,'new_lane_reservation_bytes':max(0,count-cached)*per_lane,
            'resident_observed_bytes':residency,'free_cuda_bytes':free,'total_cuda_bytes':total,
            'capability':{'backend':capability.backend,'compute_capability':capability.compute_capability,
                          'graph':capability.graph,'visible_devices':capability.visible_devices},
            'memory_estimate':{'per_lane_bytes':per_lane,'activation_bytes':estimate.activation_bytes,
                'state_bytes':estimate.state_bytes,'graph_bytes':estimate.graph_bytes,
                'pinned_bytes':count*estimate.pinned_bytes,'output_bytes':count*estimate.output_bytes,
                'scope':'Declared estimates plus observed residency, not an allocator cap'},
            'lifecycle':'Graph pools remain charged until session close; group credits until synchronized return'}
        admission['parallel_selection']=deepcopy(self.last_selection)
        return admission


def open_lane_session(*, config: PathInput, work: PathInput, policy: ParallelPolicy | None=None, plan: ParallelPlan | None=None, limits: QueueLimits | None=None) -> LaneSession:
    if policy is None:policy=ParallelPolicy(requested=2)
    native=WorkloadMemory()
    if (policy.workload.inputs != native.inputs or policy.workload.outputs != native.outputs
            or policy.workload.latent != native.latent or policy.workload.tokens != native.tokens
            or policy.workload.batch_size != 1 or policy.workload.graph_copies < 1):
        raise ValueError('Native graph lanes require fixed S2L2A224/S1GRD FP32 B1 layouts; estimates do not implement new shapes')
    if limits is None:
        from .generation_queue import INPUT_BYTES,OUTPUT_BYTES
        limits=dict(max_pending=2,max_results=2,max_input_bytes=2*INPUT_BYTES,max_output_bytes=2*OUTPUT_BYTES)
    from .profiles import require_profile
    from ..terramind_native.offline import inspect_config
    require_profile('generation')
    inspected=inspect_config(config)
    if inspected['model_id'] != 'terramind-v1-base':
        raise ValueError('LaneSession supports only the base generation topology')
    if plan is None:plan=policy.plan(limits,qualification(inspected['model_id']))
    return LaneSession(config,work,execute_model=True,allow_unqualified_batch=True,policy=policy,plan=plan,
        max_lanes=plan['planned'],max_input_bytes=limits['max_input_bytes'],max_output_bytes=limits['max_output_bytes'],
        max_workspace_bytes=policy.workspace_bytes,min_free_cuda_bytes=policy.reserve_bytes)
