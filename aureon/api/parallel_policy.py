"""Pure bounded planning; memory reservations are not an allocator hard limit."""

from collections.abc import Mapping
from dataclasses import dataclass
from .resource_policy import MemoryEstimate, WorkloadMemory, DeviceCapability
from .contracts import (
    INPUT_BYTES,
    OUTPUT_BYTES,
    ParallelFallback,
    Parallelism,
    ParallelPlan,
    QueueLimits,
    positive,
)

GIB = 1024**3
# Compatibility alias for callers using the default native profile.
LANE_RESERVATION = WorkloadMemory().estimate().workspace_bytes
FREE_RESERVE = 512 * 1024**2


@dataclass(frozen=True)
class ParallelPolicy:
    requested: Parallelism = 1
    allow_unqualified: bool = False
    fallback: ParallelFallback = "serial"
    workspace_bytes: int = 2 * GIB
    resident_bytes: int = 3 * GIB
    workload: WorkloadMemory = WorkloadMemory()
    pinned_bytes: int = 256 * 1024**2
    output_bytes: int = 256 * 1024**2
    free_reserve_bytes: int = FREE_RESERVE

    def __post_init__(self) -> None:
        if self.requested != "auto":
            positive(self.requested, "parallelism")
        if type(self.allow_unqualified) is not bool:
            raise TypeError("allow_unqualified_parallelism must be bool")
        if self.fallback not in ("serial", "error"):
            raise ValueError("parallel_fallback must be serial or error")
        positive(self.workspace_bytes, "parallel_workspace_bytes")
        positive(self.resident_bytes, "parallel_resident_bytes")
        if type(self.workload) is not WorkloadMemory:
            raise TypeError('Declare a WorkloadMemory estimate')
        positive(self.pinned_bytes, 'parallel_pinned_bytes')
        positive(self.output_bytes, 'parallel_output_bytes')
        positive(self.free_reserve_bytes, 'free_reserve_bytes')

    @property
    def estimate(self) -> MemoryEstimate:
        return self.workload.estimate()

    @property
    def reserve_bytes(self) -> int:
        return max(self.free_reserve_bytes, self.estimate.reserve_bytes)

    def bound_queue_limits(self, limits: QueueLimits) -> QueueLimits:
        """Bind retained outputs/queued inputs to host budgets before queue creation.

        The queue, not the lane session, retains these credits until consume/discard.
        Pinned estimates additionally reserve native staging copies at admission.
        """
        for key, value in limits.items():
            positive(value, key)
        result: QueueLimits = {'max_pending':limits['max_pending'], 'max_results':limits['max_results'],
                              'max_input_bytes':limits['max_input_bytes'], 'max_output_bytes':limits['max_output_bytes']}
        result['max_input_bytes'] = min(limits['max_input_bytes'], self.pinned_bytes)
        result['max_output_bytes'] = min(limits['max_output_bytes'], self.output_bytes)
        if result['max_input_bytes'] < INPUT_BYTES or result['max_output_bytes'] < OUTPUT_BYTES:
            raise MemoryError('Host budgets cannot retain one native input/result')
        return result

    def plan(
        self, limits: QueueLimits, evidence: Mapping[str, object] | None,
        *, capability: DeviceCapability | None = None,
    ) -> ParallelPlan:
        # Historical evidence is provenance, never an executable capability limit.
        # Runtime native eligibility still validates model/recipe/shape.
        for key, value in limits.items():
            positive(value, key)
        estimate = self.estimate
        if estimate.weights_bytes > self.resident_bytes:
            raise MemoryError('Declared model weights exceed the resident budget')
        credits = min(
            limits["max_pending"],
            limits["max_results"],
            limits["max_input_bytes"] // max(INPUT_BYTES, estimate.input_bytes),
            limits["max_output_bytes"] // max(OUTPUT_BYTES, estimate.output_bytes),
        )
        budget = min(self.workspace_bytes // estimate.workspace_bytes,
                     self.pinned_bytes // max(1, estimate.pinned_bytes),
                     self.output_bytes // max(1, estimate.output_bytes))
        desired = credits if self.requested == 'auto' else self.requested
        if capability is not None:
            available = max(0, capability.free_bytes - max(estimate.weights_bytes, self.resident_bytes) - self.reserve_bytes)
            budget = min(budget, available // estimate.workspace_bytes)
        planned = min(desired, credits, budget)
        reasons = []
        if planned < desired:
            if credits < desired:
                reasons.append("input_result_credits")
            if budget < desired:
                reasons.append("estimated_memory_budget")
        if capability is not None and capability.lane_rejection(dtype=self.workload.inputs.dtype):
            if self.fallback == 'error':
                raise ValueError('Capability rejected and serial fallback is disabled: ' + str(capability.lane_rejection(dtype=self.workload.inputs.dtype)))
            planned = min(planned, 1)
            reasons.append(str(capability.lane_rejection(dtype=self.workload.inputs.dtype)))
        if planned < 1:
            raise MemoryError('Estimated resources cannot admit one lane; select explicit reference execution or increase budgets')
        return {
            "requested": self.requested,
            "planned": planned,
            "admitted": 0,
            "actual": 0,
            "neural_batch": 1,
            "desired_lanes": desired,
            "queue_credit_limit": credits,
            "workspace_lane_limit": budget,
            "fallback_reason": ",".join(reasons) or None,
            "qualification": "runtime_admission_not_checked",
            "workspace_bytes": self.workspace_bytes,
            "resident_bytes": self.resident_bytes,
            "per_lane_reservation_bytes": estimate.workspace_bytes,
            "free_reserve_bytes": self.reserve_bytes,
            "memory_scope": "conservative planning reservation; not total VRAM guarantee or hard allocator cap",
        }
