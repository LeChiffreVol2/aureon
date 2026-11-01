"""Explicit-model placement with bounded resource checks; no Ray or model imports.

This planner does not reserve resources or dispatch jobs. A service must atomically
revalidate the selected worker generation and acquire its own fenced lease.
"""
from __future__ import annotations
import typing as t
from dataclasses import dataclass
import math


def _nonnegative(value: object, name: str, *, integer: bool=False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a nonnegative finite number")
    if not math.isfinite(value) or value < 0 or (integer and type(value) is not int):
        raise ValueError(name + " must be a nonnegative finite number")


@dataclass(frozen=True)
class ExecutionKey:
    """Exact execution compatibility, including immutable artifact identities."""
    model_id: str
    model_identity: str
    recipe_identity: str
    task: str
    input_modality: str
    output_modality: str
    precision_profile: str

    def __post_init__(self) -> None:
        for value in vars(self).values():
            if not isinstance(value, str) or not value.strip():
                raise ValueError("ExecutionKey fields must be explicit nonempty strings")


@dataclass(frozen=True)
class PlacementRequest:
    key: ExecutionKey
    transient_vram_bytes: int
    cold_resident_bytes: int
    host_bytes: int
    output_bytes: int
    deadline_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, ExecutionKey):
            raise TypeError("key requires ExecutionKey")
        for name in ("transient_vram_bytes", "cold_resident_bytes", "host_bytes", "output_bytes"):
            _nonnegative(getattr(self, name), name, integer=True)
        if self.deadline_seconds is not None:
            _nonnegative(self.deadline_seconds, "deadline_seconds")


@dataclass(frozen=True)
class WorkerSnapshot:
    worker_id: str
    generation: str
    device_id: str
    key: ExecutionKey
    resident: bool
    ready: bool
    free_vram_bytes: int
    free_host_bytes: int
    free_output_bytes: int
    queue_seconds: float
    execution_seconds: float
    delivery_seconds: float
    load_seconds: float
    measured_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, ExecutionKey):
            raise TypeError("key requires ExecutionKey")
        for name in ("worker_id", "generation", "device_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(name + " is required")
        if type(self.resident) is not bool or type(self.ready) is not bool:
            raise ValueError("resident and ready require booleans")
        for name in ("free_vram_bytes", "free_host_bytes", "free_output_bytes"):
            _nonnegative(getattr(self, name), name, integer=True)
        for name in ("queue_seconds", "execution_seconds", "delivery_seconds", "load_seconds", "measured_at"):
            _nonnegative(getattr(self, name), name)


def plan_placement(request: PlacementRequest, workers: t.Iterable[WorkerSnapshot], *, now: float, max_snapshot_age: float=5.0) -> t.Dict[str, t.Any]:
    """Choose lowest estimated completion time among exact eligible replicas.

    Free capacities must already deduct reservations by other requests. Memory
    estimates include graphs/tokenizers; there is no implicit eviction or model
    substitution. Times use one caller-supplied monotonic clock domain. A stale
    snapshot, unknown capability or impossible deadline is a rejected route.
    """
    if not isinstance(request, PlacementRequest):
        raise TypeError("request requires PlacementRequest")
    _nonnegative(now, "now")
    _nonnegative(max_snapshot_age, "max_snapshot_age")
    candidates, rejected, seen = [], [], set()
    for worker in workers:
        if not isinstance(worker, WorkerSnapshot):
            raise TypeError("workers requires WorkerSnapshot values")
        if worker.worker_id in seen:
            raise ValueError("Duplicate worker_id")
        seen.add(worker.worker_id)
        reasons = []
        if not worker.ready:
            reasons.append("worker_not_ready")
        if worker.key != request.key:
            reasons.append("execution_contract_mismatch")
        if not 0 <= now - worker.measured_at <= max_snapshot_age:
            reasons.append("stale_or_future_snapshot")
        vram = request.transient_vram_bytes + (0 if worker.resident else request.cold_resident_bytes)
        if worker.free_vram_bytes < vram:
            reasons.append("vram_capacity")
        if worker.free_host_bytes < request.host_bytes:
            reasons.append("host_capacity")
        if worker.free_output_bytes < request.output_bytes:
            reasons.append("output_capacity")
        seconds = (worker.queue_seconds + worker.execution_seconds + worker.delivery_seconds
                   + (0 if worker.resident else worker.load_seconds))
        if not math.isfinite(seconds):
            reasons.append("invalid_completion_estimate")
        if request.deadline_seconds is not None and seconds > request.deadline_seconds:
            reasons.append("estimated_deadline_miss")
        if reasons:
            rejected.append({"worker_id": worker.worker_id, "reasons": reasons})
        else:
            candidates.append((seconds, not worker.resident, worker.worker_id, worker, vram))
    candidates.sort(key=lambda x: x[:3])
    selected = None
    if candidates:
        seconds, _, _, worker, vram = candidates[0]
        selected = {"worker_id": worker.worker_id, "worker_generation": worker.generation,
                    "device_id": worker.device_id, "resident": worker.resident,
                    "estimated_completion_seconds": seconds, "required_vram_bytes": vram,
                    "required_host_bytes": request.host_bytes, "required_output_bytes": request.output_bytes,
                    "snapshot_measured_at": worker.measured_at}
    return {"schema": "aureon.placement_plan.v1", "execution_key": dict(vars(request.key)),
            "selected": selected, "eligible_count": len(candidates), "rejected": rejected,
            "policy": "exact_contract_then_estimated_completion_then_resident_then_worker_id",
            "reservation_acquired": False, "dispatched": False,
            "reason": "eligible_worker_selected" if selected else "no_eligible_worker"}
