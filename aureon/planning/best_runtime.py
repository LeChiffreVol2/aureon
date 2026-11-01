"""Pure cost and process-replica planning for the preserved TerraMind core.

No device discovery, model imports, process launch or global environment changes.
Each replica owns a complete model/graph allocation and sees its assigned physical
device as cuda:0. This preserves the core's device contract without changing it.
Declared memory budgets do not prove concurrency efficiency or reserve real VRAM.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
import re
from typing import Literal

BestProfile = Literal["best_fp32", "reference_fp32"]


def _integer(value: int, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(name + " must be a " + ("positive" if positive else "nonnegative") + " integer")


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be a finite nonnegative number")
    return float(value)


@dataclass(frozen=True)
class BestDeviceBudget:
    """A caller-declared device snapshot, never an implicit device query.

    Prefer a physical GPU/MIG UUID over an ordinal. A host name identifies an
    independent placement domain; this object does not establish remote access.
    """
    device: str
    total_bytes: int
    free_bytes: int
    host: str = "local"
    cuda_graph: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/_-]+)", self.device) is None:
            raise ValueError("device must be one CUDA ordinal, GPU UUID or MIG UUID")
        if not isinstance(self.host, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,252}", self.host) is None:
            raise ValueError("host must be an explicit safe placement label")
        _integer(self.total_bytes, "total_bytes", positive=True)
        _integer(self.free_bytes, "free_bytes")
        if self.free_bytes > self.total_bytes or type(self.cuda_graph) is not bool:
            raise ValueError("Invalid memory or CUDA Graph capability declaration")


@dataclass(frozen=True)
class BestReplicaPlacement:
    host: str
    device: str
    replicas: int
    capacity: int
    per_replica_bytes: int
    reserved_device_bytes: int
    unassigned_bytes: int

    def worker_environment(self) -> dict[str, str]:
        """Apply in a fresh worker before importing Torch, never in a live owner."""
        return {"CUDA_VISIBLE_DEVICES": self.device}


@dataclass(frozen=True)
class BestCapacityPlan:
    profile: BestProfile
    placements: tuple[BestReplicaPlacement, ...]
    qualification: str = "declared_capacity_only_not_measured_scaling"

    @property
    def replicas(self) -> int:
        return sum(p.replicas for p in self.placements)

    def to_dict(self) -> dict[str, object]:
        return {"schema": "aureon.best_capacity.v1", "profile": self.profile,
                "replicas": self.replicas,
                "placements": [{**asdict(p), "worker_environment": p.worker_environment()}
                               for p in self.placements],
                "qualification": self.qualification, "native_visible_device": "cuda:0",
                "allocation_scope": "full independent process/model/graph per replica; caller peak includes CUDA context and non-Torch allocations",
                "processes_launched": 0, "automatic_precision_change": False,
                "caution": "Not a VRAM reservation. Free memory may change; benchmark throughput before increasing a serving default."}


def plan_best_replicas(devices: Sequence[BestDeviceBudget], *, per_replica_bytes: int,
                       reserve_per_device_bytes: int, replicas_per_device: int = 1,
                       profile: BestProfile = "best_fp32") -> BestCapacityPlan:
    """Plan arbitrary N process replicas per device under explicit memory bounds.

    Reject overcommit instead of silently lowering the request. Per-replica peak
    is a caller-supplied measurement/estimate for the intended hardware/workload;
    a Torch allocator peak alone need not account for all device memory.
    """
    _integer(per_replica_bytes, "per_replica_bytes", positive=True)
    _integer(reserve_per_device_bytes, "reserve_per_device_bytes")
    _integer(replicas_per_device, "replicas_per_device", positive=True)
    if profile not in ("best_fp32", "reference_fp32"):
        raise ValueError("Selected generation core supports best_fp32 or reference_fp32 only")
    if not devices:
        raise ValueError("Declare at least one device")
    seen: set[tuple[str, str]] = set()
    placements = []
    for device in devices:
        if not isinstance(device, BestDeviceBudget):
            raise TypeError("Use BestDeviceBudget for each placement")
        identity = (device.host, device.device)
        if identity in seen:
            raise ValueError("Duplicate physical device in a host placement domain")
        seen.add(identity)
        if profile == "best_fp32" and not device.cuda_graph:
            raise ValueError("best_fp32 needs declared CUDA Graph support; no silent fallback")
        available = max(0, device.free_bytes - reserve_per_device_bytes)
        capacity = available // per_replica_bytes
        if replicas_per_device > capacity:
            raise ValueError(f"Requested replicas exceed declared capacity on {device.host}/{device.device}: {replicas_per_device} > {capacity}")
        placements.append(BestReplicaPlacement(device.host, device.device, replicas_per_device,
                          capacity, per_replica_bytes, reserve_per_device_bytes,
                          available - replicas_per_device * per_replica_bytes))
    return BestCapacityPlan(profile, tuple(placements))


def best_usage_cost(*, resource_seconds: float, completed_outputs: int,
                    hourly_rate: float, currency: str = "THB") -> dict[str, object]:
    """Normalize explicitly supplied resource occupancy; not a payment ledger.

    Include startup, failed attempts and export in resource_seconds when that is
    the intended boundary. For multiple paid resources sum their occupied time;
    wall time of concurrent work is not automatically aggregate resource time.
    """
    seconds = _number(resource_seconds, "resource_seconds")
    rate = _number(hourly_rate, "hourly_rate")
    _integer(completed_outputs, "completed_outputs")
    if not isinstance(currency, str) or re.fullmatch(r"[A-Z]{3}", currency) is None:
        raise ValueError("currency must be an explicit three-letter uppercase label")
    total = seconds * rate / 3600
    if not math.isfinite(total):
        raise ValueError("Cost arithmetic overflow")
    per_output = total / completed_outputs if completed_outputs else None
    per_thousand = per_output * 1000 if per_output is not None else None
    if per_thousand is not None and not math.isfinite(per_thousand):
        raise ValueError("Normalized cost arithmetic overflow")
    return {"schema": "aureon.best_usage_cost.v1", "resource_seconds": seconds,
            "completed_outputs": completed_outputs, "hourly_rate": rate, "currency": currency,
            "total": total, "per_completed_output": per_output,
            "per_1000_completed_outputs": per_thousand,
            "price_source": "caller_assumption", "billable": False,
            "scope": "Arithmetic on declared occupancy; not a provider quote, speed prediction or payment charge"}
