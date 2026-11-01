from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

@dataclass(frozen=True)
class ExecutionKey:
    model_id: str
    model_identity: str
    recipe_identity: str
    task: str
    input_modality: str
    output_modality: str
    precision_profile: str

@dataclass(frozen=True)
class PlacementRequest:
    key: ExecutionKey
    transient_vram_bytes: int
    cold_resident_bytes: int
    host_bytes: int
    output_bytes: int
    deadline_seconds: float | None = None

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

def plan_placement(request: PlacementRequest, workers: Iterable[WorkerSnapshot], *, now: float, max_snapshot_age: float = ...) -> dict[str, Any]: ...
