"""Value contracts shared by planners and owners; no model, clock or I/O access.

Array types are imported only by static checkers. Diagnostic dictionaries remain
extensible; declaring the stable fields does not freeze every runtime receipt.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypedDict
from collections.abc import Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from .memory_input import GenerationInput

ExecutionProfile: TypeAlias = Literal["recommended", "reference"]
Parallelism: TypeAlias = int | Literal["auto"]
ParallelFallback: TypeAlias = Literal["serial", "error"]
EngineState: TypeAlias = Literal["new", "ready", "closed", "failed"]
HandleState: TypeAlias = Literal[
    "reserved",
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
    "consumed",
    "discarded",
]

INPUT_BYTES = (12 * 4 + 1) * 224 * 224
OUTPUT_BYTES = (2 * 4 + 1) * 224 * 224


class QueueLimits(TypedDict):
    max_pending: int
    max_results: int
    max_input_bytes: int
    max_output_bytes: int


class ParallelPlan(TypedDict):
    requested: Parallelism
    planned: int
    admitted: int
    actual: int
    neural_batch: int
    desired_lanes: int
    queue_credit_limit: int
    workspace_lane_limit: int
    fallback_reason: str | None
    qualification: str
    workspace_bytes: int
    resident_bytes: int
    per_lane_reservation_bytes: int
    free_reserve_bytes: int
    memory_scope: str


class RecipeValues(TypedDict):
    token_steps: int
    diffusion_steps: int
    temperature: float
    top_p: float
    top_k: int
    seed: int


class NativeOptions(TypedDict):
    """Fully validated options owned by a committed generation queue entry."""
    recipe: RecipeValues
    execution_profile: ExecutionProfile
    job_id: str | None
    attempt_id: str


class ExecuteGeneration(Protocol):
    def __call__(self, input_data: GenerationInput, *, recipe: RecipeValues,
                 execution_profile: ExecutionProfile, job_id: str | None,
                 attempt_id: str) -> GenerationResult: ...


GroupRequests: TypeAlias = list[tuple["GenerationInput", NativeOptions]]
ExecuteGroup: TypeAlias = Callable[[GroupRequests], list["GenerationResult"]]


class _MemoryResult(TypedDict):
    array: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    provenance: dict[str, Any]


class NumericalCoreReceipt(TypedDict):
    manifest_sha256: str
    wheel_sha256: str
    source_archive_sha256: str
    namespace: str
    relocated_source: bool
    new_performance_qualification: bool
    execution_adapter: str
    shared_primitives: list[str]
    orchestration: str
    numerical_span_sha256: str
    delivery: str
    qualification: str


class GenerationResult(_MemoryResult, total=False):
    """Owned CPU output; optional diagnostics vary by the execution path.

    Inheritance makes required/optional keys accurate at runtime too, without
    evaluating NumPy annotations during an otherwise dependency-light import.
    """

    status: Literal["completed"]
    job_id: str
    attempt_id: str
    recipe: RecipeValues
    parallel_selection: dict[str, Any]
    numerical_core: NumericalCoreReceipt
    native_model_seconds: float
    resident_delivery_seconds: float
    complete_request_seconds: float
    startup_seconds: float
    model_load_count: int
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    timing_scope: str


class _QueueMetadata(QueueLimits):
    accepting: bool
    failed: bool
    pending: int
    queued: int
    running: bool
    reserved_results: int
    input_bytes: int
    output_bytes_reserved: int
    high_water_pending: int
    high_water_results: int
    oldest_queued_seconds: float
    dispatcher_alive: bool
    budget_scope: str


class QueueMetadata(_QueueMetadata, total=False):
    collection: dict[str, Any]


class EngineMetadata(TypedDict):
    state: EngineState
    model_id: str
    execution_profile: ExecutionProfile
    work: str
    session: dict[str, Any] | None
    queue: QueueMetadata | None
    parallel: str
    parallelism: ParallelPlan
    closing: bool


def positive(value: object, name: str) -> int:
    """Validate finite count credits, rejecting booleans instead of treating them as1."""
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def validate_timeout(timeout: object) -> float | None:
    """Validate a duration without reading the clock or changing owner state."""
    if timeout is None:
        return None
    if type(timeout) not in (int, float):
        raise ValueError("timeout must be finite and nonnegative, or None")
    # isinstance narrows for type checkers; the exact-type check above rejects bool.
    assert isinstance(timeout, (int, float))
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout must be finite and nonnegative, or None")
    return timeout
