"""Typing contracts for dynamic upstream modules and structured runtime records.

Metadata is deliberately open: versioned receipts contain heterogeneous nested
JSON and optional tensor diagnostics. Any stays at these serialization/upstream
extension boundaries; owned tensors, graph buffers and lifecycle methods use
explicit protocols. No neural dependency is imported at runtime here.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypedDict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from torch import Tensor, device, dtype

PathInput: TypeAlias = str | Path
Metadata: TypeAlias = dict[str, Any]
TensorMap: TypeAlias = dict[str, "Tensor"]
FloatArray: TypeAlias = "NDArray[np.float32]"
BoolArray: TypeAlias = "NDArray[np.bool_]"
Array: TypeAlias = "NDArray[Any]"
Cancel: TypeAlias = Callable[[], bool] | None
TensorSignature: TypeAlias = tuple[tuple[str, tuple[int, ...], "dtype", "device"], ...]
InputSnapshot: TypeAlias = tuple[FloatArray, BoolArray, Metadata, Metadata]
DecoderContext: TypeAlias = Callable[["TerraMindModel", Metadata], "AbstractContextManager[Metadata]"]

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


class Graph(Protocol):
    def replay(self) -> None: ...
    def reset(self) -> None: ...
    def pool(self) -> tuple[int, int]: ...


class GraphBinding(Protocol):
    def validate(self, inputs: TensorMap) -> None: ...
    def stage(self, inputs: TensorMap, stats: Metadata) -> TensorMap: ...
    def forward(self, buffers: TensorMap) -> Tensor: ...
    def close(self) -> None: ...


class GraphBackend(Protocol):
    def capture(self, forward: Callable[[TensorMap], Tensor], buffers: TensorMap,
                stats: Metadata) -> tuple[Graph, Tensor]: ...


class DecoderBinding(GraphBinding, Protocol):
    frequency: Tensor | None
    def namespace(self) -> Metadata: ...
    def embedding(self, timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor: ...


class PreparedState(TypedDict, total=False):
    """Request-owned state populated at the native decoder suspension boundary."""
    image: Tensor
    conditioning: Tensor
    timestep: Tensor
    scheduler: Any  # Pinned upstream DDIM scheduler, runtime identity checked.
    initial_noise: Metadata
    conditioning_identity: Metadata
    rng: Metadata
    trace: Metadata
    compatibility: Metadata
    reuse: Metadata
    reuse_decision: Metadata


class StageRunner(Protocol):
    def __call__(self, states: list[PreparedState], *, cancel: Callable[[], None]) -> Metadata: ...


class PrepareRunner(Protocol):
    def __call__(self, prepares: list[Callable[[], PreparedState]], *, cancel: Callable[[], None]) -> tuple[list[PreparedState], Metadata, Metadata]: ...


class Stream(Protocol):
    cuda_stream: int
    def wait_stream(self, stream: Stream) -> None: ...
    def wait_event(self, event: Event) -> None: ...
    def synchronize(self) -> None: ...


class Event(Protocol):
    def record(self, stream: Stream) -> None: ...
    def elapsed_time(self, end: Event) -> float: ...


class LaneBackend(Protocol):
    label: str
    def create_stream(self) -> Stream: ...
    def current(self) -> Stream: ...
    def context(self, stream: Stream) -> AbstractContextManager[None]: ...
    def wait(self, stream: Stream, origin: Stream) -> None: ...
    def record(self, tensor: Tensor, stream: Stream) -> None: ...
    def sync(self, stream: Stream) -> None: ...
    def to_lane_timesteps(self, steps: Tensor) -> Tensor: ...
    def event(self, stream: Stream) -> Event: ...
    def join(self, origin: Stream, event: Event) -> None: ...
    def stream_id(self, stream: Stream) -> int: ...
    def intervals(self, origin_event: Event, starts: list[Event], ends: list[Event]) -> list[dict[str, float]]: ...


class NativeModule(Protocol):
    """The common guarded tensor/module surface used for identity and placement."""
    training: bool

    def parameters(self, recurse: bool = True) -> Iterator[Tensor]: ...
    def named_parameters(self, prefix: str = "", recurse: bool = True,
                         remove_duplicate: bool = True) -> Iterator[tuple[str, Tensor]]: ...
    def named_buffers(self, prefix: str = "", recurse: bool = True,
                      remove_duplicate: bool = True) -> Iterator[tuple[str, Tensor]]: ...
    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Tensor]: ...


class TerraMindModel(NativeModule, Protocol):
    # TerraTorch dynamically registers and replaces sampler/tokenizer methods.
    # Their inspected version-specific extension surface is validated at runtime.
    modalities: list[str]
    output_modalities: list[str]
    sampler: Any
    tokenizer: Any
    decoding_steps: int
    standardize: bool
    temps: float
    top_p: float
    top_k: int
    mod_name_mapping: dict[str, str]
    pretraining_mean: dict[str, Tensor]
    pretraining_std: dict[str, Tensor]

    def modules(self) -> Iterator[NativeModule]: ...
    def __call__(self, data: TensorMap, **kwargs: Any) -> TensorMap: ...
    def eval(self) -> TerraMindModel: ...
    def requires_grad_(self, requires_grad: bool = True) -> TerraMindModel: ...
    def to(self, *args: Any, **kwargs: Any) -> TerraMindModel: ...
