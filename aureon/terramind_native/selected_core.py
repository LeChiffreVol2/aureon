"""Typed delivery adapter bindings to the immutable selected numerical primitives.

Verification belongs to resident construction. Neural imports belong to explicit
runtime load. No request hashing, model ownership, input storage or forward here.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

from .._reference.integrity import CoreFingerprint
from ._types import TerraMindModel, Metadata, DecoderBinding, GraphBinding

if TYPE_CHECKING:
    from ..api.contracts import NumericalCoreReceipt
    from .denoiser_graph import DenoiserGraph
    from .graph_runtime import FixedTensorGraph


NUMERICAL_SPAN_SHA256 = "2f056be1e43fbc7b351017635a5b9affafa85e6cc903949c6105a598cfe5b65e"


class Schedule(Protocol):
    def __call__(self, model: TerraMindModel, *, token_steps: int) -> AbstractContextManager[Metadata]: ...


class Reuse(Protocol):
    def __call__(self, model: TerraMindModel, *, enabled: bool) -> AbstractContextManager[Metadata]: ...


class RNG(Protocol):
    def __call__(self, seed: int, *, devices: tuple[int, ...] = ()) -> AbstractContextManager[Metadata]: ...


class Lock(Protocol):
    def acquire(self, blocking: bool = True) -> bool: ...
    def release(self) -> None: ...


def core_receipt(identity: CoreFingerprint, *, route: str) -> NumericalCoreReceipt:
    return {"manifest_sha256": identity.manifest_sha256, "wheel_sha256": identity.wheel_sha256,
            "source_archive_sha256": identity.source_archive_sha256,
            "namespace": "aureon._reference.terramind", "relocated_source": True,
            "new_performance_qualification": False, "execution_adapter": route,
            "shared_primitives": ["compat.schedule_compatibility", "rng.rng_capsule",
                "reuse.exact_conditioning_reuse", "session._decision",
                "denoiser_graph.DenoiserGraph", "denoiser_binding.S1GRDDecoderBinding",
                "graph_runtime.FixedTensorGraph"],
            "orchestration": "active typed adapter; normalized single-request numerical span checked against frozen source",
            "numerical_span_sha256": NUMERICAL_SPAN_SHA256,
            "delivery": "Memory returns owned arrays and semantic provenance; raster export is explicit. File adapters retain file delivery.",
            "qualification": "source identity only; new artifact/route parity and performance require separate evidence"}


@dataclass(frozen=True)
class SelectedPrimitives:
    identity: CoreFingerprint
    schedule: Schedule
    reuse: Reuse
    rng: RNG
    decision: Callable[[TerraMindModel, Metadata, str], Metadata]
    graph_factory: Callable[[TerraMindModel], DenoiserGraph]
    graph_eligibility: Callable[[TerraMindModel, Metadata, str], tuple[bool, str]]
    binding_factory: Callable[[TerraMindModel], DecoderBinding]
    tensor_graph_factory: Callable[[GraphBinding], FixedTensorGraph]
    patch_lock: Lock
    rng_module: ModuleType

    def receipt(self, *, route: str) -> NumericalCoreReceipt:
        return core_receipt(self.identity, route=route)


def load_primitives(identity: CoreFingerprint) -> SelectedPrimitives:
    """Called only by the explicit native load, after construction verification.

    These are the actual frozen function/class objects, not similarly named
    active implementations. The active assets loader remains responsible for
    pinned weights/source validation and constructing exactly one model.
    """
    prefix = "aureon._reference.terramind."
    compat, reuse, rng, session, graph, binding, tensor_graph = (
        import_module(prefix + name) for name in
        ("compat", "reuse", "rng", "session", "denoiser_graph", "denoiser_binding", "graph_runtime"))
    return SelectedPrimitives(identity,
        cast(Schedule, compat.schedule_compatibility), cast(Reuse, reuse.exact_conditioning_reuse),
        cast(RNG, rng.rng_capsule), cast(Callable[[TerraMindModel, Metadata, str], Metadata], session._decision),
        cast("Callable[[TerraMindModel], DenoiserGraph]", graph.DenoiserGraph),
        cast(Callable[[TerraMindModel, Metadata, str], tuple[bool, str]], graph.eligibility),
        cast(Callable[[TerraMindModel], DecoderBinding], binding.S1GRDDecoderBinding),
        cast("Callable[[GraphBinding], FixedTensorGraph]", tensor_graph.FixedTensorGraph),
        cast(Lock, graph._PATCH_LOCK), rng)
