"""In-memory evidence and explicit optional COG export; no model forward."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from collections.abc import Iterator
from ._types import TerraMindModel, Metadata, PathInput, Array
if TYPE_CHECKING:
    from ..api.contracts import GenerationResult
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import types


def array_identity(array: Array) -> Metadata:
    return {"sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "dtype": str(array.dtype), "shape": list(array.shape), "bytes": array.nbytes,
            "encoding": "C-order raw array bytes"}


@contextmanager
def trace_memory(model: TerraMindModel) -> Iterator[Metadata]:
    """Keep a token digest without serializing input/output arrays to disk."""
    sampler = model.sampler
    missing = object()
    old = sampler.__dict__.get("generate", missing)
    original = sampler.generate
    trace: Metadata = {"tokens": None, "logits": [], "methods_restored": False,
             "storage": "memory_digest_only"}
    def generate(this: Any, *args: Any, **kwargs: Any) -> dict[str, Metadata]:
        result: dict[str, Metadata] = original(*args, **kwargs)
        tokens = result["tok_sen1grd@224"]["tensor"].detach().cpu().numpy()
        trace["tokens"] = array_identity(tokens)
        return result
    sampler.generate = types.MethodType(generate, sampler)
    try:
        yield trace
    finally:
        if old is missing:
            delattr(sampler, "generate")
        else:
            sampler.generate = old
        trace["methods_restored"] = True


def export_generation(result: GenerationResult, output: PathInput) -> Metadata:
    """Explicitly export an unchanged in-memory result with its supplied grid.

    No coordinate system is guessed. No model execution or normalization occurs.
    """
    from .geo import write_cog
    provenance = deepcopy(result["provenance"])
    grid = provenance["input"].get("grid")
    if grid is None:
        raise ValueError("COG export requires grid CRS/affine on GenerationInput")
    values, mask = result["array"], result["valid_mask"]
    if (array_identity(values) != provenance["array"]
            or array_identity(mask) != provenance["valid_mask"]):
        raise ValueError("Result arrays changed after inference")
    return write_cog(output, values, mask, grid, provenance)
