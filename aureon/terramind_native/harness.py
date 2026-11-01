"""Request tracing and traceback cleanup used by the resident Engine.

The historical module name is retained for implementation compatibility.
Benchmark orchestration and paired-comparison drivers are not library code.
"""
from __future__ import annotations
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
import traceback
import types
from typing import Any, TYPE_CHECKING
from ._types import TerraMindModel, Metadata, TensorMap
from .common import binding
if TYPE_CHECKING:
    from torch import Tensor

_MISSING = object()

def capture_failure(exc: BaseException) -> str:
    """Retain diagnostic text, but detach unwound frames before owned cleanup."""
    message = "".join(traceback.format_exception(type(exc),exc,exc.__traceback__))
    pending = [exc]; seen = set()
    while pending:
        error = pending.pop()
        if id(error) in seen: continue
        seen.add(id(error))
        pending.extend(e for e in (error.__cause__,error.__context__) if e is not None)
        if error.__traceback__ is not None:
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
    return message


@contextmanager
def trace_native(model: TerraMindModel, output: Path, *, detailed: bool) -> Iterator[Metadata]:
    """Token verification for both arms; logits/stage diagnostics only in qualification."""
    import numpy as np
    import torch
    saved = []
    trace: Metadata = {"logits": [], "tokens": None, "stage_seconds_diagnostic": {}, "methods_restored": False}
    def replace(obj: Any, name: str, fn: Callable[..., Any]) -> None:
        saved.append((obj, name, obj.__dict__.get(name, _MISSING)))
        setattr(obj, name, types.MethodType(fn, obj))
    original_generate = model.sampler.generate
    def generate(this: Any, *args: Any, **kwargs: Any) -> dict[str, Metadata]:
        start = perf_counter()
        result: dict[str, Metadata] = original_generate(*args, **kwargs)
        tokens = result["tok_sen1grd@224"]["tensor"].detach().cpu().numpy().copy()
        np.save(output / "tokens.npy", tokens, allow_pickle=False)
        trace["tokens"] = binding(output / "tokens.npy")
        if detailed: trace["stage_seconds_diagnostic"]["token_generation_including_trace"] = perf_counter()-start
        return result
    replace(model.sampler, "generate", generate)
    if detailed:
        core = model.sampler.model
        original_logits = core.forward_logits
        def logits(this: Any, *args: Any, **kwargs: Any) -> TensorMap:
            values: TensorMap = original_logits(*args, **kwargs)
            step = len(trace["logits"])
            records = {}
            for name, value in values.items():
                array = value.detach().cpu().numpy()
                path = output / ("logits-%02d-%s.npy" % (step, name.replace("@", "_")))
                np.save(path, array, allow_pickle=False)
                records[name] = binding(path)
            trace["logits"].append(records)
            return values
        replace(core, "forward_logits", logits)
        tokenizer = model.tokenizer["tok_sen1grd@224"]
        original_decode = tokenizer.decode_tokens
        def decode(this: Any, *args: Any, **kwargs: Any) -> Tensor:
            torch.cuda.synchronize(); start = perf_counter()
            result: Tensor = original_decode(*args, **kwargs)
            torch.cuda.synchronize()
            trace["stage_seconds_diagnostic"]["modality_decode"] = perf_counter()-start
            return result
        replace(tokenizer, "decode_tokens", decode)
    try:
        yield trace
    finally:
        for obj, name, previous in reversed(saved):
            if previous is _MISSING: delattr(obj, name)
            else: setattr(obj, name, previous)
        saved.clear()
        trace["methods_restored"] = True

