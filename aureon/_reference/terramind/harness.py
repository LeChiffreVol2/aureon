"""Frozen finite native proof driver. Only the designated GPU owner executes it."""
from contextlib import contextmanager, nullcontext
import copy
import gc
import hashlib
import os
from pathlib import Path
import statistics
from time import perf_counter
import traceback
import types
import weakref

from .common import binding, digest, file_hash, read_json, write_json
from .compat import BASELINE_LABEL, schedule_compatibility
from .geo import load_input, write_cog
from .loader import load_native
from .reuse import exact_conditioning_reuse
from .rng import rng_capsule

_MISSING = object()


def capture_failure(exc):
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
def trace_native(model, output, *, detailed):
    """Token verification for both arms; logits/stage diagnostics only in qualification."""
    import numpy as np
    import torch
    saved = []
    trace = {"logits": [], "tokens": None, "stage_seconds_diagnostic": {}, "methods_restored": False}
    def replace(obj, name, fn):
        saved.append((obj, name, obj.__dict__.get(name, _MISSING)))
        setattr(obj, name, types.MethodType(fn, obj))
    original_generate = model.sampler.generate
    def generate(this, *args, **kwargs):
        start = perf_counter()
        result = original_generate(*args, **kwargs)
        tokens = result["tok_sen1grd@224"]["tensor"].detach().cpu().numpy().copy()
        np.save(output / "tokens.npy", tokens, allow_pickle=False)
        trace["tokens"] = binding(output / "tokens.npy")
        if detailed: trace["stage_seconds_diagnostic"]["token_generation_including_trace"] = perf_counter()-start
        return result
    replace(model.sampler, "generate", generate)
    if detailed:
        core = model.sampler.model
        original_logits = core.forward_logits
        def logits(this, *args, **kwargs):
            values = original_logits(*args, **kwargs)
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
        def decode(this, *args, **kwargs):
            torch.cuda.synchronize(); start = perf_counter()
            result = original_decode(*args, **kwargs)
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


