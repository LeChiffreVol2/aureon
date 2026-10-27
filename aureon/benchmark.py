"""Manual, correctness-gated timing of warmed normalized-tile predictors.

Nothing runs on import. Callers supply predictors, identities and optional
device synchronization. Timings cover synchronous predictor calls and copying
their NumPy outputs into one group, including transport if the callable uses
it. Numerical validation and comparison are outside the timed region. These are not
cold-start, full-image preprocessing/export, or ground-truth accuracy metrics.
Identity and environment metadata are caller declarations, not attestations.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from time import perf_counter
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class BenchmarkIdentity:
    model_id: str
    checkpoint_sha256: str
    recipe_id: str
    preprocessing_id: str
    precision: str = "float32"


def _identity(identity: BenchmarkIdentity) -> dict:
    if not isinstance(identity, BenchmarkIdentity):
        raise ValueError("identity must be a BenchmarkIdentity")
    result = asdict(identity)
    if not all(isinstance(value, str) and value.strip() for value in result.values()):
        raise ValueError("identity fields must be nonempty strings")
    digest = identity.checkpoint_sha256
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
    if identity.precision not in ("float32", "float16", "bfloat16"):
        raise ValueError("precision must explicitly be float32, float16, or bfloat16")
    return result


def _bounded_int(value: int, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return int(value)


def _input_hash(inputs: np.ndarray) -> str:
    digest = hashlib.sha256(json.dumps({"shape": list(inputs.shape), "dtype": inputs.dtype.str}).encode())
    digest.update(memoryview(inputs).cast("B"))
    return digest.hexdigest()


def benchmark_predictors(
    reference: Callable[[np.ndarray], np.ndarray],
    candidate: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    *,
    reference_identity: BenchmarkIdentity,
    candidate_identity: BenchmarkIdentity,
    reference_batch_size: int = 1,
    candidate_batch_size: int = 1,
    warmup: int = 1,
    runs: int = 5,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    min_class_agreement: float = 1.0,
    synchronize_reference: Callable[[], None] | None = None,
    synchronize_candidate: Callable[[], None] | None = None,
    environment: Mapping[str, str] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> dict:
    """Compare the same 1..16 normalized float32 [N,12,224,224] tiles.

    A predictor must synchronously return CPU float32 logits [batch,10,224,224]
    in input order. Its adapter owns any conversion from the declared compute
    precision. Scheduling batch sizes may differ; numerical/model identities
    must match exactly. The copied inputs are read-only. Warmups validate shape
    and finite outputs; every measured pair contributes to the correctness gate.

    Reference/candidate order alternates each measured run. Optional sync hooks
    execute before the clock starts and before it stops. Speedup is reference
    p50 / candidate p50 only when all logits pass atol + rtol * abs(reference)
    and class agreement passes. ``promotion_eligible`` means this numerical and
    median-latency gate passed on these inputs, not production or task approval.
    """
    identity = _identity(reference_identity)
    if identity != _identity(candidate_identity):
        raise ValueError("reference and candidate model/checkpoint/recipe/preprocessing/precision must match")
    for name, callback in (("reference", reference), ("candidate", candidate), ("clock", clock)):
        if not callable(callback):
            raise ValueError(f"{name} must be callable")
    for callback in (synchronize_reference, synchronize_candidate):
        if callback is not None and not callable(callback):
            raise ValueError("synchronize callbacks must be callable or None")
    warmup, runs = _bounded_int(warmup, "warmup", 0, 10), _bounded_int(runs, "runs", 1, 30)
    sizes = {
        "reference": _bounded_int(reference_batch_size, "reference_batch_size", 1, 16),
        "candidate": _bounded_int(candidate_batch_size, "candidate_batch_size", 1, 16),
    }
    for name, value in (("atol", atol), ("rtol", rtol), ("min_class_agreement", min_class_agreement)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if min_class_agreement > 1:
        raise ValueError("min_class_agreement must be at most 1")
    metadata = dict(environment or {})
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()):
        raise ValueError("environment must contain caller-recorded string keys and values")
    if (
        not isinstance(inputs, np.ndarray) or inputs.dtype != np.float32
        or inputs.ndim != 4 or inputs.shape[1:] != (12, 224, 224)
        or not 1 <= len(inputs) <= 16 or not np.isfinite(inputs).all()
    ):
        raise ValueError("inputs must be finite normalized float32 [N,12,224,224], with 1 <= N <= 16")
    tiles = np.array(inputs, copy=True, order="C")
    tiles.setflags(write=False)
    input_sha256 = _input_hash(tiles)
    predictors = {"reference": reference, "candidate": candidate}
    syncs = {"reference": synchronize_reference, "candidate": synchronize_candidate}
    samples = {"reference": [], "candidate": []}

    def invoke(name: str, timed: bool) -> np.ndarray:
        output = np.empty((len(tiles), 10, 224, 224), dtype=np.float32)
        sync = syncs[name]
        if sync is not None:
            sync()
        start = clock() if timed else None
        for offset in range(0, len(tiles), sizes[name]):
            batch = tiles[offset:offset + sizes[name]]
            result = predictors[name](batch)
            if not isinstance(result, np.ndarray) or result.dtype != np.float32 or result.shape != (len(batch), 10, 224, 224):
                raise ValueError(f"{name} must return float32 NumPy logits [batch,10,224,224]")
            output[offset:offset + len(batch)] = result
        if sync is not None:
            sync()
        if timed:
            elapsed = float(clock() - start)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("clock must measure a finite positive elapsed duration")
            samples[name].append(elapsed)
        if not np.isfinite(output).all():
            raise ValueError(f"{name} returned nonfinite logits")
        if _input_hash(tiles) != input_sha256:
            raise ValueError(f"{name} changed the benchmark inputs")
        return output

    for _ in range(warmup):
        invoke("reference", False)
        invoke("candidate", False)
    close = True
    maximum_error = error_sum = 0.0
    value_count = class_matches = class_count = 0
    for run in range(runs):
        order = ("reference", "candidate") if run % 2 == 0 else ("candidate", "reference")
        outputs = {name: invoke(name, True) for name in order}
        for baseline, proposed in zip(outputs["reference"], outputs["candidate"]):
            baseline64 = baseline.astype(np.float64)
            error = np.abs(proposed.astype(np.float64) - baseline64)
            close = close and bool(np.all(error <= atol + rtol * np.abs(baseline64)))
            maximum_error = max(maximum_error, float(error.max()))
            error_sum += float(error.sum())
            value_count += error.size
            class_matches += int(np.count_nonzero(np.argmax(baseline, axis=0) == np.argmax(proposed, axis=0)))
            class_count += baseline.shape[1] * baseline.shape[2]

    def summarize(name: str) -> dict:
        values = samples[name]
        p50, p95 = np.percentile(values, [50, 95])
        return {
            "batch_size": sizes[name], "predictor_calls_per_run": math.ceil(len(tiles) / sizes[name]),
            "latency_seconds": {"samples": values, "p50": float(p50), "p95": float(p95)},
            "tiles_per_second": len(tiles) * runs / sum(values),
        }

    reference_stats, candidate_stats = summarize("reference"), summarize("candidate")
    agreement = class_matches / class_count
    passed = close and agreement >= min_class_agreement
    speedup = reference_stats["latency_seconds"]["p50"] / candidate_stats["latency_seconds"]["p50"] if passed else None
    return {
        "schema_version": 1, "scope": "warm_predictor_pipeline_normalized_tiles",
        "identity": identity, "environment": metadata,
        "input": {"sha256": input_sha256, "shape": list(tiles.shape), "dtype": str(tiles.dtype)},
        "config": {"warmup": warmup, "runs": runs, "atol": atol, "rtol": rtol, "min_class_agreement": min_class_agreement},
        "reference": reference_stats, "candidate": candidate_stats,
        "correctness": {"passed": passed, "max_abs_error": maximum_error, "mean_abs_error": error_sum / value_count, "class_agreement": agreement},
        "speedup": speedup, "speedup_basis": "median_group_latency",
        "promotion_eligible": bool(passed and speedup > 1.0),
    }
