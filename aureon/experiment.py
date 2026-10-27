"""Explicit sequential profiles for GPUs that cannot hold both backends.

Capture each resident predictor separately, then compare saved measurements.
Timings cover warm synchronous normalized-tile prediction, output copies and
transport when used. They exclude loading, preprocessing, scene stitching and
export. Repeated inputs and numerical parity are not labeled task accuracy.
Nothing starts a model or server on import.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping

import numpy as np

from .benchmark import BenchmarkIdentity, _bounded_int, _identity, _input_hash


_SCOPE = "warm_predictor_pipeline_normalized_tiles"
_LOGITS_FILE = "first_logits.npy"
_PROFILE_FILE = "profile.json"
_EXECUTION_FILE = "execution.json"
_LOCAL_WORKER_STATE = "local_worker_full_state_tensor_hashes_verified_post_load"
_CONTROLLED_KEYS = ("controlled_service_dir", "controlled_service_launch_sha256",
                    "controlled_service_worker_sha256")


def _diagnostic_markers(environment):
    """Fail closed for unfamiliar values on explicit instrumentation flags."""
    return [key for key in ("forward_diagnostics", "transport_diagnostics", "engine_diagnostics",
                            "AUREON_FORWARD_DIAGNOSTICS", "AUREON_ENGINE_DIAGNOSTICS")
            if key in environment and environment[key].strip().lower() not in ("", "0", "false")]


def _tolerances(atol, rtol, min_class_agreement=1.0):
    for name, value in (("atol", atol), ("rtol", rtol), ("min_class_agreement", min_class_agreement)):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if min_class_agreement > 1:
        raise ValueError("min_class_agreement must be at most one")


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_execution_receipt(folder, record, profile_sha256):
    """A driver profile is incomplete until its postflight receipt is valid.

    Generic callable captures remain usable without a process-driver receipt.
    The data-manifest marker also covers existing run-profile captures written
    before an explicit driver marker was introduced.
    """
    environment = record["environment"]
    controlled = any(key in environment for key in _CONTROLLED_KEYS)
    if "driver" not in environment and "dataset_manifest_sha256" not in environment and not controlled:
        return None
    path = folder / _EXECUTION_FILE
    if not path.is_file() or not 0 < path.stat().st_size <= 256 * 1024:
        raise ValueError("Driver profile requires a complete bounded execution.json receipt")
    raw = path.read_bytes()
    try:
        receipt = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Driver execution receipt is not valid JSON") from exc
    if (not isinstance(receipt, dict) or receipt.get("schema") != "aureon.profile_execution.v1"
            or receipt.get("profile_sha256") != profile_sha256
            or any(receipt.get(key) is not True for key in (
                "valid_execution_receipt", "bundle_unchanged_after_capture",
                "dataset_unchanged_after_capture", "precision_contract_unchanged_after_capture",
            ))):
        raise ValueError("Driver execution receipt must verify this exact profile and successful postflight checks")
    if "dataset_manifest_sha256" in environment:
        dataset = receipt.get("dataset")
        if (not isinstance(dataset, dict)
                or dataset.get("manifest_sha256") != environment["dataset_manifest_sha256"]):
            raise ValueError("Driver execution receipt dataset identity does not match its profile")
    if controlled:
        if (environment.get("backend") != "vllm"
                or not environment.get("controlled_service_dir")
                or not environment.get("model_execution_settings_actual")
                or environment.get("server_checkpoint_attestation") != _LOCAL_WORKER_STATE
                or receipt.get("server_checkpoint_attestation") != _LOCAL_WORKER_STATE
                or receipt.get("controlled_service_evidence_unchanged") is not True):
            raise ValueError("Controlled local service requires matching worker state evidence and unchanged postflight evidence")
        for key in _CONTROLLED_KEYS[1:]:
            digest = environment.get(key, "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Controlled local service launch and worker SHA-256 identities are required")
    return hashlib.sha256(raw).hexdigest()


def _actual_execution_settings(record):
    """Use model-process settings only, never client flags or server expectations."""
    environment = record["environment"]
    controlled = record["controlled_service_verified"]
    supported = environment.get("backend") == "pytorch" or controlled
    if not supported or "model_execution_settings_actual" not in environment:
        return None
    try:
        settings = json.loads(environment["model_execution_settings_actual"])
    except (ValueError, TypeError) as exc:
        raise ValueError("Recorded actual execution settings must be JSON objects") from exc
    if not isinstance(settings, dict) or not settings:
        raise ValueError("Recorded actual execution settings must be nonempty JSON objects")
    return settings


def _differences(reference, candidate, *, atol, rtol):
    """Accumulate a tile at a time to bound float64 comparison memory."""
    close, maximum, total, count, matching, pixels = True, 0.0, 0.0, 0, 0, 0
    for original, proposed in zip(reference, candidate):
        baseline = original.astype(np.float64)
        error = np.abs(proposed.astype(np.float64) - baseline)
        close = close and bool(np.all(error <= atol + rtol * np.abs(baseline)))
        maximum = max(maximum, float(error.max()))
        total += float(error.sum())
        count += error.size
        matching += int(np.count_nonzero(original.argmax(axis=0) == proposed.argmax(axis=0)))
        pixels += original.shape[1] * original.shape[2]
    return {"allclose": close, "max_abs_error": maximum, "mean_abs_error": total / count,
            "class_agreement": matching / pixels}


def _summary(samples, *, tile_count, batch_size):
    if not isinstance(samples, list) or not samples or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value <= 0 for value in samples
    ):
        raise ValueError("Latency samples must be finite positive durations")
    p50, p95 = np.percentile(samples, [50, 95])
    return {"batch_size": batch_size, "predictor_calls_per_run": math.ceil(tile_count / batch_size),
            "latency_seconds": {"samples": samples, "p50": float(p50), "p95": float(p95)},
            "tiles_per_second": tile_count * len(samples) / sum(samples)}


def capture_profile(
    predictor: Callable[[np.ndarray], np.ndarray],
    tiles: np.ndarray,
    *,
    identity: BenchmarkIdentity,
    output_dir,
    batch_size: int = 1,
    warmup: int = 1,
    runs: int = 5,
    synchronize: Callable[[], None] | None = None,
    environment: Mapping[str, str] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    clock: Callable[[], float] = perf_counter,
) -> dict:
    """Measure one explicitly supplied predictor and persist its first logits.

    The directory must be new and its parent must exist. Input must be finite
    normalized float32 [N,12,224,224], 1 <= N <= 16. Compute precision comes from
    identity, while predictor output must be CPU float32 [N,10,224,224]. Every
    measured run after the first must preserve logits within tolerance AND all
    class IDs. A failed repeatability gate is recorded rather than discarded.
    Optional clock injection exists for unit tests, not performance claims.
    Environment values are caller declarations, never hardware attestations.
    """
    identity_record = _identity(identity)
    batch_size = _bounded_int(batch_size, "batch_size", 1, 16)
    warmup, runs = _bounded_int(warmup, "warmup", 0, 10), _bounded_int(runs, "runs", 1, 30)
    _tolerances(atol, rtol)
    if not callable(predictor) or not callable(clock) or (synchronize is not None and not callable(synchronize)):
        raise ValueError("predictor, clock and supplied synchronize must be callable")
    metadata = dict(environment or {})
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()):
        raise ValueError("environment must contain caller-recorded string keys and values")
    if (not isinstance(tiles, np.ndarray) or tiles.dtype != np.float32
            or tiles.ndim != 4 or tiles.shape[1:] != (12, 224, 224)
            or not 1 <= len(tiles) <= 16 or not np.isfinite(tiles).all()):
        raise ValueError("tiles must be finite normalized float32 [N,12,224,224], with 1 <= N <= 16")
    inputs = np.array(tiles, copy=True, order="C")
    inputs.setflags(write=False)
    input_sha256 = _input_hash(inputs)
    folder = Path(output_dir)
    folder.mkdir(parents=False, exist_ok=False)
    samples = []

    def invoke(timed):
        output = np.empty((len(inputs), 10, 224, 224), dtype=np.float32)
        if synchronize is not None:
            synchronize()
        start = clock() if timed else None
        for offset in range(0, len(inputs), batch_size):
            batch = inputs[offset:offset + batch_size]
            result = predictor(batch)
            if not isinstance(result, np.ndarray) or result.dtype != np.float32 or result.shape != (len(batch), 10, 224, 224):
                raise ValueError("predictor must return float32 NumPy logits [batch,10,224,224]")
            output[offset:offset + len(batch)] = result
        if synchronize is not None:
            synchronize()
        if timed:
            elapsed = float(clock() - start)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("clock must measure a finite positive elapsed duration")
            samples.append(elapsed)
        if not np.isfinite(output).all():
            raise ValueError("predictor returned nonfinite logits")
        if _input_hash(inputs) != input_sha256:
            raise ValueError("predictor changed the benchmark inputs")
        return output

    for _ in range(warmup):
        invoke(False)
    first = invoke(True)
    repetitions = []
    for _ in range(1, runs):
        repetitions.append(_differences(first, invoke(True), atol=atol, rtol=rtol))
    repeat_passed = all(row["allclose"] and row["class_agreement"] == 1.0 for row in repetitions)
    logits_path = folder / _LOGITS_FILE
    with logits_path.open("xb") as stream:
        np.save(stream, first, allow_pickle=False)
    report = {
        "schema_version": 1, "kind": "sequential_predictor_profile", "scope": _SCOPE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity_record, "environment": metadata,
        "diagnostic_scope": {"active": bool(_diagnostic_markers(metadata)),
                             "markers": _diagnostic_markers(metadata)},
        "input": {"sha256": input_sha256, "shape": list(inputs.shape), "dtype": str(inputs.dtype)},
        "config": {"warmup": warmup, "runs": runs, "atol": atol, "rtol": rtol},
        "measurement": _summary(samples, tile_count=len(inputs), batch_size=batch_size),
        "repeatability": {"passed": repeat_passed, "measured_comparisons": runs - 1,
                          "min_class_agreement": 1.0, "comparisons": repetitions},
        "first_logits": {"file": _LOGITS_FILE, "sha256": _hash_file(logits_path),
                         "shape": list(first.shape), "dtype": str(first.dtype)},
        "server_checkpoint_attestation": metadata.get("server_checkpoint_attestation", "not_available"),
        "limitations": ["Sequential captures are exposed to thermal and time-varying workload differences.",
                        "Warm repeated inputs; no cold start, full scene, peak memory or labeled task-quality measurement.",
                        "Model identity and hardware metadata are caller declarations; local hashes do not attest a remote service."],
    }
    with (folder / _PROFILE_FILE).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def _read_profile(folder):
    folder = Path(folder)
    path = folder / _PROFILE_FILE
    if path.stat().st_size > 256 * 1024:
        raise ValueError("Profile JSON is too large")
    raw = path.read_bytes()
    record = json.loads(raw)
    if record.get("schema_version") != 1 or record.get("kind") != "sequential_predictor_profile" or record.get("scope") != _SCOPE:
        raise ValueError("Unsupported sequential profile schema or scope")
    environment = record.get("environment")
    if (not isinstance(environment, dict)
            or not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items())):
        raise ValueError("Profile environment must contain string keys and values")
    profile_sha256 = hashlib.sha256(raw).hexdigest()
    execution_sha256 = _verify_execution_receipt(folder, record, profile_sha256)
    _identity(BenchmarkIdentity(**record["identity"]))
    source = record["input"]
    shape = source["shape"]
    if (not isinstance(shape, list) or len(shape) != 4 or shape[1:] != [12, 224, 224]
            or type(shape[0]) is not int or not 1 <= shape[0] <= 16 or source["dtype"] != "float32"):
        raise ValueError("Profile input shape or dtype is invalid")
    if not isinstance(source["sha256"], str) or len(source["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in source["sha256"]):
        raise ValueError("Profile input hash is invalid")
    config = record["config"]
    runs = _bounded_int(config["runs"], "runs", 1, 30)
    _bounded_int(config["warmup"], "warmup", 0, 10)
    _tolerances(config["atol"], config["rtol"])
    measurement = record["measurement"]
    size = _bounded_int(measurement["batch_size"], "batch_size", 1, 16)
    samples = measurement["latency_seconds"]["samples"]
    if len(samples) != runs:
        raise ValueError("Profile sample count does not match its declared runs")
    # Recompute summaries from samples instead of trusting editable p50 fields.
    record["measurement"] = _summary(samples, tile_count=shape[0], batch_size=size)
    repeat = record["repeatability"]
    if (type(repeat["passed"]) is not bool or repeat["measured_comparisons"] != runs - 1
            or len(repeat["comparisons"]) != runs - 1 or repeat["min_class_agreement"] != 1.0):
        raise ValueError("Invalid repeatability metadata")
    actual_passed = True
    for row in repeat["comparisons"]:
        if type(row["allclose"]) is not bool:
            raise ValueError("Invalid repeatability comparison")
        _tolerances(row["max_abs_error"], row["mean_abs_error"], row["class_agreement"])
        actual_passed = actual_passed and row["allclose"] and row["class_agreement"] == 1.0
    if actual_passed != repeat["passed"]:
        raise ValueError("Repeatability summary contradicts its comparisons")
    artifact = record["first_logits"]
    expected_shape = [shape[0], 10, 224, 224]
    if artifact["file"] != _LOGITS_FILE or artifact["shape"] != expected_shape or artifact["dtype"] != "float32":
        raise ValueError("Invalid first-logit artifact metadata")
    logits_path = folder / _LOGITS_FILE
    maximum_size = math.prod(expected_shape) * 4 + 65536
    if logits_path.stat().st_size > maximum_size or _hash_file(logits_path) != artifact["sha256"]:
        raise ValueError("Saved logits fail size or SHA-256 verification")
    logits = np.load(logits_path, allow_pickle=False, mmap_mode="r")
    if logits.dtype != np.float32 or list(logits.shape) != expected_shape or not np.isfinite(logits).all():
        raise ValueError("Saved logits must be finite float32 of the declared shape")
    record["profile_sha256"] = profile_sha256
    record["execution_sha256"] = execution_sha256
    record["controlled_service_verified"] = bool(
        execution_sha256 and any(key in environment for key in _CONTROLLED_KEYS)
    )
    return record, logits


def compare_profiles(reference_dir, candidate_dir, *, atol=1e-5, rtol=1e-5, min_class_agreement=1.0) -> dict:
    """Compare two explicit saved profiles without model execution or writes.

    Identities and input hashes must match exactly. A declared gpu_uuid mismatch
    is rejected. A repeatability failure or a repeat gate looser than the
    comparison tolerance suppresses speedup. Missing GPU identity is reported,
    never silently called same-hardware evidence. The caller chooses where to
    save the returned report. Driver captures must have a valid matching
    execution receipt. Recorded PyTorch and controlled local-worker model
    execution policies must match; server expectations are not actual settings.
    Instrumented profiles retain numerical comparisons and raw timings, but
    never supply clean speedup or promotion evidence, even against each other.
    """
    _tolerances(atol, rtol, min_class_agreement)
    reference, baseline = _read_profile(reference_dir)
    candidate, proposed = _read_profile(candidate_dir)
    if reference["identity"] != candidate["identity"] or reference["input"] != candidate["input"]:
        raise ValueError("Profiles must use the same model/checkpoint/recipe/preprocessing/precision and input")
    reference_gpu = reference["environment"].get("gpu_uuid")
    candidate_gpu = candidate["environment"].get("gpu_uuid")
    if reference_gpu and candidate_gpu and reference_gpu != candidate_gpu:
        raise ValueError("Profiles declare different GPU UUIDs")
    precision_settings_match = "not_verified"
    settings = [_actual_execution_settings(profile) for profile in (reference, candidate)]
    controlled_profiles = [profile for profile in (reference, candidate) if profile["controlled_service_verified"]]
    if all(value is not None for value in settings):
        if settings[0] != settings[1]:
            raise ValueError("Profiles record different actual PyTorch/local-worker precision/execution settings")
        precision_settings_match = ("same_recorded_execution_settings_with_controlled_local_worker_evidence"
                                    if controlled_profiles else "same_recorded_pytorch_execution_settings")
    difference = _differences(baseline, proposed, atol=atol, rtol=rtol)
    repeat_passed = reference["repeatability"]["passed"] and candidate["repeatability"]["passed"]
    tolerance_compatible = all(profile["config"]["atol"] <= atol and profile["config"]["rtol"] <= rtol for profile in (reference, candidate))
    passed = bool(difference["allclose"] and difference["class_agreement"] >= min_class_agreement and repeat_passed and tolerance_compatible)
    diagnostics = {name: _diagnostic_markers(profile["environment"])
                   for name, profile in (("reference", reference), ("candidate", candidate))}
    diagnostic_active = any(diagnostics.values())
    speedup = (reference["measurement"]["latency_seconds"]["p50"]
               / candidate["measurement"]["latency_seconds"]["p50"] if passed and not diagnostic_active else None)
    state_evidence = {name: (_LOCAL_WORKER_STATE if profile["controlled_service_verified"] else "not_available")
                      for name, profile in (("reference", reference), ("candidate", candidate))}
    vllm_profiles = [profile for profile in (reference, candidate) if profile["environment"].get("backend") == "vllm"]
    all_services_verified = bool(vllm_profiles and all(profile["controlled_service_verified"] for profile in vllm_profiles))
    return {
        "schema_version": 1, "comparison_kind": "sequential_predictor_profiles", "scope": _SCOPE,
        "identity": reference["identity"], "input": reference["input"],
        "config": {"atol": atol, "rtol": rtol, "min_class_agreement": min_class_agreement},
        "environment": {"reference": reference["environment"], "candidate": candidate["environment"]},
        "hardware_match": "same_caller_declared_gpu_uuid" if reference_gpu and candidate_gpu else "not_verified",
        "precision_settings_match": precision_settings_match,
        "profiles_sha256": {"reference": reference["profile_sha256"], "candidate": candidate["profile_sha256"]},
        "execution_receipts_sha256": {"reference": reference["execution_sha256"], "candidate": candidate["execution_sha256"]},
        "reference": reference["measurement"], "candidate": candidate["measurement"],
        "correctness": {"passed": passed, **difference, "repeatability_passed": repeat_passed,
                        "repeatability_tolerance_compatible": tolerance_compatible},
        "speedup": speedup, "speedup_basis": "median_group_latency",
        "diagnostic_scope": {"active": diagnostic_active, **diagnostics,
                             "timings_eligible_as_clean_evidence": not diagnostic_active},
        "speedup_suppressed_reason": ("diagnostic_instrumentation" if diagnostic_active else
                                      "numerical_or_repeatability_gate_failed" if not passed else None),
        "promotion_eligible": bool(speedup is not None and speedup > 1.0 and reference_gpu and candidate_gpu),
        "production_approved": False,
        "server_checkpoint_attestation": _LOCAL_WORKER_STATE if all_services_verified else "not_available",
        "server_state_verification": state_evidence,
        "remote_cryptographic_attestation": "not_provided",
        "limitations": [*reference["limitations"],
                        "Latency percentiles cover the full input group, not individual request latency.",
                        "For vLLM, predictor batch_size is a request-group size; actual server GPU batching is not measured here.",
                        "vLLM server expected precision settings are declarations, not worker-setting attestations.",
                        "Forward or transport diagnostic instrumentation excludes both profiles from clean speedup and promotion evidence.",
                        "Controlled local worker evidence checks post-load tensors and settings; it is not remote cryptographic attestation or proof of settings during every forward."],
    }
