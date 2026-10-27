"""Offline resident-job audit: frozen scene checks plus worker identity linkage.

No model runtime is imported. This verifies recorded evidence and its declared
local trust scope; it cannot observe historical tensor values or establish
remote attestation, a controlled shutdown, map accuracy, or speedup.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import stat
import tempfile

from ..scene_pipeline import audit as scene_audit


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[0-9a-f]{32}\Z")
_POLICY = {
    "full_hash": "startup and controlled shutdown",
    "per_job": "file metadata, tensor storage/shape/dtype/version/eval mode and precision guards; all input evidence remains hashed",
    "model_mutation_endpoints": False,
    "full_checkpoint_hash_per_job": False,
}
_REQUIRED_SOURCES = {"aureon/scene_api/worker.py", "aureon/scene_api/identity.py",
                     "aureon/scene_api/audit.py"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return isinstance(value, str) and bool(_SHA.fullmatch(value))


def _nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _resident_checks(folder, base_report):
    job_path = scene_audit._member(folder, "job.json")
    job_digest = scene_audit._sha(job_path)
    _require(base_report.get("job_receipt_sha256") == job_digest,
             "Job receipt changed after the independent scene audit")
    job = scene_audit._json(job_path)
    launch = scene_audit._json(scene_audit._member(folder, "launch.json"))
    manifest = scene_audit._json(scene_audit._member(folder, "input-manifest.json"))
    startup_path = scene_audit._member(folder, "worker-startup.json")
    startup_digest = scene_audit._sha(startup_path)
    startup = scene_audit._json(startup_path)
    worker = job["worker"]
    provenance = job["artifacts"]["provenance"]
    provenance_worker = provenance["worker"]
    _require(startup.get("schema") == "aureon.resident_scene_worker.v1"
             and startup.get("status") == "ready", "Worker startup must record ready state")
    _require(job["scope"].get("job_kind") == "resident_local_async_scene_job",
             "Expected a resident scene job")
    _require(job["files_sha256"].get("worker-startup.json") == startup_digest
             == worker.get("worker_startup_receipt_sha256")
             == provenance_worker.get("worker_startup_receipt_sha256"),
             "Worker startup file digest linkage differs")
    worker_id = startup.get("worker_id")
    _require(isinstance(worker_id, str) and _ID.fullmatch(worker_id)
             and worker_id == worker.get("worker_id") == provenance_worker.get("worker_id")
             == launch.get("worker_id"), "Resident worker IDs disagree")
    for record in (startup, worker, provenance_worker):
        _require(type(record.get("model_load_count")) is int and record["model_load_count"] == 1,
                 "Resident receipt must identify exactly one model load")
    _require(startup.get("model_name") == job.get("model_name")
             == provenance.get("model_name") == "KAO-TerraMind B"
             and provenance.get("model_id") == "kao-terramind-b", "Unexpected resident model")
    _require(_digest(startup.get("bundle_receipt_sha256"))
             and startup["bundle_receipt_sha256"] == job["bundle_receipt_sha256"]
             == launch["bundle_receipt_sha256"] == provenance["bundle_receipt_sha256"],
             "Resident bundle receipt declarations disagree")
    bundle = startup.get("bundle_files_sha256")
    _require(isinstance(bundle, dict) and set(bundle) == {"config.json", "pytorch_model.bin"}
             and all(_digest(value) for value in bundle.values())
             and bundle == provenance.get("bundle_files_sha256"),
             "Resident bundle member declarations disagree")
    sources = job["source_sha256"]
    _require(isinstance(sources, dict) and _REQUIRED_SOURCES <= sources.keys()
             and all(_digest(value) for value in sources.values())
             and sources == startup.get("code_sha256") == launch.get("source_sha256")
             == provenance.get("code_sha256"), "Resident code identity declarations disagree")
    scene_id = job.get("scene_id")
    registered = startup.get("registered_scenes")
    _require(isinstance(registered, dict) and 1 <= len(registered) <= 16
             and all(isinstance(key, str) and key and _digest(value)
                     for key, value in registered.items())
             and isinstance(scene_id, str) and scene_id == manifest.get("scene_id")
             == provenance.get("scene_id")
             and registered.get(scene_id) == job["source_manifest_sha256"],
             "Job scene was not admitted by this worker startup")
    precision = startup.get("precision")
    _require(isinstance(precision, dict) and precision == job.get("precision")
             == provenance.get("precision_actual"), "Resident precision declarations disagree")
    _require(precision.get("float32_matmul_precision") == "highest"
             and all(precision.get(key) is False for key in
                     ("matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_sdp_enabled", "cudnn_benchmark")),
             "Resident receipt violates highest-FP32 precision contract")
    policy = startup.get("identity_policy")
    _require(isinstance(policy, dict) and policy == worker.get("identity_policy")
             == provenance_worker.get("identity_policy"), "Resident identity policies disagree")
    _require(all(type(policy.get(key)) is type(value) and policy[key] == value
                 for key, value in _POLICY.items())
             and isinstance(policy.get("scope"), str) and policy["scope"].strip()
             and worker.get("full_checkpoint_hash_per_job") is False
             and provenance_worker.get("full_checkpoint_hash_per_job") is False,
             "Unsupported resident guarded-identity policy")
    _require(_nonnegative(worker.get("identity_guard_seconds")), "Missing resident guard accounting")
    for name in ("startup_parity", "reversed_input_probe"):
        gate = startup.get(name, {})
        _require(gate.get("passed") is True and gate.get("allclose") is True
                 and gate.get("atol") == 1e-5 and gate.get("rtol") == 1e-5
                 and gate.get("class_agreement") == 1.0
                 and type(gate.get("tile_count")) is int and 1 <= gate["tile_count"] <= 8,
                 "Startup numerical/freshness gate is not the fixed accepted gate: " + name)
    # Do not follow the absolute file paths in the historical metadata seal.
    # The independent scene audit rehashes the portable source snapshots.
    seals = startup.get("file_seal")
    expected = set(sources) | {"bundle/" + name for name in (*bundle, "bundle_receipt.json")}
    _require(isinstance(seals, dict) and len(seals) == len(expected),
             "Resident file seal has incomplete identity-member coverage")
    observed = set()
    for path, stamp in seals.items():
        _require(isinstance(path, str) and isinstance(stamp, list) and len(stamp) == 6
                 and all(type(value) is int and value >= 0 for value in stamp)
                 and stat.S_ISREG(stamp[5]), "Invalid resident regular-file metadata seal")
        portable = path.replace("\\", "/")
        matches = [name for name in expected if portable.endswith("/" + name)]
        _require(len(matches) == 1 and matches[0] not in observed,
                 "Unexpected or duplicate resident identity member")
        member = matches[0]
        observed.add(member)
        if member in sources:
            snapshot = scene_audit._member(folder / "source-snapshot", member)
            _require(snapshot.stat().st_size == stamp[2], "Source snapshot size differs from startup file seal")
    _require(observed == expected, "Resident file seal omits a required member")
    startup_stages = startup.get("startup_seconds")
    _require(isinstance(startup_stages, dict) and startup_stages
             and all(_nonnegative(value) for value in startup_stages.values()),
             "Invalid resident startup accounting")
    total = startup.get("startup_observed_seconds_before_receipt")
    _require(_nonnegative(total) and sum(startup_stages.values()) <= total + 1e-5,
             "Resident startup stages exceed observed startup interval")
    _require(scene_audit._sha(startup_path) == startup_digest
             and scene_audit._sha(job_path) == job_digest, "Resident receipts changed during audit")
    return {"worker_id": worker_id, "worker_startup_receipt_sha256": startup_digest,
            "model_load_count": 1, "scene_id": scene_id, "registered_scene_verified": True,
            "code_and_bundle_declarations_linked": True, "precision_declarations_verified": True,
            "file_metadata_seal_members": len(seals), "identity_policy": policy,
            "startup_gates": "Declared startup and reversed-input gates checked; startup tensors are not replayed",
            "startup_observed_seconds_before_receipt": total,
            "live_tensor_values_verified": False, "checkpoint_bytes_rehashed_by_this_audit": False,
            "controlled_shutdown_verified": False, "remote_attestation": False}


def audit_job(job_dir, output_path):
    folder, output = Path(job_dir).resolve(strict=True), Path(output_path).absolute()
    if not folder.is_dir() or not output.parent.is_dir():
        raise ValueError("Existing job directory and output parent required")
    try:
        schema=scene_audit._json(scene_audit._member(folder,'job.json')).get('schema')
    except Exception:
        schema=None  # The legacy verifier records malformed/missing evidence.
    if schema=='aureon.streaming_scene_job.v1':
        from ..scene_pipeline.stream_audit import audit_streaming_job
        if output.exists(): raise FileExistsError('Audit output must be fresh')
        try:
            return audit_streaming_job(folder,output)
        except Exception as error:
            report={'schema':'aureon.streaming_job_audit.v1','status':'failed',
                    'error_type':type(error).__name__,'error':str(error),'model_executed':False}
            with output.open('x') as stream:json.dump(report,stream,indent=2)
            return report
    with output.open("x", encoding="utf-8") as stream:
        report = {"schema": "aureon.resident_scene_job_audit.v1", "job_directory": str(folder),
                  "created_utc": datetime.now(timezone.utc).isoformat(), "model_executed": False,
                  "task_accuracy_evaluated": False, "speedup_measured": False,
                  "audit_source_sha256": scene_audit._sha(__file__)}
        try:
            with tempfile.TemporaryDirectory(prefix=".aureon-scene-audit-", dir=output.parent) as temporary:
                base = scene_audit.audit_job(folder, Path(temporary) / "scene-audit.json")
            report["scene_pipeline_audit"] = base
            _require(base.get("status") == "passed", "Independent scene pipeline audit failed: " + str(base.get("error")))
            report["checks"] = _resident_checks(folder, base)
            report["job_receipt_sha256"] = base["job_receipt_sha256"]
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline CPU audit of a completed resident AUREON scene job")
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit_job(args.job, args.output)
    print(json.dumps({"status": report["status"], "audit": str(Path(args.output).absolute())}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
