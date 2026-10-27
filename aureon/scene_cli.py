"""Explicit local scene CLI, including bounded streaming through the resident worker."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter
import uuid


def run_streaming_scene(manifest, bundle, output, *, execute_model=False, model_id="kao-terramind-b",
                        hourly_rate=1.0, currency="THB", min_free_bytes=2*1024**3):
    if execute_model is not True or sys.platform != "linux":
        raise ValueError("Streaming inference requires explicit --execute-model in Linux/WSL")
    if type(min_free_bytes) is not int or min_free_bytes < 0:
        raise ValueError("min_free_bytes must be a nonnegative integer")
    if not math.isfinite(hourly_rate) or hourly_rate < 0:
        raise ValueError("Hourly rate must be finite and nonnegative")
    from .prepare import check_scene
    from .scene_pipeline.stream_io import WindowSource
    from .scene_pipeline.streaming import budget
    from .scene_pipeline.job import write_json, file_hash
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("Output must be fresh with an existing parent")
    output = output.parent.resolve()/output.name
    manifest, bundle = Path(manifest).resolve(strict=True), Path(bundle).resolve(strict=True)
    started = perf_counter()
    checked = check_scene(manifest)
    with WindowSource(manifest) as source:
        allocation = budget(source, output)
    # A private worker bundle also resides in this output. This check reserves
    # the conservative existing streaming plan without relaxing its 8 GiB cap.
    bundle_bytes = sum((bundle/name).stat().st_size for name in ("config.json", "pytorch_model.bin", "bundle_receipt.json"))
    reservation = allocation["scratch_estimate_bytes"]+bundle_bytes
    if shutil.disk_usage(output.parent).free < min_free_bytes+reservation:
        raise ValueError("Insufficient free storage after reserving scene scratch and the private model bundle")
    from .scene_api.worker import ResidentSceneWorker
    output.mkdir()
    worker = None
    job_id = uuid.uuid4().hex
    launch = {"schema": "aureon.streaming_scene_cli.v1", "status": "running", "model_id": model_id,
              "scene_id": checked["scene_id"], "job_id": job_id, "input_preflight": checked,
              "input_preflight_seconds": perf_counter()-started, "source_sha256": file_hash(__file__),
              "created_utc": datetime.now(timezone.utc).isoformat(), "min_free_bytes": min_free_bytes,
              "storage_reservation_bytes": reservation, "memory_plan": allocation,
              "cost_scenario": {"hourly_rate": hourly_rate, "currency": currency},
              "reference_policy": "full_reference", "public_deployment": False}
    write_json(output/"cli-launch.json", launch)
    try:
        worker = ResidentSceneWorker(bundle, {checked["scene_id"]: manifest}, output/"worker",
            execute_model=True, streaming=True, model_id=model_id, hourly_rate=hourly_rate, currency=currency)
        completed = worker.run_job(checked["scene_id"], job_id, output/"job")
        cleanup = worker.close()
        worker = None
        if not cleanup.get("cleanup_passed") or not cleanup.get("full_identity_check_passed") or cleanup.get("errors"):
            raise RuntimeError("Worker cleanup/identity verification failed")
        write_json(output/"worker-cleanup.json", cleanup)
        storage_path = output/"worker/bundle-retirement.json"
        if not storage_path.exists():
            storage_path = output/"worker/bundle-retention.json"
        storage = json.loads(storage_path.read_text(encoding="utf-8"))
        seconds = perf_counter()-started
        result = {"schema": "aureon.streaming_scene_cli_result.v1", "status": "completed",
                  "job_id": job_id, "scene_id": checked["scene_id"], "job_directory": str(output/"job"),
                  "job_receipt": str(output/"job/job.json"), "parity": completed["summary"],
                  "class_map": str(output/"job/delivery/artifacts/class_map.tif"),
                  "valid_mask": str(output/"job/delivery/artifacts/valid_mask.tif"),
                  "provenance": str(output/"job/delivery/artifacts/receipt.json"),
                  "elapsed_seconds": seconds, "whole_machine_scenario_cost": seconds/3600*hourly_rate,
                  "currency": currency, "timing_scope": "Input preflight, startup, full reference, delivery and shutdown",
                  "cleanup_passed": True, "task_accuracy_evaluated": False,
                  "bundle_storage": {"status": storage["status"], "receipt": str(storage_path),
                      "storage_retirement_seconds": storage.get("storage_retirement_seconds")}}
        write_json(output/"cli-result.json", result)
        return result
    except BaseException as error:
        cleanup = None
        if worker is not None:
            try:
                cleanup = worker.close()
            except BaseException as cleanup_error:
                cleanup = {"cleanup_passed": False, "error": str(cleanup_error)}
        write_json(output/"cli-failure.json", {"status": "failed", "error": str(error), "cleanup": cleanup,
                                              "partial_artifacts_not_publishable": True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explicit FP32 KAO-TerraMind scene inference; no acquisition")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute-model", action="store_true")
    parser.add_argument("--streaming", action="store_true", help="Bounded 1..4096-pixel scene with full FP32 reference")
    parser.add_argument("--model", choices=("kao-terramind-a", "kao-terramind-b", "kao-terramind-c"), default="kao-terramind-b")
    parser.add_argument("--hourly-rate", type=float, default=1.0)
    parser.add_argument("--currency", default="THB")
    parser.add_argument("--min-free-bytes", type=int, default=2*1024**3, help="Streaming storage floor after scratch reservation")
    args = parser.parse_args(argv)
    if not args.execute_model or sys.platform != "linux":
        parser.error("Pass --execute-model in the supported Linux/WSL environment")
    try:
        if args.streaming:
            result = run_streaming_scene(args.manifest, args.bundle, args.output, execute_model=True,
                model_id=args.model, hourly_rate=args.hourly_rate, currency=args.currency, min_free_bytes=args.min_free_bytes)
        else:
            if args.model != "kao-terramind-b":
                parser.error("KAO-TerraMind A/C require --streaming")
            from .scene_pipeline.job import run_job
            receipt = run_job(args.manifest, args.bundle, args.output, execute_model=True,
                              hourly_rate=args.hourly_rate, currency=args.currency)
            result = {"job": str(Path(args.output).resolve()/"job.json"), "status": receipt["status"],
                      "parity_passed": receipt["parity"]["passed"], "warm_job_seconds": receipt["warm_job_seconds"],
                      "scenario_cost": receipt["cost"]["warm_job_cost"], "currency": receipt["cost"]["currency"]}
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"Scene inference failed: {error}\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
