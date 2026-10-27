"""Independent CPU audit of a completed scene job; never imports a model runtime."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {"launch.json", "input-manifest.json", "tile-plan.json", "normalized_tiles.npy",
            "eager_logits.npy", "candidate_logits.npy", "eager_classes.npy", "groups.json",
            "parity.json", "artifacts/receipt.json", "artifacts/class_map.tif", "artifacts/valid_mask.tif"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _member(root, name):
    if (not isinstance(name, str) or "\\" in name or not name
            or PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts):
        raise ValueError("Evidence path must remain inside its recorded directory")
    path = root / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Missing or nonlocal evidence member: " + name)
    return path


def _json(path):
    path = Path(path)
    if not 0 < path.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError("JSON evidence must be bounded")
    return json.loads(path.read_text(encoding="utf-8"))


def _declared_source(path):
    """Translate only an explicitly recorded Windows/WSL drive path."""
    if not isinstance(path, str) or not path:
        raise ValueError("Missing declared source path")
    if sys.platform == "win32" and re.match(r"^/mnt/[a-zA-Z]/", path):
        path = path[5].upper() + ":/" + path[7:]
    elif sys.platform != "win32" and PureWindowsPath(path).is_absolute():
        windows = PureWindowsPath(path)
        path = "/mnt/" + windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])
    result = Path(path)
    if not result.is_absolute() or not result.is_file():
        raise ValueError("Declared original source is unavailable locally: " + str(result))
    return result


def _assert(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    _assert(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0, "Expected finite nonnegative accounting value")
    return float(value)


def _counter(value):
    _assert(type(value) is int and value >= 0, "Expected nonnegative integer execution counter")
    return value


def _audit_group_accounting(job, groups, provenance, tile_count):
    """Verify declared dispatch and logical forwards without importing a runtime.

    Historical receipts without adapter metadata retain fixed-eight padding.
    Zero-padding tails require the named adapter, its explicit dispatch policy,
    and matching provenance; small counters alone never select that contract.
    """
    _assert(1 <= _counter(tile_count) <= 36, "Invalid scene tile count")
    _assert(isinstance(groups, list), "Group evidence must be a list")
    expected_groups = [(start, min(8, tile_count-start)) for start in range(0, tile_count, 8)]
    _assert(len(groups) == len(expected_groups) == _counter(job["groups"]), "Group count mismatch")
    has_before, has_after = "adapter_before_job" in job, "adapter_after_job" in job
    _assert(has_before == has_after, "Both adapter accounting snapshots are required")
    before, after = job.get("adapter_before_job"), job.get("adapter_after_job")
    if has_before:
        _assert(isinstance(before, dict) and isinstance(after, dict), "Invalid adapter accounting snapshots")
    tail_name = "FullEightEagerTailAdapter"
    tail_policy = "explicit ordered eager batch-one tail; zero padding forwards"
    recipe = provenance.get("recipe", {})
    _assert(isinstance(recipe, dict), "Invalid provenance recipe")
    named_tail = any(isinstance(item, dict) and item.get("adapter") == tail_name for item in (before, after))
    unpadded = named_tail or recipe.get("last_group") == tail_policy
    if unpadded:
        _assert(has_before and before.get("adapter") == after.get("adapter") == tail_name,
                "Unpadded dispatch requires the recognized adapter before and after the job")
        _assert(recipe.get("last_group") == tail_policy, "Unpadded dispatch provenance policy mismatch")
        dispatch = "count==8: fixed-eight graph; count in 1..7: ordered eager batch-one"
        for metadata in (before, after):
            _assert(metadata.get("schema") == "aureon.scene_predictor.v1"
                    and metadata.get("capacity") == 8 and type(metadata.get("capacity")) is int
                    and metadata.get("state") == "ready"
                    and metadata.get("dispatch_policy") == dispatch,
                    "Unpadded adapter dispatch metadata mismatch")
    padding = 0 if unpadded else 8 * len(groups) - tile_count
    for group, (start, count) in zip(groups, expected_groups):
        _assert(isinstance(group, dict), "Invalid group evidence")
        expected_padding = 0 if unpadded else 8-count
        _assert(tuple(_counter(group[key]) for key in ("first_tile", "real_tiles", "padding_tiles"))
                == (start, count, expected_padding), "Group order/padding mismatch")
        if unpadded or "neural_forwards" in group:
            _assert(_counter(group.get("neural_forwards")) == count + expected_padding,
                    "Group neural forward count mismatch")
        _number(group["elapsed_seconds"])
    _assert(_counter(job["padding_forwards_in_job"]) == padding, "Job padding count mismatch")
    if has_before:
        expected = {"calls_completed": len(groups), "real_tiles_completed": tile_count,
                    "padding_outputs_discarded": padding}
        if unpadded:
            graph_groups, tail_tiles = divmod(tile_count, 8)
            tail_groups = int(tail_tiles > 0)
            expected.update({
                "calls_attempted": len(groups), "real_tiles_submitted": tile_count,
                "padding_slots_submitted": 0,
                "graph_groups_attempted": graph_groups, "graph_groups_completed": graph_groups,
                "tail_groups_attempted": tail_groups, "tail_groups_completed": tail_groups,
                "eager_tile_calls_attempted": tail_tiles, "eager_tile_calls_completed": tail_tiles,
                "backend_calls_attempted": graph_groups + tail_tiles,
                "backend_calls_completed": graph_groups + tail_tiles,
                "neural_tile_forwards_submitted": tile_count,
                "neural_tile_forwards_completed": tile_count,
            })
        for key, delta in expected.items():
            _assert(_counter(after.get(key)) - _counter(before.get(key)) == delta,
                    "Adapter accounting mismatch: " + key)
    return padding


def _near(actual, expected, message, *, abs_tol=1e-9):
    _assert(math.isclose(_number(actual), _number(expected), rel_tol=1e-9, abs_tol=abs_tol), message)


def _parity(eager, candidate):
    _assert(eager.shape == candidate.shape and eager.dtype == candidate.dtype == np.float32
            and eager.ndim == 4 and 1 <= len(eager) <= 36
            and eager.shape[1:] == (10, 224, 224), "Invalid saved logit arrays")
    maximum, total, matching, allclose = 0.0, 0.0, 0, True
    for baseline, proposed in zip(eager, candidate):
        _assert(np.isfinite(baseline).all() and np.isfinite(proposed).all(), "Nonfinite saved logits")
        base64 = baseline.astype(np.float64)
        error = np.abs(proposed.astype(np.float64) - base64)
        maximum = max(maximum, float(error.max()))
        total += float(error.sum())
        allclose &= bool(np.all(error <= 1e-5 + 1e-5 * np.abs(base64)))
        matching += int(np.count_nonzero(baseline.argmax(0) == proposed.argmax(0)))
    agreement = matching / (len(eager) * 224 * 224)
    return {"atol": 1e-5, "rtol": 1e-5, "allclose": allclose,
            "class_agreement": agreement, "max_abs_error": maximum,
            "mean_abs_error": total / eager.size, "passed": allclose and agreement == 1.0,
            "bitwise_equal": eager.tobytes() == candidate.tobytes(), "tile_count": len(eager)}


def _inspect(job_dir):
    import rasterio
    from ..recipe import INPUT_UNITS, S2_BANDS, segment_tiled, tile_anchors
    from .georaster import read_s2_geotiff

    job = _json(_member(job_dir, "job.json"))
    _assert(job.get("schema") == "aureon.scene_job.v1" and job.get("status") == "completed",
            "Only completed scene jobs can be audited")
    hashes = job["files_sha256"]
    _assert(isinstance(hashes, dict) and REQUIRED <= hashes.keys(), "Required job evidence is missing")
    for name, digest in hashes.items():
        _assert(_sha(_member(job_dir, name)) == digest, "Artifact hash mismatch: " + name)
    source_hashes = job["source_sha256"]
    _assert(isinstance(source_hashes, dict) and "aureon/recipe.py" in source_hashes, "Missing recipe source identity")
    for name, digest in source_hashes.items():
        _assert(_sha(_member(job_dir / "source-snapshot", name)) == digest, "Source snapshot mismatch: " + name)
    _assert(_sha(ROOT / "aureon/recipe.py") == source_hashes["aureon/recipe.py"],
            "Current recipe differs from the job snapshot; do not execute untrusted snapshot code")
    launch = _json(_member(job_dir, "launch.json"))
    manifest = _json(_member(job_dir, "input-manifest.json"))
    delivery = _json(_member(job_dir, "artifacts/receipt.json"))
    provenance = delivery["provenance"]
    _assert(delivery == job["artifacts"] and delivery.get("status") == "completed"
            and delivery.get("format") == "COG", "Embedded delivery receipt mismatch")
    _assert(_canonical_hash(provenance) == delivery["provenance_sha256"], "Provenance digest mismatch")
    _assert(launch["job_id"] == provenance["job_id"] == job["job_id"], "Job ID mismatch")
    _assert(launch["manifest_sha256"] == provenance["source_manifest_sha256"] == job["source_manifest_sha256"],
            "Original manifest digest declarations disagree")
    _assert(_sha(job_dir / "input-manifest.json") == job["source_manifest_sha256"],
            "Original manifest raw bytes differ from the recorded digest")
    _assert(launch["source_sha256"] == provenance["code_sha256"] == source_hashes, "Code identity declarations disagree")
    _assert(launch["bundle_receipt_sha256"] == provenance["bundle_receipt_sha256"] == job["bundle_receipt_sha256"],
            "Checkpoint identity declarations disagree")
    _assert(manifest["raster"]["bands"] == list(S2_BANDS)
            and manifest["raster"]["units"] == INPUT_UNITS, "Input modality declaration mismatch")
    identity = delivery["source"]
    identity_body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    _assert(_canonical_hash(identity_body) == identity["identity_sha256"], "Source identity digest mismatch")
    raster_path = _declared_source(identity["source"]["path"])
    mask_record = identity["additional_valid_mask"]
    _assert(isinstance(mask_record, dict), "Scene job must retain its declared validity mask")
    mask_path = _declared_source(mask_record["path"])
    _assert(_sha(raster_path) == identity["source"]["sha256"] == manifest["raster"]["sha256"], "Original raster digest mismatch")
    _assert(_sha(mask_path) == mask_record["sha256"] == manifest["valid_mask"]["sha256"], "Original mask digest mismatch")
    relative_raster = PurePosixPath(manifest["raster"]["path"])
    _assert(not relative_raster.is_absolute() and ".." not in relative_raster.parts
            and "\\" not in str(relative_raster), "Invalid original manifest raster path")
    manifest_root = raster_path.parents[len(relative_raster.parts)-1]
    _assert(_member(manifest_root, str(relative_raster)).resolve() == raster_path.resolve(),
            "Declared raster cannot locate the original manifest directory")
    declarations = []
    def collect(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                declarations.append(value)
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)
    collect(manifest)
    _assert(len(declarations) <= 64, "Too many original source evidence members")
    source_bytes = 0
    for declaration in declarations:
        member = _member(manifest_root, declaration["path"])
        source_bytes += member.stat().st_size
        _assert(source_bytes <= 128*1024*1024 and _sha(member) == declaration["sha256"],
                "Original source metadata/crop evidence mismatch")
        if "bytes" in declaration:
            _assert(declaration["bytes"] == member.stat().st_size, "Original source evidence size mismatch")
    source = read_s2_geotiff(raster_path, bands=S2_BANDS, units=INPUT_UNITS,
        expected_crs=manifest["raster"]["crs"], mask_policy="all_bands_valid", valid_mask_path=mask_path)
    for key in ("image", "valid_mask", "grid_sha256", "valid_pixels", "total_pixels"):
        _assert(source.input_identity[key] == identity[key], "Reopened source identity differs: " + key)
    _assert(delivery["grid"] == identity["grid"] == source.metadata, "Source grid identity differs")
    arrays = {name: np.load(_member(job_dir, name + ".npy"), mmap_mode="r", allow_pickle=False)
              for name in ("normalized_tiles", "eager_logits", "candidate_logits", "eager_classes")}
    tiles, eager, candidate = (arrays[name] for name in ("normalized_tiles", "eager_logits", "candidate_logits"))
    _assert(tiles.dtype == np.float32 and tiles.shape == (len(eager), 12, 224, 224)
            and np.isfinite(tiles).all(), "Saved normalized tiles invalid")
    independent_parity = _parity(eager, candidate)
    _assert(independent_parity["passed"], "Independent fixed FP32 numerical/class gate failed")
    recorded_parity = _json(_member(job_dir, "parity.json"))
    _assert(recorded_parity == job["parity"], "Embedded parity receipt mismatch")
    _assert(_sha(job_dir / "parity.json") == provenance["parity_receipt_sha256"], "Parity provenance digest mismatch")
    for key, value in independent_parity.items():
        _assert(recorded_parity[key] == value, "Recorded numerical result differs: " + key)
    plan = _json(_member(job_dir, "tile-plan.json"))
    windows = [(r, c, min(r + 224, source.valid_mask.shape[0]), min(c + 224, source.valid_mask.shape[1]))
               for r in tile_anchors(source.valid_mask.shape[0]) for c in tile_anchors(source.valid_mask.shape[1])
               if source.valid_mask[r:r+224, c:c+224].any()]
    _assert(plan["windows"] == [list(window) for window in windows] and len(windows) == len(tiles), "Tile plan differs")
    _assert(hashlib.sha256(tiles.tobytes(order="C")).hexdigest() == plan["input_array_sha256"], "Normalized array digest mismatch")

    def replay(logits):
        position = 0
        def saved(values):
            nonlocal position
            end = position + len(values)
            _assert(np.array_equal(values, tiles[position:end]), "Independent recipe normalization or tile order differs")
            result = np.array(logits[position:end], copy=True)
            position = end
            return result
        result = segment_tiled(source.image, source.valid_mask, saved, bands=S2_BANDS,
                               units=INPUT_UNITS, batch_size=1, include_max_score=False)
        _assert(position == len(tiles), "Recipe replay omitted saved tiles")
        return result

    eager_result, candidate_result = replay(eager), replay(candidate)
    _assert(arrays["eager_classes"].dtype == np.uint8
            and np.array_equal(eager_result.class_map, arrays["eager_classes"])
            and np.array_equal(candidate_result.class_map, eager_result.class_map), "Independent stitched class maps differ")
    for name, expected, nodata in (("class_map.tif", candidate_result.class_map, 255),
                                  ("valid_mask.tif", source.valid_mask.astype(np.uint8), 0)):
        path = _member(job_dir, "artifacts/" + name)
        _assert(_sha(path) == delivery["files"][name]["sha256"], "COG delivery digest mismatch")
        with rasterio.open(path) as dataset:
            _assert(dataset.driver == "GTiff" and dataset.count == 1 and dataset.dtypes == ("uint8",)
                    and dataset.crs == rasterio.crs.CRS.from_user_input(source.metadata["crs"])
                    and tuple(dataset.transform)[:6] == tuple(source.metadata["transform"])
                    and dataset.nodata == nodata and np.array_equal(dataset.read(1), expected)
                    and np.array_equal(dataset.read_masks(1) > 0, source.valid_mask)
                    and dataset.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG", "COG values/grid/mask/layout differ: " + name)
            _assert(dataset.tags().get("AUREON_PROVENANCE_SHA256") == delivery["provenance_sha256"]
                    and dataset.tags().get("AUREON_INPUT_ID") == identity["identity_sha256"], "COG identity tags differ")
    groups = _json(_member(job_dir, "groups.json"))
    padding = _audit_group_accounting(job, groups, provenance, len(tiles))
    _assert(job["tile_count"] == plan["executed"] == len(tiles)
            and job["tiles_total"] == plan["total"] == eager_result.tiles_total
            and job["tiles_skipped"] == plan["skipped"] == eager_result.tiles_skipped
            and job["padding_forwards_in_job"] == padding, "Tile accounting mismatch")
    stages = job["stages_seconds"]
    total_stages = sum(_number(value) for value in stages.values())
    # Microsecond timer/residual rounding only; not a performance assertion.
    _near(total_stages, job["warm_job_seconds"], "Warm stage sum mismatch", abs_tol=1e-5)
    _near(sum(group["elapsed_seconds"] for group in groups), stages["predict_seconds"], "Group timing sum mismatch")
    crs = rasterio.crs.CRS.from_user_input(source.metadata["crs"])
    _assert(crs.is_projected, "Area accounting requires projected CRS")
    a, b, _, d, e, _ = source.metadata["transform"]
    area = int(source.valid_mask.sum()) * abs(a*e-b*d) * crs.linear_units_factor[1]**2 / 1e6
    _near(area, job["area"]["valid_area_km2"], "Valid area mismatch")
    cost = job["cost"]
    rate, warm = _number(cost["hourly_rate"]), _number(job["warm_job_seconds"])
    startup = sum(_number(value) for value in job["startup"].values())
    validation = sum(_number(value) for value in job["validation_overhead"].values())
    for key, expected in {"warm_job_seconds": warm, "cold_start_seconds": startup,
        "validation_overhead_seconds": validation, "valid_area_km2": area,
        "warm_job_cost": warm*rate/3600, "cold_start_cost": startup*rate/3600,
        "cold_job_cost": (warm+startup)*rate/3600, "validation_overhead_cost": validation*rate/3600}.items():
        _near(cost[key], expected, "Cost arithmetic mismatch: " + key)
    if area:
        _near(cost["warm_cost_per_valid_km2"], warm*rate/3600/area, "Per-area cost mismatch")
    else:
        _assert(cost["warm_cost_per_valid_km2"] is None, "Zero area must not have per-area cost")
    _assert(cost["actual_bill"] is False and job["scope"]["task_accuracy_evaluated"] is False
            and job["scope"]["repeated_speedup_comparison"] is False, "Unsupported quality/speed/billing declaration")
    return {"job_id": job["job_id"], "artifact_files_rehashed": len(hashes),
            "source_snapshot_files_rehashed": len(source_hashes), "parity": independent_parity,
            "original_source_evidence_files_rehashed": len(declarations),
            "recipe_replay": "eager and candidate logits independently replayed through unchanged recipe",
            "cog_values_grid_mask_layout_verified": True, "tile_count": len(tiles),
            "groups": len(groups), "padding_forwards": padding, "stage_sum_seconds": total_stages,
            "valid_area_km2": area, "recorded_cost_verified": True,
            "one_thb_per_hour_scenario": {"hourly_rate": 1.0, "currency": "THB",
                "warm_job_cost": warm/3600, "cold_job_cost": (warm+startup)/3600,
                "warm_cost_per_valid_km2": warm/3600/area if area else None, "actual_bill": False},
            "original_manifest_scope": "Exact original manifest raw bytes rehashed and cross-checked among launch, job and delivery receipts."}


def audit_job(job_dir, output_path):
    job_dir, output_path = Path(job_dir).resolve(strict=True), Path(output_path).absolute()
    if not job_dir.is_dir() or not output_path.parent.is_dir():
        raise ValueError("Existing job directory and output parent required")
    # Reserve the result exclusively before doing any expensive audit work.
    with output_path.open("x", encoding="utf-8") as stream:
        report = {"schema": "aureon.scene_job_audit.v1", "job_directory": str(job_dir),
                  "created_utc": datetime.now(timezone.utc).isoformat(), "model_executed": False,
                  "task_accuracy_evaluated": False, "speedup_measured": False,
                  "audit_source_sha256": _sha(__file__)}
        try:
            report["job_receipt_sha256"] = _sha(_member(job_dir, "job.json"))
            report["checks"] = _inspect(job_dir)
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline CPU audit of a completed AUREON scene job")
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit_job(args.job, args.output)
    print(json.dumps({"status": report["status"], "audit": str(Path(args.output).absolute())}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
