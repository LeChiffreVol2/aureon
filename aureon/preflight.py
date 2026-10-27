"""CPU-only environment, bundle and prepared-scene checks; never loads a model."""
import argparse
from importlib import metadata
import json
import platform
import sys


PROFILES = {
    "torch214-cu130": {"numpy": "2.5.4", "torch": "2.14.1", "torchvision": "0.29.1",
                       "terratorch": "1.2.13", "torchgeo": "0.9.0", "rasterio": "1.5.2"},
    "torch210-cu128": {"numpy": "2.2.6", "torch": "2.10.0", "torchvision": "0.25.0",
                       "terratorch": "1.2.13", "torchgeo": "0.9.0", "rasterio": "1.5.2"},
}
DEFAULT_PROFILE = "torch214-cu130"


def inspect_environment(profile=DEFAULT_PROFILE):
    if profile not in PROFILES:
        raise ValueError("Unknown environment profile: " + str(profile))
    checks = [{"name": "platform", "passed": sys.platform == "linux", "actual": sys.platform, "required": "linux"},
              {"name": "python", "passed": sys.version_info[:2] == (3, 12),
               "actual": platform.python_version(), "required": "3.12.x"}]
    for name, version in PROFILES[profile].items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        checks.append({"name": name, "actual": actual, "required": version,
                       "passed": actual is not None and actual.split("+")[0] == version})
    return {"profile": profile, "passed": all(c["passed"] for c in checks), "checks": checks,
            "cuda_status": "not_probed; verified during explicit model startup",
            "dependency_resolution": "not_tested; metadata versions are not fresh-install evidence",
            "model_imported_or_executed": False}


def preflight(*, manifest=None, bundle=None, environment=True, profile=DEFAULT_PROFILE):
    result = {"schema": "aureon.preflight.v1", "status": "passed", "model_executed": False,
              "network_access": False}
    if environment:
        result["environment"] = inspect_environment(profile)
        if not result["environment"]["passed"]:
            result["status"] = "failed"
    if manifest is not None:
        from .prepare import check_scene
        result["scene"] = check_scene(manifest)
    if bundle is not None:
        from .runtime import verify_bundle
        verified = verify_bundle(bundle)
        result["bundle"] = {"model_id": verified["model_id"], "bundle_receipt_sha256": verified["receipt_sha256"],
                            "files_sha256": verified["files_sha256"], "loaded_model_attested": False}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Optional prepared local manifest; checks every bounded pixel window")
    parser.add_argument("--bundle", help="Optional existing serving bundle; verifies file hashes on CPU")
    parser.add_argument("--input-only", action="store_true", help="Skip Linux model-environment checks")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=DEFAULT_PROFILE,
                        help="Exact dependency profile; CUDA and live model identity require explicit startup")
    args = parser.parse_args(argv)
    if args.input_only and not (args.manifest or args.bundle):
        parser.error("--input-only requires --manifest or --bundle")
    try:
        result = preflight(manifest=args.manifest, bundle=args.bundle, environment=not args.input_only, profile=args.profile)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Preflight failed: {error}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
