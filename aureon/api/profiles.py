"""Dependency profiles are metadata contracts, separate from execution profiles."""
from importlib import metadata
import platform
import sys

PROFILES = {
    "scene": {"numpy":"2.5.4", "torch":"2.14.1", "torchvision":"0.29.1",
              "terratorch":"1.2.13", "torchgeo":"0.9.0", "rasterio":"1.5.2"},
    "scene-legacy": {"numpy":"2.2.6", "torch":"2.10.0", "torchvision":"0.25.0",
                     "terratorch":"1.2.13", "torchgeo":"0.9.0", "rasterio":"1.5.2"},
    "vllm": {"numpy":"2.2.6", "torch":"2.10.0", "torchvision":"0.25.0",
             "terratorch":"1.2.13", "rasterio":"1.5.2", "vllm":"0.19.0"},
    "generation": {"numpy":"2.2.6", "torch":"2.10.0", "torchvision":"0.25.0",
                   "terratorch":"1.2.13", "rasterio":"1.5.2", "diffusers":"0.39.0",
                   "einops":"0.8.2", "huggingface-hub":"0.36.2", "timm":"1.0.30",
                   "transformers":"4.57.6"},
}


class ProfileError(ValueError):
    """Unsupported profile or mismatched exact dependency pins."""


def inspect_profile(name):
    if name not in PROFILES:
        raise ProfileError("Choose exactly one environment: " + ", ".join(PROFILES) + "; use separate environments")
    checks = [{"name":"python", "actual":platform.python_version(), "required":"3.12.x",
               "passed":sys.version_info[:2] == (3,12)},
              {"name":"platform", "actual":sys.platform, "required":"linux", "passed":sys.platform == "linux"}]
    for package, expected in PROFILES[name].items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = None
        checks.append({"name":package,"actual":actual,"required":expected,
                       "passed":actual is not None and actual.split("+")[0] == expected})
    return {"schema":"aureon.environment_profile.v1", "profile":name,
            "status":"passed" if all(c["passed"] for c in checks) else "failed", "checks":checks,
            "remedy":f"Create a separate Python 3.12 Linux environment and install aureon-inference[{name}] with its matching constraints. Do not combine scene with scene-legacy, vllm or generation.",
            "cuda":"not_probed", "model_imported":False, "fresh_resolution_proven":False}


def require_profile(name):
    result = inspect_profile(name)
    if result["status"] != "passed":
        failures = ", ".join(f"{c['name']}={c['actual']} (need {c['required']})" for c in result["checks"] if not c["passed"])
        raise ProfileError(f"Environment profile {name} mismatch: {failures}. {result['remedy']}")
    return result
