"""Bounded source or installed CPU contracts using existing dependencies only."""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_TEST = "test_generation_batch.BatchContracts.test_installed_pipeline_noise_boundary_and_ddim_steps_on_cpu"
UPSTREAM_SKIP_REASON = "Upstream pipeline imports prohibited by CPU contract runner; run this test in the dependency environment without AUREON_BLOCK_FOUNDATION_IMPORTS."


def permitted_skips(result):
    """Only this dependency-environment test is outside the no-import contract."""
    return all(test.id() == UPSTREAM_TEST and reason == UPSTREAM_SKIP_REASON
               for test, reason in result.skipped)


SUITES = {
    "numpy": ("test_single_engine_default", "test_best_session", "test_best_cli", "test_best_runtime_planning",
        "test_gpu_devices",
        "test_library_contracts", "test_generation_engine", "test_generation_concurrency",
        "test_generation_pair_queue", "test_generation_parallel_engine",
        "test_generation_generic_engine", "test_generation_execution_selection",
    ),
    "service": ("test_service_wakeup", "test_service_input_builder", "test_service_workflows", "test_replica_operations", "test_service_metrics", "test_supervisor_identity", "test_replica_service", "test_replica_capacity", "test_service_contracts",
                "test_service_lifecycle", "test_replica_metadata", "test_service_metadata"),
    "tensor": ("test_selected_memory_core", "test_best_workflows","test_model_bridge", "test_encoder_precision", "test_pool_scaling", "test_model_pool", "test_service_workflow_inputs", "test_encoder_resident_files", "test_runtime_diagnostics", "test_scene_generation", "test_generation_dataloader", "test_terramind_family_generation", "test_terramind_encoder", "test_terramind_encoder_io", "test_generation_batch", "test_generation_lanes", "test_generation_generic_lanes", "test_terramind_session", "test_terramind_graph_runtime", "test_terramind_denoiser_graph", "test_terramind_native", "test_generation_memory_delivery"),
}


class NoModelRuntime(importlib.abc.MetaPathFinder):
    def __init__(self, *, allow_cpu_tensor=False):
        self.allow_cpu_tensor = allow_cpu_tensor

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "torchvision", "terratorch", "diffusers", "transformers",
            "vllm", "rasterio",
        }:
            if not (self.allow_cpu_tensor and fullname.split(".")[0] in {"torch", "rasterio"}):
                raise AssertionError("CPU contract attempted a model/geospatial import: " + fullname)
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument("--source", type=Path, default=ROOT,
                        help="Source checkout or extracted sdist supplying tests")
    parser.add_argument("--runtime", type=Path,
                        help="Installed package prefix; defaults to the source checkout")
    parser.add_argument("--scratch", type=Path, required=True, help="Fresh owned scratch directory")
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON receipt")
    args = parser.parse_args(argv)
    source = args.source.resolve(strict=True)
    runtime = (args.runtime or source).resolve(strict=True)
    output = args.output.absolute()
    scratch = args.scratch.absolute()
    if output.exists() or scratch.exists():
        parser.error("Output and scratch must be fresh")
    if args.suite == "service" and sys.platform != "linux":
        parser.error("Service process contracts require Linux; do not silently skip this gate")
    if args.suite in {"numpy", "tensor"} and importlib.util.find_spec("numpy") is None:
        parser.error("NumPy is required in the existing interpreter; this runner never installs it")
    if args.suite == "tensor" and any(importlib.util.find_spec(name) is None for name in ("torch", "rasterio")):
        parser.error("CPU Torch and Rasterio are required for tensor/grid contracts; nothing will be installed")
    if args.suite == "tensor":
        try:
            upstream_version = importlib.metadata.version("terratorch")
        except importlib.metadata.PackageNotFoundError:
            parser.error("Tensor contracts read installed TerraTorch 1.2.13 factory source without importing it; the runner installs nothing")
        if upstream_version != "1.2.13":
            parser.error("Factory source compatibility requires TerraTorch 1.2.13")
    scratch.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    sys.path[:0] = [str(runtime), str(source / "tests"), str(source / "packaging/tests")]
    os.environ.update(PYTHONPATH=os.pathsep.join((str(runtime), str(source / "tests"), str(source / "packaging/tests"))),
                      PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      AUREON_BLOCK_FOUNDATION_IMPORTS="1",
                      AUREON_REPLICA_TEST_ROOT=str(scratch / "replicas"),
                      AUREON_SERVICE_TEST_ROOT=str(scratch / "lifecycle"),
                      TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    tempfile.tempdir = str(scratch)
    sys.meta_path.insert(0, NoModelRuntime(allow_cpu_tensor=args.suite == "tensor"))
    modules = []
    for name in SUITES[args.suite]:
        path = source / "tests" / (name + ".py")
        if not path.is_file():
            path = source / "packaging/tests" / (name + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load declared contract module: " + name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    locations = {}
    for name, module in tuple(sys.modules.items()):
        if name == "aureon" or name.startswith("aureon."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(runtime):
                raise AssertionError((name, str(path), "outside selected runtime"))
            locations[name] = str(path.relative_to(runtime))
    receipt = {
        "passed": result.wasSuccessful() and permitted_skips(result),
        "suite": args.suite, "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "modules": list(SUITES[args.suite]),
        "skip_details": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
        "foundation_imports_prohibited": True,
        "runtime": str(runtime), "test_source": str(source),
        "runtime_imports": locations, "python": sys.version,
        "numpy": importlib.metadata.version("numpy") if args.suite in {"numpy", "tensor"} else None,
        "rasterio": importlib.metadata.version("rasterio") if args.suite == "tensor" else None,
        "terratorch_source": upstream_version if args.suite == "tensor" else None,
        "model_calls": 0, "GPU_calls": 0, "dependencies_installed": False,
        "scratch": str(scratch), "scope": "CPU doubles/process contracts, not GPU qualification",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
