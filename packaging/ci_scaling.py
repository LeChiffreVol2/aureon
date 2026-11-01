"""CPU-only stage/replica protocol tests using the explicitly selected runtime."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scratch.exists() or args.output.exists():
        parser.error("Use fresh owned scratch and receipt paths")
    args.scratch.mkdir(parents=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(args.scratch.resolve())
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    runtime = args.runtime.resolve(strict=True)
    sys.path.insert(0, str(runtime))
    # The child process test must import the installed runtime, not the sdist.
    os.environ["AUREON_SCALING_TEST_RUNTIME"] = str(runtime)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"), pattern="test_scaling_runtime.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    import aureon.scaling
    receipt = {"passed": result.wasSuccessful(), "tests": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
               "runtime": str(runtime), "imported": str(aureon.scaling.__file__),
               "scope": "CPU stage ownership, admission, recovery/fencing and real loopback/separate-process JSON/blob transport",
               "model_calls": 0, "physical_multi_gpu_qualification": False,
               "external_tls_provider_qualification": False}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
