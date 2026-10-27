"""Run standard-library package/API contracts without model or native I/O tests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("test_build_review", "test_backend", "test_public_api", "test_session",
          "test_model_support", "test_service_package")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON receipt path")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        parser.error("Output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT / "packaging/tests"), str(ROOT)]
    os.environ["PYTHONPATH"] = str(ROOT)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="cpu-ci-", dir=output.parent) as scratch:
        tempfile.tempdir = scratch
        os.environ["TEMP"] = os.environ["TMP"] = scratch
        suite = unittest.defaultTestLoader.loadTestsFromNames(SUITES)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {"schema": "aureon.cpu_ci.v1", "passed": result.wasSuccessful(),
               "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "seconds": time.time() - started,
               "python": sys.version, "suites": SUITES, "model_execution": False,
               "scope": "Source CPU contracts with test doubles; not installed GPU acceptance or performance"}
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
