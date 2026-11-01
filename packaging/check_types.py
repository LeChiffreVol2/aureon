"""Check consumer contracts and pure implementation with an existing mypy installation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PURE = ("api/contracts.py", "api/parallel_policy.py", "api/lifecycle.py", "cloud/contracts.py", "serving/contracts.py", "serving/errors.py", "serving/replica_capacity.py")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--runtime", type=Path, help="Installed prefix or source tree to check")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source.resolve(strict=True)
    runtime = (args.runtime or source).resolve(strict=True)
    scratch, output = args.scratch.absolute(), args.output.absolute()
    if scratch.exists() or output.exists():
        parser.error("Scratch and output must be fresh")
    versions = {name: importlib.metadata.version(name) for name in ("mypy", "numpy")}
    if versions != {"mypy": "1.11.2", "numpy": "2.2.6"}:
        parser.error("Use existing mypy 1.11.2 and NumPy 2.2.6; this runner never installs packages")
    if not (runtime / "aureon/py.typed").is_file():
        parser.error("Selected runtime has no PEP 561 marker")
    scratch.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = scratch / "mypy.ini"
    config.write_text("[mypy]\npython_version = 3.12\nstrict = True\n"
                      "follow_imports = silent\nshow_error_codes = True\n"
                      "no_error_summary = True\npretty = False\n"
                      "mypy_path = " + str(runtime) + "\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH="", MYPYPATH="", PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cases = []
    for name in ("positive", "negative", "pure"):
        if name == "pure":
            paths = [runtime / "aureon" / p for p in PURE]
        else:
            fixture = scratch / (name + ".py")
            fixture.write_bytes((source / "packaging/typecheck" / (name + ".py")).read_bytes())
            paths = [fixture]
        command = [sys.executable, "-I", "-B", "-m", "mypy", "--config-file", str(config),
                   "--cache-dir", os.devnull, *map(str, paths)]
        result = subprocess.run(command, cwd=scratch, env=env, text=True,
                                encoding="utf-8", errors="replace", capture_output=True)
        log = result.stdout + result.stderr
        (scratch / (name + ".log")).write_text(log, encoding="utf-8")
        expected = []
        if name == "negative":
            expected = [(i, code) for i, line in enumerate(paths[0].read_text().splitlines(), 1)
                        for code in re.findall(r"# EXPECT: ([a-z-]+)", line)]
            observed = [(int(n), code) for n, code in re.findall(
                r"negative\.py:(\d+): error: .*\[([a-z-]+)\]", log)]
            error_count = len(re.findall(r": error:", log))
            passed = bool(expected) and result.returncode == 1 and sorted(observed) == sorted(expected) and error_count == len(expected)
        else:
            observed = []
            passed = result.returncode == 0
        cases.append({"name": name, "passed": passed, "returncode": result.returncode,
                      "expected_errors": expected, "observed_errors": observed, "log": log})
    receipt = {"passed": all(c["passed"] for c in cases), "tools": versions,
               "python": sys.version, "runtime": str(runtime), "test_source": str(source),
               "cases": cases, "cache_persisted": False, "dependencies_installed": False, "model_executed": False,
               "scope": "Strict public consumers and seven pure modules; full implementation coverage is checked separately by check_implementation.py"}
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
