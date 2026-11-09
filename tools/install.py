"""Install the released Best wheel with pinned dependencies; never run a model.

Preview by default. --execute creates a new environment outside the checkout.
This source companion does not change or rebuild the AUREON distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import urlopen
import venv

ROOT = Path(__file__).resolve().parents[1]
BEST = "e09e710c9e9ace17b7bf2214d8c9eff2c4ed605a7d45e8c4e649b672f6f88a64"
RELEASE_URL = "https://github.com/LeChiffreVol2/aureon/releases/download/v0.1.0/"
LOCKS = ("packaging/lock-bootstrap-generation.txt",
         "packaging/lock-generation-linux-py312-cu128.txt")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def release(root: Path = ROOT) -> dict:
    record = json.loads((root / "release/v0.1.0.json").read_text(encoding="utf-8"))
    selected = json.loads((root / "packaging/review/SELECTED-ENGINE.json").read_text(encoding="utf-8"))
    wheel = record["artifacts"]["wheel"]
    if (record["version"] != "0.1.0" or wheel["sha256"] != BEST
            or selected["wheel_sha256"] != BEST
            or wheel["filename"] != "aureon_inference-0.1.0-py3-none-any.whl"
            or wheel["url"] != RELEASE_URL + wheel["filename"]
            or type(wheel["bytes"]) is not int or wheel["bytes"] <= 0):
        raise ValueError("Release identity differs from selected Best")
    for name in LOCKS:
        if sha256(root / name) != selected["build_inputs"][name]:
            raise ValueError("Dependency lock changed: " + name)
    return wheel


def verify_archive(path: Path, row: dict) -> None:
    if (not path.is_file() or path.is_symlink() or path.stat().st_size != row["bytes"]
            or sha256(path) != row["sha256"]):
        raise ValueError("Wheel size/hash differs from the release")


def acquire(row: dict, destination: Path) -> None:
    """Bound a download, check its identity, and publish without overwriting."""
    if destination.exists() or destination.is_symlink():
        raise ValueError("Download destination already exists")
    if row["url"] != RELEASE_URL + row["filename"]:
        raise ValueError("Unexpected release URL")
    descriptor, filename = tempfile.mkstemp(prefix=".aureon-", dir=destination.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as output, urlopen(row["url"], timeout=60) as response:
            if urlparse(response.geturl()).scheme != "https":
                raise ValueError("Insecure redirect rejected")
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > row["bytes"]:
                    raise ValueError("Download exceeds pinned size")
                output.write(block)
        verify_archive(temporary, row)
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def environment() -> dict[str, str]:
    result = {key: value for key, value in os.environ.items()
              if not key.startswith(("PYTHON", "PIP_"))}
    result.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                  PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
                  CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return result


def destination_path(value: Path, root: Path = ROOT) -> Path:
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ValueError("Choose a new --prefix; existing environments are never overwritten")
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Use a prefix without symlink parents")
    if path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Choose --prefix outside the source checkout")
    return path


def run_step(command: list[str], logs: Path, name: str, steps: list[dict]) -> None:
    print("Checking " + name, flush=True)
    started = time.perf_counter()
    with (logs / (name + ".log")).open("x", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=logs.parent, env=environment(),
                                stdin=subprocess.DEVNULL, stdout=output,
                                stderr=subprocess.STDOUT, timeout=1800)
    steps.append({"step": name, "seconds": time.perf_counter() - started,
                  "exit_code": result.returncode})
    if result.returncode:
        raise RuntimeError("Step failed: " + name + "; inspect " + str(logs / (name + ".log")))


def install(prefix: Path, row: dict, wheel: Path | None,
            wheelhouse: Path | None, root: Path = ROOT) -> dict:
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12) or platform.machine() != "x86_64":
        raise ValueError("Use Linux x86_64 / WSL2 with Python 3.12")
    prefix = destination_path(prefix, root)
    if wheel is not None:
        wheel = wheel.absolute()
        verify_archive(wheel, row)
    if wheelhouse is not None:
        wheelhouse = wheelhouse.resolve(strict=True)
        if not wheelhouse.is_dir() or wheel is None:
            raise ValueError("Offline --wheelhouse requires a directory and --wheel")
    prefix.mkdir(parents=True, exist_ok=False)
    logs = prefix / "setup-logs"
    logs.mkdir()
    result: dict = {"schema": "aureon.install.v1", "passed": False,
                    "wheel_sha256": row["sha256"], "version": "0.1.0",
                    "model_calls": 0, "benchmark_run": False, "engine_rebuilt": False,
                    "gpu_qualified": False, "steps": []}
    started = time.perf_counter()
    try:
        artifact = prefix / row["filename"]
        if wheel is None:
            acquire(row, artifact)
        else:
            with wheel.open("rb") as source, artifact.open("xb") as output:
                shutil.copyfileobj(source, output)
        verify_archive(artifact, row)
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(prefix)
        python = str(prefix / "bin/python")
        pip = [python, "-I", "-B", "-m", "pip"]
        offline = ["--no-index", "--find-links", str(wheelhouse)] if wheelhouse else []
        for name, lock in zip(("bootstrap", "dependencies"), LOCKS):
            flags = ["--no-build-isolation"] if name == "dependencies" else []
            run_step(pip + ["install", "--require-hashes", "--no-compile", *offline, *flags,
                            "-r", str(root / lock)], logs, name, result["steps"])
        run_step(pip + ["install", "--no-index", "--no-deps", "--no-compile", str(artifact)],
                 logs, "engine", result["steps"])
        run_step(pip + ["check"], logs, "dependencies-consistent", result["steps"])
        run_step([python, "-I", "-B", str(root / "packaging/verify_delivery.py"),
                  "--wheel", str(artifact), "--sha256", BEST, "--environment", "generation"],
                 logs, "installed-identity", result["steps"])
        run_step([python, "-I", "-B", str(root / "examples/generation/cpu_walkthrough.py"),
                  "--output", str(prefix / "cpu-demo")], logs, "cpu-first-use", result["steps"])
        result["passed"] = True
    except Exception as error:
        result["error_type"] = type(error).__name__
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        with (prefix / "install-result.json").open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
        print("Receipt: " + str(prefix / "install-result.json"), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Create a new environment and install")
    action.add_argument("--plan", action="store_true", help="Preview only (default)")
    parser.add_argument("--prefix", type=Path, default=Path.home() / "aureon-env")
    parser.add_argument("--wheel", type=Path, help="Existing exact Best wheel instead of downloading it")
    parser.add_argument("--wheelhouse", type=Path, help="Offline directory of all hash-locked dependency archives")
    args = parser.parse_args()
    try:
        row = release()
        if not args.execute:
            print(json.dumps({"mode": "plan", "wheel": row, "prefix": str(args.prefix.expanduser()),
                              "platform": "Linux x86_64 / WSL2, Python 3.12", "dependencies": "hash-locked generation",
                              "model_calls": 0, "network_used": False, "writes": False,
                              "note": "Third-party dependencies may build from pinned source; AUREON is never rebuilt. Weights are prepared separately."}, indent=2))
            return 0
        print(json.dumps(install(args.prefix, row, args.wheel, args.wheelhouse), indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("Installation stopped: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
