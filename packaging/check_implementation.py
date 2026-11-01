"""Strict-check owned implementations; verify the declared frozen core by hash."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Optional external implementations are not AUREON's checker targets. NumPy and
# Torch keep their installed type definitions; untyped GDAL/provider boundaries are audited
# through the annotations/protocols in AUREON, not a third-party implementation run.
EXTERNAL = ("terratorch", "rasterio", "affine", "orjson", "pybase64")
# This zero-argument upstream bool query lacks a return annotation in Torch.
# Keep every owned function strict; exempt only this provider call, not torch.cuda.
UNTYPED_PROVIDER_CALLS = ("torch.cuda.is_initialized",)
# Torch 2.10 declares Event.record(self, stream=None) without any annotations;
# Torch 2.11 annotates stream. This external method is not an owned-code exemption.
UNTYPED_TORCH210_CALLS = ("torch.cuda.streams.Event.record",)


def coverage(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    complete = 0
    for node in functions:
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        args += [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
        complete += int(node.returns is not None and all(a.annotation is not None for a in args if a.arg not in {"self", "cls"}))
    return {"functions": len(functions), "fully_annotated_signatures": complete,
            "explicit_Any_names": sum(isinstance(n, ast.Name) and n.id == "Any" for n in ast.walk(tree))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT, help="Tree supplying the runtime allowlist")
    parser.add_argument("--runtime", type=Path, help="Source or installed runtime to audit")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source, runtime = args.source.resolve(strict=True), (args.runtime or args.source).resolve(strict=True)
    scratch, output = args.scratch.absolute(), args.output.absolute()
    if scratch.exists() or output.exists():
        parser.error("Scratch and output must be fresh")
    if importlib.metadata.version("mypy") != "1.11.2":
        parser.error("Use the existing mypy 1.11.2 environment; nothing will be installed")
    torch_version = importlib.metadata.version("torch")
    provider_calls = UNTYPED_PROVIDER_CALLS + (
        UNTYPED_TORCH210_CALLS if torch_version.split("+", 1)[0] == "2.10.0" else ()
    )
    scratch.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((source / "packaging/runtime-files.json").read_text())["files"]
    spec = importlib.util.spec_from_file_location("aureon_frozen_types", Path(__file__).with_name("verify_frozen_core.py"))
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    frozen = verifier.verify_frozen_runtime(runtime, inventory)
    excluded = set(frozen["frozen_python_paths"])
    names = [n for n in inventory if n.endswith(".py")]
    targets = [n for n in names if n not in excluded]
    if not names or len(names) != len(set(names)):
        parser.error("Invalid Python runtime inventory")
    mirror = scratch / "implementation"
    rows = []
    for name in sorted(names):
        original = (runtime / name).resolve(strict=True)
        if not original.is_relative_to(runtime) or not name.startswith("aureon/"):
            parser.error("Implementation outside selected runtime")
        target = mirror / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = original.read_bytes()
        target.write_bytes(data)
        rows.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "strict_target": name not in excluded, **coverage(original)})
    # Never copy .pyi files. Only exact verified frozen payloads are exempt.
    assert not list(mirror.rglob("*.pyi"))
    config = scratch / "mypy.ini"
    contents = "[mypy]\npython_version = 3.12\nplatform = linux\nstrict = True\nfollow_imports = silent\nshow_error_codes = True\nno_error_summary = True\npretty = False\n"
    contents += "untyped_calls_exclude = " + ",".join(provider_calls) + "\n"
    contents += "mypy_path = " + str(mirror) + "\n"
    for module in EXTERNAL:
        contents += f"\n[mypy-{module},{module}.*]\nfollow_imports = skip\nignore_missing_imports = True\n"
    for name in sorted(excluded):
        module = name.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        contents += f"\n[mypy-{module}]\nfollow_imports = skip\n"
    config.write_text(contents, encoding="utf-8")
    # Each check uses a fresh source mirror. Its cache cannot benefit a later
    # check; keep the durable diagnostics/identities without accumulating caches.
    command = [sys.executable, "-I", "-B", "-m", "mypy", "--config-file", str(config),
               "--cache-dir", os.devnull, *targets]
    result = subprocess.run(command, cwd=mirror, env=dict(os.environ, MYPYPATH="", PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1"),
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    log = result.stdout + result.stderr
    diagnostics = []
    for line in log.splitlines():
        match = re.match(r"(.+?):(\d+)(?::\d+)?: error: (.*?)\s+\[([a-z-]+)\]$", line)
        if match:
            diagnostics.append({"path": match[1].replace("\\", "/"), "line": int(match[2]), "message": match[3], "code": match[4]})
    for row in rows:
        row["diagnostics"] = sum(d["path"] == row["path"] for d in diagnostics)
    receipt = {"passed": result.returncode == 0, "returncode": result.returncode,
               "source": str(source), "runtime": str(runtime), "mypy": "1.11.2", "strict": True,
               "cache_persisted": False,
               "all_shipped_implementations_explicit_targets": not excluded, "facade_stubs_excluded": True,
               "all_owned_implementations_explicit_targets": True,
               "frozen_source_exception": frozen, "strict_target_count": len(targets),
               "external_implementation_skips": list(EXTERNAL),
               "untyped_provider_calls": list(provider_calls),
               "typed_providers": {name: importlib.metadata.version(name) for name in ("torch", "numpy")},
               "files": rows, "diagnostics": diagnostics, "diagnostic_counts": dict(Counter(d["code"] for d in diagnostics)),
               "raw_checker_output": log, "model_calls": 0, "dependencies_installed": False}
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "implementations": len(rows),
                      "diagnostics": len(diagnostics), "counts": receipt["diagnostic_counts"], "receipt": str(output)}))
    if result.returncode:
        # Keep failures actionable in CI even when the detailed receipt is local
        # to an ephemeral runner. Do not suppress or reclassify diagnostics.
        print(log, file=sys.stderr, end="" if log.endswith("\n") else "\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
