"""Reproduce the selected TerraMind closure from the exact wheel/source pair.

No archive extraction, dependency installation, model loading or network access.
The helper prefix is an explicitly recorded excerpt, not a whole-file copy.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

WHEEL_SHA256 = "0fa5ef2b9440fcc310a17c76f78df52278cbe2d354015be2046a8e4196715c97"
SOURCE_SHA256 = "4c133ca76190c412bf2b4c4c780a2eb5da8c65161bea810173321910a0baea78"
ORIGIN = "aureon/terramind_native/"
DESTINATION = "aureon/_reference/terramind/"
FILES = ("__init__.py", "common.py", "compat.py", "denoiser_binding.py",
         "denoiser_graph.py", "geo.py", "graph_runtime.py", "harness.py",
         "loader.py", "reuse.py", "rng.py", "session.py", "upstream-source.json")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def helper_prefix(data: bytes) -> tuple[bytes, dict[str, object]]:
    """Keep both required helpers, imports and whitespace exactly as archived."""
    boundary = data.index(b"def generate_one(")
    prefix = data[:boundary]
    original, extracted = ast.parse(data), ast.parse(prefix)
    functions = {n.name: n for n in original.body if isinstance(n, ast.FunctionDef)}
    helpers = [n for n in extracted.body if isinstance(n, ast.FunctionDef)]
    if {n.name for n in helpers} != {"capture_failure", "trace_native"}:
        raise ValueError("Unexpected helper prefix")
    checks = {}
    for node in helpers:
        expected = functions[node.name]
        if ast.dump(node, include_attributes=True) != ast.dump(expected, include_attributes=True):
            raise ValueError("Helper AST changed")
        body = ast.get_source_segment(prefix.decode(), node)
        if body != ast.get_source_segment(data.decode(), expected):
            raise ValueError("Helper body changed")
        checks[node.name] = sha(ast.dump(node, include_attributes=True).encode())
    return prefix, {"mode": "verbatim_prefix", "source_span_bytes": [0, boundary],
                    "source_span_lines": [1, len(prefix.splitlines())],
                    "helper_ast_sha256": checks,
                    "omitted_functions": sorted(set(functions) - {n.name for n in helpers})}


def payloads(wheel: Path, source: Path) -> dict[str, bytes]:
    if sha(wheel.read_bytes()) != WHEEL_SHA256 or sha(source.read_bytes()) != SOURCE_SHA256:
        raise ValueError("Use the exact selected wheel and matching source archive")
    result: dict[str, bytes] = {}
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(wheel) as archive, tarfile.open(source, "r:gz") as sdist:
        for name in FILES:
            origin = ORIGIN + name
            archived = archive.read(origin)
            member = sdist.getmember("aureon_inference-0.1.0/" + origin)
            if not member.isfile():
                raise ValueError("Source member must be a regular file")
            stream = sdist.extractfile(member)
            if stream is None or stream.read() != archived:
                raise ValueError("Selected wheel/source payload mismatch: " + name)
            data = archived
            details: dict[str, object] = {"mode": "verbatim"}
            if name == "harness.py":
                data, details = helper_prefix(archived)
            if name.endswith(".py"):
                for node in ast.walk(ast.parse(data)):
                    if isinstance(node, ast.ImportFrom) and node.level:
                        if node.level != 1 or (node.module or "") + ".py" not in FILES:
                            raise ValueError("Dependency escapes the selected closure")
                    if isinstance(node, ast.Import) and any(a.name.startswith("aureon") for a in node.names):
                        raise ValueError("Unexpected absolute runtime import")
                    if isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").startswith("aureon"):
                        raise ValueError("Unexpected absolute runtime import")
            result[name] = data
            records.append({"path": DESTINATION + name, "origin_path": origin,
                            "source_sha256": sha(archived), "source_bytes": len(archived),
                            "sha256": sha(data), "bytes": len(data), **details})
    manifest = {"schema": "aureon.selected_core_source.v1", "wheel_sha256": WHEEL_SHA256,
                "source_archive_sha256": SOURCE_SHA256, "namespace": "aureon._reference.terramind",
                "files": records,
                "relocation": "Relative imports preserved. __file__ source provenance paths and inventory change. Module process lock is separate; the owned facade holds the broad runtime lifecycle lock.",
                "qualification": "Operation-preserving source relocation; no new performance qualification.",
                "attribution": "AUREON-owned runtime, Apache-2.0; original source headers retained. TerraTorch/TerraMind remain separately installed dependencies with their own terms. No comparator or model weights included."}
    result["SOURCE.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh closure directory")
    args = parser.parse_args()
    data = payloads(args.wheel, args.source)
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        parser.error("Output must be fresh with an existing parent")
    args.output.mkdir()
    for name, content in data.items():
        (args.output / name).write_bytes(content)
    print(json.dumps({"files": len(data), "manifest_sha256": sha(data["SOURCE.json"]), "model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
