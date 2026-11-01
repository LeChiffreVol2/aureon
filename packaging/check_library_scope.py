"""Check reusable runtime boundaries without importing AUREON or a model."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PARTS = frozenset({"runpod_worker", "research", "preservation", "maintenance",
                           "publication", "internal", "runs", ".scratch", ".venv",
                           "hosted_local", "web_server", "aureon_cloud"})
PRIVATE_IMPORT_ROOTS = PRIVATE_PARTS | {"cloud", "tests", "fixtures"}
# These modules implement portable local control and guarded execution. The name
# cloud predates the public service; it does not license the rest of that folder.
LOCAL_CONTROL = frozenset({"__init__", "client", "contracts", "control", "engine", "http",
                           "io_types", "profiles", "single_pass_source", "stream_guarded"})
EXPERIMENT_MODULES = frozenset({
    "finite_scan", "finite_scan_v3", "forward_diagnostics", "microbatch_graph",
    "pinned_queue", "pytorch_fast", "queued_client", "queued_http", "registered_graph",
    "resident_protocol", "resident_slots", "serving_telemetry", "shm_engine_protocol", "server",
    "benchmark", "experiment", "comparison_cli", "vllm_worker",
})


def validate_runtime_names(names: list[str]) -> None:
    """Fail closed even when a private path is added to the build allowlist."""
    if not names or len(names) != len(set(names)):
        raise ValueError("Runtime paths must be nonempty and unique")
    for name in names:
        path = PurePosixPath(name)
        if (not name.startswith("aureon/") or path.is_absolute() or ".." in path.parts
                or "\\" in name or ":" in name or str(path) != name
                or PRIVATE_PARTS.intersection(path.parts)):
            raise ValueError("Private or invalid runtime path: " + name)
        if (any(p in {"tests", "fixtures", "__pycache__"} for p in path.parts)
                or path.name.startswith("test_") or path.suffix in {".whl", ".pt", ".pth", ".npy", ".tif", ".sqlite", ".db"}):
            raise ValueError("Tests, data and generated artifacts are source-only: " + name)
        if len(path.parts) == 2 and path.stem in EXPERIMENT_MODULES:
            raise ValueError("Disconnected experiment is not runtime: " + name)
        if path.parts[1] == "cloud" and (len(path.parts) != 3 or path.stem not in LOCAL_CONTROL):
            raise ValueError("Private cloud operation is not local serving: " + name)
        if name == "aureon/terramind_native/decoder_profile.py":
            raise ValueError("Experimental decoder timing driver is not runtime: " + name)


def import_targets(tree: ast.AST, module: str, name: str, root: Path) -> list[tuple[int, str]]:
    """Resolve static imports, including `from package import submodule` edges.

    This is an architecture guard, not a sandbox for computed dynamic imports.
    Attributes such as public classes are not misclassified as submodules.
    """
    package = module if name.endswith("/__init__.py") else module.rpartition(".")[0]
    targets: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        imported: list[str] = []
        if isinstance(node, ast.Import):
            imported = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
            imported = [base]
            for alias in node.names:
                candidate = base + "." + alias.name
                candidate_path = root.joinpath(*candidate.split("."))
                if (alias.name != "*" and (candidate_path.with_suffix(".py").is_file()
                        or (candidate_path / "__init__.py").is_file()
                        or (candidate.startswith("aureon.") and alias.name in PRIVATE_PARTS))):
                    imported.append(candidate)
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
              and ((isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                   or (isinstance(node.func, ast.Name) and node.func.id == "__import__"))):
            imported = [node.args[0].value]
        targets.extend((node.lineno, target) for target in imported)
    return targets


def inspect(root: Path) -> dict[str, object]:
    names = json.loads((root / "packaging/runtime-files.json").read_text(encoding="utf-8"))["files"]
    validate_runtime_names(names)
    modules = {n.removesuffix(".py").replace("/", ".").removesuffix(".__init__"): n
               for n in names if n.endswith(".py")}
    edges: list[dict[str, object]] = []
    for module, name in modules.items():
        source = root / name
        if source.is_symlink() or not source.resolve(strict=True).is_relative_to(root.resolve()):
            raise ValueError("Runtime must be an ordinary contained source: " + name)
        tree = ast.parse(source.read_bytes(), filename=name)
        for line, target in import_targets(tree, module, name, root):
            if target.split(".")[0] in PRIVATE_IMPORT_ROOTS:
                raise ValueError(f"Private/test runtime dependency: {name}:{line}: {target}")
            if target == "aureon" or target.startswith("aureon."):
                if target not in modules:
                    raise ValueError(f"Missing local runtime dependency: {name}:{line}: {target}")
                edges.append({"source": name, "line": line, "module": target})
        if name == "aureon/terramind_native/harness.py":
            declared = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if declared != {"capture_failure", "trace_native"}:
                raise ValueError("Request helper must not contain benchmark orchestration")
    for path in (root / "aureon").rglob("*"):
        if path.is_file() and path.relative_to(root).as_posix() not in names:
            raise ValueError("Unallowlisted runtime source: " + path.relative_to(root).as_posix())
    return {"passed": True, "runtime_payloads": len(names), "python_implementations": len(modules),
            "local_import_edges": len(edges), "portable_control_modules": len(LOCAL_CONTROL),
            "model_calls": 0, "network": False,
            "scope": "Runtime path policy and static import closure; not exhaustive security or model qualification."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output must be fresh")
    result = inspect(args.source.resolve(strict=True))
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
