"""Check Engine package ownership, extracted layers and compatibility imports.

This is a static dependency check, not a sandbox or model qualification.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tomllib
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAYERS = {
    "planning": ("routing", "resource_policy", "pool_placement"),
    "integrations": ("model_bridge", "borrowed_model", "preprocessing", "terratorch_task", "upstream_task"),
}


def compatibility_source(group: str, module: str, source: bytes) -> str:
    exports: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            exports.append(node.name)
        elif isinstance(node, ast.Assign):
            exports.extend(n.id for n in node.targets if isinstance(n, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            exports.append(node.target.id)
    lines = [f'"""Compatibility import for :mod:`aureon.{group}.{module}`."""',
             "import sys as _sys", f"from ..{group} import {module} as _implementation",
             f"from ..{group}.{module} import ("]
    lines.extend(f"    {name} as {name}," for name in sorted(exports))
    return "\n".join([*lines, ")", "", "_sys.modules[__name__] = _implementation", ""])


def validate_edge(source: str, target: str) -> None:
    """Keep planners provider-free and bridges independent of orchestration."""
    if source.startswith("aureon.planning"):
        if target != "aureon" and not target.startswith("aureon.planning"):
            if target.partition(".")[0] not in sys.stdlib_module_names:
                raise ValueError(f"Planning dependency violates its boundary: {source} -> {target}")
    if source.startswith("aureon.integrations"):
        allowed = ("aureon.integrations", "aureon.terramind_native.encoder_loader", "aureon.terramind_native.rng")
        if target.startswith("aureon.") and not any(target == p or target.startswith(p + ".") for p in allowed):
            raise ValueError(f"Integration depends on orchestration: {source} -> {target}")
        if target.partition(".")[0] in {"cloud", "maintenance", "tests", "packaging", "runs"}:
            raise ValueError(f"Integration depends on private/tooling code: {source} -> {target}")


def validate_packages(names: list[str], project: dict[str, Any]) -> None:
    """An allowlisted source package must actually be included by setuptools."""
    expected = {n.removesuffix("/__init__.py").replace("/", ".") for n in names if n.endswith("/__init__.py")}
    declared = project["tool"]["setuptools"]["packages"]
    if len(declared) != len(set(declared)) or set(declared) != expected:
        raise ValueError("Setuptools packages differ from runtime allowlist: " + str(sorted(expected ^ set(declared))))
    if "aureon/py.typed" not in names:
        raise ValueError("Typed library marker is missing")
    data = project["tool"]["setuptools"].get("package-data", {})
    for name in names:
        if name.endswith(".pyi"):
            package = name.rsplit("/", 1)[0].replace("/", ".")
            if "*.pyi" not in data.get(package, ()) and name.rsplit("/", 1)[1] not in data.get(package, ()):
                raise ValueError("Stub is not included in package data: " + name)


def import_targets(tree: ast.AST, module: str, root: Path, *, package: bool) -> list[str]:
    """Resolve ordinary and literal dynamic imports, including deferred imports."""
    targets: list[str] = []
    parent = module if package else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = node.module or ""
            if node.level:
                target = importlib.util.resolve_name("." * node.level + target, parent)
            targets.append(target)
            for alias in node.names:
                child = target + "." + alias.name
                path = root.joinpath(*child.split("."))
                if path.with_suffix(".py").is_file() or (path / "__init__.py").is_file():
                    targets.append(child)
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
              and ((isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                   or (isinstance(node.func, ast.Name) and node.func.id == "__import__"))):
            targets.append(node.args[0].value)
    return targets


def inspect(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    names = json.loads((root / "packaging/runtime-files.json").read_text(encoding="utf-8"))["files"]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    validate_packages(names, project)
    modules = {n.removesuffix(".py").replace("/", ".").removesuffix(".__init__") for n in names if n.endswith(".py")}
    edges = 0
    for name in names:
        if not name.endswith(".py") or not name.startswith(("aureon/planning/", "aureon/integrations/")):
            continue
        path = root / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("Nonlocal implementation: " + name)
        module = name.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        for target in import_targets(ast.parse(path.read_bytes()), module, root, package=name.endswith("/__init__.py")):
            validate_edge(module, target)
            if (target == "aureon" or target.startswith("aureon.")) and target not in modules:
                raise ValueError("Missing runtime dependency: " + target)
            edges += 1
    aliases = 0
    for group, members in LAYERS.items():
        for member in members:
            old, new = f"aureon/api/{member}.py", f"aureon/{group}/{member}.py"
            if old not in names or new not in names:
                raise ValueError("Missing compatibility implementation: " + member)
            expected = compatibility_source(group, member, (root / new).read_bytes())
            if ast.dump(ast.parse((root / old).read_bytes())) != ast.dump(ast.parse(expected)):
                raise ValueError("Compatibility module contains unexpected implementation: " + old)
            aliases += 1
    return {"passed": True, "version": project["project"]["version"],
            "packages": len(project["tool"]["setuptools"]["packages"]),
            "compatibility_modules": aliases, "checked_layer_imports": edges,
            "model_calls": 0, "scope": "Package/stub inclusion and extracted-layer static boundaries; not GPU acceptance."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output must be fresh")
    result = inspect(args.source)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
