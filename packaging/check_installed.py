"""Verify a noneditable installed wheel's public imports and CLI without models."""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


class BlockNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "torchvision", "terratorch", "transformers", "diffusers", "rasterio", "numpy"}:
            raise AssertionError("Installed import/help loaded a heavy dependency: " + fullname)
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Matching extracted sdist")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runtime, source = args.runtime.resolve(strict=True), args.source.resolve(strict=True)
    scratch, output = args.scratch.absolute(), args.output.absolute()
    if scratch.exists() or output.exists() or runtime == source:
        parser.error("Use separate runtime/source and fresh scratch/output paths")
    scratch.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    sys.path.insert(0, str(runtime))
    sys.meta_path.insert(0, BlockNeural())
    import aureon.api as api
    distribution = importlib.metadata.distribution("aureon-inference")
    assert Path(distribution.locate_file("")).resolve() == runtime
    assert distribution.version == "0.1.0"
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    assert direct.get("dir_info", {}).get("editable") is not True
    runtime_paths = json.loads((source / "packaging/runtime-files.json").read_text())["files"]
    for relative in runtime_paths:
        assert (runtime / relative).is_file(), relative
        assert (runtime / relative).read_bytes() == (source / relative).read_bytes(), relative
    assert len(api.__all__) == len(set(api.__all__))
    assert all(hasattr(api, name) for name in api.__all__)
    assert (runtime / "aureon/py.typed").is_file()
    imports = {}
    for name, module in tuple(sys.modules.items()):
        if name == "aureon" or name.startswith("aureon."):
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(runtime), (name, path)
            imports[name] = str(path.relative_to(runtime))
    modules = ("aureon", "aureon.api", "aureon.serving", "aureon.preflight")
    child = ("import sys,runpy; from pathlib import Path; "
             "sys.path.insert(0,sys.argv[1]); "
             "guard=runpy.run_path(sys.argv[2])['BlockNeural']; sys.meta_path.insert(0,guard()); "
             "module=sys.argv[3]; sys.argv=[module,'--help']; "
             "runpy.run_module(module,run_name='__main__')")
    commands = []
    for module in modules:
        result = subprocess.run([sys.executable, "-I", "-B", "-c", child, str(runtime),
                                 str(Path(__file__).resolve()), module], cwd=scratch,
                                env=dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES="-1"),
                                text=True, encoding="utf-8", errors="replace", capture_output=True,
                                timeout=30)
        commands.append({"module": module, "returncode": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
    receipt = {"passed": all(c["returncode"] == 0 for c in commands),
               "version": distribution.version, "runtime": str(runtime), "source": str(source),
               "runtime_files_equal": len(runtime_paths), "facade_exports": len(api.__all__),
               "runtime_imports": imports, "CLI": commands, "model_calls": 0,
               "dependencies_installed": False}
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"CLI", "runtime_imports"}}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
