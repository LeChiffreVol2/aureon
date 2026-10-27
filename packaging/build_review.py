"""Build an allowlisted local review candidate. No installation or publication."""
from __future__ import annotations

import argparse
import ast
import fnmatch
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"
EPOCH = str(int(datetime(2026, 10, 25, tzinfo=timezone.utc).timestamp()))
# Guarded v2/v3 bind these original source bytes. Their existing docstrings and
# provenance links remain; full attribution/proposed comments are shipped in
# NOTICE and SOURCE-NOTICE-OVERLAYS.json. Do not rewrite qualification hashes.
QUALIFIED_NOTICE_SOURCES = {
    "aureon/catalog.py": "5e82eff095948ba83b6ea40145bac84ece30e788e80bcdd01c9f2a9394908708",
    "aureon/recipe.py": "f1e9ac9e5eb4b15c6548509ccf4a6ed80e2ffa0213653b21beed58361fd27e1d",
    "aureon/vllm_bundle.py": "4b7d799f35ef1ce3d19c69bc61d9ccc0cae1023c0b688af1fb06f1592c07dd28",
}
FIXED = (
    'pyproject.toml',
    'packaging/README.md',
    'packaging/build_review.py',
    'packaging/build_backend.py',
    'packaging/runtime-files.json',
    'packaging/requirements-build.txt',
    'packaging/requirements-scene-linux-cu128.txt',
    'packaging/constraints-linux-py312-cu128.txt',
    'packaging/requirements-scene-linux-cu130.txt',
    'packaging/constraints-linux-py312-cu130.txt',
    'packaging/requirements-vllm-linux-cu128.txt',
    'packaging/requirements-generation-linux-cu128.txt',
    'packaging/lock-linux-py312-cu130.txt',
    'packaging/lock-scene-linux-py312-cu130.txt',
    'packaging/lock-generation-linux-py312-cu128.txt',
    'packaging/lock-bootstrap-scene.txt',
    'packaging/lock-bootstrap-generation.txt',
    'packaging/dependency-inventory.json',
    'packaging/START-HERE.md',
    'packaging/INPUTS.md',
    'packaging/ASSETS.md',
    'packaging/SUPPORT.md',
    'packaging/LOCKED-INSTALL.md',
    'packaging/ENGINE-QUICKSTART.md',
    'packaging/MODEL-SUPPORT.md',
    'packaging/PUBLIC-API.md',
    'packaging/SECURITY.md',
    'packaging/CONTRIBUTING.md',
    'packaging/ASYNC-SERVICE.md',
    'packaging/verify_delivery.py',
    'packaging/review/LICENSE',
    'packaging/review/NOTICE',
    'packaging/review/THIRD-PARTY-NOTICES.md',
    'packaging/review/REVIEW-STATUS.md',
    'packaging/review/SOURCE-NOTICE-OVERLAYS.json',
    'examples/packaged_scene/offline_job.py',
    'examples/packaged_scene/api_job.py',
    'examples/packaged_scene/aligned-input.json',
    'examples/packaged_scene/l2a-input.json',
    'packaging/tests/test_examples.py',
    'packaging/tests/test_installed_layout.py',
    'packaging/tests/test_build_review.py',
    'packaging/tests/test_public_api.py',
    'packaging/tests/test_backend.py',
    'packaging/tests/test_manifest_digest.py',
    'packaging/tests/test_session.py',
    'packaging/tests/test_model_support.py',
    'packaging/tests/test_service_package.py',
    'packaging/tests/test_installed_alpha.py',
    'packaging/ci_cpu.py',
    '.github/workflows/cpu.yml',
)
FORBIDDEN_PARTS = {".git", ".archive", ".venv", "models", "inputs", "runs", "__pycache__", "node_modules"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    if not isinstance(name, str) or not name or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise ValueError("Invalid candidate path")
    path = PurePosixPath(name)
    if "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts or str(path) != name.rstrip("/"):
        raise ValueError("Invalid relative candidate path")
    return path


def member(root, name):
    path = safe_name(name)
    resolved = root.joinpath(*path.parts)
    if any(p in FORBIDDEN_PARTS for p in path.parts) or any(part.startswith(".env") for part in path.parts):
        raise ValueError("Forbidden candidate member: " + name)
    if any(p.is_symlink() for p in [resolved, *resolved.parents] if p != root.parent):
        raise ValueError("Symlink candidate members are unsupported: " + name)
    if not resolved.is_file() or not resolved.resolve().is_relative_to(root.resolve()):
        raise ValueError("Missing/nonlocal candidate member: " + name)
    return resolved


def inspect_text(name, data):
    text = data.decode("utf-8")
    # A release inspection guard, not a complete secret or ownership audit.
    if re.search(r"(?:C:[/\\]Users[/\\]|/mnt/c/Users/[A-Za-z0-9]|/root/aureon-runtime/(?:venv|review)-)", text, re.I):
        raise ValueError("Personal runtime path in staged member: " + name)
    if re.search(r"(?:sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)", text):
        raise ValueError("Credential-like content in candidate member: " + name)


def stage(output):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Output must be new with an existing parent")
    runtime = json.loads((ROOT / "packaging/runtime-files.json").read_text(encoding="utf-8"))["files"]
    if len(runtime) != len(set(runtime)) or any(not n.startswith("aureon/") for n in runtime):
        raise ValueError("Invalid runtime allowlist")
    import tomllib
    project = tomllib.loads(member(ROOT, "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extra = [p for p in project["license-files"] if p not in FIXED]
    if any(any(c in p for c in "*?[") for p in project["license-files"]):
        raise ValueError("Legal file paths must be explicit")
    if not extra:
        raise ValueError("Third-party license inventory is required")
    names = list(FIXED) + runtime + extra
    if len(names) != len(set(names)):
        raise ValueError("Duplicate candidate paths")
    overlays = json.loads(member(ROOT, "packaging/review/SOURCE-NOTICE-OVERLAYS.json").read_text(encoding="utf-8"))["overlays"]
    for name, prefix in overlays.items():
        if name not in runtime or not name.endswith(".py") or not isinstance(prefix, str):
            raise ValueError("Overlay must target an allowlisted Python file")
        if not prefix.endswith("\n") or any(line.strip() and not line.lstrip().startswith("#") for line in prefix.splitlines()):
            raise ValueError("Only comment/blank notice prefixes are permitted")
    selected = []
    for name in names:
        original = member(ROOT, name).read_bytes()
        prefix = overlays.get(name, "").encode("utf-8")
        if name in QUALIFIED_NOTICE_SOURCES:
            if sha(original) != QUALIFIED_NOTICE_SOURCES[name]:
                raise ValueError("Qualified source bytes changed: " + name)
            prefix = b""
        # Idempotent when the builder is run from its own staged source tree.
        data = original if prefix and original.startswith(prefix) else prefix + original
        if prefix and ast.dump(ast.parse(original)) != ast.dump(ast.parse(data)):
            raise ValueError("Notice overlay changed Python syntax tree: " + name)
        inspect_text(name, data)
        selected.append((name, name, original, data))
    output.mkdir()
    source = output / "source"
    source.mkdir()
    entries = []
    for origin, name, original, data in selected:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        entries.append({"origin": origin, "path": name, "original_sha256": sha(original),
                        "staged_sha256": sha(data), "bytes": len(data),
                        "notice_prefix_added": original != data,
                        "qualified_notice_source_preserved": name in QUALIFIED_NOTICE_SOURCES})
    inventory = {"schema": "aureon.review_source_inventory.v1", "version": VERSION,
                 "review_only": True, "published": False, "files": entries}
    (source / "SOURCE-INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline="\n")
    # Include exactly staged review material in the sdist. Runtime package data
    # remains explicitly selected in pyproject, with no environment/model files.
    manifest = "\n".join("include " + e["path"] for e in entries) + "\ninclude SOURCE-INVENTORY.json\ninclude MANIFEST.in\n"
    (source / "MANIFEST.in").write_text(manifest, encoding="utf-8", newline="\n")
    return source, inventory


def audit_distributions(source, dist, inventory, *, selected=None):
    runtime = {e["path"] for e in inventory["files"] if e["path"].startswith("aureon/")}
    source_paths = {e["path"] for e in inventory["files"]} | {"SOURCE-INVENTORY.json", "MANIFEST.in", "PKG-INFO"}
    staged = {e["path"]: e["staged_sha256"] for e in inventory["files"]}
    generated = {
        "SOURCE-INVENTORY.json": (json.dumps(inventory, indent=2) + "\n").encode(),
        "MANIFEST.in": ("\n".join("include " + e["path"] for e in inventory["files"]) + "\ninclude SOURCE-INVENTORY.json\ninclude MANIFEST.in\n").encode(),
    }
    import tomllib
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    legal = {n for n in staged if any(fnmatch.fnmatchcase(n, p) for p in project["license-files"])}
    dist_info = "aureon_inference-" + VERSION + ".dist-info"
    sdist_root = "aureon_inference-" + VERSION
    reports = []
    wheels, sdists = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    if selected is None and (len(wheels) != 1 or len(sdists) != 1):
        raise ValueError("Expected one wheel and one sdist")
    for artifact in selected if selected is not None else wheels + sdists:
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                # ZipInfo may normalize Windows separators or truncate NULs.
                # Validate the original member spelling before normalized names.
                for info in archive.infolist():
                    safe_name(info.orig_filename)
                names = archive.namelist()
                normalized = [str(safe_name(n)) for n in names]
                if len(normalized) != len(set(normalized)):
                    raise ValueError("Duplicate wheel members")
                contents = {name: archive.read(name) for name in names if not name.endswith("/")}
            present = {n for n in contents if n.startswith("aureon/")}
            if present != runtime:
                raise ValueError("Wheel runtime differs from allowlist")
            for name, data in contents.items():
                if not name.startswith("aureon/") and not name.startswith(dist_info + "/"):
                    raise ValueError("Unexpected wheel content: " + name)
                if name in staged and sha(data) != staged[name]:
                    raise ValueError("Wheel bytes differ from staged source")
            license_prefix = dist_info + "/licenses/"
            expected_wheel = runtime | {license_prefix+n for n in legal} | {
                dist_info+"/"+n for n in ("METADATA","WHEEL","entry_points.txt","top_level.txt","RECORD")}
            if set(contents) != expected_wheel:
                raise ValueError("Wheel member set differs from exact distribution allowlist")
            wheel_legal = {n[len(license_prefix):] for n in contents if n.startswith(license_prefix)}
            if wheel_legal != legal:
                raise ValueError("Wheel legal inventory differs from staged declaration")
            for name in legal:
                if sha(contents[license_prefix + name]) != staged[name]:
                    raise ValueError("Wheel legal bytes differ from staged source")
            meta = next(data.decode() for n, data in contents.items() if n.endswith(".dist-info/METADATA"))
            if "Classifier: Development Status :: 3 - Alpha" not in meta or "License-Expression: Apache-2.0" not in meta:
                raise ValueError("Review/license metadata missing")
            for required in ("LICENSE", "NOTICE", "THIRD-PARTY-NOTICES.md", "REVIEW-STATUS.md", "SOURCE-NOTICE-OVERLAYS.json"):
                if not any(n.endswith("/licenses/packaging/review/" + required) for n in contents):
                    raise ValueError("Wheel license material missing: " + required)
        else:
            with tarfile.open(artifact, "r:gz") as archive:
                members = archive.getmembers()
                normalized = [str(safe_name(m.name)) for m in members]
                if len(normalized) != len(set(normalized)) or any(PurePosixPath(n).parts[0] != sdist_root for n in normalized):
                    raise ValueError("Duplicate or unexpected sdist roots/members")
                if any(m.issym() or m.islnk() or not (m.isdir() or m.isfile()) for m in members):
                    raise ValueError("Unexpected sdist member type")
                contents = {str(PurePosixPath(m.name).relative_to(PurePosixPath(m.name).parts[0])): archive.extractfile(m).read()
                            for m in members if m.isfile()}
            for name, data in contents.items():
                generated_metadata = {"setup.cfg"} | {"aureon_inference.egg-info/"+n for n in (
                    "PKG-INFO","SOURCES.txt","dependency_links.txt","entry_points.txt","requires.txt","top_level.txt")}
                if name not in source_paths | generated_metadata:
                    raise ValueError("Unexpected sdist content: " + name)
                if name in staged and sha(data) != staged[name]:
                    raise ValueError("Sdist bytes differ from staged source")
            if not set(staged).issubset(contents):
                raise ValueError("Sdist is missing staged members")
            if any(contents.get(name) != data for name, data in generated.items()):
                raise ValueError("Sdist generated inventory/manifest differs from staged declaration")
        for name, data in contents.items():
            if any(p in FORBIDDEN_PARTS for p in PurePosixPath(name).parts) or name.endswith((".pt", ".bin", ".npy", ".tif", ".png", ".pyc")):
                raise ValueError("Forbidden archive member: " + name)
            inspect_text(name, data)
        reports.append({"file": artifact.name, "bytes": artifact.stat().st_size,
                        "sha256": sha(artifact.read_bytes()), "member_count": len(contents)})
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if importlib.metadata.version("setuptools") != "84.2.1":
        parser.error("Use a separate build environment with setuptools==84.2.1; do not modify an existing runtime")
    output = args.output.absolute()
    source, inventory = stage(output)
    dist = output / "dist"
    dist.mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SOURCE_DATE_EPOCH=EPOCH)
    with (output / "build.log").open("xb") as log:
        # Each PEP 517 hook gets a fresh process: setuptools commands retain
        # distribution options when unrelated hooks share one interpreter.
        for hook in ("build_sdist", "build_wheel"):
            code = "from setuptools import build_meta; import sys; getattr(build_meta,sys.argv[1])(sys.argv[2])"
            subprocess.run([sys.executable, "-B", "-c", code, hook, str(dist)], cwd=source, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    report = {"schema": "aureon.review_build.v1", "status": "passed", "published": False,
              "version": VERSION, "setuptools": importlib.metadata.version("setuptools"),
              "source_files": len(inventory["files"]), "artifacts": audit_distributions(source, dist, inventory),
              "scope": "Allowlist, identity and archive content checks; installation/model acceptance is recorded separately"}
    with (output / "build-report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
