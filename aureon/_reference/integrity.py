"""Verify the exact declared frozen closure before import, build or type exclusion."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

MANIFEST_SHA256 = "42f56ee519e8f7ba6343d4339a9adb2b6dd9486fd6e8a42444f17d4a2b5530cf"
WHEEL_SHA256 = "0fa5ef2b9440fcc310a17c76f78df52278cbe2d354015be2046a8e4196715c97"
SOURCE_SHA256 = "4c133ca76190c412bf2b4c4c780a2eb5da8c65161bea810173321910a0baea78"
PREFIX = "aureon/_reference/terramind/"
FILENAMES = ("__init__.py", "common.py", "compat.py", "denoiser_binding.py",
             "denoiser_graph.py", "geo.py", "graph_runtime.py", "harness.py",
             "loader.py", "reuse.py", "rng.py", "session.py", "upstream-source.json")


class _File(TypedDict):
    path: str
    sha256: str
    bytes: int


class _Manifest(TypedDict):
    files: list[_File]
    wheel_sha256: str
    source_archive_sha256: str


@dataclass(frozen=True)
class CoreFingerprint:
    """Verified source identity, not a performance or model-weights attestation."""

    manifest_sha256: str
    wheel_sha256: str
    source_archive_sha256: str
    frozen_python_paths: tuple[str, ...]

    def metadata(self) -> dict[str, object]:
        return {"manifest_sha256": self.manifest_sha256, "wheel_sha256": self.wheel_sha256,
                "source_archive_sha256": self.source_archive_sha256,
                "namespace": "aureon._reference.terramind",
                "relocated_source": True, "new_performance_qualification": False}


def _read(root: Path, name: str) -> bytes:
    path = root / name
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError("Frozen source cannot contain symlinks: " + name)
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Missing/nonlocal frozen source: " + name)
    return path.read_bytes()


def verify_core(root: Path | None = None) -> CoreFingerprint:
    """Fail closed on missing, modified or additional Python files.

    The manifest itself is pinned before its declarations can exempt any source
    from strict typing. Only the twelve listed frozen implementations qualify.
    """
    root = (Path(__file__).resolve().parents[2] if root is None else root).resolve(strict=True)
    data = _read(root, PREFIX + "SOURCE.json")
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Selected-core manifest hash mismatch")
    manifest = cast(_Manifest, json.loads(data))
    expected = {PREFIX + name for name in FILENAMES}
    if {row["path"] for row in manifest["files"]} != expected or len(manifest["files"]) != len(expected):
        raise ValueError("Selected-core inventory mismatch")
    if manifest["wheel_sha256"] != WHEEL_SHA256 or manifest["source_archive_sha256"] != SOURCE_SHA256:
        raise ValueError("Selected archive identity mismatch")
    python_paths = {name for name in expected if name.endswith(".py")}
    actual = {p.relative_to(root).as_posix() for p in (root / PREFIX).rglob("*.py")}
    if actual != python_paths:
        raise ValueError("Undeclared or missing frozen Python source")
    for row in manifest["files"]:
        payload = _read(root, row["path"])
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("Frozen payload hash mismatch: " + row["path"])
    return CoreFingerprint(MANIFEST_SHA256, WHEEL_SHA256, SOURCE_SHA256, tuple(sorted(python_paths)))
