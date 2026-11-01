"""Offline, bounded Sentinel-2 input preparation. No model or network imports."""

from __future__ import annotations
from pathlib import Path
from ..value_types import JsonObject, Pathish


def prepare_scene(spec: str | Path | JsonObject, output: Pathish, *, base_dir: Pathish | None=None) -> JsonObject:
    """Prepare a local spec (mapping or JSON path) into a fresh directory."""
    from .local import prepare_scene as prepare
    return prepare(spec, output, base_dir=base_dir)


def check_scene(manifest: Pathish) -> JsonObject:
    """Validate every prepared pixel and identity on CPU before registration."""
    from .local import check_scene as check
    return check(manifest)


__all__ = ["prepare_scene", "check_scene"]
