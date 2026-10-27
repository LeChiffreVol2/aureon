"""Offline, bounded Sentinel-2 input preparation. No model or network imports."""


def prepare_scene(spec, output, *, base_dir=None):
    """Prepare a local spec (mapping or JSON path) into a fresh directory."""
    from .local import prepare_scene as prepare
    return prepare(spec, output, base_dir=base_dir)


def check_scene(manifest):
    """Validate every prepared pixel and identity on CPU before registration."""
    from .local import check_scene as check
    return check(manifest)


__all__ = ["prepare_scene", "check_scene"]
