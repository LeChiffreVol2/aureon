"""Offline local-asset configuration and stdlib-only preflight; no tensor loading."""
import os
from importlib import metadata
from pathlib import Path

from .common import INPUT_BANDS, MODEL_SHA, TOKENIZER_SHA, binding, read_json, verify_file, write_json

ASSETS = {
    "model": {"sha256": MODEL_SHA, "bytes": 1518713958,
              "revision": "fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a"},
    "tokenizer": {"sha256": TOKENIZER_SHA, "bytes": 1146593994,
                  "revision": "38937fd35012c2dcf81987b400cec725b5eb0466"},
}


def _relative(path, parent):
    # Same-drive relative references keep the generated configuration relocatable
    # together with its local assets. Cross-drive users must colocate first.
    return Path(os.path.relpath(Path(path).resolve(strict=True), parent)).as_posix()


def create_config(*, model, tokenizer, input_manifest, output, seed=17, token_steps=8):
    """Bind existing pinned local assets; never download, copy weights or overwrite."""
    from ..api.profiles import PROFILES, require_profile
    from .common import validate_recipe
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("Config output must be new under an existing directory")
    output = output.parent.resolve() / output.name
    require_profile("generation")
    dependencies = {name: metadata.version(name) for name in PROFILES["generation"]}
    recipe = validate_recipe({"token_steps": token_steps, "diffusion_steps": 10,
                              "temperature": 1.0, "top_p": 0.8, "top_k": 0, "seed": seed})
    records = {}
    for name, path in (("model", model), ("tokenizer", tokenizer)):
        record = {**ASSETS[name], "path": _relative(path, output.parent)}
        verify_file(record, output.parent)
        records[name] = record
    source_path = Path(__file__).with_name("upstream-source.json")
    for name, path in (("source_manifest", source_path), ("input_manifest", input_manifest)):
        records[name] = {**binding(path), "path": _relative(path, output.parent)}
    _input_metadata(Path(input_manifest).resolve(strict=True))
    config = {"schema": "aureon.terramind_native.v1", **records,
              "dependencies": dependencies, "recipe": recipe}
    write_json(output, config)
    return {"schema": "aureon.generation_config_created.v1", "status": "created",
            "config": binding(output), "model_loaded": False, "downloads": False}


def _input_metadata(path):
    value = read_json(path)
    if (value.get("schema") != "aureon.terramind_native_input.v1"
            or value.get("bands") != INPUT_BANDS
            or value.get("units") != "surface_reflectance_x10000"
            or value.get("grid", {}).get("shape") != [224, 224]):
        raise ValueError("Expected prepared all-valid 224x224 S2L2A input contract")
    for name in ("image", "mask"):
        verify_file(value[name], path.parent)
    return value


def inspect_config(config):
    """Hash assets/source/input and inspect dependency metadata without model imports.

    Pixel validity and affine/CRS validation are performed by existing session
    admission before model construction; this check is not numerical qualification.
    """
    from ..api.profiles import require_profile
    from .loader import verify_config
    environment = require_profile("generation")
    value, paths, recipe = verify_config(config)
    for name in ("model", "tokenizer"):
        if any(value[name].get(key) != expected for key, expected in ASSETS[name].items()):
            raise ValueError("Pinned asset revision/size differs: " + name)
    # Config source inventory must be the reviewed manifest, not any arbitrary
    # list of 37 files satisfying the low-level experiment loader's shape check.
    if read_json(paths["source_manifest"])["files"] != read_json(Path(__file__).with_name("upstream-source.json"))["files"]:
        raise ValueError("Upstream TerraMind source inventory differs from packaged pins")
    _input_metadata(paths["input_manifest"])
    return {"schema": "aureon.generation_preflight.v1", "status": "passed",
            "environment": environment, "config": binding(config), "recipe": recipe,
            "assets_and_upstream_source": "hash_verified", "model_loaded": False,
            "cuda": "not_probed", "pixel_validity": "checked_by_session_before_loading"}
