"""Strict local native loading; upstream permissive checkpoint filters are unused."""
from contextlib import ExitStack
import importlib.metadata as metadata
import os
from pathlib import Path
from unittest.mock import patch

from .common import MODEL_SHA, TOKENIZER_SHA, digest, file_hash, read_json, validate_recipe, verify_file


def verify_config(config_path):
    config_path = Path(config_path).resolve(strict=True)
    config = read_json(config_path)
    expected = {"schema", "model", "tokenizer", "source_manifest", "dependencies", "input_manifest", "recipe"}
    if set(config) != expected or config["schema"] != "aureon.terramind_native.v1":
        raise ValueError("Unexpected native config schema/fields")
    recipe = validate_recipe(config["recipe"])
    paths = {name: verify_file(config[name], config_path.parent)
             for name in ("model", "tokenizer", "source_manifest", "input_manifest")}
    if config["model"]["sha256"] != MODEL_SHA or config["tokenizer"]["sha256"] != TOKENIZER_SHA:
        raise ValueError("Only the pinned TerraMind base and S1GRD tokenizer are supported")
    for name in ("model", "tokenizer"):
        if not config[name].get("revision") or type(config[name].get("bytes")) is not int:
            raise ValueError("Immutable artifact revision and exact byte size required")
    required = {"terratorch", "torch", "numpy", "rasterio", "diffusers", "timm", "einops", "huggingface-hub"}
    if not required.issubset(config["dependencies"]): raise ValueError("Incomplete dependency lock")
    for name, version in config["dependencies"].items():
        if metadata.version(name) != version: raise ValueError("Dependency differs: " + name)
    source = read_json(paths["source_manifest"])
    base = Path(metadata.distribution("terratorch").locate_file(""))
    if len(source["files"]) != 37: raise ValueError("Expected inspected 37-file TerraMind source inventory")
    for item in source["files"]:
        verify_file(item, base)
    return config, paths, recipe


def select_state(expected, checkpoint, *, allow_inactive_embeddings):
    """Reject every absent/mismatched active tensor; report unused modalities."""
    if not isinstance(checkpoint, dict) or not all(isinstance(k, str) for k in checkpoint):
        raise ValueError("Expected a flat restricted tensor state dictionary")
    missing = sorted(set(expected) - set(checkpoint))
    if missing: raise ValueError("Missing active checkpoint tensors: " + repr(missing[:12]))
    extra = sorted(set(checkpoint) - set(expected))
    if extra and (not allow_inactive_embeddings or any(
            not k.startswith(("encoder_embeddings.", "decoder_embeddings.")) for k in extra)):
        raise ValueError("Unexpected checkpoint tensors: " + repr(extra[:12]))
    selected = {}
    for key, target in expected.items():
        value = checkpoint[key]
        if not hasattr(value, "shape") or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError("Checkpoint shape/dtype mismatch: " + key)
        selected[key] = value
    # Extra active-modality keys are errors too; only whole unused modalities are omitted.
    active = {tuple(k.split(".")[:2]) for k in expected if k.startswith(("encoder_embeddings.", "decoder_embeddings."))}
    if any(tuple(k.split(".")[:2]) in active for k in extra):
        raise ValueError("Unexpected key inside an active modality")
    return selected, extra


def strict_load(module, path, *, allow_inactive_embeddings=False):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = module.state_dict()
    selected, unused = select_state(expected, checkpoint, allow_inactive_embeddings=allow_inactive_embeddings)
    loaded = module.load_state_dict(selected, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys: raise ValueError("Strict checkpoint load failed")
    records = []
    import hashlib
    for name, tensor in module.state_dict().items():
        if not torch.equal(tensor, selected[name]): raise ValueError("Loaded value mismatch: " + name)
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite loaded tensor: " + name)
        records.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "sha256": hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()})
    return {"matched_all_active_tensor_values": True, "tensor_count": len(records),
            "loaded_tensor_manifest_sha256": digest(records), "tensors": records,
            "unused_inactive_modality_keys": unused}


def verify_embedding_aliases(model):
    # TerraMind wraps the mapping in a fresh ModuleDict, retaining its modules.
    # Container identity is not required; every active embedding module is.
    for name in ("encoder_embeddings", "decoder_embeddings"):
        outer = getattr(model, name)
        inner = getattr(model.sampler.model, name)
        if set(outer) != set(inner) or any(outer[k] is not inner[k] for k in outer):
            raise ValueError("Upstream active embedding alias relationship changed")


def load_native(config_path):
    config, paths, recipe = verify_config(config_path)
    # The harness is a dedicated offline process, not a shared application server.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from terratorch.models.backbones.terramind.model import terramind_register as registry
    from terratorch.models.backbones.terramind.tokenizer import tokenizer_register
    from .rng import rng_capsule
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with rng_capsule(0), ExitStack() as stack:
            for target in (registry, tokenizer_register):
                stack.enter_context(patch.object(target, "hf_hub_download", side_effect=RuntimeError("Implicit download forbidden")))
            model = registry.terramind_v1_base_generate(
                pretrained=False, modalities=["S2L2A"], output_modalities=["S1GRD"],
                standardize=True, img_size=224, decoding_steps=recipe["token_steps"],
                temps=recipe["temperature"], top_p=recipe["top_p"], top_k=recipe["top_k"],
                timesteps=recipe["diffusion_steps"])
        if set(model.tokenizer) != {"tok_sen1grd@224"}:
            raise ValueError("Unexpected tokenizer set")
        if set(model.sampler.model.encoder_embeddings) != {"untok_sen2l2a@224"}:
            raise ValueError("Unexpected conditioning encoder modalities")
        verify_embedding_aliases(model)
        model_receipt = strict_load(model.sampler.model, paths["model"], allow_inactive_embeddings=True)
        tokenizer_receipt = strict_load(model.tokenizer["tok_sen1grd@224"], paths["tokenizer"])
        model.eval().requires_grad_(False)
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in model.state_dict().values()):
            raise ValueError("Native model is not entirely FP32")
        # A changed file after construction invalidates the binding before GPU dispatch.
        for name in ("model", "tokenizer"):
            if file_hash(paths[name]) != config[name]["sha256"]: raise ValueError("Artifact changed during load")
        return model, {"model": model_receipt, "tokenizer": tokenizer_receipt,
                       "config_sha256": file_hash(config_path), "recipe": recipe,
                       "pretrained_downloads": False, "fallback_random_tensors": 0}, config, paths
    finally:
        # Optional-dependency import tracebacks can retain this completed frame.
        # The return tuple already owns the successful result; do not leave a
        # second model owner in this frame on either success or failure.
        model = None
        torch.set_default_dtype(previous_dtype)
