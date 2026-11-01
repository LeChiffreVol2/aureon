"""Request-local exact encoder reuse, without modifying upstream files/classes."""
from contextlib import contextmanager
import types

_MISSING = object()


@contextmanager
def exact_conditioning_reuse(model, *, enabled):
    import torch
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Use torch.no_grad(), not inference_mode, for version-checked encoder reuse")
    core = model.sampler.model
    receipt = {"requested": enabled, "eligible": False, "encoder_calls": 0, "cache_hits": 0,
               "cache_released": False, "fallbacks": 0}
    eligible = (enabled and not model.training and not torch.is_grad_enabled()
                and set(core.encoder_embeddings) == {"untok_sen2l2a@224"}
                and list(model.modalities) == ["S2L2A"]
                and list(model.output_modalities) == ["tok_sen1grd@224"])
    receipt["eligible"] = eligible
    old_attribute = core.__dict__.get("forward_encoder", _MISSING)
    original = core.forward_encoder
    cache = {}
    versions = tuple(p._version for p in core.parameters())

    def forward(this, x, mask):
        # Exact values and masks are checked on every hit, including device/dtype.
        # No approximation, changed-conditioning reuse or cross-request cache.
        safe = eligible and not this.training and not torch.is_grad_enabled()
        safe = safe and tuple(p._version for p in this.parameters()) == versions
        if safe and cache:
            same = all(a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
                       and torch.equal(a, b) for a,b in ((x, cache["x"]), (mask, cache["mask"])))
            if same and cache["value"]._version == cache["version"]:
                receipt["cache_hits"] += 1
                return cache["value"]
            receipt["fallbacks"] += 1
            cache.clear()
        receipt["encoder_calls"] += 1
        value = original(x, mask)
        if safe:
            cache.update(x=x.clone(), mask=mask.clone(), value=value, version=value._version)
        return value

    core.forward_encoder = types.MethodType(forward, core)
    try:
        yield receipt
    finally:
        if old_attribute is _MISSING: delattr(core, "forward_encoder")
        else: core.forward_encoder = old_attribute
        cache.clear()
        receipt["cache_released"] = True
