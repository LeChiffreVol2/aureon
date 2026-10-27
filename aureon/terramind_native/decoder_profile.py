"""Experimental exact timestep transport for the pinned S1GRD decoder.

The scheduler keeps its original CPU timesteps. Only the decoder receives views
of one request-local device copy instead of copying each scalar itself. The
installed decoder, scheduler, neural arithmetic and random draws are unchanged.
"""
from contextlib import contextmanager
from functools import partial
from time import perf_counter
import types

PROFILE_ID = "s1grd-ddim10-request-timestep-table-v1"
BASELINE_ID = "s1grd-ddim10-native-timestep-transport-v1"
_MODULE = "terratorch.models.backbones.terramind.tokenizer.models.unet.unet"
_MISSING = object()


class _TimestepTable:
    """Exact CPU schedule guards; no GPU equality checks or host scalar reads."""
    def __init__(self, receipt):
        self.receipt = receipt
        self.cpu = self.snapshot = self.device_table = None
        self.views = ()
        self.index = 0
        self.disabled = False

    def _fallback(self, value, reason):
        self.receipt["fallbacks"] += 1
        self.receipt["fallback_reasons"].append(reason)
        self.disabled = True
        return value

    def stage(self, timestep, schedule, device):
        import torch
        if self.disabled:
            return self._fallback(timestep, "request_staging_disabled")
        if (not isinstance(timestep, torch.Tensor) or timestep.device.type != "cpu"
                or timestep.dtype != torch.int64 or timestep.ndim != 0
                or not isinstance(schedule, torch.Tensor) or schedule.device.type != "cpu"
                or schedule.dtype != torch.int64 or tuple(schedule.shape) != (10,)):
            return self._fallback(timestep, "unsupported_timestep_or_schedule")
        if self.cpu is None:
            # A private snapshot detects even value changes that bypass _version.
            self.cpu = schedule
            self.snapshot = schedule.clone()
            self.device_table = schedule.to(device=device, copy=True)
            self.views = self.device_table.unbind()
            self.receipt["table_copies"] += 1
            self.receipt["host_to_device_table_copies"] += int(device.type == "cuda")
            self.receipt["staged_bytes"] = self.device_table.numel() * self.device_table.element_size()
        if (schedule is not self.cpu or not torch.equal(schedule, self.snapshot)
                or self.device_table.device != device or self.index >= len(self.views)
                or not torch.equal(timestep, self.snapshot[self.index])):
            return self._fallback(timestep, "schedule_device_or_order_changed")
        value = self.views[self.index]
        self.index += 1
        self.receipt["staged_scalar_uses"] += 1
        return value

    def clear(self):
        self.cpu = self.snapshot = self.device_table = None
        self.views = ()
        self.index = 0
        self.disabled = False
        self.receipt["cache_released"] = True


def make_decoder_context(*, enabled=False):
    """Factory for TerraMindSession's per-request decoder_context hook.

    Both variants use identical decode-wall/CUDA-event instrumentation. It is
    diagnostic overhead in the comparison, not a production speed claim.
    """
    if type(enabled) is not bool:
        raise TypeError("enabled must be an explicit bool")
    return partial(decoder_context, enabled=enabled)


@contextmanager
def decoder_context(model, recipe, *, enabled=False):
    import torch
    if (recipe["diffusion_steps"] != 10 or recipe["token_steps"] not in (1, 8)
            or model.training or torch.is_grad_enabled()
            or list(model.modalities) != ["S2L2A"]
            or list(model.output_modalities) != ["tok_sen1grd@224"]):
        raise ValueError("Decoder profile requires the qualified eval/no_grad S2L2A to S1GRD ten-step path")
    tokenizer = model.tokenizer["tok_sen1grd@224"]
    decoder = tokenizer.decoder
    if (type(decoder).__name__ != "PatchedUNetCondCat" or type(decoder).__module__ != _MODULE
            or decoder.training):
        raise ValueError("Unsupported decoder implementation")
    receipt = {
        "profile_id": PROFILE_ID if enabled else BASELINE_ID,
        "enabled": enabled, "decoder_calls": 0, "decode_calls": 0,
        "table_copies": 0, "host_to_device_table_copies": 0,
        "staged_scalar_uses": 0, "staged_bytes": 0,
        "fallbacks": 0, "fallback_reasons": [], "cache_released": False,
        "restored": False, "decode_wall_seconds": 0.0,
        "denoiser_gpu_ms": None, "denoiser_host_submit_seconds": 0.0,
        "timing_scope": "Whole decode_tokens synchronized wall span; nested decoder CUDA event sum and host-submit span are overlapping diagnostics, not additive stages",
        "numerical_change": "None: same decoder forward and CPU scheduler; only the decoder timestep transport changes",
    }
    table = _TimestepTable(receipt)
    saved_forward = decoder.__dict__.get("forward", _MISSING)
    saved_decode = tokenizer.__dict__.get("decode_tokens", _MISSING)
    original_forward = decoder.forward
    original_decode = tokenizer.decode_tokens
    events = []

    def forward(this, sample, timestep, *args, **kwargs):
        receipt["decoder_calls"] += 1
        start = perf_counter()
        if enabled:
            timestep = table.stage(timestep, tokenizer.noise_scheduler.timesteps, sample.device)
        if sample.is_cuda:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
        result = original_forward(sample, timestep, *args, **kwargs)
        if sample.is_cuda:
            end.record(); events.append((begin, end))
        receipt["denoiser_host_submit_seconds"] += perf_counter() - start
        return result

    def decode(this, tokens, *args, **kwargs):
        receipt["decode_calls"] += 1
        if tokens.is_cuda: torch.cuda.synchronize(tokens.device)
        start = perf_counter()
        try:
            return original_decode(tokens, *args, **kwargs)
        finally:
            try:
                if tokens.is_cuda: torch.cuda.synchronize(tokens.device)
                receipt["decode_wall_seconds"] += perf_counter() - start
            finally:
                table.clear()

    decoder.forward = types.MethodType(forward, decoder)
    tokenizer.decode_tokens = types.MethodType(decode, tokenizer)
    try:
        yield receipt
    finally:
        if saved_forward is _MISSING: delattr(decoder, "forward")
        else: decoder.forward = saved_forward
        if saved_decode is _MISSING: delattr(tokenizer, "decode_tokens")
        else: tokenizer.decode_tokens = saved_decode
        # decode's finally synchronized outstanding work before event collection.
        try:
            if events: receipt["denoiser_gpu_ms"] = sum(a.elapsed_time(b) for a, b in events)
        finally:
            events.clear()
            table.clear()
            receipt["restored"] = True
