"""Disclosed compatibility fix for the inspected TerraTorch ROAR call boundary.

TerraMindGeneration.forward passes total num_tokens=196 to every iteration.
GenerationSampler.generate supports num_tokens=None to honor its own schedule.
This context changes that argument only for the fixed eight-step profile, equally
in both arms. Upstream source and sampler numerical implementation are unchanged.
"""
from contextlib import contextmanager
from numbers import Integral
import types

COMPATIBILITY_ID = "terramind-roar-schedule-counts-v1"
BASELINE_LABEL = "Direct installed TerraTorch with disclosed multi-step schedule compatibility fix"
_MISSING = object()


@contextmanager
def schedule_compatibility(model, *, token_steps):
    if token_steps not in (1,8) or type(token_steps) is not int:
        raise ValueError("Compatibility is restricted to the frozen one/eight-step profiles")
    if list(model.modalities) != ["S2L2A"] or list(model.output_modalities) != ["tok_sen1grd@224"]:
        raise ValueError("Compatibility requires the fixed single-output native path")
    sampler = model.sampler
    original = sampler.generate
    previous = sampler.__dict__.get("generate",_MISSING)
    receipt = {"id":COMPATIBILITY_ID,"baseline_label":BASELINE_LABEL,"calls":0,
               "token_steps":token_steps,"applied":False,"method_restored":False}

    def generate(this, mod_dict, schedule, top_k=0., top_p=0., tokenizer=None,
                 verbose=False, seed=None, num_tokens=None):
        if num_tokens != 196 or len(schedule) != token_steps:
            raise ValueError("Unexpected total-token override or schedule length")
        if tuple(mod_dict["tok_sen1grd@224"]["tensor"].shape) != (1,196):
            raise ValueError("Compatibility requires exactly 196 single-sample target tokens")
        counts = []
        for step in schedule:
            count = step["num_tokens"]
            if (not isinstance(count,Integral) or isinstance(count,bool) or count <= 0
                    or step["target_domain"] != "tok_sen1grd@224" or step["scheme"] != "roar"
                    or step["cfg_scale"] != 1. or step["temperature"] != 1.):
                raise ValueError("Unexpected/empty native schedule step")
            counts.append(int(count))
        expected = [196] if token_steps == 1 else [25,25,25,25,24,24,24,24]
        if counts != expected or sum(counts) != 196:
            raise ValueError("Schedule does not match the inspected 196-token partition")
        forwarded = None if token_steps == 8 else num_tokens
        receipt.update(calls=receipt["calls"]+1,applied=token_steps==8,
                       schedule_counts=counts,received_num_tokens=int(num_tokens),forwarded_num_tokens=forwarded)
        return original(mod_dict,schedule,top_k=top_k,top_p=top_p,tokenizer=tokenizer,
                        verbose=verbose,seed=seed,num_tokens=forwarded)

    sampler.generate = types.MethodType(generate,sampler)
    try:
        yield receipt
    finally:
        if previous is _MISSING: delattr(sampler,"generate")
        else: sampler.generate = previous
        receipt["method_restored"] = True
