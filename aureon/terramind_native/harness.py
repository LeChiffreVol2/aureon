"""Frozen finite native proof driver. Only the designated GPU owner executes it."""
from contextlib import contextmanager, nullcontext
import copy
import gc
import hashlib
import os
from pathlib import Path
import statistics
from time import perf_counter
import traceback
import types
import weakref

from .common import binding, digest, file_hash, read_json, write_json
from .compat import BASELINE_LABEL, schedule_compatibility
from .geo import load_input, write_cog
from .loader import load_native
from .reuse import exact_conditioning_reuse
from .rng import rng_capsule

_MISSING = object()


def capture_failure(exc):
    """Retain diagnostic text, but detach unwound frames before owned cleanup."""
    message = "".join(traceback.format_exception(type(exc),exc,exc.__traceback__))
    pending = [exc]; seen = set()
    while pending:
        error = pending.pop()
        if id(error) in seen: continue
        seen.add(id(error))
        pending.extend(e for e in (error.__cause__,error.__context__) if e is not None)
        if error.__traceback__ is not None:
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
    return message


@contextmanager
def trace_native(model, output, *, detailed):
    """Token verification for both arms; logits/stage diagnostics only in qualification."""
    import numpy as np
    import torch
    saved = []
    trace = {"logits": [], "tokens": None, "stage_seconds_diagnostic": {}, "methods_restored": False}
    def replace(obj, name, fn):
        saved.append((obj, name, obj.__dict__.get(name, _MISSING)))
        setattr(obj, name, types.MethodType(fn, obj))
    original_generate = model.sampler.generate
    def generate(this, *args, **kwargs):
        start = perf_counter()
        result = original_generate(*args, **kwargs)
        tokens = result["tok_sen1grd@224"]["tensor"].detach().cpu().numpy().copy()
        np.save(output / "tokens.npy", tokens, allow_pickle=False)
        trace["tokens"] = binding(output / "tokens.npy")
        if detailed: trace["stage_seconds_diagnostic"]["token_generation_including_trace"] = perf_counter()-start
        return result
    replace(model.sampler, "generate", generate)
    if detailed:
        core = model.sampler.model
        original_logits = core.forward_logits
        def logits(this, *args, **kwargs):
            values = original_logits(*args, **kwargs)
            step = len(trace["logits"])
            records = {}
            for name, value in values.items():
                array = value.detach().cpu().numpy()
                path = output / ("logits-%02d-%s.npy" % (step, name.replace("@", "_")))
                np.save(path, array, allow_pickle=False)
                records[name] = binding(path)
            trace["logits"].append(records)
            return values
        replace(core, "forward_logits", logits)
        tokenizer = model.tokenizer["tok_sen1grd@224"]
        original_decode = tokenizer.decode_tokens
        def decode(this, *args, **kwargs):
            torch.cuda.synchronize(); start = perf_counter()
            result = original_decode(*args, **kwargs)
            torch.cuda.synchronize()
            trace["stage_seconds_diagnostic"]["modality_decode"] = perf_counter()-start
            return result
        replace(tokenizer, "decode_tokens", decode)
    try:
        yield trace
    finally:
        for obj, name, previous in reversed(saved):
            if previous is _MISSING: delattr(obj, name)
            else: setattr(obj, name, previous)
        saved.clear()
        trace["methods_restored"] = True


def generate_one(model, tensor, mask, input_metadata, recipe, output, *, arm, detailed, provenance_base):
    import numpy as np
    import torch
    output = Path(output); output.mkdir(exist_ok=False)
    tokenizer = model.tokenizer["tok_sen1grd@224"]
    original_scheduler = tokenizer.noise_scheduler
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    cache_stats = {"requested": False, "instrumentation": "direct timed arm has no encoder hook"}
    try:
        # A new wrapper alone would retain mutable timestep buffers. Own a full scheduler clone.
        tokenizer.noise_scheduler = copy.deepcopy(original_scheduler)
        with torch.no_grad(), rng_capsule(recipe["seed"], devices=(0,)) as rng, \
                schedule_compatibility(model,token_steps=recipe["token_steps"]) as compatibility, \
                trace_native(model, output, detailed=detailed) as trace:
            reuse = exact_conditioning_reuse(model, enabled=arm == "aureon_exact_reuse") if detailed or arm == "aureon_exact_reuse" else nullcontext(cache_stats)
            with reuse as cache_stats:
                native_start = perf_counter()
                result = model({"S2L2A": tensor}, standardize=True, timesteps=recipe["diffusion_steps"], verbose=False)
                torch.cuda.synchronize()
                native_seconds = perf_counter()-native_start
                if set(result) != {"S1GRD"}: raise ValueError("Unexpected native output modalities")
                values = result["S1GRD"].detach().cpu().numpy()[0].copy()
                del result
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        if peak_allocated > int(6.5*1024**3) or peak_reserved > 7*1024**3:
            raise RuntimeError("Frozen CUDA memory cap exceeded")
        if not trace["methods_restored"] or not rng["restored"] or not compatibility["method_restored"]:
            raise RuntimeError("Native request restoration failed")
        np.save(output / "raw.npy", values, allow_pickle=False)
        provenance = {**provenance_base, "arm": arm, "recipe": recipe,
            "synthetic": True, "sensor_observation": False, "input": input_metadata,
            "output": {"modality": "S1GRD", "channels": ["VV","VH"], "units": "dB",
                       "standardize": True, "postprocessing": "Upstream unstandardization only; no display stretch or clipping"},
            "scientific_quality": {"status": "not_evaluated"},
            "physical_quality": {"status": "not_evaluated"},
            "numerical_fidelity": {"status": "pending_separate_pair_comparison"},
            "raw": binding(output / "raw.npy"), "rng": rng, "reuse": cache_stats,
            "compatibility":compatibility,"baseline_label":BASELINE_LABEL}
        cog = write_cog(output / "synthetic-s1grd.tif", values, mask, input_metadata["grid"], provenance)
        write_json(output / "provenance.json", provenance)
        # Delivery includes durable files and reopening both the raster (above) and provenance.
        if read_json(output / "provenance.json")["raw"]["sha256"] != file_hash(output / "raw.npy"):
            raise RuntimeError("Provenance round-trip mismatch")
        delivered_seconds = perf_counter()-start
        record = {"arm": arm, "recipe": recipe, "detailed": detailed,
            "complete_delivered_seconds": delivered_seconds, "native_model_seconds": native_seconds,
            "other_delivery_seconds": delivered_seconds-native_seconds,
            "peak_cuda_allocated_bytes": peak_allocated, "peak_cuda_reserved_bytes": peak_reserved,
            "raw": binding(output / "raw.npy"), "cog": cog, "provenance": binding(output / "provenance.json"),
            "rng": rng, "trace": trace, "reuse": cache_stats,"compatibility":compatibility,
            "baseline_label":BASELINE_LABEL,
            "timing_scope": "Per-call scheduler/RNG/cache setup, native forward, required token evidence, raw/COG/provenance writes and independent reopen; excludes startup/input preparation; no fsync durability claim",
            "scheduler_restored": False}
    finally:
        tokenizer.noise_scheduler = original_scheduler
    record["scheduler_restored"] = tokenizer.noise_scheduler is original_scheduler
    write_json(output / "receipt.json", record)
    return record


def compare_pair(direct, candidate, *, detailed):
    import numpy as np
    raw_exact = np.array_equal(np.load(direct["raw"]["path"], allow_pickle=False), np.load(candidate["raw"]["path"], allow_pickle=False))
    tokens_exact = direct["trace"]["tokens"]["sha256"] == candidate["trace"]["tokens"]["sha256"]
    logits_exact = True
    if detailed:
        left, right = direct["trace"]["logits"], candidate["trace"]["logits"]
        logits_exact = len(left) == len(right) and bool(left)
        if logits_exact:
            logits_exact = all(set(a) == set(b) and all(a[k]["sha256"] == b[k]["sha256"] for k in a) for a,b in zip(left,right))
    rng_exact = direct["rng"]["request_post_rng_sha256"] == candidate["rng"]["request_post_rng_sha256"]
    n = candidate["recipe"]["token_steps"]
    mechanism = candidate["reuse"].get("encoder_calls") == 1 and candidate["reuse"].get("cache_hits") == n-1 and candidate["reuse"].get("fallbacks") == 0
    if detailed: mechanism = mechanism and direct["reuse"]["encoder_calls"] == n
    checks = {"raw_fp32_exact": raw_exact, "tokens_exact": tokens_exact, "logits_exact": logits_exact if detailed else None,
              "post_rng_exact": rng_exact, "caller_rng_restored": direct["rng"]["restored"] and candidate["rng"]["restored"],
              "mechanism_counts_match": mechanism, "request_cache_released": candidate["reuse"].get("cache_released") is True,
              "scheduler_restored": direct["scheduler_restored"] and candidate["scheduler_restored"]}
    checks["identical_disclosed_compatibility"] = direct["compatibility"] == candidate["compatibility"]
    return {"passed": all(v for v in checks.values() if v is not None), "checks": checks,
            "direct_delivered_seconds": direct["complete_delivered_seconds"],
            "candidate_delivered_seconds": candidate["complete_delivered_seconds"],
            "relative_reduction": 1-candidate["complete_delivered_seconds"]/direct["complete_delivered_seconds"]}


def run_screen(config_path, spec_path, expected_spec_sha256, output):
    import numpy as np
    import torch
    if os.environ.get("AUREON_NATIVE_GPU_OWNER") != "Inference Engine": raise RuntimeError("Designated GPU-owner process required")
    if file_hash(spec_path) != expected_spec_sha256: raise ValueError("Frozen screen hash mismatch")
    spec = read_json(spec_path)
    if (spec.get("schema") != "aureon.terramind_native_screen_spec.v2"
            or spec.get("amendment",{}).get("baseline_label") != BASELINE_LABEL):
        raise ValueError("Disclosed schedule repair requires the frozen corrective-attempt specification")
    if spec["total_model_call_cap"] != 16 or spec["qualification"]["seed"] != 17 or spec["timing"]["seeds"] != [101,202,303]:
        raise ValueError("Unsupported finite screen")
    if [(p["token_decoding_steps"],p["diffusion_timesteps"]) for p in spec["profiles"]] != [(1,10),(8,10)]:
        raise ValueError("Unsupported frozen profiles")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("Exactly one visible CUDA GPU required")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    model = tensor = None; calls = 0; records = []; pairs = []; error = None; owned_refs = {}
    start = perf_counter()
    old_precision = torch.get_float32_matmul_precision()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    old_tf32 = torch.backends.cudnn.allow_tf32
    old_benchmark = torch.backends.cudnn.benchmark
    torch.set_float32_matmul_precision("highest"); torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    def progress(event):
        import json
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps({"elapsed_seconds": perf_counter()-start, "calls_started": calls, **event})+"\n")
    try:
        model, loaded, config, paths = load_native(config_path)
        owned_refs = {"model":weakref.ref(model),"core":weakref.ref(model.sampler.model),
                      "tokenizer":weakref.ref(model.tokenizer["tok_sen1grd@224"])}
        # Imports may set a global matmul policy. Reassert after all model imports.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if (torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32): raise RuntimeError("Highest FP32 policy not active")
        write_json(output / "loaded-identity.json", loaded)
        image, mask, input_metadata = load_input(paths["input_manifest"])
        write_json(output / "execution-binding.json", {"config": binding(config_path), "spec": binding(spec_path),
            "input_manifest": binding(paths["input_manifest"]), "code": [binding(p) for p in sorted(Path(__file__).parent.glob("*.py"))],
            "precision": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                          "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "autocast": False,
                          "torch_cpu_threads": torch.get_num_threads(), "gradient_mode": "no_grad"},
            "gpu": torch.cuda.get_device_name(0), "source_manifest": config["source_manifest"]})
        model.to("cuda:0")
        tensor = torch.from_numpy(image.copy()).unsqueeze(0).to("cuda:0")
        owned_refs["conditioning_tensor"] = weakref.ref(tensor)
        torch.cuda.synchronize()
        startup_seconds = perf_counter()-start
        base = {"model": config["model"], "tokenizer": config["tokenizer"], "source_manifest": config["source_manifest"],
                "dependencies": config["dependencies"], "input_manifest": config["input_manifest"], "screen_sha256": expected_spec_sha256}
        # Qualification of both profiles precedes every timed call.
        schedule = [(p,17,["direct","aureon_exact_reuse"],True) for p in spec["profiles"]]
        for p in spec["profiles"]:
            orders = spec["timing"]["orders"][p["id"]]
            for seed, order in zip(spec["timing"]["seeds"], orders):
                arms = order.split(",")
                if sorted(arms) != ["aureon_exact_reuse","direct"]: raise ValueError("Invalid paired order")
                schedule.append((p,seed,arms,False))
        if sum(len(s[2]) for s in schedule) != 16: raise ValueError("Invalid call budget")
        for profile, seed, arms, detailed in schedule:
            recipe = {**config["recipe"], "token_steps": profile["token_decoding_steps"], "diffusion_steps":10, "seed":seed}
            model.decoding_steps = recipe["token_steps"]
            current = {}
            for arm in arms:
                if calls >= 16 or perf_counter()-start > 590: raise RuntimeError("Finite call/time cap reached")
                calls += 1
                name = "%02d-%s-%s-seed%d" % (calls,profile["id"],arm,seed)
                progress({"event":"call_start", "name":name})
                record = generate_one(model, tensor, mask, input_metadata, recipe, output/name,
                                      arm=arm, detailed=detailed, provenance_base=base)
                records.append(record); current[arm] = record
                progress({"event":"call_done", "name":name, "peak_cuda_allocated_bytes":record["peak_cuda_allocated_bytes"],
                          "peak_cuda_reserved_bytes":record["peak_cuda_reserved_bytes"]})
            pair = {"profile": profile["id"], "seed":seed, "detailed":detailed,
                    **compare_pair(current["direct"],current["aureon_exact_reuse"],detailed=detailed)}
            pairs.append(pair); write_json(output/("pair-%02d.json" % len(pairs)), pair)
            if not pair["passed"]: raise RuntimeError("Exact native pair failed; no subsequent calls")
        result = {"status":"passed_finite_native_screen", "calls_started":calls, "startup_seconds":startup_seconds,
                  "pairs":pairs, "scientific_quality":"not_evaluated", "physical_quality":"not_evaluated"}
        result["profiles"] = {}
        for p in spec["profiles"]:
            rates = [x["relative_reduction"] for x in pairs if not x["detailed"] and x["profile"] == p["id"]]
            result["profiles"][p["id"]] = {"paired_reductions":rates, "median_paired_reduction":statistics.median(rates),
                "all_three_positive":all(x>0 for x in rates), "promising_local_screen":statistics.median(rates)>=.1 and all(x>0 for x in rates)}
    except Exception as exc:
        error = capture_failure(exc)
        result = {"status":"failed", "error":error, "calls_started":calls, "pairs":pairs}
    finally:
        cleanup_start = perf_counter()
        model = tensor = None
        gc.collect()
        objects_alive = {name:ref() is not None for name,ref in owned_refs.items()}
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            clear = getattr(torch._C,"_cuda_clearCublasWorkspaces",None)
            if clear is not None: clear()
            torch.cuda.empty_cache()
        cleanup = {"cuda_allocated_bytes":torch.cuda.memory_allocated(), "cuda_reserved_bytes":torch.cuda.memory_reserved(),
                   "seconds":perf_counter()-cleanup_start,"owned_objects_alive":objects_alive}
        cleanup["passed"] = (cleanup["cuda_allocated_bytes"] == cleanup["cuda_reserved_bytes"] == 0
                             and not any(objects_alive.values()))
        torch.set_float32_matmul_precision(old_precision); torch.backends.cudnn.allow_tf32 = old_tf32
        torch.backends.cudnn.benchmark = old_benchmark
        torch.set_num_threads(old_threads)
    result.update(cleanup=cleanup, occupied_seconds=perf_counter()-start,
                  timing_limitations="Three pairs, one 224px crop and one GPU; not significance, tail latency, scientific quality or broad competitor superiority")
    if not cleanup["passed"]: result["status"] = "failed_cleanup"
    write_json(output/"result.json", result)
    return result
