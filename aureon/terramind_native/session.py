"""Experimental resident TerraMind lifecycle. Importing this module is CPU-light."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import gc
from pathlib import Path
import re
import threading
from time import perf_counter
import uuid
import weakref

from .common import binding, canonical, file_hash, read_json, validate_recipe, verify_file, write_json


class SessionBusy(RuntimeError):
    """A call is in flight or another TerraMindSession owns this process."""


class SessionClosed(RuntimeError):
    """A terminal session cannot run another request."""


class SessionFailed(RuntimeError):
    """Execution failed; the session is terminal and cleanup was attempted."""


_PROCESS_OWNER = threading.Lock()
_INHERIT = object()


def _identifier(value, name):
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value)
            or value.endswith(".") or value.split(".")[0].upper() in
            {"CON", "PRN", "AUX", "NUL", *["COM%d" % i for i in range(1, 10)], *["LPT%d" % i for i in range(1, 10)]}):
        raise ValueError(name + " must be a safe 1..80 character identifier")
    return value


def _reuse_mode(value):
    if value not in ("auto", "off"):
        raise ValueError("reuse must be 'auto' or 'off'")
    return value


def _denoiser_mode(value):
    if value not in ("native", "graph"):
        raise ValueError("denoiser must be 'native' or 'graph'")
    return value


def _input(path):
    """Validate the entire supported input before model construction/admission."""
    import math
    from rasterio.crs import CRS
    from .geo import load_input
    identity = binding(path)
    image, mask, metadata = load_input(path)
    grid = metadata["grid"]
    transform = grid["transform"]
    if (grid["shape"] != [224, 224] or len(transform) != 6
            or not all(type(v) in (int, float) and math.isfinite(v) for v in transform)
            or transform[0]*transform[4] - transform[1]*transform[3] == 0):
        raise ValueError("Invalid 224x224 affine grid")
    CRS.from_user_input(grid["crs"])
    for name in ("image", "mask"):
        verify_file(metadata[name], path.parent)
    if binding(path) != identity:
        raise ValueError("Input manifest changed during admission")
    return image, mask, metadata, identity


def _decision(model, recipe, mode):
    eligible = (recipe["token_steps"] == 8 and not model.training
                and set(model.sampler.model.encoder_embeddings) == {"untok_sen2l2a@224"}
                and list(model.modalities) == ["S2L2A"]
                and list(model.output_modalities) == ["tok_sen1grd@224"])
    enabled = mode == "auto" and eligible
    reason = ("explicit_off" if mode == "off" else "no_repeated_encoder" if recipe["token_steps"] == 1
              else "eligible_eight_step_profile" if eligible else "unsupported_topology_or_mode")
    return {"mode": mode, "enabled": enabled, "reason": reason, "token_steps": recipe["token_steps"]}


class _NativeRuntime:
    """All methods execute on one session-owned thread. CPU is a test seam only."""
    device = "cuda:0"

    def __init__(self):
        self.model = None
        self.torch = None
        self.before = None
        self.refs = {}
        self.denoiser_graph = None

    def _sync(self):
        if self.device == "cuda:0": self.torch.cuda.synchronize()

    def _memory(self, peak=False):
        if self.device != "cuda:0": return {"allocated_bytes": 0, "reserved_bytes": 0}
        cuda = self.torch.cuda
        return {"allocated_bytes": cuda.max_memory_allocated() if peak else cuda.memory_allocated(),
                "reserved_bytes": cuda.max_memory_reserved() if peak else cuda.memory_reserved()}

    def _policy(self):
        t = self.torch
        t.set_float32_matmul_precision("highest")
        t.backends.cuda.matmul.allow_tf32 = False
        t.backends.cudnn.allow_tf32 = False
        t.backends.cudnn.benchmark = False
        t.backends.cuda.enable_cudnn_sdp(False)
        t.backends.cuda.enable_flash_sdp(True)
        t.backends.cuda.enable_mem_efficient_sdp(True)
        t.backends.cuda.enable_math_sdp(True)

    def _assert_policy(self):
        t = self.torch
        if (t.get_float32_matmul_precision() != "highest" or t.backends.cuda.matmul.allow_tf32
                or t.backends.cudnn.allow_tf32 or t.is_autocast_enabled()
                or t.is_grad_enabled() or self.model.training or t.backends.cuda.cudnn_sdp_enabled()
                or not t.backends.cuda.flash_sdp_enabled() or not t.backends.cuda.mem_efficient_sdp_enabled()
                or not t.backends.cuda.math_sdp_enabled()):
            raise RuntimeError("Native request left highest-FP32 eval/no-grad policy")

    def _versions(self):
        return tuple((id(p), p._version) for p in self.model.parameters())

    def load(self, config_path, work):
        import torch
        from .loader import load_native
        self.torch = torch
        self.before = {"precision": torch.get_float32_matmul_precision(), "threads": torch.get_num_threads(),
                       "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_tf32": torch.backends.cudnn.allow_tf32, "benchmark": torch.backends.cudnn.benchmark,
                       "sdp": [torch.backends.cuda.cudnn_sdp_enabled(),torch.backends.cuda.flash_sdp_enabled(),
                               torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.math_sdp_enabled()]}
        if self.device == "cuda:0" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError("TerraMindSession requires exactly one visible CUDA GPU")
        if any(self._memory().values()):
            raise RuntimeError("Session requires an empty owned-process CUDA allocator")
        torch.set_num_threads(1)
        self._policy()
        self.model, loaded, self.config, self.paths = load_native(config_path)
        self.refs = {"model": weakref.ref(self.model), "core": weakref.ref(self.model.sampler.model),
                     "tokenizer": weakref.ref(self.model.tokenizer["tok_sen1grd@224"])}
        self._policy()  # Dependency imports may change global precision.
        write_json(work / "loaded-identity.json", loaded)
        self.loaded_identity = binding(work / "loaded-identity.json")
        self.model.to(self.device)
        self._sync()
        self.versions = self._versions()

    def run(self, image, mask, metadata, input_identity, recipe, output, *, mode, decoder_context,
            capture_logits, session_identity, job_id, attempt_id, denoiser="native"):
        import numpy as np
        from .compat import BASELINE_LABEL, schedule_compatibility
        from .geo import write_cog
        from .harness import trace_native
        from .reuse import exact_conditioning_reuse
        from .rng import rng_capsule
        t = self.torch
        if self._versions() != self.versions:
            raise RuntimeError("Resident model parameters changed")
        self._policy()
        self._sync()
        if self.device == "cuda:0": t.cuda.reset_peak_memory_stats()
        start = perf_counter()
        tensor = t.from_numpy(image.copy()).unsqueeze(0).to(self.device)
        tokenizer = self.model.tokenizer["tok_sen1grd@224"]
        original_scheduler = tokenizer.noise_scheduler
        original_decoding_steps = self.model.decoding_steps
        self.model.decoding_steps = recipe["token_steps"]
        decision = _decision(self.model, recipe, mode)
        stats = {"requested": False, "cache_hits": 0, "cache_released": True,
                 "instrumentation": "disabled profile has no encoder hook"}
        result = values = None
        try:
            tokenizer.noise_scheduler = deepcopy(original_scheduler)
            devices = (0,) if self.device == "cuda:0" else ()
            with t.no_grad(), rng_capsule(recipe["seed"], devices=devices) as rng, \
                    schedule_compatibility(self.model, token_steps=recipe["token_steps"]) as compatibility, \
                    trace_native(self.model, output, detailed=capture_logits) as trace:
                reuse = (exact_conditioning_reuse(self.model, enabled=decision["enabled"])
                         if decision["enabled"] or capture_logits else nullcontext(stats))
                if denoiser == "graph":
                    from .denoiser_graph import DenoiserGraph, eligibility
                    eligible, reason = eligibility(self.model, recipe, self.device)
                    if eligible:
                        if self.denoiser_graph is None: self.denoiser_graph = DenoiserGraph(self.model)
                        decoder = self.denoiser_graph.request(self.model)
                    else:
                        decoder = nullcontext({"profile_id":"native", "enabled":False, "restored":True,
                                               "requested":"graph", "fallback":True, "reason":reason})
                else:
                    decoder = (decoder_context(self.model, dict(recipe)) if decoder_context is not None else
                               nullcontext({"profile_id": "native", "enabled": False, "restored": True}))
                with reuse as stats, decoder as decoder_receipt:
                    if not isinstance(decoder_receipt, dict) or not isinstance(decoder_receipt.get("profile_id"), str):
                        raise ValueError("Decoder context must yield a receipt with profile_id")
                    self._assert_policy()
                    self._sync(); native_start = perf_counter()
                    result = self.model({"S2L2A": tensor}, standardize=True,
                                        timesteps=recipe["diffusion_steps"], verbose=False)
                    self._sync(); native_seconds = perf_counter() - native_start
                    self._assert_policy()
                    if set(result) != {"S1GRD"}: raise ValueError("Unexpected native output modalities")
                    values = result["S1GRD"].detach().cpu().numpy()[0].copy()
                    result = None
                self._assert_policy()
            if (not rng["restored"] or not compatibility["method_restored"] or not trace["methods_restored"]
                    or not stats["cache_released"] or decoder_receipt.get("restored") is not True):
                raise RuntimeError("Request state restoration failed")
            if self._versions() != self.versions:
                raise RuntimeError("Decoder context changed resident model parameters")
            canonical(decoder_receipt)  # Never retain tensor/model objects in a public receipt.
            peak = self._memory(peak=True)
        finally:
            tokenizer.noise_scheduler = original_scheduler
            self.model.decoding_steps = original_decoding_steps
            tensor = result = None
        scheduler_restored = tokenizer.noise_scheduler is original_scheduler
        decoding_steps_restored = self.model.decoding_steps == original_decoding_steps
        np.save(output / "raw.npy", values, allow_pickle=False)
        provenance = {"schema": "aureon.terramind_session_job.v1", "session": session_identity,
                      "job_id": job_id, "attempt_id": attempt_id, "recipe": recipe,
                      "model": self.config["model"], "tokenizer": self.config["tokenizer"],
                      "source_manifest": self.config["source_manifest"], "dependencies": self.config["dependencies"],
                      "loaded_identity": self.loaded_identity, "input_manifest": input_identity, "input": metadata,
                      "synthetic": True, "sensor_observation": False,
                      "output": {"modality": "S1GRD", "channels": ["VV", "VH"], "units": "dB",
                                 "postprocessing": "Upstream unstandardization; no clipping or display conversion"},
                      "scientific_quality": "not_evaluated", "physical_quality": "not_evaluated",
                      "numerical_fidelity": "not_evaluated_per_job", "raw": binding(output / "raw.npy"),
                      "baseline_label": BASELINE_LABEL, "compatibility": compatibility, "rng": rng,
                      "reuse_decision": decision, "reuse": stats, "decoder": decoder_receipt,
                      "denoiser_mode": denoiser,
                      "precision": {"parameters": "float32", "matmul": "highest", "tf32": False, "autocast": False}}
        cog = write_cog(output / "synthetic-s1grd.tif", values, mask, metadata["grid"], provenance)
        write_json(output / "provenance.json", provenance)
        if read_json(output / "provenance.json")["raw"]["sha256"] != file_hash(output / "raw.npy"):
            raise RuntimeError("Provenance round-trip mismatch")
        return {"status": "completed", "job_id": job_id, "attempt_id": attempt_id, "recipe": recipe,
                "raw": binding(output / "raw.npy"), "cog": cog, "provenance": binding(output / "provenance.json"),
                "trace": trace, "capture_logits": capture_logits, "rng": rng, "compatibility": compatibility,
                "baseline_label": BASELINE_LABEL, "reuse_decision": decision, "reuse": stats,
                "decoder": deepcopy(decoder_receipt), "scheduler_restored": scheduler_restored,
                "decoding_steps_restored": decoding_steps_restored,
                "denoiser_mode": denoiser,
                "resident_delivery_seconds": perf_counter()-start, "native_model_seconds": native_seconds,
                "peak_cuda_allocated_bytes": peak["allocated_bytes"], "peak_cuda_reserved_bytes": peak["reserved_bytes"],
                "timing_scope": "Resident delivery includes input device transfer, request contexts, forward, token/raw/COG/provenance writes and reopening; excludes admission and startup; no fsync claim"}

    def close(self):
        graph_receipt = None
        graph_error = None
        if self.denoiser_graph is not None:
            try: graph_receipt = self.denoiser_graph.close()
            except BaseException as exc:
                from .harness import capture_failure
                graph_error = capture_failure(exc)
                graph_receipt = {**self.denoiser_graph.total, "cleanup_error":graph_error}
            finally: self.denoiser_graph = None
        self.model = None
        gc.collect()
        alive = {k: ref() is not None for k, ref in self.refs.items()}
        try:
            if self.torch is not None and self.device == "cuda:0" and self.torch.cuda.is_initialized():
                self._sync()
                clear = getattr(self.torch._C, "_cuda_clearCublasWorkspaces", None)
                if clear is not None: clear()
                self.torch.cuda.empty_cache()
            memory = self._memory() if self.torch is not None else {"allocated_bytes": 0, "reserved_bytes": 0}
        finally:
            if self.before is not None:
                t, old = self.torch, self.before
                t.backends.cuda.matmul.allow_tf32 = old["matmul_tf32"]
                t.backends.cudnn.allow_tf32 = old["cudnn_tf32"]
                # Restore the aggregate policy last: setting allow_tf32 after
                # restoring "medium" changes only CUDA's setting and leaves
                # it inconsistent with the saved CPU matmul policy in Torch2.10.
                t.set_float32_matmul_precision(old["precision"])
                t.backends.cudnn.benchmark = old["benchmark"]
                for setter,value in zip((t.backends.cuda.enable_cudnn_sdp,t.backends.cuda.enable_flash_sdp,
                                         t.backends.cuda.enable_mem_efficient_sdp,t.backends.cuda.enable_math_sdp),old["sdp"]):
                    setter(value)
                t.set_num_threads(old["threads"])
        return {"passed": graph_error is None and not any(alive.values()) and not any(memory.values()), "owned_objects_alive": alive,
                "cuda_allocated_bytes": memory["allocated_bytes"], "cuda_reserved_bytes": memory["reserved_bytes"],
                "denoiser_graph": graph_receipt}


class TerraMindSession:
    """One lazy resident model and serial owner thread; explicit close is required.

    Only one TerraMindSession may own the process runtime. This guard does not
    coordinate unrelated Torch code or the separate KAO-TerraMind B runtime.
    """

    def __init__(self, config, work, *, execute_model=False, reuse="auto", decoder_context=None, denoiser="native"):
        if execute_model is not True: raise ValueError("Pass execute_model=True explicitly")
        self._reuse = _reuse_mode(reuse)
        if decoder_context is not None and not callable(decoder_context): raise TypeError("decoder_context must be callable")
        self._decoder_context = decoder_context
        self._denoiser = _denoiser_mode(denoiser)
        if denoiser == "graph" and decoder_context is not None:
            raise ValueError("denoiser='graph' cannot compose with a custom decoder_context")
        self._config_path = Path(config).resolve(strict=True)
        self._config_identity = binding(self._config_path)
        config_value = read_json(self._config_path)
        self._recipe = validate_recipe(config_value["recipe"])
        self._input_path = (self._config_path.parent / config_value["input_manifest"]["path"]).resolve()
        work = Path(work).absolute()
        if work.exists() or work.is_symlink() or not work.parent.is_dir():
            raise ValueError("work must be a fresh directory with an existing parent")
        self.work = work.parent.resolve() / work.name
        self._gate = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aureon-terramind")
        self._state = "new"; self._runtime = None; self._owns_process = False; self._cleanup = None
        self._attempts = set(); self._loads = 0; self._load_attempts = 0; self._completed = 0
        self._owner_thread = None
        self.work.mkdir(); (self.work / "jobs").mkdir()
        self._identity = {"schema": "aureon.terramind_session.v1", "session_id": uuid.uuid4().hex,
                          "config": self._config_identity, "reuse_default": reuse,
                          "denoiser_default": denoiser,
                          "source": [binding(p) for p in sorted(Path(__file__).parent.glob("*.py"))],
                          "scope": "Experimental source-only TerraMind 1.0 base S2L2A to synthetic S1GRD; accepted B wheel unchanged"}
        write_json(self.work / "session.json", self._identity)

    def metadata(self):
        return {"state": self._state, "model_load_count": self._loads, "model_load_attempts": self._load_attempts,
                "jobs_completed": self._completed, "owner_thread_id": self._owner_thread,
                "session_id": self._identity["session_id"], "work": str(self.work)}

    def run(self, input_manifest=None, *, job_id, attempt_id, recipe=None, reuse=None,
            decoder_context=_INHERIT, capture_logits=False, denoiser=None):
        if not self._gate.acquire(blocking=False): raise SessionBusy("One call is already in flight; nothing queued")
        start = perf_counter()
        try:
            if self._state in ("failed", "closed"): raise SessionClosed("Create a new session after close or execution failure")
            job_id = _identifier(job_id, "job_id"); attempt_id = _identifier(attempt_id, "attempt_id")
            if (job_id, attempt_id) in self._attempts: raise FileExistsError("This job attempt was already admitted")
            recipe = validate_recipe(self._recipe if recipe is None else recipe)
            mode = _reuse_mode(self._reuse if reuse is None else reuse)
            hook = self._decoder_context if decoder_context is _INHERIT else decoder_context
            denoiser_mode = _denoiser_mode(self._denoiser if denoiser is None else denoiser)
            if denoiser_mode == "graph" and hook is not None:
                raise ValueError("denoiser='graph' cannot compose with a custom decoder_context")
            if hook is not None and not callable(hook): raise TypeError("decoder_context must be callable or None")
            if type(capture_logits) is not bool: raise TypeError("capture_logits must be bool")
            path = self._input_path if input_manifest is None else Path(input_manifest).resolve(strict=True)
            return self._executor.submit(self._run_owned, path, job_id, attempt_id, recipe, mode,
                                         hook, capture_logits, start, denoiser_mode).result()
        finally:
            self._gate.release()

    def _run_owned(self, path, job_id, attempt_id, recipe, mode, hook, capture_logits, request_start, denoiser="native"):
        self._owner_thread = threading.get_ident()
        if binding(self._config_path) != self._config_identity: raise ValueError("Session config changed")
        image, mask, metadata, input_identity = _input(path)
        output = self.work / "jobs" / job_id / attempt_id
        if any(p.is_symlink() for p in (self.work, self.work / "jobs", output.parent, output)):
            raise ValueError("Output paths cannot contain symlinks")
        if output.exists(): raise FileExistsError(output)
        acquired_here = not self._owns_process
        if acquired_here:
            if not _PROCESS_OWNER.acquire(blocking=False): raise SessionBusy("Another TerraMindSession owns the process")
            self._owns_process = True
        try:
            output.parent.mkdir(exist_ok=True)
            output.mkdir(exist_ok=False)
        except BaseException:
            if acquired_here:
                self._owns_process = False; _PROCESS_OWNER.release()
            raise
        self._attempts.add((job_id, attempt_id))
        self._state = "running"
        try:
            startup = 0.
            if self._runtime is None:
                start = perf_counter(); self._load_attempts += 1
                self._runtime = _NativeRuntime()
                self._runtime.load(self._config_path, self.work)
                self._loads += 1; startup = perf_counter()-start
            record = self._runtime.run(image, mask, metadata, input_identity, recipe, output, mode=mode,
                                       decoder_context=hook, capture_logits=capture_logits,
                                       session_identity=self._identity, job_id=job_id, attempt_id=attempt_id, denoiser=denoiser)
            record.update(complete_request_seconds=perf_counter()-request_start, startup_seconds=startup,
                          model_load_count=self._loads, owner_thread_id=self._owner_thread)
            write_json(output / "receipt.json", record)
            self._completed += 1; self._state = "ready"
            return deepcopy(record)
        except BaseException as exc:
            from .harness import capture_failure
            error = capture_failure(exc)
            self._state = "failed"
            try:
                write_json(output / "failure.json", {"status": "failed", "job_id": job_id, "attempt_id": attempt_id,
                                                      "error": error, "model_load_count": self._loads})
            finally:
                self._close_owned()
            raise SessionFailed("Native execution failed; session terminated. " + error) from None

    def _close_owned(self):
        if self._cleanup is not None: return self._cleanup
        self._owner_thread = threading.get_ident()
        start = perf_counter()
        try:
            cleanup = (self._runtime.close() if self._runtime is not None else
                       {"passed": True, "owned_objects_alive": {}, "cuda_allocated_bytes": 0, "cuda_reserved_bytes": 0})
        except BaseException as exc:
            from .harness import capture_failure
            cleanup = {"passed": False, "error": capture_failure(exc)}
        finally:
            self._runtime = None
            if self._owns_process:
                self._owns_process = False; _PROCESS_OWNER.release()
        cleanup.update(schema="aureon.terramind_session_cleanup.v1", model_load_count=self._loads,
                       model_load_attempts=self._load_attempts, jobs_completed=self._completed,
                       owner_thread_id=self._owner_thread, seconds=perf_counter()-start,
                       execution_failed=self._state == "failed")
        self._cleanup = cleanup
        if self._state != "failed": self._state = "closed" if cleanup["passed"] else "failed"
        write_json(self.work / "session-cleanup.json", cleanup)
        return cleanup

    def close(self):
        if not self._gate.acquire(blocking=False): raise SessionBusy("Cannot close during an in-flight call")
        try:
            if self._cleanup is None:
                self._cleanup = self._executor.submit(self._close_owned).result()
            self._executor.shutdown(wait=True)
            result = deepcopy(self._cleanup)
            if not result["passed"]: raise SessionFailed("Session cleanup failed; inspect session-cleanup.json")
            return result
        finally:
            self._gate.release()

    def __enter__(self):
        if self._state in ("failed", "closed"): raise SessionClosed("Session is terminal")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except SessionFailed as cleanup_error:
            if exc is None: raise
            exc.add_note(str(cleanup_error))
        return False
