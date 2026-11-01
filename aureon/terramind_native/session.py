"""Experimental resident TerraMind lifecycle. Importing this module is CPU-light."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING, cast, Literal, Self
from collections.abc import Mapping
from types import ModuleType, TracebackType
from ._types import TerraMindModel, Metadata, PathInput, FloatArray, BoolArray, InputSnapshot, DecoderContext
if TYPE_CHECKING:
    from ..api.contracts import GenerationResult
    from ..api.memory_input import GenerationInput
    from .denoiser_graph import DenoiserGraph
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
from .execution import completed_selection, plan_generation_execution, validate_selection
from .loader import config_model_id
from .families import get_terramind_spec
from .._reference.integrity import CoreFingerprint, verify_core
from .selected_core import SelectedPrimitives, core_receipt


class SessionBusy(RuntimeError):
    """A call is in flight or another TerraMindSession owns this process."""


class SessionClosed(RuntimeError):
    """A terminal session cannot run another request."""


class SessionFailed(RuntimeError):
    """Execution failed; the session is terminal and cleanup was attempted."""


from .._reference.terramind.session import _PROCESS_OWNER as _PROCESS_OWNER  # Shared resident ownership.
_INHERIT = object()


def _identifier(value: object, name: str) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value)
            or value.endswith(".") or value.split(".")[0].upper() in
            {"CON", "PRN", "AUX", "NUL", *["COM%d" % i for i in range(1, 10)], *["LPT%d" % i for i in range(1, 10)]}):
        raise ValueError(name + " must be a safe 1..80 character identifier")
    return value


def _reuse_mode(value: str) -> str:
    if value not in ("auto", "off"):
        raise ValueError("reuse must be 'auto' or 'off'")
    return value


def _denoiser_mode(value: str) -> str:
    if value not in ("native", "graph"):
        raise ValueError("denoiser must be 'native' or 'graph'")
    return value


def _input(path: Path) -> InputSnapshot:
    """Validate the entire supported input before model construction/admission."""
    import math
    from rasterio.crs import CRS  # Optional upstream typing is declared in the implementation-checker boundary policy.
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


def _decision(model: TerraMindModel, recipe: Metadata, mode: str) -> Metadata:
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
    model: TerraMindModel | None
    torch: ModuleType | None
    before: Metadata | None
    refs: dict[str, weakref.ReferenceType[Any]]
    denoiser_graph: DenoiserGraph | None
    config: Metadata
    paths: dict[str, Path]
    loaded_identity: Metadata

    def __init__(self) -> None:
        self.model = None
        self.torch = None
        self.before = None
        self.refs = {}
        self.denoiser_graph = None
        self.core_identity: CoreFingerprint | None = None
        self.numerical_core: SelectedPrimitives | None = None

    def _sync(self) -> None:
        if self.device == "cuda:0": cast("ModuleType", self.torch).cuda.synchronize()

    def _memory(self, peak: bool=False) -> dict[str, int]:
        if self.device != "cuda:0": return {"allocated_bytes": 0, "reserved_bytes": 0}
        cuda = cast("ModuleType", self.torch).cuda
        return {"allocated_bytes": cuda.max_memory_allocated() if peak else cuda.memory_allocated(),
                "reserved_bytes": cuda.max_memory_reserved() if peak else cuda.memory_reserved()}

    def _policy(self) -> None:
        t = cast("ModuleType", self.torch)
        t.set_float32_matmul_precision("highest")
        t.backends.cuda.matmul.allow_tf32 = False
        t.backends.cudnn.allow_tf32 = False
        t.backends.cudnn.benchmark = False
        t.backends.cuda.enable_cudnn_sdp(False)
        t.backends.cuda.enable_flash_sdp(True)
        t.backends.cuda.enable_mem_efficient_sdp(True)
        t.backends.cuda.enable_math_sdp(True)

    def _assert_policy(self) -> None:
        t = cast("ModuleType", self.torch)
        if (t.get_float32_matmul_precision() != "highest" or t.backends.cuda.matmul.allow_tf32
                or t.backends.cudnn.allow_tf32 or t.is_autocast_enabled()
                or t.is_grad_enabled() or cast("TerraMindModel", self.model).training or t.backends.cuda.cudnn_sdp_enabled()
                or not t.backends.cuda.flash_sdp_enabled() or not t.backends.cuda.mem_efficient_sdp_enabled()
                or not t.backends.cuda.math_sdp_enabled()):
            raise RuntimeError("Native request left highest-FP32 eval/no-grad policy")

    def _versions(self) -> tuple[tuple[int, int], ...]:
        return tuple((id(p), p._version) for p in cast("TerraMindModel", self.model).parameters())

    def load(self, config_path: PathInput, work: Path) -> None:
        import torch
        from .loader import load_native
        self.torch = torch
        if config_model_id(read_json(config_path)) == "terramind-v1-base":
            from .selected_core import load_primitives
            self.numerical_core = load_primitives(self.core_identity or verify_core())
        self.before = {"precision": torch.get_float32_matmul_precision(), "threads": torch.get_num_threads(),
                       "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_tf32": torch.backends.cudnn.allow_tf32, "benchmark": torch.backends.cudnn.benchmark,
                       "sdp": [torch.backends.cuda.cudnn_sdp_enabled(),torch.backends.cuda.flash_sdp_enabled(),  # type: ignore[no-untyped-call]  # PyTorch policy queries lack annotations.
                               torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.math_sdp_enabled()]}  # type: ignore[no-untyped-call]  # PyTorch policy queries lack annotations.
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

    def run(self, image: FloatArray, mask: BoolArray, metadata: Metadata, input_identity: Metadata, recipe: Metadata, output: Path, *, mode: str, decoder_context: DecoderContext | None,
            capture_logits: bool, session_identity: Metadata, job_id: str, attempt_id: str, denoiser: str="native", execution_plan: Metadata | None=None,
            delivery: str="files") -> Metadata:
        import numpy as np
        from .compat import BASELINE_LABEL, schedule_compatibility
        from .geo import write_cog
        from .harness import trace_native
        from .reuse import exact_conditioning_reuse
        from .rng import rng_capsule
        from .memory_output import array_identity, trace_memory
        from .denoiser_graph import DenoiserGraph, eligibility
        core = self.numerical_core
        schedule = core.schedule if core is not None else schedule_compatibility
        rng_context = core.rng if core is not None else rng_capsule
        reuse_context = core.reuse if core is not None else exact_conditioning_reuse
        decision_function = core.decision if core is not None else _decision
        graph_factory = core.graph_factory if core is not None else DenoiserGraph
        graph_eligibility = core.graph_eligibility if core is not None else eligibility
        t = cast("ModuleType", self.torch)
        if self._versions() != self.versions:
            raise RuntimeError("Resident model parameters changed")
        self._policy()
        self._sync()
        if self.device == "cuda:0": t.cuda.reset_peak_memory_stats()
        start = perf_counter()
        tensor = t.from_numpy(image.copy()).unsqueeze(0).to(self.device)
        tokenizer = cast("TerraMindModel", self.model).tokenizer["tok_sen1grd@224"]
        original_scheduler = tokenizer.noise_scheduler
        original_decoding_steps = cast("TerraMindModel", self.model).decoding_steps
        cast("TerraMindModel", self.model).decoding_steps = recipe["token_steps"]
        decision = decision_function(cast("TerraMindModel", self.model), recipe, mode)
        stats: Metadata = {"requested": False, "cache_hits": 0, "cache_released": True,
                 "instrumentation": "disabled profile has no encoder hook"}
        result = values = None
        from contextlib import AbstractContextManager
        decoder: AbstractContextManager[Metadata]
        try:
            tokenizer.noise_scheduler = deepcopy(original_scheduler)
            devices = (0,) if self.device == "cuda:0" else ()
            with t.no_grad(), rng_context(recipe["seed"], devices=devices) as rng, \
                    schedule(cast("TerraMindModel", self.model), token_steps=recipe["token_steps"]) as compatibility, \
                    (trace_memory(cast("TerraMindModel", self.model)) if delivery == "memory" else
                     trace_native(cast("TerraMindModel", self.model), output, detailed=capture_logits)) as trace:
                reuse = (reuse_context(cast("TerraMindModel", self.model), enabled=decision["enabled"])
                         if decision["enabled"] or capture_logits else nullcontext(stats))
                if denoiser == "graph":
                    eligible, reason = graph_eligibility(cast("TerraMindModel", self.model), recipe, self.device)
                    if eligible:
                        if self.denoiser_graph is None: self.denoiser_graph = graph_factory(cast("TerraMindModel", self.model))
                        decoder = self.denoiser_graph.request(cast("TerraMindModel", self.model))
                    else:
                        decoder = nullcontext({"profile_id":"native", "enabled":False, "restored":True,
                                               "requested":"graph", "fallback":True, "reason":reason})
                else:
                    decoder = (decoder_context(cast("TerraMindModel", self.model), dict(recipe)) if decoder_context is not None else
                               nullcontext({"profile_id": "native", "enabled": False, "restored": True}))
                with reuse as stats, decoder as decoder_receipt:
                    if not isinstance(decoder_receipt, dict) or not isinstance(decoder_receipt.get("profile_id"), str):
                        raise ValueError("Decoder context must yield a receipt with profile_id")
                    self._assert_policy()
                    self._sync(); native_start = perf_counter()
                    result = cast("TerraMindModel", self.model)({"S2L2A": tensor}, standardize=True,
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
            cast("TerraMindModel", self.model).decoding_steps = original_decoding_steps
            tensor = result = None
        scheduler_restored = tokenizer.noise_scheduler is original_scheduler
        decoding_steps_restored = cast("TerraMindModel", self.model).decoding_steps == original_decoding_steps
        if values.shape != (2, 224, 224) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Expected finite two-channel FP32 native output")
        if delivery == "files":
            np.save(output / "raw.npy", values, allow_pickle=False)
        selection = ({"execution_selection": completed_selection(execution_plan, decision, decoder_receipt)}
                     if execution_plan is not None else {})
        shared_core = ({"numerical_core": core.receipt(route="memory" if delivery == "memory" else "active_file_adapter")}
                       if core is not None else {})
        provenance = {**shared_core, "schema": "aureon.terramind_session_job.v1", "session": session_identity,
                      "model_id": config_model_id(self.config),
                      **selection,
                      "job_id": job_id, "attempt_id": attempt_id, "recipe": recipe,
                      "model": self.config["model"], "tokenizer": self.config["tokenizer"],
                      "source_manifest": self.config["source_manifest"], "dependencies": self.config["dependencies"],
                      "loaded_identity": self.loaded_identity,
                      **({"input_identity": input_identity} if delivery == "memory" else {"input_manifest": input_identity}),
                      "input": metadata,
                      "synthetic": True, "sensor_observation": False,
                      "output": {"modality": "S1GRD", "channels": ["VV", "VH"], "units": "dB",
                                 "postprocessing": "Upstream unstandardization; no clipping or display conversion"},
                      "scientific_quality": "not_evaluated", "physical_quality": "not_evaluated",
                      "numerical_fidelity": "not_evaluated_per_job",
                      **({"array": array_identity(values), "valid_mask": array_identity(mask), "delivery": "memory"}
                         if delivery == "memory" else {"raw": binding(output / "raw.npy")}),
                      "baseline_label": BASELINE_LABEL, "compatibility": compatibility, "rng": rng,
                      "reuse_decision": decision, "reuse": stats, "decoder": decoder_receipt,
                      "denoiser_mode": denoiser,
                      "precision": {"parameters": "float32", "matmul": "highest", "tf32": False, "autocast": False}}
        if delivery == "files":
            cog = write_cog(output / "synthetic-s1grd.tif", values, mask, metadata["grid"], provenance)
            write_json(output / "provenance.json", provenance)
            if read_json(output / "provenance.json")["raw"]["sha256"] != file_hash(output / "raw.npy"):
                raise RuntimeError("Provenance round-trip mismatch")
            artifacts: Metadata = {"raw": binding(output / "raw.npy"), "cog": cog, "provenance": binding(output / "provenance.json")}
        else:
            artifacts = {"array": values, "valid_mask": mask.copy(), "provenance": provenance}
        return {**shared_core, "status": "completed", "job_id": job_id, "attempt_id": attempt_id, "recipe": recipe,
                "model_id": config_model_id(self.config),
                **selection,
                **artifacts,
                "trace": trace, "capture_logits": capture_logits, "rng": rng, "compatibility": compatibility,
                "baseline_label": BASELINE_LABEL, "reuse_decision": decision, "reuse": stats,
                "decoder": deepcopy(decoder_receipt), "scheduler_restored": scheduler_restored,
                "decoding_steps_restored": decoding_steps_restored,
                "denoiser_mode": denoiser,
                "resident_delivery_seconds": perf_counter()-start, "native_model_seconds": native_seconds,
                "peak_cuda_allocated_bytes": peak["allocated_bytes"], "peak_cuda_reserved_bytes": peak["reserved_bytes"],
                "timing_scope": ("Memory delivery includes device transfer, contexts, forward and array/token hashes; excludes admission/startup; not a benchmark"
                                 if delivery == "memory" else "Resident delivery includes input device transfer, request contexts, forward, token/raw/COG/provenance writes and reopening; excludes admission and startup; no fsync claim")}

    def close(self) -> Metadata:
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
            if cast("ModuleType", self.torch) is not None and self.device == "cuda:0" and cast("ModuleType", self.torch).cuda.is_initialized():
                self._sync()
                clear = getattr(cast("ModuleType", self.torch)._C, "_cuda_clearCublasWorkspaces", None)
                if clear is not None: clear()
                cast("ModuleType", self.torch).cuda.empty_cache()
            memory = self._memory() if cast("ModuleType", self.torch) is not None else {"allocated_bytes": 0, "reserved_bytes": 0}
        finally:
            if self.before is not None:
                t, old = cast("ModuleType", self.torch), self.before
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

    _runtime: _NativeRuntime | None
    _cleanup: Metadata | None
    _attempts: set[tuple[str, str]]
    _owner_thread: int | None

    def __init__(self, config: PathInput, work: PathInput, *, execute_model: bool=False, reuse: str="auto", decoder_context: DecoderContext | None=None,
                 denoiser: str="native", execution_profile: str | None=None, model_id: str | None=None) -> None:
        if execute_model is not True: raise ValueError("Pass execute_model=True explicitly")
        validate_selection(execution_profile, reuse, denoiser)
        if execution_profile is not None and decoder_context is not None:
            raise ValueError("Named execution profiles cannot compose with a custom decoder_context")
        self._execution_profile = execution_profile
        self._reuse = _reuse_mode(reuse)
        if decoder_context is not None and not callable(decoder_context): raise TypeError("decoder_context must be callable")
        self._decoder_context = decoder_context
        self._denoiser = _denoiser_mode(denoiser)
        if denoiser == "graph" and decoder_context is not None:
            raise ValueError("denoiser='graph' cannot compose with a custom decoder_context")
        self._config_path = Path(config).resolve(strict=True)
        self._config_identity = binding(self._config_path)
        config_value = read_json(self._config_path)
        self.model_id = config_model_id(config_value)
        self._core_identity = verify_core() if self.model_id == "terramind-v1-base" else None
        if model_id is not None and model_id != self.model_id:
            raise ValueError("Requested model_id differs from generation config")
        self._recipe = validate_recipe(config_value["recipe"])
        self._input_path = ((self._config_path.parent / config_value["input_manifest"]["path"]).resolve()
                            if "input_manifest" in config_value else None)
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
                          "model_id": self.model_id,
                          **({"numerical_core": core_receipt(self._core_identity, route="resident_memory_or_group")}
                             if self._core_identity is not None else {}),
                          "config": self._config_identity, "reuse_default": reuse,
                          "denoiser_default": denoiser,
                          "execution_profile": execution_profile,
                          "source": [binding(p) for p in sorted(Path(__file__).parent.glob("*.py"))],
                          "scope": "TerraMind 1.0 " + get_terramind_spec(self.model_id).size + " S2L2A to synthetic S1GRD; GPU qualification is artifact-specific"}
        write_json(self.work / "session.json", self._identity)

    def metadata(self) -> Metadata:
        return {**({"numerical_core": core_receipt(self._core_identity, route="resident_memory_or_group")}
                   if self._core_identity is not None else {}), "state": self._state, "model_load_count": self._loads, "model_load_attempts": self._load_attempts,
                "model_id": self.model_id,
                "jobs_completed": self._completed, "owner_thread_id": self._owner_thread,
                "session_id": self._identity["session_id"], "work": str(self.work)}

    def generate(self, input_data: GenerationInput, *, job_id: str | None=None, attempt_id: str="1", recipe: Mapping[str, Any] | None=None,
                 execution_profile: object=_INHERIT, reuse: str | None=None, denoiser: str | None=None) -> GenerationResult:
        """Generate into owned CPU arrays, without input/output raster or NPY files.

        Small session/job identity and cleanup JSON records remain in work.
        GenerationInput snapshots data before admission; no grid is required.
        """
        from ..api.memory_input import GenerationInput
        if not isinstance(input_data, GenerationInput):
            raise TypeError("generate requires GenerationInput with explicit input semantics")
        return cast("GenerationResult", self.run(input_data, job_id=uuid.uuid4().hex if job_id is None else job_id, attempt_id=attempt_id,
                        recipe=recipe, execution_profile=execution_profile, reuse=reuse, denoiser=denoiser))

    def run(self, input_manifest: GenerationInput | PathInput | None=None, *, job_id: str, attempt_id: str, recipe: Mapping[str, Any] | None=None, reuse: str | None=None,
            decoder_context: object=_INHERIT, capture_logits: bool=False, denoiser: str | None=None, execution_profile: object=_INHERIT) -> Metadata:
        if not self._gate.acquire(blocking=False): raise SessionBusy("One call is already in flight; nothing queued")
        start = perf_counter()
        try:
            if self._state in ("failed", "closed"): raise SessionClosed("Create a new session after close or execution failure")
            job_id = _identifier(job_id, "job_id"); attempt_id = _identifier(attempt_id, "attempt_id")
            if (job_id, attempt_id) in self._attempts: raise FileExistsError("This job attempt was already admitted")
            recipe = validate_recipe(self._recipe if recipe is None else recipe)
            mode = _reuse_mode(self._reuse if reuse is None else reuse)
            hook = self._decoder_context if decoder_context is _INHERIT else cast("DecoderContext | None", decoder_context)
            denoiser_mode = _denoiser_mode(self._denoiser if denoiser is None else denoiser)
            profile = self._execution_profile if execution_profile is _INHERIT else cast("str | None", execution_profile)
            validate_selection(profile, mode, denoiser_mode)
            execution_plan = None
            if profile is not None:
                if hook is not None:
                    raise ValueError("Named execution profiles cannot compose with a custom decoder_context")
                execution_plan = plan_generation_execution(execution_profile=profile, recipe=recipe, model_id=self.model_id)
                mode = execution_plan["planned"]["reuse"]
                denoiser_mode = execution_plan["planned"]["denoiser"]
            if denoiser_mode == "graph" and hook is not None:
                raise ValueError("denoiser='graph' cannot compose with a custom decoder_context")
            if hook is not None and not callable(hook): raise TypeError("decoder_context must be callable or None")
            if type(capture_logits) is not bool: raise TypeError("capture_logits must be bool")
            from ..api.memory_input import GenerationInput
            if isinstance(input_manifest, GenerationInput):
                if capture_logits:
                    raise ValueError("In-memory generation does not persist diagnostic logits")
                path: Path | InputSnapshot | None = input_manifest.snapshot()
            else:
                path = self._input_path if input_manifest is None else Path(input_manifest).resolve(strict=True)
                if path is None:
                    raise ValueError("Assets-only config requires GenerationInput or an explicit input manifest")
            return self._executor.submit(self._run_owned, cast("Path | InputSnapshot", path), job_id, attempt_id, recipe, mode,
                                         hook, capture_logits, start, denoiser_mode, execution_plan).result()
        finally:
            self._gate.release()

    def _runtime_factory(self) -> _NativeRuntime:
        return _NativeRuntime()

    def _run_owned(self, path: Path | InputSnapshot, job_id: str, attempt_id: str, recipe: Metadata, mode: str, hook: DecoderContext | None, capture_logits: bool, request_start: float,
                   denoiser: str="native", execution_plan: Metadata | None=None) -> Metadata:
        self._owner_thread = threading.get_ident()
        if binding(self._config_path) != self._config_identity: raise ValueError("Session config changed")
        memory = isinstance(path, tuple)
        image, mask, metadata, input_identity = cast(InputSnapshot, path) if memory else _input(cast(Path, path))
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
                self._runtime = self._runtime_factory()
                self._runtime.core_identity = self._core_identity
                self._runtime.load(self._config_path, self.work)
                self._loads += 1; startup = perf_counter()-start
            selection_options: Metadata = {"execution_plan": execution_plan} if execution_plan is not None else {}
            if memory:
                selection_options["delivery"] = "memory"
            record = self._runtime.run(image, mask, metadata, input_identity, recipe, output, mode=mode,
                                       decoder_context=hook, capture_logits=capture_logits,
                                       session_identity=self._identity, job_id=job_id, attempt_id=attempt_id,
                                       denoiser=denoiser, **selection_options)
            record.update(complete_request_seconds=perf_counter()-request_start, startup_seconds=startup,
                          model_load_count=self._loads, owner_thread_id=self._owner_thread)
            saved_record = ({k: v for k, v in record.items() if k not in ("array", "valid_mask")}
                            if memory else record)
            write_json(output / "receipt.json", saved_record)
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

    def _close_owned(self) -> Metadata:
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

    def close(self) -> Metadata:
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

    def __enter__(self) -> Self:
        if self._state in ("failed", "closed"): raise SessionClosed("Session is terminal")
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> Literal[False]:
        try:
            self.close()
        except SessionFailed as cleanup_error:
            if exc is None: raise
            exc.add_note(str(cleanup_error))
        return False
