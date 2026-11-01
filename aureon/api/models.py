"""CPU-safe capabilities and thin dispatch to the existing model runtimes."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, cast, overload
from ..terramind_native._types import Metadata
from .v1 import SceneResult
from .contracts import GenerationResult
if TYPE_CHECKING:
    from .best import BestTerraMindSession, BestProfile
    from .session import EngineSession
    from ..terramind_native.session import TerraMindSession
    from .memory_input import GenerationInput
    from .terramind_encoder import TerraMindEncoder
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .profiles import require_profile
from .v1 import ExecutionRequired, SceneRequest, run_scene
from ..terramind_native.execution import plan_generation_execution as plan_generation_execution, validate_selection

from ..terramind_native.families import (GENERATION_MODELS, BACKBONE_MODELS, TIM_MODELS, get_terramind_spec)

GENERATION_MODEL = "terramind-v1-base"


def list_models() -> list[Metadata]:
    """Return detached capability metadata; never inspect assets or import models."""
    result = []
    for variant in "abc":
        resident = variant == "b"
        result.append({
            "model_id": "kao-terramind-" + variant,
            "name": "KAO-TerraMind " + variant.upper(), "family": "KAO-TerraMind",
            "task": "semantic-segmentation", "input": "prepared S2L2A scene",
            "outputs": ["class_map.tif", "valid_mask.tif", "provenance.json"],
            "result_schema": "aureon.engine_result.v1",
            "full_reference_environments": ["scene", "scene-legacy"] if resident else ["scene-legacy"],
            "resident": resident, "guarded": resident,
            "resident_environment": "scene" if resident else None,
            "guarded_requires": "artifact-bound qualification JSON" if resident else None,
            "status": "implemented; model qualification is artifact/runtime specific",
        })
    for model_id in GENERATION_MODELS:
        spec = get_terramind_spec(model_id)
        result.append({
            "model_id": model_id, "name": "TerraMind 1.0 " + spec.size, "family": "TerraMind",
            "size": spec.size, "mode": spec.mode, "upstream_name": spec.upstream_name,
            "task": "modality-generation", "input": "prepared all-valid 224x224 S2L2A",
            "outputs": ["synthetic-s1grd.tif", "raw.npy", "provenance.json"],
            "provenance_schema": "aureon.terramind_session_job.v1",
            "result_schema": "aureon.generation_result.v1",
            "cleanup_schema": "aureon.terramind_session_cleanup.v1",
            "environment": "generation", "resident": True, "synthetic": True,
            "http_tasks": ["generation", "generation_scene"],
            "scene_api": "run_prepared_scene" if spec.size == "base" else "run_generation_scene_with_engine",
            "scene_scope": "prepared raster tiling with explicit validity/context; not monolithic-generation equivalence",
            "denoiser_default": "graph" if spec.size == "base" else "native", "denoiser_options": ["native", "graph"],
            "execution_profile_options": ["recommended", "reference"],
            "execution_profile_default": "recommended" if spec.size == "base" else None,
            "default_file_runtime": "selected_core" if spec.size == "base" else "family_runtime",
            "default_file_scope": "base S2L2A224 to synthetic S1GRD, FP32, 8 token / 10 diffusion steps" if spec.size == "base" else "family-specific evidence; no base performance inheritance",
            "memory_numerical_core": "frozen selected primitives with audited active memory delivery adapter" if spec.size == "base" else "family-specific implementation",
            "memory_core_qualification": "new route/artifact parity and performance require their own receipt",
            "execution_profile_status": "default base file route uses the preserved selected core; reference is explicit diagnostics",
            "memory_input": "GenerationInput: CPU FP32 12x224x224 S2L2A, all-valid; optional grid",
            "memory_output": ("input_kind=\"array\" or GenerationEngine exposes generate -> array/valid_mask/provenance; default file session exposes run" if spec.size == "base"
                              else "session.generate returns array/valid_mask/provenance; optional explicit export_generation"),
            "memory_status": ("prior base source qualification is artifact-bound" if spec.size == "base"
                              else "real-weight acceptance recorded for a previous exact artifact; current build requires its own evidence"),
            "graph_profile": ("base224px / 8 token / 10 diffusion / CUDA FP32; runtime eligibility and fallback apply"
                              if spec.size == "base" else "recommended selects native; explicit optimization requires separate qualification"),
            "reuse": "auto only for eligible repeated conditioning; off supported",
            "status": "implemented; model qualification is artifact/runtime specific",
            "scientific_quality": "not_evaluated", "new_variant_gpu_qualified": False,
        })
    for model_id in (*BACKBONE_MODELS, *TIM_MODELS):
        spec = get_terramind_spec(model_id)
        result.append({
            "model_id": model_id, "name": "TerraMind 1.0 " + spec.size + " " + spec.mode,
            "family": "TerraMind", "size": spec.size, "mode": spec.mode,
            "upstream_name": spec.upstream_name, "task": "feature-extraction",
            "input": "explicit full-band modality tensors with upstream preprocessing; continuous FP32 or LULC class indices",
            "outputs": ["feature tensors", "provenance"], "environment": "generation",
            "resident": True, "api": "TerraMindEncoder.predict", "http_service": True,
            "http_task": "features", "http_profile": "fp32-features-v1",
            "http_scope": "operator-bound resident workflow; qualification remains artifact/recipe specific",
            "synthetic": spec.mode == "tim", "precision": "highest_fp32",
            "status": "implemented; previous exact-artifact FP32 acceptance recorded; see support guide",
            "scientific_quality": "not_evaluated", "new_variant_gpu_qualified": False,
            "output_semantics": "backbone features; downstream prediction needs a task head",
        })
    return result


def inspect_model(model_id: str) -> Metadata:
    for item in list_models():
        if item["model_id"] == model_id:
            return deepcopy(item)
    raise ValueError("Unknown model_id: " + str(model_id))


@dataclass(frozen=True)
class GenerationRequest:
    config: str | Path
    output: str | Path
    job_id: str
    attempt_id: str
    model_id: str = GENERATION_MODEL
    environment: str = "generation"
    input_manifest: str | Path | None = None
    recipe: Metadata | None = None
    reuse: str | None = None
    denoiser: str | None = None
    execution_profile: str | None = None
    input_data: object | None = None

    def __post_init__(self) -> None:
        _generation_options(self.model_id, self.environment, self.reuse or "auto", self.denoiser or "native")
        _generation_dispatch(self.model_id, "array" if self.input_data is not None else "file",
                             self.execution_profile, self.reuse, self.denoiser)
        if self.input_data is not None:
            from .memory_input import GenerationInput
            if not isinstance(self.input_data, GenerationInput):
                raise TypeError("input_data requires GenerationInput")
            if self.input_manifest is not None:
                raise ValueError("Choose input_data or input_manifest, not both")
        from ..terramind_native.session import _identifier
        _identifier(self.job_id, "job_id")
        _identifier(self.attempt_id, "attempt_id")
        if self.recipe is not None:
            from ..terramind_native.common import validate_recipe
            object.__setattr__(self, "recipe", validate_recipe(self.recipe))


def _generation_options(model_id: str, environment: str, reuse: str, denoiser: str) -> None:
    if model_id not in GENERATION_MODELS or environment != "generation":
        raise ValueError("Generation requires a TerraMind 1.0 generation model in the separate generation environment")
    if reuse not in ("auto", "off") or denoiser not in ("native", "graph"):
        raise ValueError("Choose reuse auto/off and denoiser native/graph")


@dataclass(frozen=True)
class _GenerationDispatch:
    runtime: Literal["selected_core", "family_runtime", "array_runtime"]
    profile: BestProfile | None
    reuse: str
    denoiser: str


def _generation_dispatch(model_id: str, input_kind: str, execution_profile: str | None,
                         reuse: str | None, denoiser: str | None) -> _GenerationDispatch:
    """Pure selection once at request/session construction; no runtime probing."""
    if input_kind not in ("file", "array"):
        raise ValueError("input_kind must be file or array")
    if reuse not in (None, "auto", "off") or denoiser not in (None, "native", "graph"):
        raise ValueError("Choose reuse auto/off and denoiser native/graph")
    if execution_profile not in (None, "recommended", "reference"):
        raise ValueError("execution_profile must be recommended or reference")
    if model_id != GENERATION_MODEL or input_kind == "array":
        resolved_reuse, resolved_denoiser = reuse or "auto", denoiser or "native"
        validate_selection(execution_profile, resolved_reuse, resolved_denoiser)
        return _GenerationDispatch("array_runtime" if input_kind == "array" else "family_runtime",
                                   None, resolved_reuse, resolved_denoiser)
    reference = execution_profile == "reference" or (execution_profile is None and reuse == "off" and denoiser == "native")
    if reference:
        if reuse not in (None, "off") or denoiser not in (None, "native"):
            raise ValueError("The reference profile owns reuse/denoiser: off/native")
        return _GenerationDispatch("selected_core", "reference_fp32", "off", "native")
    if reuse not in (None, "auto") or denoiser not in (None, "graph"):
        raise ValueError("Default base file execution owns reuse/denoiser: auto/graph; use execution_profile='reference' for diagnostics")
    return _GenerationDispatch("selected_core", "best_fp32", "auto", "graph")


def _require_base_file_config(config: str | Path) -> None:
    """Read the small config once; original loader owns asset/source hash checks."""
    from ..terramind_native.common import read_json
    value = read_json(config)
    if value.get("model_id", GENERATION_MODEL) != GENERATION_MODEL:
        raise ValueError("Configured checkpoint model_id differs from requested model_id")
    expected = {"schema", "model", "tokenizer", "source_manifest", "dependencies", "input_manifest", "recipe"}
    if value.get("schema") != "aureon.terramind_native.v1" or set(value) != expected:
        raise ValueError("Base file execution requires a generation-config with a seed input manifest; assets-only inputs use input_kind='array' or GenerationEngine")


def open_model_session(model_id: str, *, execute_model: bool=False, **options: Any) -> BestTerraMindSession | TerraMindSession | EngineSession | TerraMindEncoder:
    """Open the selected base file core; other family/array contracts stay explicit.

    A/C are one-shot full-reference scene jobs, not resident-B aliases. The two
    resident families need separate processes/environments and allocator ownership.
    """
    spec = inspect_model(model_id)
    if not spec["resident"]:
        raise ValueError("KAO-TerraMind A/C use run_model(SceneRequest), not a resident session")
    if model_id in (*BACKBONE_MODELS, *TIM_MODELS):
        allowed = {"model_file", "modalities", "tokenizer_files", "tim_modalities", "device", "merge_method", "tim_decoding_steps", "tim_temperature", "tim_top_p", "tim_top_k", "tim_chained_generations"}
        if set(options) - allowed or not {"model_file", "modalities"}.issubset(options):
            raise ValueError("TerraMind encoder requires model_file/modalities and explicit supported encoder options")
        from .terramind_encoder import TerraMindEncoder
        return TerraMindEncoder(model_id, execute_model=execute_model, **options)
    if model_id in GENERATION_MODELS:
        allowed = {"config", "work", "environment", "reuse", "denoiser", "execution_profile", "input_kind"}
        if set(options) - allowed or not {"config", "work"}.issubset(options):
            raise ValueError("TerraMind requires config/work; optional environment/reuse/denoiser/execution_profile/input_kind only")
        environment = options.pop("environment", "generation")
        input_kind = options.pop("input_kind", "file")
        if (model_id == GENERATION_MODEL and input_kind == "array"
                and all(options.get(name) is None for name in ("execution_profile", "reuse", "denoiser"))):
            options["execution_profile"] = "recommended"
        dispatch = _generation_dispatch(model_id, input_kind,
                                        options.get("execution_profile"), options.get("reuse"), options.get("denoiser"))
        _generation_options(model_id, environment, dispatch.reuse, dispatch.denoiser)
        if execute_model is not True:
            raise ExecutionRequired("Pass execute_model=True explicitly; nothing loaded")
        if dispatch.runtime == "selected_core":
            _require_base_file_config(options["config"])
            from .best import BestTerraMindSession
            return BestTerraMindSession(options["config"], options["work"], execute_model=True,
                                       profile=cast("BestProfile", dispatch.profile))
        require_profile(environment)
        options["reuse"], options["denoiser"] = dispatch.reuse, dispatch.denoiser
        from ..terramind_native.offline import inspect_config
        configured = inspect_config(options["config"])
        if configured.get("model_id", GENERATION_MODEL) != model_id:
            raise ValueError("Configured checkpoint model_id differs from requested model_id")
        from ..terramind_native import TerraMindSession
        return TerraMindSession(model_id=model_id, execute_model=True, **options)
    allowed = {"bundle", "work", "profile_id", "environment", "qualification"}
    if set(options) - allowed or not {"bundle", "work", "profile_id"}.issubset(options):
        raise ValueError("KAO-TerraMind B session requires bundle/work/profile_id and optional environment/qualification")
    if execute_model is not True:
        raise ExecutionRequired("Pass execute_model=True explicitly; nothing loaded")
    from .session import EngineSession
    return EngineSession(model_id=model_id, execute_model=True, **options)


@overload
def run_model(request: SceneRequest, *, execute_model: bool = False) -> SceneResult: ...
@overload
def run_model(request: GenerationRequest, *, execute_model: bool = False) -> Metadata: ...

def run_model(request: SceneRequest | GenerationRequest, *, execute_model: bool=False) -> SceneResult | Metadata:
    """Dispatch typed tasks without changing their native output/artifact schemas."""
    if isinstance(request, SceneRequest):
        spec = inspect_model(request.model_id)
        if request.profile_id == "fp32-full-reference-v1" and request.environment not in spec["full_reference_environments"]:
            raise ValueError("KAO-TerraMind A/C require scene-legacy for the supported full-reference route")
        return run_scene(request, execute_model=execute_model)
    if not isinstance(request, GenerationRequest):
        raise TypeError("Expected SceneRequest or GenerationRequest")
    _generation_options(request.model_id, request.environment, request.reuse or "auto", request.denoiser or "native")
    if execute_model is not True:
        raise ExecutionRequired("Pass execute_model=True explicitly; nothing loaded")
    selection_options = ({"execution_profile": request.execution_profile}
                         if request.execution_profile is not None else {})
    with cast("BestTerraMindSession | TerraMindSession", open_model_session(request.model_id, config=request.config, work=request.output,
                            environment=request.environment, reuse=request.reuse,
                            denoiser=request.denoiser, input_kind="array" if request.input_data is not None else "file",
                            execute_model=True, **selection_options)) as session:
        if request.input_data is not None:
            record: GenerationResult | Metadata = cast("TerraMindSession", session).generate(cast("GenerationInput", request.input_data), job_id=request.job_id,
                                      attempt_id=request.attempt_id, recipe=request.recipe)
        else:
            record = session.run(request.input_manifest, job_id=request.job_id,
                                 attempt_id=request.attempt_id, recipe=request.recipe)
    return {"schema": "aureon.generation_result.v1", "model_id": request.model_id,
            "task": "modality-generation", "status": "completed", "synthetic": True,
            "job": record, "cleanup": session.close()}
