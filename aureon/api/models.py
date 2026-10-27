"""CPU-safe capabilities and thin dispatch to the existing model runtimes."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .profiles import require_profile
from .v1 import ExecutionRequired, SceneRequest, run_scene

GENERATION_MODEL = "terramind-v1-base"


def list_models():
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
    result.append({
        "model_id": GENERATION_MODEL, "name": "TerraMind 1.0 base", "family": "TerraMind",
        "task": "modality-generation", "input": "prepared all-valid 224x224 S2L2A",
        "outputs": ["synthetic-s1grd.tif", "raw.npy", "provenance.json"],
        "provenance_schema": "aureon.terramind_session_job.v1",
        "result_schema": "aureon.generation_result.v1",
        "cleanup_schema": "aureon.terramind_session_cleanup.v1",
        "environment": "generation", "resident": True, "synthetic": True,
        "denoiser_default": "native", "denoiser_options": ["native", "graph"],
        "graph_profile": "224px / 8 token / 10 diffusion / CUDA FP32; runtime eligibility and fallback apply",
        "reuse": "auto only for eligible repeated conditioning; off supported",
        "status": "implemented; model qualification is artifact/runtime specific",
        "scientific_quality": "not_evaluated",
    })
    return result


def inspect_model(model_id):
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
    recipe: dict | None = None
    reuse: str = "auto"
    denoiser: str = "native"

    def __post_init__(self):
        _generation_options(self.model_id, self.environment, self.reuse, self.denoiser)
        from ..terramind_native.session import _identifier
        _identifier(self.job_id, "job_id")
        _identifier(self.attempt_id, "attempt_id")
        if self.recipe is not None:
            from ..terramind_native.common import validate_recipe
            object.__setattr__(self, "recipe", validate_recipe(self.recipe))


def _generation_options(model_id, environment, reuse, denoiser):
    if model_id != GENERATION_MODEL or environment != "generation":
        raise ValueError("Generation requires terramind-v1-base in the separate generation environment")
    if reuse not in ("auto", "off") or denoiser not in ("native", "graph"):
        raise ValueError("Choose reuse auto/off and denoiser native/graph")


def open_model_session(model_id, *, execute_model=False, **options):
    """Return the original resident session; callers own close/context management.

    A/C are one-shot full-reference scene jobs, not resident-B aliases. The two
    resident families need separate processes/environments and allocator ownership.
    """
    spec = inspect_model(model_id)
    if not spec["resident"]:
        raise ValueError("KAO-TerraMind A/C use run_model(SceneRequest), not a resident session")
    if model_id == GENERATION_MODEL:
        allowed = {"config", "work", "environment", "reuse", "denoiser"}
        if set(options) - allowed or not {"config", "work"}.issubset(options):
            raise ValueError("TerraMind requires config/work; optional environment/reuse/denoiser only")
        environment = options.pop("environment", "generation")
        _generation_options(model_id, environment, options.get("reuse", "auto"), options.get("denoiser", "native"))
        if execute_model is not True:
            raise ExecutionRequired("Pass execute_model=True explicitly; nothing loaded")
        require_profile(environment)
        from ..terramind_native.offline import inspect_config
        inspect_config(options["config"])
        from ..terramind_native import TerraMindSession
        return TerraMindSession(execute_model=True, **options)
    allowed = {"bundle", "work", "profile_id", "environment", "qualification"}
    if set(options) - allowed or not {"bundle", "work", "profile_id"}.issubset(options):
        raise ValueError("KAO-TerraMind B session requires bundle/work/profile_id and optional environment/qualification")
    if execute_model is not True:
        raise ExecutionRequired("Pass execute_model=True explicitly; nothing loaded")
    from .session import EngineSession
    return EngineSession(model_id=model_id, execute_model=True, **options)


def run_model(request, *, execute_model=False):
    """Dispatch typed tasks without changing their native output/artifact schemas."""
    if isinstance(request, SceneRequest):
        spec = inspect_model(request.model_id)
        if request.profile_id == "fp32-full-reference-v1" and request.environment not in spec["full_reference_environments"]:
            raise ValueError("KAO-TerraMind A/C require scene-legacy for the supported full-reference route")
        return run_scene(request, execute_model=execute_model)
    if not isinstance(request, GenerationRequest):
        raise TypeError("Expected SceneRequest or GenerationRequest")
    _generation_options(request.model_id, request.environment, request.reuse, request.denoiser)
    if execute_model is not True:
        raise ExecutionRequired("Pass execute_model=True explicitly; nothing loaded")
    with open_model_session(request.model_id, config=request.config, work=request.output,
                            environment=request.environment, reuse=request.reuse,
                            denoiser=request.denoiser, execute_model=True) as session:
        record = session.run(request.input_manifest, job_id=request.job_id,
                             attempt_id=request.attempt_id, recipe=request.recipe)
    return {"schema": "aureon.generation_result.v1", "model_id": request.model_id,
            "task": "modality-generation", "status": "completed", "synthetic": True,
            "job": record, "cleanup": session.close()}
