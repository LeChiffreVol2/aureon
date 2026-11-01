"""CPU-only execution planning. Plans do not attest CUDA or numerical parity."""
from __future__ import annotations
import typing as t
from .common import validate_recipe
from .families import get_terramind_spec

EXECUTION_PROFILES = ("recommended", "reference")


def validate_selection(execution_profile: str | None, reuse: str="auto", denoiser: str="native") -> None:
    if execution_profile is not None and execution_profile not in EXECUTION_PROFILES:
        raise ValueError("execution_profile must be recommended, reference, or None for expert controls")
    if execution_profile is not None and (reuse != "auto" or denoiser != "native"):
        raise ValueError("A named execution_profile owns reuse/denoiser; omit expert overrides or use execution_profile=None")


def plan_generation_execution(*, execution_profile: str | None="recommended", recipe: t.Mapping[str,t.Any],
                              reuse: str="auto", denoiser: str="native", model_id: str="terramind-v1-base") -> dict[str,t.Any]:
    """Plan a pinned S2L2A224 -> S1GRD path without inspecting a GPU.

    Actual inputs/assets/dependencies remain subject to existing admission.
    Runtime topology, CUDA tensors and graph signatures are checked at execution.
    """
    validate_selection(execution_profile, reuse, denoiser)
    spec = get_terramind_spec(model_id)
    if spec.mode != "generation":
        raise ValueError("Execution planning requires a generation model_id")
    if reuse not in ("auto", "off") or denoiser not in ("native", "graph"):
        raise ValueError("Choose reuse auto/off and denoiser native/graph")
    recipe = validate_recipe(recipe)
    reasons = []
    if execution_profile == "reference":
        reuse, denoiser = "off", "native"
    elif execution_profile == "recommended":
        scoped = spec.size == "base" and recipe["token_steps"] == 8 and recipe["diffusion_steps"] == 10
        reuse, denoiser = ("auto", "graph") if scoped else ("off", "native")
        if spec.size != "base":
            reasons.append("model_size_not_gpu_qualified")
        elif not scoped:
            reasons.append("recommended_scope_requires_8_token_and_10_diffusion_steps")
    return {"schema": "aureon.generation_execution_selection.v1", "phase": "plan",
            "model_id": model_id,
            "requested": execution_profile or "expert",
            "planned": {"reuse": reuse, "denoiser": denoiser}, "effective": None,
            "fallback": bool(reasons), "reasons": reasons,
            "eligibility": {"recipe_scope": not reasons, "runtime": "not_checked"},
            "cuda": "not_probed", "model_loaded": False,
            "qualification": "not_established_by_selection", "precision": "highest_fp32"}


def completed_selection(plan: dict[str,t.Any], reuse_decision: dict[str,t.Any], decoder: dict[str,t.Any]) -> dict[str,t.Any]:
    """Report observed mechanisms, including admission/signature fallbacks."""
    from copy import deepcopy
    result = deepcopy(plan)
    actual_reuse = "auto" if reuse_decision["enabled"] else "off"
    actual_denoiser = "graph" if decoder.get("enabled") and not decoder.get("fallback") else "native"
    reasons = result["reasons"]
    if plan["planned"]["reuse"] == "auto" and actual_reuse == "off":
        reasons.append("reuse: " + reuse_decision["reason"])
    if plan["planned"]["denoiser"] == "graph" and actual_denoiser == "native":
        reasons.append("denoiser: " + decoder.get("reason", "not_enabled"))
    result.update(phase="completed", effective={"reuse": actual_reuse, "denoiser": actual_denoiser},
                  fallback=bool(reasons), cuda="see_runtime_receipts", model_loaded=True)
    result["eligibility"]["runtime"] = {"reuse": deepcopy(reuse_decision),
                                          "denoiser": {k: decoder[k] for k in
                                                       ("profile_id", "enabled", "fallback", "reason") if k in decoder}}
    return result
