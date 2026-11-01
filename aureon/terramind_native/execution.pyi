from typing import Any, Literal
from ..api.contracts import ExecutionProfile, RecipeValues

def validate_selection(execution_profile: ExecutionProfile | None,
                       reuse: Literal["auto", "off"] = ...,
                       denoiser: Literal["native", "graph"] = ...) -> None: ...
def plan_generation_execution(*, execution_profile: ExecutionProfile | None = ...,
                              recipe: RecipeValues, reuse: Literal["auto", "off"] = ...,
                              denoiser: Literal["native", "graph"] = ..., model_id: str = ...) -> dict[str, Any]: ...
def completed_selection(plan: dict[str, Any], reuse_decision: dict[str, Any],
                        decoder: dict[str, Any]) -> dict[str, Any]: ...
