from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload
from .contracts import ExecutionProfile, RecipeValues
from .memory_input import GenerationInput
from .session import EngineSession
from .terramind_encoder import TerraMindEncoder
from collections.abc import Mapping, Sequence
from ..terramind_native.encoder_loader import MergeMethod
from .v1 import SceneRequest, SceneResult
from ..terramind_native.session import TerraMindSession
from .best import BestTerraMindSession

GENERATION_MODEL: Literal["terramind-v1-base"]
def list_models() -> list[dict[str, Any]]: ...
def inspect_model(model_id: str) -> dict[str, Any]: ...

@dataclass(frozen=True)
class GenerationRequest:
    config: str | Path
    output: str | Path
    job_id: str
    attempt_id: str
    model_id: str = ...
    environment: str = ...
    input_manifest: str | Path | None = ...
    recipe: RecipeValues | None = ...
    reuse: Literal["auto", "off"] | None = ...
    denoiser: Literal["native", "graph"] | None = ...
    execution_profile: ExecutionProfile | None = ...
    input_data: GenerationInput | None = ...

@overload
def run_model(request: SceneRequest, *, execute_model: bool = ...) -> SceneResult: ...
@overload
def run_model(request: GenerationRequest, *, execute_model: bool = ...) -> dict[str, Any]: ...
@overload
def open_model_session(model_id: Literal["terramind-v1-base"], *, config: str | Path,
                       work: str | Path, execute_model: bool = ..., environment: Literal["generation"] = ...,
                       reuse: Literal["auto", "off"] | None = ..., denoiser: Literal["native", "graph"] | None = ...,
                       execution_profile: ExecutionProfile | None = ..., input_kind: Literal["file"] = ...) -> BestTerraMindSession: ...
@overload
def open_model_session(model_id: Literal["terramind-v1-base"], *, config: str | Path,
                       work: str | Path, input_kind: Literal["array"], execute_model: bool = ...,
                       environment: Literal["generation"] = ..., reuse: Literal["auto", "off"] | None = ...,
                       denoiser: Literal["native", "graph"] | None = ...,
                       execution_profile: ExecutionProfile | None = ...) -> TerraMindSession: ...
@overload
def open_model_session(model_id: Literal["terramind-v1-tiny", "terramind-v1-small", "terramind-v1-large"], *, config: str | Path,
                       work: str | Path, execute_model: bool = ..., environment: Literal["generation"] = ...,
                       reuse: Literal["auto", "off"] | None = ..., denoiser: Literal["native", "graph"] | None = ...,
                       execution_profile: ExecutionProfile | None = ..., input_kind: Literal["file", "array"] = ...) -> TerraMindSession: ...
@overload
def open_model_session(model_id: Literal["kao-terramind-b"], *, bundle: str | Path,
                       work: str | Path, profile_id: str, execute_model: bool = ...,
                       environment: Literal["scene"] = ..., qualification: str | Path | None = ...) -> EngineSession: ...

@overload
def open_model_session(model_id: Literal["terramind-v1-tiny-backbone", "terramind-v1-small-backbone", "terramind-v1-base-backbone", "terramind-v1-large-backbone", "terramind-v1-tiny-tim", "terramind-v1-small-tim", "terramind-v1-base-tim", "terramind-v1-large-tim"], *,
                       model_file: str | Path, modalities: Sequence[str],
                       tokenizer_files: Mapping[str, str | Path] | None = ...,
                       tim_modalities: Sequence[str] | None = ..., device: str = ...,
                       merge_method: MergeMethod = ..., execute_model: bool = ...,
                       tim_decoding_steps: int = ..., tim_temperature: float = ...,
                       tim_top_p: float = ..., tim_top_k: int = ...,
                       tim_chained_generations: bool = ...) -> TerraMindEncoder: ...
