from dataclasses import dataclass
from pathlib import Path

class ExecutionRequired(ValueError): ...

@dataclass(frozen=True)
class SceneRequest:
    prepared_input: str | Path
    bundle: str | Path
    output: str | Path
    model_id: str
    profile_id: str
    environment: str = ...
    qualification: str | Path | None = ...

@dataclass(frozen=True)
class SceneResult:
    output: Path
    job_receipt: Path
    class_map: Path
    valid_mask: Path
    provenance: Path
    profile_id: str
    numerical_status: str
    receipt: Path

def run_scene(request: SceneRequest, *, execute_model: bool = ...) -> SceneResult: ...
