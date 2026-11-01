from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self
from .contracts import EngineMetadata, ExecutionProfile, GenerationResult, ParallelFallback, Parallelism
from .generation_bridges import GenerationJob, GenerationRecipe
from .generation_queue import GenerationHandle, GenerationStream
from .memory_input import GenerationInput
from .parallel_policy import ParallelPolicy

class GenerationEngine:
    model: str
    work: Path
    def __init__(self, model: str = ..., *, model_file: str | PathLike[str],
                 tokenizer_file: str | PathLike[str], work: str | PathLike[str],
                 execute_model: bool = ..., execution_profile: ExecutionProfile = ...,
                 max_pending: int | None = ..., max_results: int | None = ...,
                 max_input_bytes: int | None = ..., max_output_bytes: int | None = ...,
                 parallel: Literal["off", "two-lane"] | None = ..., pair_wait_ms: float = ...,
                 parallelism: Parallelism = ..., parallel_wait_ms: float | None = ...,
                 allow_unqualified_parallelism: bool = ..., parallel_fallback: ParallelFallback = ...,
                 parallel_workspace_bytes: int = ..., parallel_resident_bytes: int = ...,
                 parallel_policy: ParallelPolicy | None = ...) -> None: ...
    def generate(self, input_data: GenerationInput | GenerationJob, *, seed: int = ...,
                 recipe: GenerationRecipe | None = ..., execution_profile: ExecutionProfile | None = ...,
                 job_id: str | None = ..., attempt_id: str = ...) -> GenerationResult: ...
    def submit(self, input_data: GenerationInput | GenerationJob, *, seed: int = ...,
               recipe: GenerationRecipe | None = ..., execution_profile: ExecutionProfile | None = ...,
               job_id: str | None = ..., attempt_id: str = ...) -> GenerationHandle: ...
    def generate_many(self, iterable: Iterable[Any], *, max_in_flight: int | None = ...,
                      adapter: Callable[[Any], GenerationInput | GenerationJob] | None = ...,
                      seed: int = ..., recipe: GenerationRecipe | None = ...,
                      execution_profile: ExecutionProfile | None = ...) -> GenerationStream: ...
    def binding(self) -> dict[str, object]: ...
    def metadata(self) -> EngineMetadata: ...
    def capabilities(self) -> dict[str, Any]: ...
    def close(self, *, cancel_pending: bool = ..., timeout: float | None = ...) -> dict[str, Any]: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> Literal[False]: ...
