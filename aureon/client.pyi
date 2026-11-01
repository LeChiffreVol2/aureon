from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

class ClientError(RuntimeError): ...
class APIError(ClientError):
    status: int
    code: str
    retry_after: str | None
    def __init__(self, status: int, code: str, message: str, retry_after: str | None) -> None: ...
class SubmissionUncertain(ClientError):
    scene_id: str
    idempotency_key: str
    def __init__(self, scene_id: str, idempotency_key: str) -> None: ...
class JobNotCompleted(ClientError):
    job: Job
    def __init__(self, job: Job) -> None: ...
@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    sha256: str
    bytes: int
@dataclass(frozen=True)
class Job:
    job_id: str
    scene_id: str
    status: str
    artifacts: Mapping[str, Artifact]
    reused: bool | None = ...
@dataclass(frozen=True)
class Delivery:
    job_id: str
    directory: Path
    files: Mapping[str, Path]
    receipt: Path
class SceneClient:
    timeout: float
    max_artifact_bytes: int
    def __init__(self, base_url: str, *, token: str, timeout: float = ...,
                 max_artifact_bytes: int = ...) -> None: ...
    @classmethod
    def from_token_file(cls, base_url: str, token_file: str | Path, *, timeout: float = ...,
                        max_artifact_bytes: int = ...) -> Self: ...
    def info(self) -> dict[str, Any]: ...
    def submit(self, scene_id: str, *, idempotency_key: str) -> Job: ...
    def get(self, job_id: str, *, timeout: float | None = ...) -> Job: ...
    def cancel(self, job_id: str) -> Job: ...
    def wait(self, job_id: str, *, timeout: float = ..., poll_interval: float = ...) -> Job: ...
    def download(self, job_id: str, output_dir: str | Path, *, expected_source_sha256: str | None = ...,
                 expected_bundle_sha256: str | None = ...) -> Delivery: ...
