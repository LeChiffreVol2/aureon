"""Structural contracts for resident workers; importing these never imports torch.

Metadata seals inspect identities only. Protocol properties are read-only so a
concrete torch module/tensor may expose more specific attribute types.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NotRequired, Protocol, TypeAlias, TypedDict

FileStamp: TypeAlias = tuple[int, int, int, int, int, int]
TensorStamp: TypeAlias = tuple[int, tuple[int, ...], str, str, int, bool]


class SealedTensor(Protocol):
    def data_ptr(self) -> int: ...
    @property
    def shape(self) -> Sequence[int]: ...
    @property
    def dtype(self) -> object: ...
    @property
    def device(self) -> object: ...
    @property
    def _version(self) -> int: ...
    @property
    def requires_grad(self) -> bool: ...


class EvaluatedModule(Protocol):
    @property
    def training(self) -> bool: ...


class SealedModel(Protocol):
    def modules(self) -> Iterable[EvaluatedModule]: ...
    def state_dict(self, *, keep_vars: bool) -> Mapping[str, SealedTensor]: ...


class WindowsFileLock(Protocol):
    """The small msvcrt surface, also checkable from a Linux mypy target."""
    LK_NBLCK: int
    LK_UNLCK: int
    def locking(self, fd: int, mode: int, nbytes: int, /) -> None: ...


class ArtifactDeclaration(TypedDict):
    path: str
    sha256: str
    bytes: int


class JobError(TypedDict):
    type: str
    message: str


class CompletedReceipt(TypedDict):
    status: str
    download_artifacts: dict[str, ArtifactDeclaration]
    job_id: NotRequired[str]
    summary: NotRequired[object]
    worker_id: NotRequired[str]


class ResultExtra(TypedDict, total=False):
    summary: object
    worker_id: str


class StatusExtra(ResultExtra, total=False):
    error: JobError
    reason: str
    download_artifacts: dict[str, ArtifactDeclaration]
    recovered_after_crash: bool


class SceneJob(StatusExtra):
    schema: str
    job_id: str
    scene_id: str
    status: str
    created_utc: str
    updated_utc: str
    request_digest: str
    idempotency_key_sha256: str
    attempts: NotRequired[int]
    started_utc: NotRequired[str]
    finished_utc: NotRequired[str]
    payload_expires_utc: NotRequired[str]
    key_expires_utc: NotRequired[str]
    previous_status: NotRequired[str]


class PublicArtifact(TypedDict):
    url: str
    sha256: str
    bytes: int


class PublicJob(TypedDict):
    """Fields used by transports; additional status metadata stays on the wire."""
    job_id: str
    artifacts: dict[str, PublicArtifact]


class Submission(TypedDict):
    job: PublicJob
    reused: bool


class RetentionPolicy(TypedDict):
    payload_ttl_seconds: float
    key_ttl_seconds: float
    max_attempts: int
    min_free_bytes: int


class SceneWorker(Protocol):
    def metadata(self) -> dict[str, object]: ...
    def run_job(self, scene_id: str, job_id: str, folder: Path) -> CompletedReceipt: ...
    def close(self) -> dict[str, object] | None: ...
