"""Local-control values and errors; no filesystem, database, client or model imports.

The control module re-exports these names for compatibility. These defaults and
the JSON encoding are unchanged; this module does not create or migrate state.
"""
import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NotRequired, Protocol, TypedDict
from dataclasses import dataclass
from typing import BinaryIO
from ..serving.contracts import ReplicaSettings
if TYPE_CHECKING:
    from ..scene_api.worker import ResidentSceneWorker
    from ..serving.workflow_config import WorkflowBinding


@dataclass
class StreamBody:
    """An already-authorized bounded file descriptor owned by the HTTP response."""
    stream: BinaryIO
    bytes: int
    sha256: str


class ServiceRoute(TypedDict):
    model_id: str
    task: Literal['generation', 'scene', 'generation_scene', 'features']
    environment: str
    profile_id: str
    reuse: Literal['auto', 'off'] | None
    denoiser: Literal['native', 'graph'] | None
    execution: str
    qualification_sha256: str | None
    workflow: NotRequired['WorkflowBinding']


class ControlContract(TypedDict):
    schema: str
    model: str
    profile: str
    execution: str
    max_active: int
    max_running: int
    max_attempts: int
    max_file_bytes: int
    max_source_bytes: int
    prepared_reserve_bytes: int
    max_tenant_storage_reservation_bytes: int
    max_global_storage_reservation_bytes: int
    max_rows: int
    max_artifact_bytes: int
    max_result_bytes: int
    input_ttl_seconds: int
    output_ttl_seconds: int
    ticket_seconds: int
    lease_seconds: int
    min_free_bytes: int
    request_deadline_seconds: NotRequired[int]
    qualification_sha256: NotRequired[str]
    route: NotRequired[ServiceRoute]
    replicas: NotRequired[ReplicaSettings]


class UploadFile(TypedDict):
    name: str
    bytes: int
    sha256: str


class UploadBody(TypedDict):
    files: list[UploadFile]
    spec: dict[str, object]


class Qualification(TypedDict):
    schema: str
    status: str
    profile_id: str
    model_id: str
    source_sha256: dict[str, str]
    runtime_source_sha256: dict[str, str]
    bundle_receipt_sha256: str
    packages: dict[str, str]
    cases: list[dict[str, object]]
    precision: dict[str, object]
    gpu_name: str


class QualificationBinding(TypedDict):
    record: Qualification
    sha256: str


class GuardedExecutor(Protocol):
    work: Path
    worker: "ResidentSceneWorker | None"
    def execute(self, manifest: str | PathLike[str], job_id: str, *,
                qualification: Mapping[str, object] | None = None) -> dict[str, Path]: ...
    def close(self) -> dict[str, object] | None: ...


class GuardedFactory(Protocol):
    def __call__(self, bundle: str | PathLike[str], work: str | PathLike[str], qualification: QualificationBinding,
                 *, execute_model: bool = False, qualification_mode: bool = False) -> GuardedExecutor: ...


class GuardedModule(Protocol):
    def verify_qualification(self, record: Qualification) -> Qualification: ...
    @property
    def ResidentExecutor(self) -> GuardedFactory: ...


class ControlHealth(TypedDict):
    ready: bool
    state: str
    checks: dict[str, str]


class HealthReport(TypedDict):
    schema: str
    scope: str
    control_plane: ControlHealth
    execution_readiness: dict[str, str]
    limitations: list[str]
    action: str | None

MODEL: Final = 'kao-terramind-b'
PROFILE: Final = 'fake-control-v1'
CONTRACT: Final[ControlContract] = {
    'schema': 'aureon.cloud.local.v1', 'model': MODEL, 'profile': PROFILE,
    'execution': 'fake', 'max_active': 3, 'max_running': 1, 'max_attempts': 3,
    'max_file_bytes': 1024**3, 'max_source_bytes': 1024**3, 'prepared_reserve_bytes': 1024**3,
    'max_tenant_storage_reservation_bytes': 2*1024**3,
    'max_global_storage_reservation_bytes': 8*1024**3, 'max_rows': 10000,
    'max_artifact_bytes': 16*1024**2, 'max_result_bytes': 64*1024**2,
    'input_ttl_seconds': 3600, 'output_ttl_seconds': 3600, 'ticket_seconds': 300,
    'lease_seconds': 60, 'min_free_bytes': 32*1024**3,
}


class Error(ValueError):
    """Structured control error; messages/details are not safe log payloads."""
    def __init__(self, status: int, code: str, message: str, details: object = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.details = details


def require(condition: object, code: str = 'invalid_request', status: int = 400) -> None:
    if not condition:
        raise Error(status, code, code.replace('_', ' '))


def encoded(value: object) -> bytes:
    """Canonical JSON bytes; reject nonfinite values without performing I/O."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
