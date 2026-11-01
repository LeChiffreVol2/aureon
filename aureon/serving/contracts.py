"""Typed service boundary records. Importing them performs no runtime work.

TypedDict describes existing wire/state shapes; it does not validate input or
authorize execution. Validation remains explicit before transactional admission.
"""
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypeAlias, TypedDict
if TYPE_CHECKING:
    from .backend import Backend
    from .control import Control

PathInput: TypeAlias = str | PathLike[str]
ArtifactFiles: TypeAlias = dict[str, Path]


class FileIdentity(TypedDict):
    path: str
    bytes: int
    sha256: str


class UploadResult(TypedDict):
    input_id: str
    manifest_sha256: str
    upload_seconds: float
    files: NotRequired[int]
    reused_input: NotRequired[bool]


class ProcessIdentity(TypedDict):
    pid: int
    start_ticks: str | None
    platform: NotRequired[str]
    boot_id: NotRequired[str]
    state: NotRequired[str]


class ManifestMember(TypedDict):
    path: str
    sha256: str
    bytes: NotRequired[int]


class PreparedManifest(TypedDict):
    schema: str
    image: NotRequired[ManifestMember]
    mask: NotRequired[ManifestMember]
    raster: NotRequired[ManifestMember]
    valid_mask: NotRequired[ManifestMember]


class PreparedSpec(TypedDict):
    schema: str
    task: str
    manifest: str
    members: dict[str, str]


class WorkerUpdate(TypedDict, total=False):
    pid: int
    model_loaded: bool
    jobs_completed: int
    device: object
    execution_stage: str
    preparation_state: str
    elapsed_clock: str
    state: str
    active_job: str | None
    replicas_available: int


class WorkerStatus(WorkerUpdate, total=False):
    owner_id: str
    heartbeat: float


class WorkerStatusOptions(WorkerUpdate, total=False):
    terminal: bool


class BackendOptions(TypedDict, total=False):
    bundle: PathInput | None
    config: PathInput | None
    execute_model: bool
    device: str | None


class BackendFactory(Protocol):
    def __call__(self, control: 'Control', work: PathInput, *, bundle: PathInput | None = None,
                 config: PathInput | None = None, execute_model: bool = False,
                 device: str | None = None) -> 'Backend': ...


class Termination(TypedDict):
    term_targets: list[ProcessIdentity]
    kill_targets: list[ProcessIdentity]
    forced: bool


class SupervisorAttempt(TypedDict):
    index: int
    fresh_work: str
    started_seconds: float
    termination: Termination | None
    ready_file: str | None
    process: NotRequired[ProcessIdentity | None]
    descendant_cleanup: NotRequired[Termination]
    exit_code: NotRequired[int | None]
    elapsed_seconds: NotRequired[float]
    scratch_bytes: NotRequired[int]
    worker_result: NotRequired[dict[str, object]]


class SecurityPolicy(TypedDict):
    request_burst: int
    requests_per_second: int
    active_keys_per_tenant: int
    key_records: int
    identity_provider: str
    public_signup: str
    billing: str


class TestPaymentEvent(TypedDict):
    tenant: str
    units: int
    kind: str
    sequence: int


class Readiness(TypedDict):
    schema: str
    ready: bool
    draining: bool
    worker: WorkerStatus
    model_loaded: NotRequired[bool]
    lazy_model_start: bool
    capabilities: dict[str, object]


class ServiceMetrics(TypedDict):
    schema: str
    scope: str
    observed_at_epoch: float
    execution: str
    model_id: str
    draining: bool
    jobs: dict[str, int]
    inputs: dict[str, int]
    attempts: dict[str, int]
    workers: dict[str, int]
    queue: dict[str, int]
    oldest_queued_age_seconds: float | None
    input_reserved_bytes: int
    input_reservation_limit_bytes: int
    record_capacity: dict[str, int]
    retention: dict[str, int]
    usage: dict[str, dict[str, int]]
    committed_usage_rows: int
    billable: bool
    semantics: str


class DeviceAttestation(TypedDict):
    uuid: str
    cuda_index: int
    attested: bool


class CudaProcess(Protocol):
    def is_initialized(self) -> bool: ...
    def memory_allocated(self) -> int: ...
    def memory_reserved(self) -> int: ...
    def synchronize(self) -> None: ...
    def empty_cache(self) -> None: ...


class TorchProcess(Protocol):
    @property
    def cuda(self) -> CudaProcess: ...


class HostIdentity(TypedDict):
    machine: str
    uid: int
    pid_namespace: str


class ReplicaLimits(TypedDict):
    max_workers: int
    max_running: int
    max_active: int
    max_queued: int
    max_rows: int
    max_tenants: int
    max_worker_generations: int
    max_tenant_active: int
    max_tenant_running: int
    max_tenant_queued: int
    max_tenant_storage_reservation_bytes: int
    max_global_storage_reservation_bytes: int


class CapacityPolicy(TypedDict):
    limits: ReplicaLimits
    hard_outstanding_bound: str
    queued_admission: str
    tenant_queued_admission: str
    running: str
    fairness: str
    worker_queue: Literal[0]
    requests_per_worker: Literal[1]
    HTTP_lanes_per_worker: Literal[1]
    HTTP_parallel_enabled: Literal[False]
    library_lanes: str


class PlacementAuthority(TypedDict):
    schema: Literal['aureon.placement_authority.v1']
    id: str
    kind: Literal['single_host_sqlite']
    host: HostIdentity
    transaction: Literal['BEGIN IMMEDIATE']
    multi_node_enabled: Literal[False]
    external_authority_required: list[str]
    external_adapter_installed: Literal[False]


class WorkerBudget(TypedDict):
    id: str
    device: str
    vram_bytes: int
    resident_vram_bytes: int
    resident_host_bytes: int
    transient_vram_bytes: int
    transient_host_bytes: int
    scratch_bytes: int
    lanes: NotRequired[int]


class ReplicaTopology(TypedDict):
    host_bytes: int
    output_bytes: int
    controller_host_bytes: int
    parser_host_bytes: int
    workers: list[WorkerBudget]


class ReplicaSettings(TypedDict):
    resources: ReplicaTopology
    host: HostIdentity
    execution: dict[str, object]
    limits: NotRequired[ReplicaLimits]
    authority: NotRequired[PlacementAuthority]


class ResourceUsage(TypedDict):
    host_bytes: int
    output_bytes: int
    device_vram_bytes: dict[str, int]


class ReplicaWorkerSnapshot(TypedDict):
    worker: str
    generation: str
    device: str
    state: str
    heartbeat: float
    scratch_bytes: int


class TenantStatus(TypedDict):
    counts: dict[str, dict[str, int]]
    queue: dict[str, int]
    admission: "TenantAdmission"


class TenantAdmission(TypedDict):
    reasons: list[dict[str, str]]
    scope: str


class Placement(TypedDict):
    schema: Literal['aureon.replica_placement.v1']
    authority: PlacementAuthority | None
    host: HostIdentity
    state_instance: str
    replica: str
    generation: str
    device: str
    model: str
    profile: str
    execution_binding_sha256: str
    lanes: Literal[1]
    max_running_requests: Literal[1]
    local_queue_slots: Literal[0]


class BaseRequest(TypedDict):
    input_id: str
    model_id: str
    profile_id: str


class ServiceRequest(BaseRequest):
    task: Literal['generation', 'scene', 'generation_scene', 'features']
    parameters: dict[str, int]


class BaseLease(TypedDict):
    job_id: str
    attempt: int
    token: str
    manifest: str
    worker_owner: NotRequired[str]
    replica_worker: NotRequired[str]
    replica_generation: NotRequired[str]
    replica_device: NotRequired[str]
    placement: NotRequired[Placement]


class AttemptLease(BaseLease):
    request: ServiceRequest


class SafeError(TypedDict):
    code: str
    type: str


class ModuleBinding(TypedDict):
    path: str
    sha256: str


class InstalledBinding(TypedDict):
    version: str
    root: str
    modules: dict[str, ModuleBinding]


class BackendCleanup(TypedDict):
    schema: Literal['aureon.async_worker_cleanup.v1']
    jobs_completed: int
    model_load_count: int
    engine_cleanup: dict[str, object] | None
    installed: InstalledBinding
    passed: bool
    close_error: SafeError | None
    cuda_allocated_bytes: int
    cuda_reserved_bytes: int
