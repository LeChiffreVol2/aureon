"""Small local job facade; delegates numerical work to the existing Engine."""
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import uuid
from .profiles import require_profile

MODELS = ("kao-terramind-a", "kao-terramind-b", "kao-terramind-c")
PROFILES = ("fp32-full-reference-v1", "fp32-guarded-v2", "fp32-stream-guarded-v3")


class ExecutionRequired(ValueError):
    """Explicit model execution opt-in is required."""


@dataclass(frozen=True)
class SceneRequest:
    prepared_input: str | Path
    bundle: str | Path
    output: str | Path
    model_id: str
    profile_id: str
    environment: str = "scene"
    qualification: str | Path | None = None

    def __post_init__(self):
        if self.model_id not in MODELS or self.profile_id not in PROFILES:
            raise ValueError("Choose a declared KAO-TerraMind model and execution profile")
        if self.environment not in ("scene", "scene-legacy"):
            raise ValueError("Scene jobs use scene or scene-legacy; vllm is a separate adapter environment")
        if self.profile_id != PROFILES[0]:
            if self.model_id != "kao-terramind-b" or self.environment != "scene" or self.qualification is None:
                raise ValueError("Guarded v2/v3 require KAO-TerraMind B, the scene environment and an explicit qualification JSON")
        elif self.qualification is not None:
            raise ValueError("Full-reference jobs do not consume a guarded qualification")


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


def _json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(4*1024**2+1)
    if len(raw) > 4*1024**2:
        raise ValueError("JSON exceeds the 4 MiB contract limit")
    return json.loads(raw)


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_scene(request: SceneRequest, *, execute_model=False) -> SceneResult:
    """Run one bounded scene, close its worker, then commit the facade receipt.

    No retry or acquisition. Output must be fresh; failures retain diagnostics
    without engine-result.json. This call is synchronous; SceneClient connects
    to the existing full-reference local v1 API for asynchronous jobs.
    """
    if execute_model is not True:
        raise ExecutionRequired("Pass execute_model=True explicitly; no model was imported or started")
    if not isinstance(request, SceneRequest):
        raise TypeError("Expected SceneRequest")
    environment = require_profile(request.environment)
    manifest = Path(request.prepared_input).resolve(strict=True)
    bundle = Path(request.bundle).resolve(strict=True)
    output = Path(request.output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("Output must be a new directory under an existing parent")
    output = output.parent.resolve()/output.name
    source_hash = _hash(manifest)
    cleanup = None
    if request.profile_id == PROFILES[0]:
        from ..scene_cli import run_streaming_scene
        result = run_streaming_scene(manifest,bundle,output,execute_model=True,model_id=request.model_id)
        paths = {"job.json":Path(result["job_receipt"]), "class_map.tif":Path(result["class_map"]),
                 "valid_mask.tif":Path(result["valid_mask"]), "provenance.json":Path(result["provenance"])}
        cleanup = _json(output/"worker-cleanup.json")
        numerical = "passed"
    else:
        from ..cloud.profiles import engine_for
        engine = engine_for(request.profile_id)
        record = _json(request.qualification)
        if record.get("profile_id") != request.profile_id or record.get("model_id") != request.model_id:
            raise ValueError("Qualification model/profile differs from the request")
        engine.verify_qualification(record)
        canonical = json.dumps(record,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        qualification = {"record":record,"sha256":hashlib.sha256(canonical).hexdigest()}
        executor = engine.ResidentExecutor(bundle,output,qualification,execute_model=True)
        try:
            paths = executor.execute(manifest,uuid.uuid4().hex)
        except BaseException:
            try:
                cleanup = executor.close()
            finally:
                if output.is_dir():
                    _write(output/"engine-failure.json",{"status":"failed","partial_artifacts_not_publishable":True,"cleanup":cleanup})
            raise
        else:
            cleanup = executor.close()
            _write(output/"worker-cleanup.json",cleanup)
        numerical = "not_evaluated"
    if not cleanup or not cleanup.get("cleanup_passed") or not cleanup.get("full_identity_check_passed") or cleanup.get("errors"):
        raise RuntimeError("Owned worker cleanup/identity check failed; no facade success receipt")
    job = _json(paths["job.json"])
    if job.get("status") != "completed" or _hash(manifest) != source_hash:
        raise RuntimeError("Incomplete job or prepared manifest changed; no facade success receipt")
    if request.profile_id != PROFILES[0] and (job.get("source_manifest_sha256") != source_hash or
            job.get("validation",{}).get("numerical",{}).get("status") != "not_evaluated"):
        raise RuntimeError("Guarded identity/validation declaration mismatch")
    artifacts = {name:{"path":str(path),"bytes":Path(path).stat().st_size,"sha256":_hash(path)} for name,path in paths.items()}
    receipt = output/"engine-result.json"
    _write(receipt,{"schema":"aureon.engine_result.v1","status":"completed",
        "package_version":metadata.version("aureon-inference"),"model_id":request.model_id,
        "profile_id":request.profile_id,"environment":environment,"source_manifest_sha256":source_hash,
        "numerical_status":numerical,"task_accuracy":"not_evaluated","artifacts":artifacts,
        "cleanup":cleanup,"scope":"One local Engine job with owned shutdown; no speed or scientific accuracy claim"})
    return SceneResult(output,Path(paths["job.json"]),Path(paths["class_map.tif"]),Path(paths["valid_mask.tif"]),
                       Path(paths["provenance.json"]),request.profile_id,numerical,receipt)
