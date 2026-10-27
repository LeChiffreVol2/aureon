"""Resident public lifecycle; numerical execution stays in existing workers."""
from concurrent.futures import Future
from copy import deepcopy
from importlib import metadata as packages
import json
from pathlib import Path
from queue import Queue
import re
import threading
import traceback
import uuid

from .profiles import require_profile
from .v1 import ExecutionRequired, SceneResult, PROFILES, _hash, _json, _write


class SessionBusy(RuntimeError):
    """One call is in flight, or another session owns the process runtime."""


class SessionClosed(RuntimeError):
    """The session is closed or failed and cannot accept another job."""


class SessionFailed(RuntimeError):
    """Internal execution failed; owner-thread cleanup was attempted."""


_PROCESS_OWNER = threading.Lock()


class _Backend:
    """Private bridge to existing serial workers, constructed on the owner."""

    def __init__(self, bundle, work, profile, qualification):
        self.bundle, self.work, self.profile = bundle, work, profile
        self.worker = self.executor = None
        if profile != PROFILES[0]:
            from ..cloud.profiles import engine_for
            self.executor = engine_for(profile).ResidentExecutor(
                bundle, work, qualification, execute_model=True)
        else:
            work.mkdir()

    def run(self, manifest, job_id, attempt_id):
        attempt = self.work/attempt_id
        attempt.mkdir(exist_ok=True)
        if self.executor is not None:
            # The existing executor binds logical job IDs to receipt identity.
            # Give each attempt a fresh parent without changing those receipts.
            self.executor.work = attempt
            try:
                return self.executor.execute(manifest, job_id)
            finally:
                self.executor.work = self.work
        from ..prepare import check_scene
        checked = check_scene(manifest)
        from ..scene_api.worker import ResidentSceneWorker
        scene = checked["scene_id"]
        if self.worker is None:
            self.worker = ResidentSceneWorker(self.bundle, {scene: manifest}, self.work/"owner",
                execute_model=True, streaming=True, model_id="kao-terramind-b")
        else:
            _, _, digest = self.worker._manifest(manifest)
            # At most one registered request; replacement occurs on the owner
            # after the preceding job returned. No numerical loop is copied.
            self.worker.scenes = {scene: (manifest, digest)}
        output = attempt/job_id
        result = self.worker.run_job(scene, job_id, output)
        return {name: output/item["path"] for name, item in result["download_artifacts"].items()}

    def metadata(self):
        worker = self.executor.worker if self.executor is not None else self.worker
        return worker.metadata() if worker is not None else None

    def close(self):
        if self.executor is not None:
            return self.executor.close()
        if self.worker is not None:
            result = self.worker.close()
            self.worker = None
            return result
        return None


class EngineSession:
    """Explicit KAO-TerraMind B resident session with one in-flight call.

    Calls may originate on different threads. Overlap raises SessionBusy rather
    than queuing callers. All backend operations use one dedicated owner thread.
    Use a context manager or close explicitly. A job result does not attest
    subsequent shutdown; inspect the separate cleanup receipt.
    """

    def __init__(self, bundle, work, *, model_id="kao-terramind-b", profile_id,
                 environment="scene", qualification=None, execute_model=False):
        if execute_model is not True:
            raise ExecutionRequired("Pass execute_model=True explicitly; no model was imported or started")
        if model_id != "kao-terramind-b" or environment != "scene" or profile_id not in PROFILES:
            raise ValueError("EngineSession supports KAO-TerraMind B in scene with declared FP32 profiles only")
        if (profile_id == PROFILES[0]) != (qualification is None):
            raise ValueError("Guarded profiles require qualification; full-reference must omit it")
        self._environment = require_profile(environment)
        self._bundle = Path(bundle).resolve(strict=True)
        if not self._bundle.is_dir():
            raise ValueError("Bundle must be an existing directory")
        self._qualification = None
        if qualification is not None:
            from ..cloud.profiles import engine_for
            record = _json(Path(qualification).resolve(strict=True))
            if record.get("profile_id") != profile_id or record.get("model_id") != model_id:
                raise ValueError("Qualification model/profile mismatch")
            engine_for(profile_id).verify_qualification(record)
            import hashlib
            raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            self._qualification = {"record": deepcopy(record), "sha256": hashlib.sha256(raw).hexdigest()}
        work = Path(work).absolute()
        if work.exists() or work.is_symlink() or not work.parent.is_dir():
            raise ValueError("Session work must be a fresh directory with an existing parent")
        self._work = work.parent.resolve()/work.name
        if self._work.is_relative_to(self._bundle):
            raise ValueError("Session work must be outside the immutable model bundle")
        self._profile = profile_id
        self._id = uuid.uuid4().hex
        self._state = "new"
        self._worker_metadata = None
        self._cleanup = None
        self._backend = None
        self._owns_process = False
        self._gate = threading.Lock()
        self._commands = Queue(maxsize=1)
        self._thread = None
        self._work.mkdir()
        (self._work/"admissions").mkdir()
        _write(self._work/"session.json", {"schema": "aureon.engine_session.v1", "session_id": self._id,
            "model_id": model_id, "profile_id": profile_id, "environment": self._environment,
            "qualification_sha256": self._qualification["sha256"] if self._qualification else None,
            "scope": "Resident serial KAO-TerraMind B; job success and session shutdown are separate"})

    def _call(self, command, payload=None):
        if not self._gate.acquire(blocking=False):
            raise SessionBusy("One session call is already in flight; no call was queued")
        try:
            if command == "close" and self._cleanup is not None:
                if self._thread is not None and self._thread.is_alive():
                    self._commands.put_nowait(("stop", None, None))
                    self._thread.join()
                self._state = "closed"
                return deepcopy(self._cleanup)
            if command == "metadata":
                return self._metadata()
            if self._state in ("failed", "closed") and command != "close":
                raise SessionClosed("Session is terminal; create a new session explicitly")
            if command == "run":
                manifest, job_id, attempt_id, current = payload
                if job_id is None:
                    job_id = uuid.uuid4().hex
                if not isinstance(job_id, str) or re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
                    raise ValueError("job_id must be exactly 32 lowercase hexadecimal characters")
                attempt_id = job_id if attempt_id is None else attempt_id
                if not isinstance(attempt_id, str) or re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None:
                    raise ValueError("attempt_id must be exactly 32 lowercase hexadecimal characters")
                if current is not None and not callable(current):
                    raise TypeError("is_current_attempt must be a callable or None")
                manifest = Path(manifest).resolve(strict=True)
                if not manifest.is_file() or manifest.is_relative_to(self._work):
                    raise ValueError("Prepared input must be an existing file outside session work")
                declaration = _json(manifest)
                if not isinstance(declaration, dict) or not isinstance(declaration.get("scene_id"), str):
                    raise ValueError("Prepared manifest requires a scene_id")
                output = self._work/"backend"/attempt_id/job_id
                if any(p.is_symlink() for p in (self._work, output.parent.parent, output.parent)):
                    raise ValueError("Session output ancestors cannot be symlinks")
                if output.exists() or output.is_symlink():
                    raise FileExistsError("Job output already exists")
                if (self._work/"admissions"/(job_id+"."+attempt_id+".json")).exists():
                    raise FileExistsError("Job attempt was already admitted")
                payload = (manifest, job_id, attempt_id, current)
            if self._thread is None:
                self._thread = threading.Thread(target=self._owner, name="aureon-engine-session-"+self._id,
                                                daemon=False)
                self._thread.start()
            future = Future()
            self._commands.put_nowait((command, payload, future))
            try:
                return future.result()
            finally:
                if command == "close":
                    self._thread.join()
        finally:
            self._gate.release()

    def _owner(self):
        while True:
            command, payload, future = self._commands.get()
            if command == "stop":
                return
            try:
                result = self._shutdown() if command == "close" else self._run(*payload)
            except SessionBusy as error:
                future.set_exception(error)
            except BaseException as error:
                detail = {"error_type": type(error).__name__, "error": str(error),
                          "traceback": traceback.format_exc()}
                # Traceback frames can own CUDA tensors. Keep text evidence,
                # clear frames before cleanup, and return a fresh public error.
                traceback.clear_frames(error.__traceback__)
                error.__traceback__ = None
                self._state = "failed"
                cleanup = self._shutdown(failed=True)
                detail["cleanup"] = cleanup
                try:
                    _write(self._work/"session-failure.json", detail)
                except OSError:
                    pass
                future.set_exception(SessionFailed(detail["error_type"]+": "+detail["error"]))
            else:
                future.set_result(result)
            if command == "close":
                return

    def _run(self, manifest, job_id, attempt_id, current):
        if current is not None and current() is not True:
            raise RuntimeError("Attempt is stale before execution")
        if not self._owns_process:
            if not _PROCESS_OWNER.acquire(blocking=False):
                raise SessionBusy("Another EngineSession owns the process runtime")
            self._owns_process = True
        source_hash = _hash(manifest)
        _write(self._work/"admissions"/(job_id+"."+attempt_id+".json"), {"job_id": job_id,
            "attempt_id": attempt_id, "source_manifest_sha256": source_hash, "manifest": str(manifest)})
        self._state = "running"
        # The canonical bundle is verified before model construction; the
        # existing worker also snapshots and attests every loaded tensor.
        if self._backend is None:
            from ..runtime import verify_bundle
            verified = verify_bundle(self._bundle)
            if verified["model_id"] != "kao-terramind-b":
                raise ValueError("Bundle is not KAO-TerraMind B")
            if self._qualification and verified["receipt_sha256"] != self._qualification["record"]["bundle_receipt_sha256"]:
                raise ValueError("Bundle differs from the guarded qualification")
            self._backend = _Backend(self._bundle, self._work/"backend", self._profile, self._qualification)
        if _hash(manifest) != source_hash:
            raise RuntimeError("Prepared manifest changed before model work")
        paths = self._backend.run(manifest, job_id, attempt_id)
        current_worker = self._backend.metadata()
        if not current_worker or current_worker["model_load_count"] != 1:
            raise RuntimeError("Expected exactly one resident model load")
        if self._worker_metadata and current_worker["worker_id"] != self._worker_metadata["worker_id"]:
            raise RuntimeError("Resident worker identity changed")
        self._worker_metadata = current_worker
        output = self._work/"backend"/attempt_id/job_id
        expected = {"job.json", "class_map.tif", "valid_mask.tif", "provenance.json"}
        if set(paths) != expected:
            raise RuntimeError("Incomplete Engine artifact set")
        for path in paths.values():
            path = Path(path)
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
                raise RuntimeError("Job artifact escapes its owned output")
        job = _json(paths["job.json"])
        numerical = "passed" if self._profile == PROFILES[0] else "not_evaluated"
        if (job.get("status") != "completed" or job.get("job_id") != job_id
                or job.get("source_manifest_sha256") != source_hash or _hash(manifest) != source_hash):
            raise RuntimeError("Job/input identity mismatch")
        if self._profile != PROFILES[0] and job.get("validation", {}).get("numerical", {}).get("status") != numerical:
            raise RuntimeError("Guarded numerical status mismatch")
        if current is not None and current() is not True:
            raise RuntimeError("Attempt is stale after execution; no public success receipt")
        receipt = output/"engine-result.json"
        _write(receipt, {"schema": "aureon.engine_session_result.v1", "status": "completed",
            "session_id": self._id, "job_id": job_id, "attempt_id": attempt_id, "worker": self._worker_metadata,
            "package_version": packages.version("aureon-inference"), "model_id": "kao-terramind-b",
            "profile_id": self._profile, "numerical_status": numerical, "task_accuracy": "not_evaluated",
            "source_manifest_sha256": source_hash,
            "artifacts": {name: {"path": str(path), "bytes": Path(path).stat().st_size, "sha256": _hash(path)}
                          for name, path in paths.items()},
            "session_shutdown": "not_evaluated", "scope": "Completed job artifacts; session still owns resident resources"})
        self._state = "ready"
        return SceneResult(output, Path(paths["job.json"]), Path(paths["class_map.tif"]),
                           Path(paths["valid_mask.tif"]), Path(paths["provenance.json"]),
                           self._profile, numerical, receipt)

    def _shutdown(self, *, failed=False):
        if self._cleanup is None:
            result = None
            try:
                if self._backend is not None:
                    result = self._backend.close()
            except BaseException as error:
                result = {"cleanup_passed": False, "full_identity_check_passed": False,
                          "errors": [{"error_type": type(error).__name__, "error": str(error)}]}
                traceback.clear_frames(error.__traceback__)
                error.__traceback__ = None
            if result is None:
                result = {"cleanup_passed": self._backend is None, "full_identity_check_passed": None,
                          "errors": [], "model_load_count": 0, "model_started": False}
            self._cleanup = {**result, "session_schema": "aureon.engine_session_cleanup.v1",
                             "session_id": self._id, "session_state": "failed" if failed else "closed"}
            self._backend = None
            # A failed release keeps the process ownership lock held. Another
            # session must not reuse an uncertain CUDA/precision environment.
            if self._owns_process and result.get("cleanup_passed") and not result.get("errors"):
                _PROCESS_OWNER.release()
                self._owns_process = False
            try:
                _write(self._work/"session-cleanup.json", self._cleanup)
            except OSError as error:
                self._cleanup = {**self._cleanup, "cleanup_receipt_persisted": False,
                                 "receipt_error": str(error)}
        self._state = "failed" if failed else "closed"
        return deepcopy(self._cleanup)

    def _metadata(self):
        worker = deepcopy(self._worker_metadata)
        return {"schema": "aureon.engine_session_metadata.v1", "session_id": self._id,
                "state": self._state, "profile_id": self._profile, "model_id": "kao-terramind-b",
                "worker": worker, "worker_id": worker.get("worker_id") if worker else None,
                "model_load_count": worker.get("model_load_count", 0) if worker else 0,
                "owner_thread_id": self._thread.ident if self._thread else None,
                "maximum_in_flight_calls": 1, "queued_calls": 0}

    def run(self, prepared_input, *, job_id=None, attempt_id=None, is_current_attempt=None):
        return self._call("run", (prepared_input, job_id, attempt_id, is_current_attempt))

    def metadata(self):
        return self._call("metadata")

    def close(self):
        return self._call("close")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        cleanup = self.close()
        if exc_type is None and (not cleanup.get("cleanup_passed") or cleanup.get("errors")
                                 or cleanup.get("cleanup_receipt_persisted") is False):
            raise SessionFailed("Session shutdown failed; inspect session-cleanup.json")
        return False
