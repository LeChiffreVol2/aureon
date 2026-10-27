"""Bounded loopback HTTP jobs around one explicitly constructed worker.

The factory, metadata, execution and close methods run only on the owner
thread. HTTP threads operate on persisted state snapshots and completed files.
This module imports no model runtime and performs no acquisition or retries.
"""

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import threading
import time
import uuid

from .console import ConsoleAssets, HEADERS as CONSOLE_HEADERS


ARTIFACT_NAMES = frozenset(("class_map.tif", "valid_mask.tif", "provenance.json", "job.json"))
MAX_BODY = 16 * 1024
MAX_ARTIFACT = 16 * 1024 * 1024
_JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class APIError(ValueError):
    def __init__(self, status, code, message, *, retry_after=None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _snapshot_json(value, maximum=128 * 1024):
    raw = _json_bytes(value)
    if len(raw) > maximum:
        raise ValueError("Worker metadata/receipt exceeds its bounded JSON size")
    return json.loads(raw)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact_path(folder, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive
            or any(part in (".", "..") for part in relative.split("/"))):
        raise ValueError("Artifact paths must be relative files within this job")
    if folder.is_symlink():
        raise ValueError("Job output directory must not be a symbolic link")
    folder = folder.resolve(strict=True)
    path = folder / relative
    if (path.is_symlink() or not path.is_file()
            or not path.resolve(strict=True).is_relative_to(folder)):
        raise ValueError("Artifact escapes its job or is unavailable")
    return path


def _artifact_bytes(folder, declaration):
    if (not isinstance(declaration, dict) or set(declaration) != {"path", "sha256", "bytes"}
            or not isinstance(declaration["sha256"], str) or not _SHA.fullmatch(declaration["sha256"])
            or type(declaration["bytes"]) is not int or not 0 <= declaration["bytes"] <= MAX_ARTIFACT):
        raise ValueError("Artifact must declare a bounded byte length, relative path and SHA-256")
    path = _artifact_path(folder, declaration["path"])
    # Snapshot bounded bytes before serving: a later mutation cannot change the
    # response after its identity check. Concurrent snapshots are HTTP-bounded.
    with path.open("rb") as stream:
        values = stream.read(MAX_ARTIFACT + 1)
    if (len(values) != declaration["bytes"]
            or hashlib.sha256(values).hexdigest() != declaration["sha256"]):
        raise ValueError("Completed artifact failed its byte length or SHA-256 check")
    return values


class LocalJobService:
    """One lifetime-bounded job registry and one FIFO/model owner.

    ``output_dir`` must be new; jobs are not silently resumed after restart.
    Status files live beside immutable job outputs, under status/. max_jobs is
    a lifetime bound, preserving every admitted idempotency mapping. Queue
    capacity counts waiting work, in addition to at most one running job.
    """

    def __init__(self, output_dir, worker_factory, scene_ids, queue_capacity=2, max_jobs=32):
        if not callable(worker_factory):
            raise ValueError("worker_factory must be callable")
        if (type(queue_capacity) is not int or not 1 <= queue_capacity <= 32
                or type(max_jobs) is not int or not queue_capacity <= max_jobs <= 1024):
            raise ValueError("Declare bounded queue_capacity 1..32 and max_jobs capacity..1024")
        if isinstance(scene_ids, str):
            raise ValueError("scene_ids must be a collection of registered strings")
        scenes = tuple(scene_ids)
        if (not scenes or len(scenes) > 64
                or any(not isinstance(scene, str) or not 1 <= len(scene) <= 256 for scene in scenes)
                or len(set(scenes)) != len(scenes)):
            raise ValueError("Register 1..64 unique scene IDs")
        self.output_dir = Path(output_dir).absolute()
        if not self.output_dir.parent.is_dir():
            raise ValueError("Service output parent must exist")
        self.output_dir.mkdir(exist_ok=False)
        (self.output_dir / "jobs").mkdir()
        (self.output_dir / "status").mkdir()
        self.scene_ids, self.queue_capacity, self.max_jobs = scenes, queue_capacity, max_jobs
        self._factory, self._worker_metadata, self._worker_shutdown = worker_factory, {}, None
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._jobs, self._keys, self._queue = {}, {}, deque()
        self._stop = False
        self._state, self._error, self._running_id = "initializing", None, None
        self._init_count = 0
        self._persist_service()
        self._thread = threading.Thread(target=self._owner, name="aureon-scene-owner", daemon=False)
        self._thread.start()

    def _service_snapshot(self):
        return {"schema": "aureon.local_scene_service.v1", "state": self._state,
                "scene_ids": list(self.scene_ids), "queue_capacity": self.queue_capacity,
                "queued_jobs": len(self._queue), "running_job_id": self._running_id,
                "max_jobs": self.max_jobs, "admitted_jobs": len(self._jobs),
                "worker_factory_calls": self._init_count, "worker": deepcopy(self._worker_metadata),
                "worker_shutdown": deepcopy(self._worker_shutdown),
                "admission_open": self._state == "ready" and not self._stop,
                "error": deepcopy(self._error)}

    def _persist_service(self):
        _atomic_json(self.output_dir / "service.json", self._service_snapshot())

    def _public_job(self, job):
        result = {key: deepcopy(value) for key, value in job.items()
                  if key not in ("request_digest", "idempotency_key_sha256", "download_artifacts")}
        result["artifacts"] = {name: {"url": f"/v1/jobs/{job['job_id']}/artifacts/{name}",
                                      "sha256": item["sha256"], "bytes": item["bytes"]}
                               for name, item in job.get("download_artifacts", {}).items()}
        return result

    def _persist_job(self, job):
        _atomic_json(self.output_dir / "status" / (job["job_id"] + ".json"), job)

    def snapshot(self):
        with self._lock:
            return self._service_snapshot()

    def wait_ready(self, timeout=30):
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._state == "initializing":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return self._state == "ready" and not self._stop

    def submit(self, request, idempotency_key):
        if (not isinstance(request, dict) or set(request) != {"scene_id"}
                or not isinstance(request["scene_id"], str) or not 1 <= len(request["scene_id"]) <= 256):
            raise APIError(400, "invalid_request", "Supply exactly one registered scene_id string")
        if (not isinstance(idempotency_key, str)
                or not re.fullmatch(r"[\x21-\x7e]{1,128}", idempotency_key)):
            raise APIError(400, "invalid_idempotency_key", "Idempotency-Key must contain 1..128 visible ASCII characters")
        request_digest = hashlib.sha256(_json_bytes(request)).hexdigest()
        key_hash = hashlib.sha256(idempotency_key.encode("ascii")).hexdigest()
        with self._condition:
            if key_hash in self._keys:
                job = self._jobs[self._keys[key_hash]]
                if job["request_digest"] != request_digest:
                    raise APIError(409, "idempotency_conflict", "This Idempotency-Key is bound to a different request")
                return {"job": self._public_job(job), "reused": True}
            if request["scene_id"] not in self.scene_ids:
                raise APIError(400, "unknown_scene", "Use a registered scene ID from /v1/info")
            if self._state != "ready" or self._stop:
                raise APIError(503, "unavailable", "The model worker is not accepting jobs", retry_after=1)
            if len(self._queue) >= self.queue_capacity:
                raise APIError(429, "queue_full", "The bounded waiting queue is full", retry_after=1)
            if len(self._jobs) >= self.max_jobs:
                raise APIError(429, "job_limit_reached", "This service reached its lifetime job registry limit", retry_after=1)
            identifier, now = uuid.uuid4().hex, _utc()
            job = {"schema": "aureon.local_scene_job_status.v1", "job_id": identifier,
                   "scene_id": request["scene_id"], "status": "queued", "created_utc": now,
                   "updated_utc": now, "request_digest": request_digest,
                   "idempotency_key_sha256": key_hash}
            self._persist_job(job)
            self._jobs[identifier], self._keys[key_hash] = job, identifier
            self._queue.append(identifier)
            try:
                self._persist_service()
            except BaseException as error:
                # A failed durable admission must never silently start work.
                self._queue.remove(identifier)
                self._stop, self._state = True, "unavailable"
                self._error = {"type": type(error).__name__, "message": "Service status persistence failed"}
                try:
                    self._set_status(job, "failed", error=self._error)
                    self._cancel_waiting("status_persistence_failed")
                except BaseException:
                    pass
                self._condition.notify_all()
                raise APIError(503, "status_persistence_failed", "Durable job admission failed") from error
            self._condition.notify_all()
            return {"job": self._public_job(job), "reused": False}

    def get_job(self, identifier):
        with self._lock:
            if identifier not in self._jobs:
                raise APIError(404, "job_not_found", "Unknown job ID")
            return self._public_job(self._jobs[identifier])

    def cancel(self, identifier):
        with self._condition:
            if identifier not in self._jobs:
                raise APIError(404, "job_not_found", "Unknown job ID")
            job = self._jobs[identifier]
            if job["status"] == "cancelled":
                return self._public_job(job)
            if job["status"] != "queued":
                raise APIError(409, "not_cancellable", "Only queued jobs can be cancelled")
            self._queue.remove(identifier)
            self._set_status(job, "cancelled", reason="client_requested")
            self._persist_service()
            return self._public_job(job)

    def artifact(self, identifier, name):
        with self._lock:
            if identifier not in self._jobs or name not in ARTIFACT_NAMES:
                raise APIError(404, "artifact_not_found", "Unknown job or artifact")
            job = self._jobs[identifier]
            if job["status"] != "completed":
                raise APIError(409, "job_not_completed", "Artifacts are available only for completed jobs")
            declaration = deepcopy(job["download_artifacts"][name])
        try:
            data = _artifact_bytes(self.output_dir / "jobs" / identifier, declaration)
        except (OSError, ValueError) as error:
            raise APIError(409, "artifact_identity_failed", "Completed artifact identity validation failed") from error
        return data, declaration["sha256"]

    def _set_status(self, job, state, **extra):
        job.update(status=state, updated_utc=_utc(), **extra)
        if state == "running":
            job["started_utc"] = job["updated_utc"]
        elif state in ("completed", "failed", "cancelled"):
            job["finished_utc"] = job["updated_utc"]
        self._persist_job(job)

    def _cancel_waiting(self, reason):
        while self._queue:
            self._set_status(self._jobs[self._queue.popleft()], "cancelled", reason=reason)

    def _worker_snapshot(self, worker):
        value = worker.metadata()
        if not isinstance(value, dict):
            raise ValueError("Worker metadata must be a dictionary")
        return _snapshot_json(value, 64 * 1024)

    def _completed_result(self, result, identifier):
        if not isinstance(result, dict) or result.get("status") != "completed":
            raise ValueError("Worker must return an explicitly completed receipt")
        if result.get("job_id", identifier) != identifier:
            raise ValueError("Worker returned a different job identity")
        files = result.get("download_artifacts")
        if not isinstance(files, dict) or set(files) != ARTIFACT_NAMES:
            raise ValueError("Completed worker receipt must expose the exact four allowlisted artifacts")
        for declaration in files.values():
            _artifact_bytes(self._job_folder(identifier), declaration)
        return _snapshot_json(result)

    def _job_folder(self, identifier):
        return self.output_dir / "jobs" / identifier

    def _run_worker_job(self, worker, job, identifier):
        return worker.run_job(job["scene_id"], identifier, self._job_folder(identifier))

    def _maintenance(self):
        """Owner-only housekeeping hook; legacy evidence is never retired."""

    def _owner(self):
        worker = None
        try:
            self._init_count += 1
            worker = self._factory()
            metadata = self._worker_snapshot(worker)
            if metadata.get("state") != "ready":
                raise ValueError("Initialized worker must report ready")
            with self._condition:
                self._worker_metadata = metadata
                self._state = "stopping" if self._stop else "ready"
                self._persist_service()
                self._condition.notify_all()
            while True:
                with self._condition:
                    self._maintenance()
                    while not self._queue and not self._stop:
                        self._condition.wait(timeout=1)
                        self._maintenance()
                    if self._stop:
                        break
                    identifier = self._queue.popleft()
                    job = self._jobs[identifier]
                    self._running_id = identifier
                    self._set_status(job, "running")
                    self._persist_service()
                error, result = None, None
                try:
                    result = self._run_worker_job(worker, job, identifier)
                    result = self._completed_result(result, identifier)
                except BaseException as caught:
                    error = {"type": type(caught).__name__, "message": str(caught)[:1000]}
                # Only this thread reads worker health, including after failure.
                try:
                    metadata = self._worker_snapshot(worker)
                except BaseException as caught:
                    metadata = {"state": "failed", "error": {"type": type(caught).__name__,
                                                               "message": str(caught)[:1000]}}
                fatal = metadata.get("state") != "ready"
                if fatal and error is None:
                    error = {"type": "WorkerUnavailable", "message": "Worker failed its health/identity contract"}
                with self._condition:
                    self._worker_metadata = metadata
                    if error is None:
                        extra = {key: result[key] for key in ("summary", "worker_id") if key in result}
                        self._set_status(job, "completed", download_artifacts=result["download_artifacts"], **extra)
                    else:
                        self._set_status(job, "failed", error=error)
                    self._running_id = None
                    if fatal:
                        self._stop, self._state, self._error = True, "unavailable", error
                        self._cancel_waiting("worker_unavailable")
                    self._persist_service()
                    self._condition.notify_all()
        except BaseException as caught:
            with self._condition:
                self._state, self._stop = "unavailable", True
                self._error = {"type": type(caught).__name__, "message": str(caught)[:1000]}
                try:
                    if self._running_id is not None:
                        self._set_status(self._jobs[self._running_id], "failed", error=self._error)
                        self._running_id = None
                    self._cancel_waiting("worker_unavailable")
                    self._persist_service()
                except BaseException:
                    pass
                self._condition.notify_all()
        finally:
            close_error = None
            if worker is not None:
                try:
                    closed = worker.close()
                    if closed is not None:
                        closed = _snapshot_json(closed, 64 * 1024)
                        with self._lock:
                            self._worker_shutdown = closed
                    metadata = self._worker_snapshot(worker)
                    with self._lock:
                        self._worker_metadata = metadata
                    if (isinstance(closed, dict)
                            and (closed.get("full_identity_check_passed") is False or closed.get("errors"))):
                        raise ValueError("Worker shutdown reported identity or cleanup errors")
                except BaseException as caught:
                    close_error = {"type": type(caught).__name__, "message": str(caught)[:1000]}
            with self._condition:
                if close_error:
                    self._state, self._error = "unavailable", close_error
                elif self._state != "unavailable":
                    self._state = "closed"
                try:
                    self._persist_service()
                except BaseException:
                    pass
                self._condition.notify_all()

    def shutdown(self, *, timeout=None):
        """Seal admission, cancel queued jobs, let running work finish, then close."""
        with self._condition:
            self._stop = True
            if self._state not in ("closed", "unavailable"):
                self._state = "stopping"
            self._cancel_waiting("service_shutdown")
            self._persist_service()
            self._condition.notify_all()
        if threading.current_thread() is self._thread:
            return False
        self._thread.join(timeout)
        return not self._thread.is_alive()


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(unused):
        raise ValueError("Nonfinite JSON value")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def _range(value, size):
    if value is None:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=([0-9]{0,20})-([0-9]{0,20})", value)
    if not match or not any(match.groups()) or size == 0:
        raise APIError(416, "invalid_range", "Supply one satisfiable byte range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise APIError(416, "invalid_range", "Byte suffix must be positive")
        return max(0, size - length), size - 1, True
    start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise APIError(416, "invalid_range", "Requested byte range is unsatisfiable")
    return start, end, True


class _BoundedServer(ThreadingHTTPServer):
    # server_close joins admitted HTTP handlers before durable storage closes.
    daemon_threads = False
    request_queue_size = 8

    def process_request(self, request, client_address):
        if not self.connection_slots.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\nRetry-After: 1\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connection_slots.release()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AUREON-Local-Scene"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, unused_format, *unused_args):
        # Tokens and user request paths never enter an implicit access log.
        pass

    def handle_expect_100(self):
        self._error(APIError(417, "unsupported_expectation", "Expect is unsupported for small JSON jobs"))
        return False

    def _reply(self, status, body, content_type="application/json", extra=None):
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", content_type)
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, status, value, extra=None):
        self._reply(status, _json_bytes(value), extra=extra)

    def _error(self, error, extra=None):
        headers = dict(extra or {})
        if error.retry_after is not None:
            headers["Retry-After"] = str(error.retry_after)
        if error.status == 401:
            headers["WWW-Authenticate"] = 'Bearer realm="AUREON local scene API"'
        self._json(error.status, {"error": {"code": error.code, "message": error.message}}, headers)

    def _single(self, name, *, required=False):
        values = self.headers.get_all(name, [])
        if len(values) > 1 or (required and len(values) != 1):
            raise APIError(400, "invalid_headers", f"Supply exactly one {name} header")
        return values[0] if values else None

    def _preflight(self, *, public_get=False):
        host = self._single("Host", required=True)
        if host not in self.server.allowed_hosts:
            raise APIError(403, "invalid_host", "Host must identify this loopback listener")
        origin = self._single("Origin")
        if origin is not None and origin != "http://" + host:
            raise APIError(403, "invalid_origin", "Only this listener's origin is permitted")
        if self.headers.get_all("Transfer-Encoding"):
            raise APIError(400, "unsupported_transfer_encoding", "Transfer-Encoding is unsupported")
        if not public_get:
            authorization = self._single("Authorization") or ""
            expected = "Bearer " + self.server.bearer_token
            if len(authorization) > 512 or not hmac.compare_digest(authorization.encode(), expected.encode()):
                raise APIError(401, "unauthorized", "A valid bearer token is required")

    def _dispatch(self, method):
        try:
            asset = self.server.console_assets.get(self.path) if self.server.console_assets is not None else None
            self._preflight(public_get=method == "GET" and (self.path == "/healthz" or asset is not None))
            if method == 'GET' and asset is not None:
                body, mime, digest = asset
                self._reply(200, body, mime, {**CONSOLE_HEADERS, 'ETag': '"'+digest+'"'})
                return
            if method == "GET" and self.path == "/healthz":
                state = self.server.service.snapshot()["state"]
                self._json(200 if state == "ready" else 503, {"status": state})
                return
            if method == "GET" and self.path == "/v1/info":
                self._json(200, self.server.service.snapshot())
                return
            if method == "POST" and self.path == "/v1/jobs":
                length = self._single("Content-Length", required=True)
                if not re.fullmatch(r"[0-9]{1,8}", length) or not 1 <= int(length) <= MAX_BODY:
                    raise APIError(413, "body_too_large", "Declare a JSON body of 1..16384 bytes")
                content_type = self._single("Content-Type", required=True)
                if content_type.lower() not in ("application/json", "application/json; charset=utf-8"):
                    raise APIError(415, "invalid_content_type", "Content-Type must be application/json")
                raw = self.rfile.read(int(length))
                if len(raw) != int(length):
                    raise APIError(400, "truncated_body", "Incomplete JSON body")
                try:
                    request = _strict_json(raw)
                except (ValueError, UnicodeError, RecursionError) as error:
                    raise APIError(400, "invalid_json", "Supply strict UTF-8 JSON without duplicate keys") from error
                result = self.server.service.submit(request, self._single("Idempotency-Key", required=True))
                self._json(200 if result["reused"] else 202, result,
                           {"Location": "/v1/jobs/" + result["job"]["job_id"]})
                return
            matched = re.fullmatch(r"/v1/jobs/([0-9a-f]{32})", self.path)
            if matched and method in ("GET", "DELETE"):
                job = (self.server.service.get_job(matched[1]) if method == "GET"
                       else self.server.service.cancel(matched[1]))
                self._json(200, {"job": job})
                return
            matched = re.fullmatch(r"/v1/jobs/([0-9a-f]{32})/artifacts/([^/]+)", self.path)
            if method == "GET" and matched:
                name = matched[2]
                data, digest = self.server.service.artifact(matched[1], name)
                try:
                    start, end, partial = _range(self._single("Range"), len(data))
                except APIError as error:
                    self._error(error, {"Content-Range": f"bytes */{len(data)}"})
                    return
                headers = {"ETag": '"' + digest + '"', "Accept-Ranges": "bytes",
                           "Content-Disposition": 'attachment; filename="' + name + '"'}
                if partial:
                    headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
                self._reply(206 if partial else 200, data[start:end + 1],
                            "image/tiff" if name.endswith(".tif") else "application/json", headers)
                return
            raise APIError(404, "not_found", "Unknown local API endpoint")
        except APIError as error:
            self._error(error)
        except (TimeoutError, ConnectionError, BrokenPipeError):
            self.close_connection = True
        except Exception:
            self._error(APIError(500, "internal_error", "Local API request failed"))

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")


def create_server(service, token, port=0, *, console_assets=None):
    """Create an authenticated IPv4 loopback listener; caller starts/stops it."""
    if not isinstance(service, LocalJobService):
        raise ValueError("Supply the explicit LocalJobService")
    if console_assets is not None and not isinstance(console_assets, ConsoleAssets):
        raise ValueError('Console assets must be an explicit bounded startup snapshot')
    if (not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token)
            or type(port) is not int or not 0 <= port <= 65535):
        raise ValueError("Use a random URL-safe token of 32..128 characters and a valid port")
    server = _BoundedServer(("127.0.0.1", port), _Handler)
    server.service, server.bearer_token = service, token
    server.console_assets = console_assets
    server.connection_slots = threading.BoundedSemaphore(8)
    server.allowed_hosts = {f"127.0.0.1:{server.server_port}", f"localhost:{server.server_port}"}
    return server
