"""Dependency-free client for AUREON's explicit async scene API.

Importing or constructing a client performs no I/O or model execution. POSTs
are never retried automatically. Downloads are bounded and verified against
the completed job, with a commit receipt written only after every check passes.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import http.client
import ipaddress
import json
import math
from pathlib import Path
import re
import stat
import time
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit


ARTIFACT_NAMES = ("job.json", "provenance.json", "class_map.tif", "valid_mask.tif")
_JOB = re.compile(r"[0-9a-f]{32}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_STATES = {"queued", "running", "completed", "failed", "cancelled", "expired"}
_JSON_LIMIT = 1024 * 1024
_CHUNK = 64 * 1024


class ClientError(RuntimeError):
    """API transport, schema or artifact verification failed."""


class APIError(ClientError):
    def __init__(self, status: int, code: str, message: str, retry_after: str | None):
        super().__init__(f"HTTP {status} ({code}): {message}")
        self.status, self.code, self.retry_after = status, code, retry_after


class SubmissionUncertain(ClientError):
    """A submission may have reached the server; reuse its original key."""

    def __init__(self, scene_id: str, idempotency_key: str):
        super().__init__("Submission outcome is unknown. Check the service and explicitly "
                         "resubmit the same scene with the same idempotency key; do not create a new key.")
        self.scene_id, self.idempotency_key = scene_id, idempotency_key


class JobNotCompleted(ClientError):
    def __init__(self, job: Job):
        super().__init__(f"Job {job.job_id} is {job.status}")
        self.job = job


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
    reused: bool | None = None


@dataclass(frozen=True)
class Delivery:
    job_id: str
    directory: Path
    files: Mapping[str, Path]
    receipt: Path


def _require(condition, message):
    if not condition:
        raise ClientError(message)


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a finite positive number")
    return float(value)


def _job_id(value):
    if not isinstance(value, str) or not _JOB.fullmatch(value):
        raise ValueError("job_id must be 32 lowercase hexadecimal characters")
    return value


def _decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(unused):
        raise ValueError("non-finite JSON constant")
    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ClientError("Response is not strict UTF-8 JSON") from None
    _require(isinstance(result, dict), "Expected a JSON object")
    return result


class SceneClient:
    """Synchronous Python interface to asynchronous server jobs.

    HTTP is restricted to loopback. HTTPS uses normal certificate verification.
    No proxies, cookies, redirects, auto-downloads or automatic POST retries.
    The current server selects its model at startup, not in each job request.
    """

    def __init__(self, base_url: str, *, token: str, timeout: float = 30,
                 max_artifact_bytes: int = 512 * 1024 * 1024):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or any(ord(c) < 33 or ord(c) == 127 for c in base_url)):
            raise ValueError("Use an HTTP(S) origin without credentials, path, query or fragment")
        host = parsed.hostname
        if parsed.scheme == "http":
            if host == "localhost":
                host = "127.0.0.1"
            try:
                local = ipaddress.ip_address(host).is_loopback
            except ValueError:
                local = False
            if not local:
                raise ValueError("Plain HTTP is restricted to loopback; remote endpoints require HTTPS")
        if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token):
            raise ValueError("Supply a URL-safe bearer token of 32..128 characters")
        if type(max_artifact_bytes) is not int or not 1 <= max_artifact_bytes <= 2**40:
            raise ValueError("Declare max_artifact_bytes between 1 and 2**40")
        self._scheme, self._host = parsed.scheme, host
        self._port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if not 1 <= self._port <= 65535:
            raise ValueError("Invalid endpoint port")
        self._token = token
        self.timeout = _seconds(timeout, "timeout")
        self.max_artifact_bytes = max_artifact_bytes

    @classmethod
    def from_token_file(cls, base_url, token_file, **kwargs):
        path = Path(token_file)
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or not 1 <= details.st_size <= 4096:
            raise ValueError("Token must be in a bounded regular file")
        with path.open("rb") as stream:
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise ValueError("Token file is too large")
        return cls(base_url, token=raw.decode("ascii").strip(), **kwargs)

    @contextmanager
    def _response(self, method, path, *, payload=None, key=None, timeout=None):
        headers = {"Authorization": "Bearer " + self._token, "Connection": "close",
                   "Accept-Encoding": "identity"}
        body = None
        if payload is not None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if key is not None:
            headers["Idempotency-Key"] = key
        connection_type = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        connection = connection_type(self._host, self._port, timeout=timeout or self.timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                data = self._read(response, _JSON_LIMIT)
                error = _decode(data).get("error", {})
                if not isinstance(error, dict):
                    error = {}
                code = error.get("code", "api_error")
                if not isinstance(code, str) or not re.fullmatch(r"[a-z0-9_]{1,80}", code):
                    code = "api_error"
                message = str(error.get("message", "Request rejected")).replace(self._token, "[redacted]")[:500]
                raise APIError(response.status, code, message, response.getheader("Retry-After"))
            yield response
        finally:
            connection.close()

    @staticmethod
    def _length(response, limit):
        values = response.headers.get_all("Content-Length", [])
        _require(len(values) == 1 and re.fullmatch(r"[0-9]{1,16}", values[0]) is not None,
                 "Expected one bounded Content-Length")
        size = int(values[0])
        _require(size <= limit and not response.getheader("Transfer-Encoding")
                 and response.getheader("Content-Encoding", "identity") == "identity",
                 "Response exceeds its limit or uses unsupported encoding")
        return size

    @classmethod
    def _read(cls, response, limit):
        size = cls._length(response, limit)
        data = response.read(size + 1)
        _require(len(data) == size, "Truncated response")
        return data

    def _json(self, method, path, **kwargs):
        with self._response(method, path, **kwargs) as response:
            _require(response.status in {200, 202}, "Unexpected successful API status")
            return _decode(self._read(response, _JSON_LIMIT))

    def _job(self, value, *, expected_id=None, expected_scene=None):
        raw = value.get("job")
        _require(isinstance(raw, dict), "Missing job object")
        identifier, scene, status = raw.get("job_id"), raw.get("scene_id"), raw.get("status")
        _require(isinstance(identifier, str) and _JOB.fullmatch(identifier), "Invalid job identity")
        _require(isinstance(scene, str) and 1 <= len(scene) <= 256 and isinstance(status, str) and status in _STATES,
                 "Invalid job state or scene")
        _require(expected_id is None or identifier == expected_id, "Server returned a different job")
        _require(expected_scene is None or scene == expected_scene, "Server returned a different scene")
        artifacts = raw.get("artifacts", {})
        _require(isinstance(artifacts, dict), "Invalid artifacts object")
        _require(set(artifacts) == set(ARTIFACT_NAMES) if status == "completed" else not artifacts,
                 "Unexpected artifact set for job state")
        parsed = {}
        for name, item in artifacts.items():
            url = f"/v1/jobs/{identifier}/artifacts/{name}"
            _require(isinstance(item, dict) and item.get("url") == url
                     and isinstance(item.get("sha256"), str) and _SHA.fullmatch(item["sha256"])
                     and type(item.get("bytes")) is int and 0 <= item["bytes"] <= self.max_artifact_bytes,
                     "Invalid artifact identity, URL or size")
            parsed[name] = Artifact(name, url, item["sha256"], item["bytes"])
        reused = value.get("reused")
        _require(reused is None or type(reused) is bool, "Invalid idempotency replay indicator")
        return Job(identifier, scene, status, MappingProxyType(parsed), reused)

    def info(self):
        """Read registered scenes and actual resident-worker metadata."""
        return self._json("GET", "/v1/info")

    def submit(self, scene_id: str, *, idempotency_key: str) -> Job:
        """Explicitly submit once; retain the key before calling this method."""
        if not isinstance(scene_id, str) or not 1 <= len(scene_id) <= 256:
            raise ValueError("Supply a registered scene_id of 1..256 characters")
        if not isinstance(idempotency_key, str) or not re.fullmatch(r"[\x21-\x7e]{1,128}", idempotency_key):
            raise ValueError("Idempotency key must have 1..128 visible ASCII characters")
        try:
            return self._job(self._json("POST", "/v1/jobs", payload={"scene_id": scene_id}, key=idempotency_key),
                             expected_scene=scene_id)
        except APIError:
            raise
        except (OSError, http.client.HTTPException, ClientError):
            raise SubmissionUncertain(scene_id, idempotency_key) from None

    def get(self, job_id: str, *, timeout=None) -> Job:
        identifier = _job_id(job_id)
        if timeout is not None:
            timeout = _seconds(timeout, "timeout")
        return self._job(self._json("GET", "/v1/jobs/" + identifier, timeout=timeout), expected_id=identifier)

    def cancel(self, job_id: str) -> Job:
        """Request queued-job cancellation; server conflicts remain visible."""
        identifier = _job_id(job_id)
        return self._job(self._json("DELETE", "/v1/jobs/" + identifier), expected_id=identifier)

    def wait(self, job_id: str, *, timeout: float = 600, poll_interval: float = 0.5) -> Job:
        """Poll until completed; timeout never resubmits or cancels a job."""
        identifier = _job_id(job_id)
        deadline = time.monotonic() + _seconds(timeout, "timeout")
        interval = _seconds(poll_interval, "poll_interval")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for job {identifier}; job remains on server")
            job = self.get(identifier, timeout=min(self.timeout, remaining))
            if job.status == "completed":
                return job
            if job.status in {"failed", "cancelled", "expired"}:
                raise JobNotCompleted(job)
            time.sleep(min(interval, max(0, deadline - time.monotonic())))

    def download(self, job_id: str, output_dir, *, expected_source_sha256=None,
                 expected_bundle_sha256=None) -> Delivery:
        """Download a complete, verified scene into a fresh directory.

        Verifies transport hashes and linked receipts, not pixel/model accuracy.
        Failure leaves a clearly uncommitted partial directory, never a success
        receipt. A subsequent attempt must use a new directory.
        """
        job = self.get(job_id)
        if job.status != "completed":
            raise JobNotCompleted(job)
        for digest in (expected_source_sha256, expected_bundle_sha256):
            if digest is not None and (not isinstance(digest, str) or not _SHA.fullmatch(digest)):
                raise ValueError("Expected identities must be lowercase SHA-256 digests")
        directory = Path(output_dir).absolute()
        if directory.exists() or directory.is_symlink() or not directory.parent.is_dir():
            raise ValueError("Use a fresh output directory under an existing parent")
        directory.mkdir(exist_ok=False)
        files = {}
        for name in ARTIFACT_NAMES:
            descriptor = job.artifacts[name]
            path = directory / name
            temporary = directory / (name + ".partial")
            digest = hashlib.sha256()
            try:
                with self._response("GET", descriptor.url) as response:
                    _require(response.status == 200, "Expected a complete artifact response")
                    size = self._length(response, self.max_artifact_bytes)
                    _require(size == descriptor.bytes and response.getheader("ETag") == '"' + descriptor.sha256 + '"',
                             "Artifact headers differ from completed job")
                    remaining = size
                    with temporary.open("xb") as stream:
                        while remaining:
                            chunk = response.read(min(_CHUNK, remaining))
                            _require(bool(chunk), "Truncated artifact download")
                            stream.write(chunk)
                            digest.update(chunk)
                            remaining -= len(chunk)
                    _require(digest.hexdigest() == descriptor.sha256, "Artifact SHA-256 mismatch")
                # The directory is exclusively created by this download.
                temporary.rename(path)
                files[name] = path
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        self._verify_receipts(job, files, expected_source_sha256, expected_bundle_sha256)
        receipt = directory / "download.json"
        record = {"schema": "aureon.client_download.v1", "status": "completed", "job_id": job.job_id,
                  "scene_id": job.scene_id, "transport_and_receipt_identity_verified": True,
                  "independent_pixel_or_model_audit_performed": False,
                  "files": {name: {"sha256": item.sha256, "bytes": item.bytes}
                            for name, item in job.artifacts.items()}}
        pending = directory / "download.json.partial"
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        pending.rename(receipt)
        return Delivery(job.job_id, directory, MappingProxyType(files), receipt)

    @staticmethod
    def _verify_receipts(job, files, expected_source, expected_bundle):
        def read(path):
            with path.open("rb") as stream:
                data = stream.read(_JSON_LIMIT + 1)
            _require(len(data) <= _JSON_LIMIT, "Receipt JSON exceeds its size limit")
            return _decode(data)
        execution, delivery = read(files["job.json"]), read(files["provenance.json"])
        _require(execution.get("schema") in {"aureon.scene_job.v1", "aureon.streaming_scene_job.v1"}
                 and execution.get("status") == "completed" and execution.get("job_id") == job.job_id
                 and execution.get("scene_id") == job.scene_id, "Execution receipt identity mismatch")
        delivery_schema = {"aureon.scene_job.v1": "aureon.georaster_delivery.v1",
                           "aureon.streaming_scene_job.v1": "aureon.streaming_delivery.v1"}[execution["schema"]]
        _require(delivery.get("schema") == delivery_schema and delivery.get("status") == "completed"
                 and delivery.get("format") == "COG", "Unexpected delivery receipt")
        provenance = delivery.get("provenance")
        _require(isinstance(provenance, dict) and provenance.get("job_id") == job.job_id
                 and provenance.get("scene_id") == job.scene_id, "Provenance identity mismatch")
        for key, expected in (("source_manifest_sha256", expected_source), ("bundle_receipt_sha256", expected_bundle)):
            actual = execution.get(key)
            _require(isinstance(actual, str) and _SHA.fullmatch(actual) and provenance.get(key) == actual,
                     "Source/model receipt identities disagree")
            _require(expected is None or expected == actual, "Source/model identity differs from caller expectation")
        _require(isinstance(execution.get("parity"), dict) and execution["parity"].get("passed") is True,
                 "Execution receipt does not record passed parity")
        prefix = "delivery/" if execution["schema"] == "aureon.streaming_scene_job.v1" else ""
        recorded = execution.get("files_sha256", {})
        _require(isinstance(recorded, dict) and isinstance(delivery.get("files"), dict), "Missing file identities")
        for name in ("class_map.tif", "valid_mask.tif", "provenance.json"):
            relative = prefix + "artifacts/" + ("receipt.json" if name == "provenance.json" else name)
            descriptor = job.artifacts[name]
            _require(recorded.get(relative) == descriptor.sha256, "Artifact differs from execution receipt")
            if name.endswith(".tif"):
                item = delivery["files"].get(name, {})
                _require(isinstance(item, dict) and item.get("sha256") == descriptor.sha256
                         and item.get("bytes") == descriptor.bytes, "COG differs from delivery receipt")


def main(argv=None):
    """Explicit CLI client; requires a separately started, authorized worker."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=30)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Read service and registered-scene metadata")
    run = commands.add_parser("run", help="Explicitly submit once, wait, and verify downloads")
    run.add_argument("--execute-model", action="store_true")
    run.add_argument("--scene-id", required=True)
    run.add_argument("--idempotency-key", required=True, help="Keep this key for any explicit resubmission")
    fetch = commands.add_parser("fetch", help="Wait for an existing job and verify downloads")
    fetch.add_argument("--job-id", required=True)
    cancel = commands.add_parser("cancel", help="Request cancellation of a queued job")
    cancel.add_argument("--job-id", required=True)
    for command in (run, fetch):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--wait-seconds", type=float, default=600)
        command.add_argument("--poll-seconds", type=float, default=0.5)
        command.add_argument("--max-artifact-bytes", type=int, default=512 * 1024 * 1024)
        command.add_argument("--expected-source-sha256")
        command.add_argument("--expected-bundle-sha256")
    args = parser.parse_args(argv)
    if args.command == "run" and not args.execute_model:
        parser.error("Submitting a model job requires --execute-model")
    try:
        if args.command in {"run", "fetch"}:
            _seconds(args.wait_seconds, "wait-seconds")
            _seconds(args.poll_seconds, "poll-seconds")
            if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
                raise ValueError("Use a fresh output directory under an existing parent")
            for identity in (args.expected_source_sha256, args.expected_bundle_sha256):
                if identity is not None and not _SHA.fullmatch(identity):
                    raise ValueError("Expected identities must be lowercase SHA-256 digests")
        client = SceneClient.from_token_file(args.endpoint, args.token_file,
                    timeout=args.request_timeout,
                    max_artifact_bytes=getattr(args, "max_artifact_bytes", 512 * 1024 * 1024))
        if args.command == "info":
            print(json.dumps(client.info(), indent=2, allow_nan=False))
            return 0
        if args.command == "cancel":
            job = client.cancel(args.job_id)
            print(json.dumps({"job_id": job.job_id, "status": job.status}))
            return 0
        if args.command == "run":
            job = client.submit(args.scene_id, idempotency_key=args.idempotency_key)
            print(json.dumps({"job_id": job.job_id, "status": job.status, "reused": job.reused}), flush=True)
            identifier = job.job_id
        else:
            identifier = args.job_id
        client.wait(identifier, timeout=args.wait_seconds, poll_interval=args.poll_seconds)
        delivery = client.download(identifier, args.output,
                    expected_source_sha256=args.expected_source_sha256,
                    expected_bundle_sha256=args.expected_bundle_sha256)
        print(json.dumps({"job_id": identifier, "status": "download_verified", "receipt": str(delivery.receipt)}))
        return 0
    except (ClientError, OSError, http.client.HTTPException, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
