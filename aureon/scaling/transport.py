"""Authenticated JSON/content transport, independent from model execution.

The stdlib reference server binds loopback only. For separate hosts, supply TLS via
an authenticated reverse proxy/tunnel. This is an operator replica protocol, not a
public tenant API: workers are trusted, receive scoped worker keys, and no pickle,
Python import names, arbitrary URLs or filesystem paths cross this boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPMessage
import hmac
import json
import os
from pathlib import Path
import ssl
import socket
import threading
import time
import math
from typing import BinaryIO, IO, cast
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .artifacts import ArtifactStore, digest_name
from .broker import Broker, Job, Lease, canonical, identifier, object_json


def _job(value: object) -> Job:
    if not isinstance(value, dict) or set(value) != {"job_id", "tenant", "route", "payload_json", "seed", "required_bytes", "max_attempts"}:
        raise ValueError("Invalid job schema")
    return Job(cast(str, value["job_id"]), cast(str, value["tenant"]), cast(str, value["route"]),
               cast(str, value["payload_json"]), cast(int, value["seed"]), cast(int, value["required_bytes"]), cast(int, value["max_attempts"]))


def _lease(value: object) -> Lease:
    if not isinstance(value, dict) or set(value) != {"job", "worker_id", "token", "attempt", "expires_at"}:
        raise ValueError("Invalid lease schema")
    if (type(value["worker_id"]) is not str or type(value["token"]) is not str
            or type(value["attempt"]) is not int or type(value["expires_at"]) not in (int, float)):
        raise ValueError("Invalid lease fields")
    return Lease(_job(value["job"]), value["worker_id"], value["token"], value["attempt"], value["expires_at"])


class _BoundedServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], limit: int) -> None:
        self._slots = threading.BoundedSemaphore(limit)
        super().__init__(address, handler)

    def process_request(self, request: socket.socket | tuple[bytes, socket.socket], client_address: tuple[str, int]) -> None:
        if not isinstance(request, socket.socket):
            raise TypeError("This coordinator accepts TCP connections only")
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: socket.socket | tuple[bytes, socket.socket], client_address: tuple[str, int]) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class CoordinatorServer:
    """Explicit start/close; caller owns broker/store. Tokens are never logged."""
    def __init__(self, broker: Broker, artifacts: ArtifactStore, *, client_key: str,
                 worker_keys: Mapping[str, str], port: int = 0, max_connections: int = 16) -> None:
        keys = [client_key, *worker_keys.values()]
        if (not worker_keys or any(not isinstance(k, str) or len(k) < 32 or not k.isascii() for k in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError("Distinct client and per-worker random keys of >=32 ASCII characters required")
        if "client" in worker_keys:
            raise ValueError("The client identity is reserved")
        for worker_id in worker_keys:
            identifier(worker_id)
        if type(max_connections) is not int or not 1 <= max_connections <= 256:
            raise ValueError("Declare 1..256 connections")
        workers = dict(worker_keys)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(15.)

            def _identity(self) -> str:
                auth = self.headers.get("Authorization", "")
                if hmac.compare_digest(auth, "Bearer " + client_key):
                    return "client"
                for worker, key in workers.items():
                    if hmac.compare_digest(auth, "Bearer " + key):
                        return worker
                raise PermissionError("Invalid credential")

            def _reply(self, status: int, data: object) -> None:
                body = canonical(data, 131072).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _size(self, limit: int) -> int:
                if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
                    raise ValueError("One explicit Content-Length is required")
                size = int(self.headers["Content-Length"])
                if not 0 <= size <= limit:
                    raise ValueError("Request too large")
                return size

            def do_POST(self) -> None:
                try:
                    identity = self._identity()
                    if self.path != "/v1/control":
                        raise ValueError("Unknown endpoint")
                    data = object_json(self.rfile.read(self._size(65536)).decode("utf-8"))
                    operation = data.get("op")
                    result: object
                    if identity == "client":
                        if operation == "submit":
                            result = broker.submit(_job(data["job"]))
                        elif operation == "status":
                            result = broker.status(identifier(cast(str, data["job_id"])))
                        elif operation == "cancel":
                            broker.cancel(identifier(cast(str, data["job_id"])))
                            result = {"ok": True}
                        else:
                            raise PermissionError("Client operation not allowed")
                    elif operation == "claim":
                        lease = broker.claim(identity)
                        result = asdict(lease) if lease else None
                    else:
                        lease = _lease(data["lease"])
                        if lease.worker_id != identity:
                            raise PermissionError("Worker does not own this lease")
                        if operation == "heartbeat":
                            result = {"expires_at": broker.heartbeat(lease)}
                        elif operation == "complete":
                            result_data = data["result"]
                            if not isinstance(result_data, dict):
                                raise ValueError("Result must be metadata object")
                            references = result_data.get("artifacts")
                            if not isinstance(references, dict) or len(references) > 64:
                                raise ValueError("Result requires a bounded artifacts name-to-SHA256 map")
                            for name, digest in references.items():
                                identifier(name)
                                with artifacts.open(digest):
                                    pass
                            broker.complete(lease, cast(dict[str, object], result_data))
                            result = {"ok": True}
                        elif operation == "fail":
                            if type(data.get("retry", False)) is not bool:
                                raise ValueError("Retry must be boolean")
                            broker.fail(lease, cast(str, data["error"]), retry=cast(bool, data.get("retry", False)))
                            result = {"ok": True}
                        else:
                            raise PermissionError("Worker operation not allowed")
                    self._reply(200, {"result": result})
                except (PermissionError, ValueError, KeyError, RuntimeError, BufferError, TypeError, OSError) as error:
                    self._reply(403 if isinstance(error, PermissionError) else 409,
                                {"error": type(error).__name__})

            def do_PUT(self) -> None:
                try:
                    self._identity()
                    if not self.path.startswith("/v1/artifacts/"):
                        raise ValueError("Unknown endpoint")
                    digest = digest_name(self.path[len("/v1/artifacts/"):])
                    artifacts.put(self.rfile, size=self._size(artifacts.max_artifact_bytes), digest=digest)
                    self._reply(200, {"sha256": digest})
                except (PermissionError, ValueError, BufferError, OSError) as error:
                    self._reply(403 if isinstance(error, PermissionError) else 409, {"error": type(error).__name__})

            def do_GET(self) -> None:
                try:
                    self._identity()
                    if not self.path.startswith("/v1/artifacts/"):
                        raise ValueError("Unknown endpoint")
                    digest = digest_name(self.path[len("/v1/artifacts/"):])
                    with artifacts.open(digest) as source:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Length", str(os.fstat(source.fileno()).st_size))
                        self.end_headers()
                        while chunk := source.read(65536):
                            self.wfile.write(chunk)
                except (PermissionError, ValueError, OSError) as error:
                    self._reply(403 if isinstance(error, PermissionError) else 409, {"error": type(error).__name__})

        self._server = _BoundedServer(("127.0.0.1", port), Handler, max_connections)
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def url(self) -> str:
        return "http://127.0.0.1:" + str(self._server.server_port)

    def start(self) -> None:
        if self._thread is not None or self._closed:
            raise RuntimeError("Server already started or closed")
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._server.shutdown()
            self._thread.join()
        self._server.server_close()
        self._closed = True


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str,
                         headers: HTTPMessage, newurl: str) -> None:
        raise ValueError("Coordinator redirects are not followed")


class RemoteClient:
    def __init__(self, url: str, key: str, *, allow_loopback_http: bool = False,
                 timeout: float = 20., tls_context: ssl.SSLContext | None = None,
                 max_artifact_bytes: int = 64 * 1024**2) -> None:
        parts = urlsplit(url)
        if (parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/")
                or not parts.hostname or parts.scheme not in ("https", "http")):
            raise ValueError("Explicit coordinator origin required")
        if parts.scheme == "http" and not (allow_loopback_http and parts.hostname in ("127.0.0.1", "::1")):
            raise ValueError("Non-loopback worker transport requires verified HTTPS")
        if tls_context is not None and (tls_context.verify_mode != ssl.CERT_REQUIRED or not tls_context.check_hostname):
            raise ValueError("TLS certificate and hostname verification must remain enabled")
        if type(max_artifact_bytes) is not int or max_artifact_bytes < 1 or not 0 < timeout <= 3600:
            raise ValueError("Positive bounded transfer settings required")
        self._url, self._key, self._timeout = url.rstrip("/"), key, timeout
        self.max_artifact_bytes = max_artifact_bytes
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect(),
                                                   urllib.request.HTTPSHandler(context=tls_context))

    def _request(self, path: str, *, method: str, data: bytes | BinaryIO | None = None, size: int | None = None) -> urllib.request.Request:
        request = urllib.request.Request(self._url + path, data=data, method=method)
        request.add_header("Authorization", "Bearer " + self._key)
        if size is not None:
            request.add_header("Content-Length", str(size))
        return request

    def call(self, operation: str, **values: object) -> object:
        body = canonical({"op": operation, **values}).encode()
        request = self._request("/v1/control", method="POST", data=body)
        request.add_header("Content-Type", "application/json")
        with self._opener.open(request, timeout=self._timeout) as response:
            raw = response.read(131073)
        if len(raw) > 131072:
            raise ValueError("Coordinator response exceeds bound")
        return object_json(raw.decode())["result"]

    def upload(self, source: BinaryIO, *, size: int, digest: str) -> None:
        digest_name(digest)
        if type(size) is not int or not 0 <= size <= self.max_artifact_bytes:
            raise ValueError("Artifact exceeds client budget")
        request = self._request("/v1/artifacts/" + digest, method="PUT", data=source, size=size)
        with self._opener.open(request, timeout=self._timeout) as response:
            object_json(response.read(4096).decode())

    def upload_file(self, path: str | Path) -> str:
        """Stream an explicit local artifact; server verifies the same content digest."""
        with Path(path).open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            if size > self.max_artifact_bytes:
                raise ValueError("Artifact exceeds client budget")
            digest = hashlib.file_digest(source, "sha256").hexdigest()
            source.seek(0)
            self.upload(source, size=size, digest=digest)
        return digest

    def download(self, digest: str, destination: str | Path) -> None:
        """Write a new caller-selected path; no replacement and no server filenames."""
        digest_name(digest)
        path = Path(destination)
        request = self._request("/v1/artifacts/" + digest, method="GET")
        with self._opener.open(request, timeout=self._timeout) as response:
            size = int(response.headers.get("Content-Length", "-1"))
            if not 0 <= size <= self.max_artifact_bytes:
                raise ValueError("Artifact exceeds download budget")
            with path.open("xb") as output:
                checksum, total = hashlib.sha256(), 0
                while chunk := response.read(min(65536, size + 1 - total)):
                    total += len(chunk)
                    if total > size:
                        raise ValueError("Artifact length changed")
                    checksum.update(chunk)
                    output.write(chunk)
                if total != size or checksum.hexdigest() != digest:
                    raise ValueError("Downloaded artifact identity mismatch; incomplete path retained")

    def claim(self) -> Lease | None:
        data = self.call("claim")
        return _lease(data) if data is not None else None

    def submit(self, job: Job) -> RemoteJob:
        """Idempotent submission; retry this same Job after an uncertain network result."""
        self.call("submit", job=asdict(job))
        return RemoteJob(self, job.job_id)

    def job(self, job_id: str) -> RemoteJob:
        """Reattach a handle without a network request or resubmission."""
        return RemoteJob(self, identifier(job_id))


class RemoteJob:
    """Typed SDK handle over the transport; polling is always explicitly requested."""
    def __init__(self, client: RemoteClient, job_id: str) -> None:
        self._client, self.job_id = client, identifier(job_id)

    def status(self) -> dict[str, object]:
        value = self._client.call("status", job_id=self.job_id)
        if not isinstance(value, dict) or value.get("job_id") != self.job_id:
            raise ValueError("Coordinator returned a mismatched job")
        return cast(dict[str, object], value)

    def cancel(self) -> None:
        self._client.call("cancel", job_id=self.job_id)

    def result(self, *, timeout: float, poll_seconds: float = .1) -> dict[str, object]:
        if not math.isfinite(timeout) or timeout < 0 or not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("Explicit finite timeout and positive polling interval required")
        deadline = time.monotonic() + timeout
        while True:
            status = self.status()
            if status["state"] == "succeeded":
                result = status["result"]
                if not isinstance(result, dict):
                    raise ValueError("Completed job has no result metadata")
                return cast(dict[str, object], result)
            if status["state"] in ("failed", "cancelled"):
                raise RuntimeError("Remote job " + str(status["state"]) + ": " + str(status["error"]))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Remote job is still pending; its handle remains usable")
            time.sleep(min(poll_seconds, remaining))

    def download_artifacts(self, directory: str | Path, *, timeout: float = 0.) -> dict[str, Path]:
        result = self.result(timeout=timeout)
        references = result.get("artifacts")
        if not isinstance(references, dict):
            raise ValueError("Result has no artifact manifest")
        names = [identifier(name) for name in references]
        reserved = {"con", "prn", "aux", "nul", *("com" + str(n) for n in range(1, 10)),
                    *("lpt" + str(n) for n in range(1, 10))}
        if (len({name.casefold() for name in names}) != len(names)
                or any(name.endswith(".") or name.split(".")[0].casefold() in reserved for name in names)):
            raise ValueError("Artifact names must be portable distinct filenames")
        for digest in references.values():
            digest_name(digest)
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        downloaded: dict[str, Path] = {}
        for name, digest in references.items():
            filename = identifier(name)
            path = root / filename
            self._client.download(digest_name(digest), path)
            downloaded[name] = path
        return downloaded
