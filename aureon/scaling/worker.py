"""Explicit resident worker loop; heartbeat does not move native model ownership."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import math
import threading
import time
from typing import Protocol

from .broker import Job, Lease, canonical
from .transport import RemoteClient


class WorkerTransport(Protocol):
    def claim(self) -> Lease | None: ...
    def call(self, operation: str, **values: object) -> object: ...


@dataclass(frozen=True)
class WorkContext:
    job: Job
    cancelled: threading.Event
    attempt: int = 1

    def check(self) -> None:
        if self.cancelled.is_set():
            raise RuntimeError("Lease lost or request cancelled; do not deliver this attempt")


class ResidentWorker:
    """Run trusted already-created adapters in the calling thread.

    One worker instance has one active callback. For multiple CUDA contexts, start
    isolated processes and choose visible GPU before constructing resident models.
    Handlers must use the request seed, be retry-safe and return bounded metadata.
    A failed/lost attempt may leave immutable blobs but cannot commit stale output.
    """
    def __init__(self, transport: WorkerTransport, handlers: Mapping[str, Callable[[WorkContext], Mapping[str, object]]],
                 *, heartbeat_seconds: float = 10.) -> None:
        if not handlers or not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
            raise ValueError("Explicit handlers and heartbeat interval required")
        self._transport, self._handlers = transport, dict(handlers)
        self._interval = heartbeat_seconds
        self._gate = threading.Lock()

    def run_once(self) -> bool:
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("Resident worker already running")
        try:
            return self._run_once()
        finally:
            self._gate.release()

    def _run_once(self) -> bool:
        lease = self._transport.claim()
        if lease is None:
            return False
        stopped, cancelled = threading.Event(), threading.Event()
        context = WorkContext(lease.job, cancelled, lease.attempt)
        interval = min(self._interval, max(.05, (lease.expires_at - time.time()) / 3))

        def heartbeat() -> None:
            while not stopped.wait(interval):
                try:
                    self._transport.call("heartbeat", lease=asdict(lease))
                except Exception:
                    cancelled.set()
                    return

        thread = threading.Thread(target=heartbeat, daemon=True, name="aureon-worker-heartbeat")
        thread.start()
        try:
            context.check()
            handler = self._handlers.get(lease.job.route)
            if handler is None:
                raise ValueError("Worker has no handler for its declared route")
            result = dict(handler(context))
            canonical(result)
            context.check()
            self._transport.call("complete", lease=asdict(lease), result=result)
        except Exception as error:
            try:
                self._transport.call("fail", lease=asdict(lease), error=type(error).__name__, retry=False)
            except Exception:
                # Cancellation/expiry/fencing may already have transferred ownership.
                pass
            raise
        finally:
            stopped.set()
            thread.join()
        return True

    def run_until_idle(self, *, max_jobs: int) -> int:
        """Bounded explicit execution, never an automatic persistent background job."""
        if type(max_jobs) is not int or max_jobs < 1:
            raise ValueError("Positive job bound required")
        count = 0
        while count < max_jobs and self.run_once():
            count += 1
        return count


def remote_worker(client: RemoteClient,
                  handlers: Mapping[str, Callable[[WorkContext], Mapping[str, object]]],
                  *, heartbeat_seconds: float = 10.) -> ResidentWorker:
    """Convenience constructor; no model creation, downloads, or execution."""
    return ResidentWorker(client, handlers, heartbeat_seconds=heartbeat_seconds)
