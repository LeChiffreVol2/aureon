"""Bounded stage execution with tenant fairness and explicit object ownership.

Memory credits are conservative caller/adapter declarations, not allocator limits.
Each request reserves its maximum live bytes across stages until result consumption.
This avoids deadlock when several requests expand while all input credits are held.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
import math
import threading
import time
from types import TracebackType
from typing import Generic, TypeVar, Self

T = TypeVar("T")


@dataclass(frozen=True)
class StageContext:
    request_id: str
    tenant: str
    seed: int
    cancelled: threading.Event
    deadline: float | None

    def check(self) -> None:
        if self.cancelled.is_set():
            raise CancelledError()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Request deadline elapsed")


@dataclass(frozen=True)
class Stage(Generic[T]):
    name: str
    run: Callable[[T, StageContext], T]
    workers: int = 1
    finalize: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not self.name or type(self.workers) is not int or not 1 <= self.workers <= 256:
            raise ValueError("Named stage requires 1..256 explicitly safe workers")


class PipelineHandle(Generic[T]):
    def __init__(self, future: Future[T], cancel: threading.Event,
                 release: Callable[[], None]) -> None:
        self._future: Future[T] | None = future
        self._cancel = cancel
        self._release: Callable[[], None] | None = release
        self._lock = threading.Lock()

    def _release_once(self) -> None:
        with self._lock:
            release, self._release = self._release, None
        if release is not None:
            release()

    def cancel(self) -> None:
        """Cooperative; native work is never killed. A cancelled result is discarded."""
        self._cancel.set()

    def result(self, timeout: float | None = None) -> T:
        with self._lock:
            future = self._future
        if future is None:
            raise RuntimeError("Result was discarded")
        try:
            return future.result(timeout)
        finally:
            if future.done():
                self._release_once()

    def discard(self) -> None:
        """Release an unneeded completed or future result without retaining payloads."""
        self.cancel()
        with self._lock:
            future, self._future = self._future, None
        if future is not None:
            future.add_done_callback(lambda _: self._release_once())

    def done(self) -> bool:
        with self._lock:
            return self._future is None or self._future.done()


@dataclass
class _Item(Generic[T]):
    context: StageContext
    value: T
    future: Future[T]
    reservation: int


class Pipeline(Generic[T]):
    """Execute trusted adapter stages; no model imports, HTTP, or device selection.

    Stage callbacks must honor cancellation between native operations and account
    for all live request buffers. Shared resident weights/graph pools need separate
    device admission. Multiple workers are an explicit thread-safety declaration.
    Caller transfers exclusive ownership of each submitted value until completion.
    """
    def __init__(self, stages: Sequence[Stage[T]], *, max_requests: int,
                 max_live_bytes: int) -> None:
        if not stages or len({s.name for s in stages}) != len(stages):
            raise ValueError("Declare distinct stages")
        if any(type(n) is not int or n < 1 for n in (max_requests, max_live_bytes)):
            raise ValueError("Positive request and memory limits required")
        self._stages = tuple(stages)
        self._max_requests, self._max_bytes = max_requests, max_live_bytes
        self._cv = threading.Condition()
        self._queues: list[dict[str, deque[_Item[T]]]] = [{} for _ in stages]
        self._turns: list[deque[str]] = [deque() for _ in stages]
        self._live: dict[str, _Item[T]] = {}
        self._bytes = 0
        self._closed = False
        self._active = 0
        self._cleanup_errors: list[BaseException] = []
        self._timings: dict[str, list[float]] = {s.name: [0., 0.] for s in stages}
        self._threads = [threading.Thread(target=self._run, args=(i,), daemon=True,
                        name="aureon-stage-" + stage.name)
                        for i, stage in enumerate(stages) for _ in range(stage.workers)]
        try:
            for thread in self._threads:
                thread.start()
        except BaseException:
            with self._cv:
                self._closed = True
                self._cv.notify_all()
            for thread in self._threads:
                if thread.ident is not None:
                    thread.join()
            raise

    def submit(self, request_id: str, value: T, *, tenant: str, seed: int,
               reservation_bytes: int, deadline: float | None = None) -> PipelineHandle[T]:
        if not request_id or not tenant or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Explicit request, tenant and nonnegative seed required")
        if type(reservation_bytes) is not int or not 1 <= reservation_bytes <= self._max_bytes:
            raise ValueError("Request does not fit live-memory budget")
        if deadline is not None and (not math.isfinite(deadline) or deadline <= time.monotonic()):
            raise ValueError("Deadline must be a future monotonic instant")
        with self._cv:
            if self._closed:
                raise RuntimeError("Pipeline closed")
            if request_id in self._live:
                raise ValueError("Duplicate live request id")
            if len(self._live) >= self._max_requests or self._bytes + reservation_bytes > self._max_bytes:
                raise BufferError("Pipeline backpressure: consume results or retry later")
            context = StageContext(request_id, tenant, seed, threading.Event(), deadline)
            item = _Item(context, value, Future[T](), reservation_bytes)
            self._live[request_id] = item
            self._bytes += reservation_bytes
            self._enqueue(0, item)
            return PipelineHandle(item.future, context.cancelled, lambda: self._release(item))

    def _enqueue(self, index: int, item: _Item[T]) -> None:
        tenant = item.context.tenant
        if tenant not in self._queues[index]:
            self._queues[index][tenant] = deque()
            self._turns[index].append(tenant)
        self._queues[index][tenant].append(item)
        self._cv.notify_all()

    def _release(self, item: _Item[T]) -> None:
        with self._cv:
            if self._live.get(item.context.request_id) is item and item.future.done():
                del self._live[item.context.request_id]
                self._bytes -= item.reservation
                self._cv.notify_all()

    def _run(self, index: int) -> None:
        try:
            self._run_stages(index)
        finally:
            finalize = self._stages[index].finalize
            if finalize is not None:
                try:
                    finalize()
                except BaseException as error:
                    with self._cv:
                        self._cleanup_errors.append(error)

    def _run_stages(self, index: int) -> None:
        while True:
            with self._cv:
                while not self._turns[index]:
                    if self._closed and self._active == 0 and not any(self._turns):
                        return
                    self._cv.wait()
                tenant = self._turns[index].popleft()
                queue = self._queues[index][tenant]
                item = queue.popleft()
                if queue:
                    self._turns[index].append(tenant)
                else:
                    del self._queues[index][tenant]
                self._active += 1
            started = time.monotonic()
            try:
                self._execute(index, item)
            except BaseException as error:
                item.future.set_exception(error)
            finally:
                with self._cv:
                    self._active -= 1
                    timing = self._timings[self._stages[index].name]
                    timing[0] += 1
                    timing[1] += time.monotonic() - started
                    self._cv.notify_all()
            # An idle worker must not retain its previous request's tensors.
            del item

    def _execute(self, index: int, item: _Item[T]) -> None:
        item.context.check()
        value = self._stages[index].run(item.value, item.context)
        item.context.check()
        item.value = value
        if index == len(self._stages) - 1:
            item.future.set_result(value)
        else:
            with self._cv:
                self._enqueue(index + 1, item)

    def metrics(self) -> dict[str, object]:
        with self._cv:
            return {"live_requests": len(self._live), "reserved_bytes": self._bytes,
                    "active_stages": self._active, "closed": self._closed,
                    "stage_seconds": {k: v[1] for k, v in self._timings.items()},
                    "stage_calls": {k: int(v[0]) for k, v in self._timings.items()},
                    "accounting": "declared_peak_request_bytes_not_allocator_cap"}

    def close(self, *, cancel: bool = False) -> None:
        if threading.current_thread() in self._threads:
            raise RuntimeError("Close the pipeline from its owner, not a stage callback")
        with self._cv:
            self._closed = True
            if cancel:
                for item in self._live.values():
                    item.context.cancelled.set()
            self._cv.notify_all()
        for thread in self._threads:
            thread.join()
        with self._cv:
            self._live.clear()
            self._bytes = 0
            if self._cleanup_errors:
                raise BaseExceptionGroup("Stage owner cleanup failed", self._cleanup_errors)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        try:
            self.close(cancel=exc is not None)
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note("Stage owner cleanup also failed: " + repr(cleanup))
