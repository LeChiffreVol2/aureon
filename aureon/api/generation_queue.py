"""Bounded serial dispatch and single-consumer result ownership; stdlib only."""
from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator
from typing import Any, TYPE_CHECKING, cast
from ..terramind_native._types import Array
from .contracts import NativeOptions, ExecuteGeneration
if TYPE_CHECKING:
    from .generation_engine import GenerationEngine
    from .memory_input import GenerationInput

from collections import deque
from concurrent.futures import CancelledError, TimeoutError
import threading
from time import monotonic
from types import TracebackType
from typing import Literal

from .contracts import (
    INPUT_BYTES as INPUT_BYTES, OUTPUT_BYTES as OUTPUT_BYTES, GenerationResult, HandleState, QueueMetadata,
    positive, validate_timeout,
)
from .lifecycle import close_preserving_error
from ..terramind_native.session import SessionBusy, SessionClosed, SessionFailed


def deadline(timeout: float | None) -> float | None:
    duration = validate_timeout(timeout)
    return None if duration is None else monotonic() + duration


class GenerationHandle:
    """One admitted result. result() transfers ownership once; discard releases it.

    Cancellation applies before execution only. There are no callbacks on the
    model owner thread. Dropping a handle is not consumption: use result/discard.
    """
    _input: GenerationInput | None
    _options: NativeOptions | None
    _result: GenerationResult | None
    _error: tuple[type[BaseException], str] | None

    def __init__(self, owner: GenerationQueue) -> None:
        self._owner = owner
        self._state: HandleState = "reserved"
        self._input = self._options = self._result = self._error = None
        self._discard = False
        self._submitted = monotonic()

    @property
    def state(self) -> HandleState:
        with self._owner._condition:
            return self._state

    def done(self) -> bool:
        return self.state in ("completed", "failed", "cancelled", "consumed", "discarded")

    def running(self) -> bool:
        return self.state == "running"

    def cancel(self) -> bool:
        with self._owner._condition:
            return self._owner._cancel(self)

    def discard(self) -> None:
        """Drop owned output, or finish and drop a running result; never preempt GPU."""
        with self._owner._condition:
            if self._owner._cancel(self):
                return
            if self._state == "running":
                self._discard = True
            elif self._state in ("completed", "failed"):
                self._result = self._error = None
                self._state = "discarded"
                self._owner._release_result(self)

    def result(self, timeout: float | None = None) -> GenerationResult:
        end = deadline(timeout)
        with self._owner._condition:
            while self._state in ("reserved", "queued", "running"):
                remaining = None if end is None else end - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Generation result is not ready; request remains admitted")
                self._owner._condition.wait(remaining)
            if self._state == "cancelled":
                raise CancelledError("Request cancelled before execution")
            if self._state in ("consumed", "discarded"):
                raise RuntimeError("Result ownership has already been transferred or discarded")
            result, error = self._result, self._error
            self._result = self._error = None
            self._state = "consumed"
            self._owner._release_result(self)
        if error is not None:
            error_type, message = error
            raise error_type(message)
        return cast(GenerationResult, result)


class GenerationQueue:
    """Internal reusable credits/dispatcher. Calls only the existing serial facade."""
    _queue: deque[GenerationHandle]
    _handles: set[GenerationHandle]
    _active: GenerationHandle | tuple[GenerationHandle, ...] | None
    _thread: threading.Thread | None

    def __init__(self, execute: ExecuteGeneration, terminal: Callable[[], bool], *, max_pending: int, max_results: int,
                 max_input_bytes: int, max_output_bytes: int) -> None:
        self._execute, self._terminal = execute, terminal
        self.max_pending = positive(max_pending, "max_pending")
        self.max_results = positive(max_results, "max_results")
        self.max_input_bytes = positive(max_input_bytes, "max_input_bytes")
        self.max_output_bytes = positive(max_output_bytes, "max_output_bytes")
        if max_input_bytes < INPUT_BYTES or max_output_bytes < OUTPUT_BYTES:
            raise ValueError("Byte credits must admit at least one fixed-profile input and result")
        self._condition = threading.Condition()
        self._queue = deque()
        self._handles = set()
        self._pending = self._results = 0
        self._active = None
        self._accepting = True
        self._failed = False
        self._thread = None
        self._high_pending = self._high_results = 0

    def reserve(self) -> GenerationHandle:
        with self._condition:
            if not self._accepting or self._terminal():
                raise SessionClosed("GenerationEngine queue is closed or failed")
            if (self._pending >= self.max_pending or self._results >= self.max_results
                    or (self._pending + 1) * INPUT_BYTES > self.max_input_bytes
                    or (self._results + 1) * OUTPUT_BYTES > self.max_output_bytes):
                raise SessionBusy("Generation credits exhausted; consume/discard results before submitting")
            handle = GenerationHandle(self)
            self._handles.add(handle)
            self._pending += 1
            self._results += 1
            self._high_pending = max(self._high_pending, self._pending)
            self._high_results = max(self._high_results, self._results)
            return handle

    def commit(self, handle: GenerationHandle, input_data: GenerationInput, options: NativeOptions) -> GenerationHandle:
        with self._condition:
            if handle._owner is not self or handle._state != "reserved" or not self._accepting:
                self._cancel(handle)
                raise SessionClosed("Reservation was cancelled during preparation")
            handle._input, handle._options = input_data, options
            handle._state = "queued"
            self._queue.append(handle)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="aureon-generation-dispatch", daemon=True)
                try:
                    self._thread.start()
                except BaseException:
                    self._thread = None
                    self._cancel(handle)
                    raise
            self._condition.notify_all()
        return handle

    def _release_result(self, handle: GenerationHandle) -> None:
        if handle in self._handles:
            self._handles.remove(handle)
            self._results -= 1
            self._condition.notify_all()

    def _cancel(self, handle: GenerationHandle) -> bool:
        if handle._owner is not self or handle._state not in ("reserved", "queued"):
            return False
        if handle._state == "queued":
            self._queue.remove(handle)
        handle._input = handle._options = None
        handle._state = "cancelled"
        self._pending -= 1
        self._release_result(handle)
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and self._accepting:
                    self._condition.wait()
                if not self._queue:
                    return
                handle = self._queue.popleft()
                handle._state = "running"
                self._active = handle
                data, options = handle._input, handle._options
                handle._input = handle._options = None
            result = error = None
            fatal = False
            try:
                result = self._execute(cast("GenerationInput", data), **cast(NativeOptions, options))
                # Bound array payloads; JSON receipts/runtime workspace are separate.
                size = sum(cast("dict[str, Array]", result)[key].nbytes for key in ("array", "valid_mask"))
                if size > OUTPUT_BYTES:
                    raise SessionFailed("Runtime output exceeds reserved fixed-profile payload credit")
            except BaseException as exc:
                fatal = self._terminal() or type(exc) not in (ValueError, TypeError, FileExistsError, SessionBusy)
                kind = type(exc) if type(exc) in (ValueError, TypeError, FileExistsError, SessionBusy) and not fatal else SessionFailed
                # Never retain exception tracebacks holding full inputs/runtime state.
                error = (kind, str(exc))
                result = None
            finally:
                data = options = None
            with self._condition:
                self._active = None
                self._pending -= 1
                if handle._discard:
                    handle._state = "discarded"
                    self._release_result(handle)
                else:
                    handle._state = "failed" if error else "completed"
                    handle._result, handle._error = result, error
                result = error = None
                if fatal:
                    self._failed = True
                    self._accepting = False
                    for waiting in list(self._handles):
                        if waiting._state in ("reserved", "queued"):
                            if waiting._state == "queued":
                                self._queue.remove(waiting)
                            waiting._input = waiting._options = None
                            waiting._state = "failed"
                            waiting._error = (SessionFailed, "Resident execution failed; queued request was not executed")
                            self._pending -= 1
                self._condition.notify_all()

    def shutdown(self, *, cancel_pending: bool = False, timeout: float | None = None) -> None:
        if type(cancel_pending) is not bool:
            raise TypeError("cancel_pending must be bool")
        end = deadline(timeout)
        with self._condition:
            self._accepting = False
            for handle in list(self._handles):
                # Preparation belongs to the caller and cannot hold close hostage.
                if handle._state == "reserved" or (cancel_pending and handle._state == "queued"):
                    self._cancel(handle)
            self._condition.notify_all()
            worker = self._thread
        if worker is not None:
            if worker is threading.current_thread():
                raise SessionBusy("Cannot close the dispatcher from its execution thread")
            worker.join(None if end is None else max(0, end - monotonic()))
            if worker.is_alive():
                raise TimeoutError("Close timed out; admission is stopped, active work is not interrupted")

    def metadata(self) -> QueueMetadata:
        with self._condition:
            return {"accepting": self._accepting, "failed": self._failed,
                    "pending": self._pending, "queued": len(self._queue), "running": self._active is not None,
                    "reserved_results": self._results, "input_bytes": self._pending * INPUT_BYTES,
                    "output_bytes_reserved": self._results * OUTPUT_BYTES,
                    "max_pending": self.max_pending, "max_results": self.max_results,
                    "max_input_bytes": self.max_input_bytes, "max_output_bytes": self.max_output_bytes,
                    "high_water_pending": self._high_pending, "high_water_results": self._high_results,
                    "oldest_queued_seconds": max(0, monotonic()-self._queue[0]._submitted) if self._queue else 0,
                    "dispatcher_alive": self._thread is not None and self._thread.is_alive(),
                    "budget_scope": "engine-held array payloads; excludes model/workspace/JSON and caller-owned results"}


class GenerationStream:
    """Lazy ordered stream. Explicit close/context exit retires only its own work."""
    _iterator: Iterator[Any] | None
    _handles: deque[GenerationHandle]
    _queue: GenerationQueue | None
    _iterable: Iterable[Any] | None

    def __init__(self, engine: GenerationEngine, iterable: Iterable[Any], *, max_in_flight: int, adapter: Callable[[Any], Any] | None, options: dict[str, Any]) -> None:
        self._engine, self._iterable, self._adapter, self._options = engine, iterable, adapter, options
        self._iterator = None
        self._window = positive(max_in_flight, "max_in_flight")
        self._handles = deque()
        self._exhausted = self._closed = False
        self._queue = None
        self._gate = threading.Lock()

    def __iter__(self) -> GenerationStream:
        return self

    def __next__(self) -> GenerationResult:
        if not self._gate.acquire(blocking=False):
            raise SessionBusy("One consumer owns this generation stream")
        try:
            if self._closed: raise StopIteration
            if self._queue is None:
                self._queue = self._engine._submission_queue()
            queue = self._queue
            while len(self._handles) < self._window and not self._exhausted:
                try:
                    handle = queue.reserve()
                except SessionBusy:
                    if self._handles: break
                    raise
                except SessionClosed as error:
                    if self._handles:
                        self._exhausted = True
                        break
                    self._retire(primary_error=error)
                    raise
                try:
                    if self._iterator is None: self._iterator = iter(cast(Iterable[Any], self._iterable))
                    try:
                        value = next(self._iterator)
                    except StopIteration:
                        self._exhausted = True
                        handle.cancel()
                        break
                    if self._adapter is not None: value = self._adapter(value)
                    value, options = self._engine._prepare_submission(value, self._options)
                    queue.commit(handle, value, options)
                    value = None
                    self._handles.append(handle)
                except BaseException as error:
                    handle.discard()
                    self._retire(primary_error=error)
                    raise
            if not self._handles:
                self._retire()
                raise StopIteration
            # Do not remove it before result(): error/interrupt cleanup still owns it.
            handle = self._handles[0]
            try:
                result = handle.result()
            except BaseException as error:
                self._retire(primary_error=error)
                raise
            self._handles.popleft()
            return cast(GenerationResult, result)
        finally:
            self._gate.release()

    def _retire(self, *, primary_error: BaseException | None = None) -> None:
        self._closed = True
        for handle in self._handles: handle.discard()
        self._handles.clear()
        iterator, self._iterator, self._iterable = self._iterator, None, None
        if iterator is not None and hasattr(iterator, "close"):
            close_preserving_error(iterator.close, primary_error, context="Input iterator cleanup failed")

    def close(self) -> None:
        if not self._gate.acquire(blocking=False):
            raise SessionBusy("Cannot close stream while next() is in progress")
        try: self._retire()
        finally: self._gate.release()

    def __enter__(self) -> GenerationStream:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> Literal[False]:
        close_preserving_error(self.close, exc, context="Generation stream cleanup failed")
        return False
