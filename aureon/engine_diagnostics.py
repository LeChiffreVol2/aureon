"""Opt-in request-correlated vLLM 0.19 host events, never clean timings.

No Torch/vLLM import, file creation or instrumentation occurs on module import.
Set AUREON_ENGINE_DIAGNOSTICS=1 and AUREON_SERVICE_RUN to an existing folder.
The scheduler classes resolve lazily through vLLM's --scheduler-cls option.
No hook synchronizes CUDA, changes inputs/results, retries or caches requests.

Events use perf_counter_ns and include Linux boot identity. Compare timestamps
only from the same clock implementation and boot. Output receipt/dequeue spans
include queueing and serialization, not uniquely isolated IPC transfer time.
Writes are bounded but introduce diagnostic host overhead; exclude these runs
from clean speed claims. Source: vLLM v0.19.0 v1/{engine,core/sched,worker}.
"""

import atexit
from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import threading
from time import get_clock_info, perf_counter_ns


DEFAULT_EVENT_LIMIT = 4096
MAX_EVENT_LIMIT = 16384
MAX_REQUEST_IDS = 16
MAX_REQUEST_ID_LENGTH = 256
MAX_EVENT_BYTES = 8192
_WRITERS = {}
_WRITER_LOCK = threading.RLock()
_WORKER_REQUEST_IDS = ContextVar('aureon_diagnostic_worker_request_ids', default=())


def enabled():
    return os.environ.get('AUREON_ENGINE_DIAGNOSTICS') == '1'


def _request_ids(values):
    values = tuple(values)
    if (len(values) > MAX_REQUEST_IDS or any(not isinstance(value, str) or not value
            or len(value) > MAX_REQUEST_ID_LENGTH for value in values)):
        raise ValueError('Diagnostic request IDs exceed the supported bounded contract')
    return values


def _scheduler_ids(output):
    values = getattr(output, 'num_scheduled_tokens', None)
    if not isinstance(values, Mapping):
        raise ValueError('Unexpected vLLM SchedulerOutput.num_scheduled_tokens')
    return _request_ids(values.keys())


def _outputs_ids(outputs):
    return _request_ids(item.request_id for item in getattr(outputs, 'outputs', ()))


class EventWriter:
    """Exclusive JSONL sink with fixed request/event/line bounds and a cap marker.

    Each event is flushed for crash visibility. This intentionally favors
    recoverable diagnostics over an assertion of negligible overhead.
    """

    def __init__(self, folder, component, *, limit=DEFAULT_EVENT_LIMIT):
        if type(limit) is not int or not 1 <= limit <= MAX_EVENT_LIMIT:
            raise ValueError('Diagnostic event limit must be an integer from 1 to 16384')
        if not isinstance(component, str) or not re.fullmatch('[a-z][a-z0-9_-]{0,31}', component):
            raise ValueError('Invalid diagnostic component name')
        folder = Path(folder).resolve(strict=True)
        if not folder.is_dir():
            raise ValueError('Diagnostic folder must already exist')
        self.path = folder / f'engine-diagnostics-{component}-pid{os.getpid()}.jsonl'
        self.component = component
        self.limit = limit
        self.count = self.dropped = 0
        self.closed = False
        self._lock = threading.RLock()
        self._stream = self.path.open('x', encoding='utf-8')
        boot_path = Path('/proc/sys/kernel/random/boot_id')
        boot_id = boot_path.read_text().strip() if boot_path.is_file() else None
        self._write({'schema': 'aureon.engine_events.v1', 'event': 'diagnostic_started',
                     'component': component, 'pid': os.getpid(), 'monotonic_ns': perf_counter_ns(),
                     'clock': 'perf_counter_ns', 'clock_implementation': get_clock_info('perf_counter').implementation,
                     'linux_boot_id': boot_id, 'event_limit': limit,
                     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'cuda_synchronization_added': False, 'host_io_overhead_included': True,
                     'scope': 'instrumented_host_boundaries; not_gpu_kernel_time_or_clean_benchmark'})

    def _write(self, value):
        line = json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n'
        if len(line.encode('utf-8')) > MAX_EVENT_BYTES:
            raise ValueError('Diagnostic event exceeds its line budget')
        self._stream.write(line)
        self._stream.flush()

    def emit(self, event, request_ids=(), *, timestamp_ns=None, **fields):
        with self._lock:
            if self.closed:
                return False
            if self.count >= self.limit:
                self.dropped += 1
                if self.dropped == 1:
                    self._write({'event': 'diagnostic_limit_reached', 'component': self.component,
                                 'pid': os.getpid(), 'monotonic_ns': perf_counter_ns(), 'event_limit': self.limit})
                return False
            if not isinstance(event, str) or not re.fullmatch('[a-z][a-z0-9_]{0,63}', event):
                raise ValueError('Invalid diagnostic event name')
            request_ids = _request_ids(request_ids)
            stamp = perf_counter_ns() if timestamp_ns is None else timestamp_ns
            if type(stamp) is not int or stamp < 0:
                raise ValueError('Diagnostic timestamps must be nonnegative integer nanoseconds')
            if (len(fields) > 12 or any(not isinstance(key, str) or len(key) > 64
                    or not isinstance(value, (str, int, bool, type(None)))
                    or (isinstance(value, str) and len(value) > MAX_REQUEST_ID_LENGTH)
                    for key, value in fields.items())):
                raise ValueError('Diagnostic fields must be bounded scalar metadata')
            reserved = {'event', 'component', 'pid', 'monotonic_ns', 'index', 'request_ids'}
            if reserved.intersection(fields):
                raise ValueError('Diagnostic fields cannot replace event identity')
            self._write({'event': event, 'component': self.component, 'pid': os.getpid(),
                         'monotonic_ns': stamp, 'index': self.count, 'request_ids': list(request_ids), **fields})
            self.count += 1
            return True

    def close(self):
        with self._lock:
            if not self.closed:
                self._write({'event': 'diagnostic_closed', 'component': self.component,
                             'pid': os.getpid(), 'monotonic_ns': perf_counter_ns(),
                             'events_written': self.count, 'events_dropped': self.dropped})
                self._stream.close()
                self.closed = True


def get_writer(component, folder=None):
    if not enabled():
        return None
    folder = Path(folder or os.environ['AUREON_SERVICE_RUN']).resolve(strict=True)
    key = (os.getpid(), str(folder), component)
    with _WRITER_LOCK:
        if key not in _WRITERS:
            writer = EventWriter(folder, component)
            _WRITERS[key] = writer
            atexit.register(writer.close)
        return _WRITERS[key]


def emit_api_event(event, request_id, **fields):
    """Gateway hook: request_id is external, before vLLM adds its random suffix."""
    writer = get_writer('api')
    if writer is not None:
        return writer.emit(event, (request_id,), request_id_namespace='external', **fields)
    return False


def install_api_engine_diagnostics(engine_client, *, writer=None):
    """Wrap an existing AsyncLLM's core methods without changing their results.

    API send events expose the explicit internal/external ID mapping. The output
    event is after native socket decode and API queue dequeue; it does not claim
    to isolate ZMQ wire time from local async scheduling.
    """
    if writer is None:
        writer = get_writer('api')
    if writer is None:
        return None
    core = engine_client.engine_core
    existing = getattr(core, '_aureon_engine_diagnostics', None)
    if existing is not None:
        return existing
    methods = {name: getattr(core, name) for name in ('add_request_async', '_send_input', 'get_output_async')}
    if not all(callable(method) for method in methods.values()):
        raise ValueError('Unsupported vLLM asynchronous engine client interface')

    @wraps(methods['add_request_async'])
    async def add_request(request, *args, **kwargs):
        ids = _request_ids((request.request_id,))
        external = getattr(request, 'external_req_id', None)
        writer.emit('api_core_add_begin', ids, external_request_id=external)
        try:
            result = await methods['add_request_async'](request, *args, **kwargs)
        except BaseException:
            writer.emit('api_core_add_error', ids)
            raise
        writer.emit('api_core_add_end', ids, external_request_id=external)
        return result

    @wraps(methods['_send_input'])
    def send_input(request_type, request, *args, **kwargs):
        # Utility/abort calls have no singular request_id; preserve them exactly.
        request_id = getattr(request, 'request_id', None)
        if request_id is None:
            return methods['_send_input'](request_type, request, *args, **kwargs)
        ids = _request_ids((request_id,))
        writer.emit('api_input_encode_begin', ids)
        result = methods['_send_input'](request_type, request, *args, **kwargs)
        writer.emit('api_input_encode_enqueued', ids)
        return result  # Preserve the exact native awaitable/future and callbacks.

    @wraps(methods['get_output_async'])
    async def get_output(*args, **kwargs):
        result = await methods['get_output_async'](*args, **kwargs)
        ids = _outputs_ids(result)
        if ids:
            writer.emit('api_output_dequeued', ids)
        return result

    core.add_request_async = add_request
    core._send_input = send_input
    core.get_output_async = get_output
    core._aureon_engine_diagnostics = writer
    writer.emit('api_hooks_installed')
    return writer


def install_worker_diagnostics(worker, folder=None, *, writer=None):
    """Worker host execution, multimodal extraction/H2D enqueue and pool hooks.

    With synchronous pooling, the native _pool includes D2H and synchronization.
    Async pooling can return before GPU completion. No new synchronization is
    added, and return objects (including async wrappers) are preserved exactly.
    """
    if writer is None:
        writer = get_writer('worker', folder)
    if writer is None:
        return None
    existing = getattr(worker, '_aureon_engine_diagnostics', None)
    if existing is not None:
        return existing
    original = worker.execute_model
    runner = worker.model_runner

    @wraps(original)
    def execute(scheduler_output, *args, **kwargs):
        ids = _scheduler_ids(scheduler_output)
        if not ids:
            return original(scheduler_output, *args, **kwargs)
        token = _WORKER_REQUEST_IDS.set(ids)
        writer.emit('worker_execute_begin', ids)
        try:
            result = original(scheduler_output, *args, **kwargs)
            writer.emit('worker_execute_end', ids, result_type=type(result).__name__,
                        async_completion_may_remain=hasattr(result, 'get_output'))
            return result
        except BaseException:
            writer.emit('worker_execute_error', ids)
            raise
        finally:
            _WORKER_REQUEST_IDS.reset(token)

    def wrap_stage(original_method, stage):
        @wraps(original_method)
        def wrapped(*args, **kwargs):
            ids = _WORKER_REQUEST_IDS.get()
            if not ids:
                return original_method(*args, **kwargs)
            writer.emit(stage + '_begin', ids)
            try:
                result = original_method(*args, **kwargs)
            except BaseException:
                writer.emit(stage + '_error', ids)
                raise
            writer.emit(stage + '_end', ids)
            return result
        return wrapped

    # Validate the complete interface before installing any wrapper.
    stages = {'_extract_mm_kwargs': 'worker_multimodal_extract_h2d_host', '_pool': 'worker_pool_output'}
    originals = {name: getattr(runner, name) for name in stages}
    if not callable(original) or not all(callable(value) for value in originals.values()):
        raise ValueError('Unsupported vLLM worker/model-runner interface')
    for name, stage in stages.items():
        setattr(runner, name, wrap_stage(originals[name], stage))
    worker.execute_model = execute
    worker._aureon_engine_diagnostics = writer
    writer.emit('worker_hooks_installed', async_scheduling=bool(getattr(runner, 'use_async_scheduling', False)))
    return writer


def _make_scheduler_class(base, *, writer_factory=None):
    """Wrap the chosen native scheduler; a factory enables dependency-free tests."""
    class InstrumentedScheduler(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._aureon_event_writer = writer_factory() if writer_factory else get_writer('scheduler')

        def add_request(self, request, *args, **kwargs):
            writer = self._aureon_event_writer
            if writer is None:
                return super().add_request(request, *args, **kwargs)
            ids = _request_ids((request.request_id,))
            writer.emit('scheduler_admission_begin', ids)
            result = super().add_request(request, *args, **kwargs)
            writer.emit('scheduler_admitted', ids)
            return result

        def schedule(self, *args, **kwargs):
            writer = self._aureon_event_writer
            if writer is None:
                return super().schedule(*args, **kwargs)
            began = perf_counter_ns()
            result = super().schedule(*args, **kwargs)
            ended = perf_counter_ns()
            ids = _scheduler_ids(result)
            if ids:
                writer.emit('scheduler_dispatched', ids, timestamp_ns=ended,
                            schedule_call_start_ns=began, schedule_call_end_ns=ended)
            return result

        def update_from_output(self, scheduler_output, model_runner_output, *args, **kwargs):
            writer = self._aureon_event_writer
            if writer is None:
                return super().update_from_output(scheduler_output, model_runner_output, *args, **kwargs)
            ids = _scheduler_ids(scheduler_output)
            writer.emit('scheduler_output_update_begin', ids)
            result = super().update_from_output(scheduler_output, model_runner_output, *args, **kwargs)
            writer.emit('scheduler_output_update_end', ids)
            return result
    return InstrumentedScheduler


def __getattr__(name):
    if name not in ('DiagnosticScheduler', 'DiagnosticAsyncScheduler'):
        raise AttributeError(name)
    if metadata.version('vllm') != '0.19.0':
        raise RuntimeError('Scheduler diagnostics are pinned to the inspected vLLM 0.19.0 interface')
    if name == 'DiagnosticAsyncScheduler':
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler as base
    else:
        from vllm.v1.core.sched.scheduler import Scheduler as base
    cls = _make_scheduler_class(base)
    cls.__name__ = cls.__qualname__ = name
    cls.__module__ = __name__
    globals()[name] = cls
    return cls
