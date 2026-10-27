"""Bounded opt-in host/fault diagnostics; imports never inspect CUDA."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from time import perf_counter
import uuid


def fault_counts():
    try:
        import resource
    except ImportError:
        return None
    process = resource.getrusage(resource.RUSAGE_SELF)
    result = {'process_minor': process.ru_minflt, 'process_major': process.ru_majflt}
    if hasattr(resource, 'RUSAGE_THREAD'):
        thread = resource.getrusage(resource.RUSAGE_THREAD)
        result.update(thread_minor=thread.ru_minflt, thread_major=thread.ru_majflt)
    return result


def delta(before, after):
    if before is None or after is None:
        return None
    return {key: after[key] - value for key, value in before.items() if key in after}


class DiagnosticSink:
    def __init__(self, folder, role, model_id, *, limit=64, clock=perf_counter, faults=fault_counts):
        if role not in ('client', 'worker') or model_id not in {'kao-terramind-a', 'kao-terramind-b', 'kao-terramind-c'}:
            raise ValueError('Diagnostics require an explicit client/worker KAO-TerraMind identity')
        if type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError('Diagnostic request limit must be 1..64')
        self.folder = Path(folder).resolve(strict=True)
        if not self.folder.is_dir():
            raise ValueError('Use an existing dedicated diagnostic directory')
        self.role, self.model_id, self.limit = role, model_id, limit
        self.clock, self.faults = clock, faults
        self._lock, self.started = threading.Lock(), 0

    def begin(self, request=None):
        with self._lock:
            if self.started >= self.limit:
                raise RuntimeError('Explicit diagnostic request budget exhausted; no unrecorded continuation')
            self.started += 1
        return RequestTrace(self, request or uuid.uuid4().hex)


class RequestTrace:
    def __init__(self, sink, request):
        self.sink = sink
        self.request, self.client = request, None
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.start_faults = sink.faults()
        self.start = sink.clock()
        self.intervals, self.extra = [], {}
        self.finished = False

    def bind(self, descriptor):
        for key in ('request', 'client'):
            if not isinstance(descriptor.get(key), str) or not re.fullmatch('[0-9a-f]{32}', descriptor[key]):
                raise ValueError('Diagnostic correlation requires exact request/client UUIDs')
        if descriptor.get('model') != self.sink.model_id:
            raise ValueError('Diagnostic model identity differs')
        self.request, self.client = descriptor['request'], descriptor['client']

    @contextmanager
    def span(self, name):
        before = self.sink.faults()
        start = self.sink.clock()
        error_type = None
        try:
            yield
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            end = self.sink.clock()
            self.intervals.append({'name': name, 'start_offset_seconds': start - self.start,
                'duration_seconds': end - start, 'fault_delta': delta(before, self.sink.faults()),
                'error_type': error_type})

    def finish(self, *, status, error=None):
        if self.finished:
            raise RuntimeError('Diagnostic request already finalized')
        self.finished = True
        duration = self.sink.clock() - self.start
        record = {'schema': 'aureon.phase5_request_diagnostic.v1', 'role': self.sink.role,
            'model_id': self.sink.model_id, 'request': self.request, 'client': self.client,
            'pid': os.getpid(), 'thread_id': threading.get_native_id(), 'started_utc': self.started_utc,
            'status': status, 'error': error, 'host_total_seconds_before_record_write': duration,
            'fault_delta': delta(self.start_faults, self.sink.faults()), 'host_intervals': self.intervals,
            'diagnostic_only': True, 'clean_speed_eligible': False, 'promotion_eligible': False,
            'task_accuracy_evaluated': False, 'request_group_size': 8, 'neural_batch_size': 1,
            'clock': 'process-local perf_counter; do not subtract timestamps across processes',
            'fault_scope': 'getrusage cumulative deltas; process counters include other threads, thread counters cover this host thread',
            'timing_scope': 'instrumented diagnostic request; interval/fault probes and record I/O add overhead; not clean speed evidence',
            'record_io_in_host_total': False, **self.extra}
        paths = [Path(__file__), Path(__file__).with_name('phase5_client.py'), Path(__file__).with_name('phase5_worker.py')]
        record['diagnostic_source_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in paths if path.is_file()}
        path = self.sink.folder / f'phase5-{self.sink.role}-{self.request}.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return record
