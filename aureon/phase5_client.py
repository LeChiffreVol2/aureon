"""Explicit diagnostic-only descriptor client; no clean speed evidence."""
import http.client
import json
import uuid

import numpy as np

from .phase5_trace import DiagnosticSink
from .shm_engine_transport import SharedMemoryEnginePoolingPredictor, PATH
from .shm_files import SharedSlots, decode_completion, INPUT_SHAPE, MAX_CONTROL_BYTES, SCHEMA


class DiagnosticShmEngineClient(SharedMemoryEnginePoolingPredictor):
    collect_diagnostics = True
    diagnostic_only = True
    clean_speed_eligible = False

    def __init__(self, model_name, *, diagnostics_dir, limit=64, **kwargs):
        super().__init__(model_name, **kwargs)
        self.diagnostics = DiagnosticSink(diagnostics_dir, 'client', model_name, limit=limit)

    def __call__(self, values):
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('<f4')
                or values.shape != (8, *INPUT_SHAPE)):
            raise ValueError('Expected exactly eight little-endian FP32 tiles')
        with self._lock:
            if self._closed or self._failure is not None:
                raise RuntimeError('Diagnostic shared-memory client is closed or permanently poisoned')
            trace = self.diagnostics.begin(uuid.uuid4().hex)
            trace.extra['transport_diagnostics'] = True
            trace.extra['copy_bytes'] = {'input': 8 * 12 * 224 * 224 * 4, 'output': 8 * 10 * 224 * 224 * 4}
            try:
                if self._slots is None:
                    with trace.span('client_slot_create'):
                        self._slots = SharedSlots.create(8)
                trace.client = self._slots.client
                with trace.span('client_input_copy'):
                    self._slots.write('input', values)
                with trace.span('client_input_finite'):
                    view = self._slots._array('input')
                    try:
                        finite = bool(np.isfinite(view).all())
                    finally:
                        del view
                    if not finite:
                        raise ValueError('Diagnostic input must be finite')
                with trace.span('client_control_encode'):
                    descriptor = self._slots.descriptor(self.model_name, 8, trace.request)
                    trace.bind(descriptor)
                    body = json.dumps(descriptor, separators=(',', ':'), allow_nan=False).encode('ascii')
                    if len(body) > MAX_CONTROL_BYTES:
                        raise ValueError('Descriptor exceeds control bound')
                with trace.span('client_http_control_round_trip'):
                    if self._connection is None:
                        self._connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.timeout)
                    self._connection.request('POST', PATH, body=body, headers={'Content-Type': 'application/json'})
                    response = self._connection.getresponse()
                    raw = response.read(MAX_CONTROL_BYTES + 1)
                    if (response.status != 200 or len(raw) > MAX_CONTROL_BYTES
                            or response.getheader('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json'):
                        raise RuntimeError('Invalid bounded diagnostic control response; no retry')
                    expected = {'schema': SCHEMA, 'request': trace.request, 'client': self._slots.client,
                                'model': self.model_name, 'count': 8, 'status': 'completed'}
                    decode_completion(raw, expected)
                with trace.span('client_output_copy'):
                    result = self._slots.snapshot('output', 8)
                with trace.span('client_output_finite'):
                    if not np.isfinite(result).all():
                        raise ValueError('Diagnostic output must be finite')
                if response.will_close:
                    self._connection.close()
                    self._connection = None
                self.completed_calls += 1
                self.completed_tiles += 8
                trace.finish(status='completed')
                return result
            except BaseException as error:
                # Poison permanently, but do not retain input arrays in a traceback.
                self._failure = RuntimeError(f'{type(error).__name__}: {error}')
                if self._connection is not None:
                    self._connection.close()
                    self._connection = None
                if not trace.finished:
                    trace.finish(status='failed', error={'type': type(error).__name__, 'message': str(error)[:512]})
                raise

    def metadata(self):
        return {**super().metadata(), 'diagnostic_only': True, 'clean_speed_eligible': False,
                'transport_diagnostics': True, 'diagnostic_limit': self.diagnostics.limit,
                'diagnostic_requests_started': self.diagnostics.started}
