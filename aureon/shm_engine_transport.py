"""Opt-in eight-tile local shared-memory client with worker-bound completion."""
import http.client
import json
import math
import threading
from urllib.parse import urlsplit
import uuid

import numpy as np

from .shm_files import SharedSlots, require_linux, decode_completion, MODEL_IDS, INPUT_SHAPE, MAX_CONTROL_BYTES, SCHEMA

PATH = '/aureon/pooling-shm-engine'


class SharedMemoryEnginePoolingPredictor:
    """One owned mmap snapshot, one in-flight request, independent returned logits.

    A failure poisons this client permanently. Close unlinks without truncation;
    a worker that already attached retains valid mappings, while a late worker
    attach fails safely. This client never retries, recreates or reuses a failed
    slot pair, and never exposes mutable mapped storage to its caller.
    """
    transport = 'shared_memory_engine'
    request_path = PATH
    dtype = 'float32'
    max_in_flight = 1
    collect_diagnostics = False
    reuse_connections = True
    input_format = response_format = 'shared_fp32'
    json_backend = 'stdlib_control_only'
    base64_backend = 'not_used'

    @staticmethod
    def _all_finite(values):
        return bool(np.isfinite(values).all())

    def _write_input(self, values):
        self._slots.write('input', values)
        view = self._slots._array('input')
        try:
            return self._all_finite(view)
        finally:
            del view

    def _read_output(self):
        result = self._slots.snapshot('output', 8)
        if not self._all_finite(result):
            raise ValueError('Engine shared-memory output contains nonfinite logits')
        return result

    def _validate_destination(self, values, destination):
        if destination is not None:
            raise ValueError('This predictor does not support caller-owned output buffers')

    def _read_output_into(self, destination):
        raise NotImplementedError('Caller-owned output requires an explicit implementation')

    def __init__(self, model_name, *, base_url='http://127.0.0.1:8000', capacity=8, timeout=120):
        require_linux()
        url = urlsplit(base_url)
        if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password
                or url.path not in ('', '/') or url.query or url.fragment):
            raise ValueError('Engine shared memory requires an explicit local http://127.0.0.1:PORT service')
        if model_name not in MODEL_IDS:
            raise ValueError('Supply a KAO-TerraMind A/B/C model identity')
        if type(capacity) is not int or capacity != 8:
            raise ValueError('Engine shared memory requires fixed capacity eight')
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError('Timeout must be finite and in (0,300] seconds')
        self.model_name, self.capacity, self.timeout, self.port = model_name, capacity, timeout, url.port or 80
        self._lock = threading.Lock()
        self._slots = self._connection = None
        self._failure = None
        self._closed = False
        self.completed_calls = self.completed_tiles = 0
        self.last_request_id = None

    def __call__(self, values):
        return self._predict(values, None)

    def _predict(self, values, destination):
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('<f4')
                or values.shape != (8, *INPUT_SHAPE)):
            raise ValueError('Expected exactly eight little-endian float32 [8,12,224,224] tiles')
        with self._lock:
            if self._closed:
                raise RuntimeError('Engine shared-memory predictor is closed')
            if self._failure is not None:
                raise RuntimeError('Engine shared-memory slots are permanently poisoned after a failed request') from self._failure
            # Invalid destinations fail before request admission and do not
            # poison a healthy client. Subclasses must retain exclusive ownership.
            self._validate_destination(values, destination)
            try:
                if self._slots is None:
                    self._slots = SharedSlots.create(self.capacity)
                # Copy once into our private snapshot. Validation reads that
                # snapshot, never a potentially changed caller array.
                finite = self._write_input(values)
                if not finite:
                    raise ValueError('Engine shared-memory input must be finite')
                request = uuid.uuid4().hex
                self.last_request_id = request
                descriptor = self._slots.descriptor(self.model_name, 8, request)
                body = json.dumps(descriptor, separators=(',', ':'), allow_nan=False).encode('ascii')
                if len(body) > MAX_CONTROL_BYTES:
                    raise ValueError('Engine shared-memory descriptor exceeds its control bound')
                if self._connection is None:
                    self._connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.timeout)
                self._connection.request('POST', self.request_path, body=body, headers={'Content-Type': 'application/json'})
                response = self._connection.getresponse()
                raw = response.read(MAX_CONTROL_BYTES + 1)
                if (response.status != 200 or len(raw) > MAX_CONTROL_BYTES
                        or response.getheader('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json'):
                    raise RuntimeError('Engine shared-memory pooling returned an invalid bounded control response; no retry')
                expected = {'schema': SCHEMA, 'request': request, 'client': self._slots.client,
                            'model': self.model_name, 'count': 8, 'status': 'completed'}
                decode_completion(raw, expected)
                result = self._read_output() if destination is None else self._read_output_into(destination)
                if response.will_close:
                    self._connection.close()
                    self._connection = None
                self.completed_calls += 1
                self.completed_tiles += 8
                return result
            except BaseException as error:
                self._failure = error
                if self._connection is not None:
                    self._connection.close()
                    self._connection = None
                raise

    def metadata(self):
        with self._lock:
            return {'schema': 'aureon.shm_engine_client.v1', 'transport': self.transport, 'model_id': self.model_name,
                'capacity': self.capacity, 'required_count': 8,
                'state': 'closed' if self._closed else 'poisoned' if self._failure else 'ready',
                'slots': dict(self._slots.identities) if self._slots else None,
                'completed_calls': self.completed_calls, 'completed_tiles': self.completed_tiles,
                'maximum_control_bytes': MAX_CONTROL_BYTES, 'in_flight_per_client': 1,
                'output_cache': False, 'retry': False, 'dtype': self.dtype,
                'scope': 'Same-UID Linux /dev/shm; native vLLM carries bounded descriptor/ACK tensors only',
                'copies': 'caller -> owned input mmap; worker staging/transfers -> output mmap -> independent client output',
                'owned_client_input_copies_per_call': 1, 'worker_ack_required': True,
                'failure': str(self._failure) if self._failure is not None else None}

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._slots is not None:
                self._slots.close(unlink=True)
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Engine shared-memory predictor is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
