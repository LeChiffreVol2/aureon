"""Opt-in Linux-local shared-memory pooling client; no model or server startup."""
import http.client
import json
import math
import threading
from urllib.parse import urlsplit
import uuid

import numpy as np

from .shm_files import SharedSlots, require_linux, decode_completion, MODEL_IDS, INPUT_SHAPE, MAX_CONTROL_BYTES, SCHEMA

PATH = '/aureon/pooling-shm'


class SharedMemoryPoolingPredictor:
    """One in-flight group per private mmap pair; never retries failed requests.

    Any transport/server/response failure permanently poisons this instance.
    close() unlinks its owned names without truncating them; already attached
    server maps stay valid until native cancellation/completion has drained.
    Callers must explicitly construct a new client after a failure.
    """
    transport = 'shared_memory'
    dtype = 'float32'
    max_in_flight = 1
    collect_diagnostics = False
    reuse_connections = True
    input_format = response_format = 'shared_fp32'
    json_backend = 'stdlib_control_only'
    base64_backend = 'not_used'

    def __init__(self, model_name, *, base_url='http://127.0.0.1:8000', capacity=8, timeout=120):
        require_linux()
        url = urlsplit(base_url)
        if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password
                or url.path not in ('', '/') or url.query or url.fragment):
            raise ValueError('Shared-memory pooling requires an explicit local http://127.0.0.1:PORT service')
        if model_name not in MODEL_IDS:
            raise ValueError('Supply a KAO-TerraMind A/B/C model identity')
        if type(capacity) is not int or not 1 <= capacity <= 8:
            raise ValueError('Capacity must be an integer from 1 to 8')
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError('Timeout must be finite and in (0,300] seconds')
        self.model_name, self.capacity, self.timeout, self.port = model_name, capacity, timeout, url.port or 80
        self._lock = threading.Lock()
        self._slots = self._connection = None
        self._failure = None
        self._closed = False
        self.completed_calls = self.completed_tiles = 0

    def __call__(self, values):
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('<f4') or values.ndim != 4
                or values.shape[1:] != INPUT_SHAPE or not 1 <= len(values) <= self.capacity):
            raise ValueError('Expected little-endian float32 [N,12,224,224] within capacity')
        with self._lock:
            if self._closed:
                raise RuntimeError('Shared-memory predictor is closed')
            if self._failure is not None:
                raise RuntimeError('Shared-memory slots are permanently poisoned after a failed request') from self._failure
            # The snapshot is private even if the caller later mutates values.
            snapshot = values.copy(order='C')
            if not np.isfinite(snapshot).all():
                raise ValueError('Shared-memory inputs must be finite')
            try:
                if self._slots is None:
                    self._slots = SharedSlots.create(self.capacity)
                self._slots.write('input', snapshot)
                request = uuid.uuid4().hex
                descriptor = self._slots.descriptor(self.model_name, len(values), request)
                body = json.dumps(descriptor, separators=(',', ':')).encode()
                if self._connection is None:
                    self._connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.timeout)
                self._connection.request('POST', PATH, body=body, headers={'Content-Type': 'application/json'})
                response = self._connection.getresponse()
                raw = response.read(MAX_CONTROL_BYTES + 1)
                if (response.status != 200 or len(raw) > MAX_CONTROL_BYTES
                        or response.getheader('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json'):
                    raise RuntimeError('Shared-memory pooling returned an invalid bounded control response; no retry')
                expected = {'schema': SCHEMA, 'request': request, 'client': self._slots.client,
                            'model': self.model_name, 'count': len(values), 'status': 'completed'}
                decode_completion(raw, expected)
                result = self._slots.snapshot('output', len(values))
                if not np.isfinite(result).all():
                    raise ValueError('Shared-memory output contains nonfinite logits')
                if response.will_close:
                    self._connection.close()
                    self._connection = None
                self.completed_calls += 1
                self.completed_tiles += len(values)
                return result
            except BaseException as error:
                self._failure = error
                if self._connection is not None:
                    self._connection.close()
                    self._connection = None
                raise

    def metadata(self):
        with self._lock:
            return {'schema': 'aureon.shm_client.v1', 'transport': self.transport, 'model_id': self.model_name,
                'capacity': self.capacity, 'state': 'closed' if self._closed else 'poisoned' if self._failure else 'ready',
                'slots': dict(self._slots.identities) if self._slots else None, 'completed_calls': self.completed_calls,
                'completed_tiles': self.completed_tiles, 'maximum_control_bytes': MAX_CONTROL_BYTES,
                'in_flight_per_client': 1, 'output_cache': False, 'retry': False, 'dtype': self.dtype,
                'scope': 'Same-UID Linux /dev/shm; HTTP carries only descriptors and completion identity',
                'copies': 'owned client input snapshot -> mmap; owned server snapshot -> native packed engine; CPU logits -> mmap -> independent client output',
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
            raise RuntimeError('Shared-memory predictor is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
