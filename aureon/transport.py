# AUREON review candidate - interoperability provenance notice.
# Wire-format reference: the separately installed vLLM 0.19.0 pooling protocol
# and TerraTorch serving documentation. AUREON implements this bounded client;
# the vLLM server/protocol implementation is not vendored in this file.
# See NOTICE and THIRD-PARTY-NOTICES.md for source and dependency scope.

"""Bounded tensor client for a separately started local TerraTorch/vLLM service.

Protocol: https://torchgeo.org/terratorch/stable/guide/vllm/serving_a_model_tensor/
Wire fields follow vLLM 0.19.0's pooling/base/protocol.py. Raw outputs use an
explicit little-endian float32 encoding independently of model precision.
This client does not start services, download weights, or retry failed requests.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
import base64
import binascii
import http.client
import io
import json
import math
import threading
from time import perf_counter
from urllib.parse import urlsplit

import numpy as np

TILE_SHAPE = (12, 224, 224)
OUTPUT_SHAPE = (10, 224, 224)
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
LOGIT_BYTES = 10 * 224 * 224 * 4
MAX_DIAGNOSTIC_CALLS = 128
MAX_GROUP_SIZE = 8


class VLLMPoolingPredictor:
    """Callable for recipe.segment_tiled; concurrency is bounded, not a speed claim.

The served config must use input key S2L2A and return raw float32 logits.
model_name must equal the separately started service's served model name.
Opt-in connection reuse retains a bounded worker pool; close the predictor or
use its context manager when finished. Failed requests are never retried.
Optional diagnostics record client wall time, including overlapping server and
network waits; they are not CPU attribution or an uninstrumented benchmark.
The binary transport requires the explicitly enabled AUREON server middleware;
it uses little-endian FP32 NPY input and raw logits, bypassing both JSON codecs
and base64 codecs. Native JSON /pooling remains the default.
"""

    def __init__(self, model_name, *, base_url='http://127.0.0.1:8000',
                 dtype='float32', max_in_flight=4, timeout=60, reuse_connections=False,
                 json_backend='stdlib', collect_diagnostics=False, response_format='base64',
                 input_format='torch', base64_backend='stdlib', transport='json'):
        url = urlsplit(base_url)
        if (url.scheme != 'http' or url.hostname != '127.0.0.1'
                or url.username or url.password or url.path not in ('', '/')
                or url.query or url.fragment):
            raise ValueError('Use a local http://127.0.0.1:PORT vLLM service.')
        if not isinstance(model_name, str) or not model_name.strip() or len(model_name) > 256:
            raise ValueError('Supply the exact served model name.')
        if dtype not in ('float16', 'float32'):
            raise ValueError('dtype must be float16 or float32.')
        if type(max_in_flight) is not int or not 1 <= max_in_flight <= 16:
            raise ValueError('max_in_flight must be an integer from 1 to 16.')
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
            raise ValueError('timeout must be in (0, 300] seconds.')
        if type(reuse_connections) is not bool:
            raise ValueError('reuse_connections must be a boolean.')
        if type(collect_diagnostics) is not bool:
            raise ValueError('collect_diagnostics must be a boolean.')
        if json_backend not in ('stdlib', 'orjson'):
            raise ValueError('json_backend must be stdlib or orjson.')
        if response_format not in ('base64', 'bytes_only'):
            raise ValueError('response_format must be base64 or bytes_only.')
        if input_format not in ('torch', 'npy'):
            raise ValueError('input_format must be torch or npy.')
        if base64_backend not in ('stdlib', 'pybase64'):
            raise ValueError('base64_backend must be stdlib or pybase64.')
        if transport not in ('json', 'binary', 'binary_group', 'binary_pack'):
            raise ValueError('transport must be json, binary, binary_group or binary_pack.')
        if transport != 'json' and dtype != 'float32':
            raise ValueError('The AUREON binary transport requires float32 model inputs.')
        if transport == 'binary_group' and max_in_flight > MAX_GROUP_SIZE:
            raise ValueError('Grouped transport supports at most eight concurrent native engine requests.')
        self._base64_encode, self._base64_decode = base64.b64encode, base64.b64decode
        if base64_backend == 'pybase64' and transport == 'json':
            try:
                import pybase64
            except ImportError as exc:
                raise ValueError('base64_backend=pybase64 requires an explicitly installed pybase64 package.') from exc
            self._base64_encode, self._base64_decode = pybase64.b64encode, pybase64.b64decode
        self._json_dumps, self._json_loads = json.dumps, json.loads
        if json_backend == 'orjson' and transport == 'json':
            try:
                import orjson
            except ImportError as exc:
                raise ValueError('json_backend=orjson requires an explicitly installed orjson package.') from exc
            self._json_dumps, self._json_loads = orjson.dumps, orjson.loads
        self.port = url.port or 80
        self.model_name = model_name
        self.dtype = dtype
        self.max_in_flight = 1 if transport == 'binary_pack' else max_in_flight
        self.timeout = timeout
        self.reuse_connections = reuse_connections
        self.transport = transport
        self.json_backend = json_backend if transport == 'json' else 'not_used'
        self.collect_diagnostics = collect_diagnostics
        self.response_format = response_format if transport == 'json' else 'bytes_only'
        self.input_format = input_format if transport == 'json' else 'npy'
        self.base64_backend = base64_backend if transport == 'json' else 'not_used'
        self._executor = None
        self._local = threading.local()
        self._connections = set()
        self._state = threading.Condition()
        self._active_calls = 0
        self._closing = False
        self._closed = False
        self._diagnostics = deque(maxlen=MAX_DIAGNOSTIC_CALLS)
        self._diagnostic_dropped_calls = 0
        self._diagnostic_next_call = 0

    def drain_diagnostics(self):
        """Atomically take completed calls; ongoing calls appear on a later drain.

Records are bounded to 128 calls. Concurrent tile intervals overlap, so their
sum is work across requests, not elapsed batch time or exclusive CPU time.
No payloads, input tensors, or logits are retained in diagnostic records.
"""
        with self._state:
            calls = list(self._diagnostics)
            dropped = self._diagnostic_dropped_calls
            self._diagnostics.clear()
            self._diagnostic_dropped_calls = 0
        return {'schema': 'aureon.transport_diagnostics.v1',
                'scope': 'instrumented_client_wall_intervals',
                'json_backend': self.json_backend, 'response_format': self.response_format,
                'input_format': self.input_format, 'base64_backend': self.base64_backend,
                'transport': self.transport,
                'collect_diagnostics': self.collect_diagnostics,
                'calls': calls, 'dropped_call_count': dropped,
                'limitations': [
                    'HTTP intervals include transport, server queueing and inference; they are not CPU time.',
                    'Concurrent tile intervals overlap and must not be summed as batch elapsed time.',
                    'Diagnostic overhead is included; use disabled diagnostics for clean timing claims.',
                    'Request/response sizes count HTTP body bytes only, excluding headers and TCP overhead.',
                ]}

    def _diagnostic_record(self):
        return getattr(self._local, 'diagnostic_record', None) if self.collect_diagnostics else None

    @staticmethod
    def _stage_end(record, name, started):
        if record is not None:
            ended = perf_counter()
            record['stage_wall_seconds'][name] = ended - started
            return ended
        return None

    def __enter__(self):
        with self._state:
            if self._closing or self._closed:
                raise RuntimeError('The vLLM predictor is closed.')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        """Finish active predictions, then release all worker and socket resources."""
        with self._state:
            if self._closed:
                return
            if self._closing:
                self._state.wait_for(lambda: self._closed)
                return
            self._closing = True
            self._state.notify_all()
            self._state.wait_for(lambda: self._active_calls == 0)
            executor = self._executor
            self._executor = None
        try:
            if executor is not None:
                executor.shutdown(wait=True)
            with self._state:
                connections = list(self._connections)
                self._connections.clear()
            for connection in connections:
                connection.close()
        finally:
            with self._state:
                self._closed = True
                self._state.notify_all()

    def _connection(self):
        if not self.reuse_connections:
            return http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.timeout)
        with self._state:
            if self._closed or (self._closing and not self._active_calls):
                raise RuntimeError('The vLLM predictor is closed.')
            connection = getattr(self._local, 'connection', None)
            if connection is None:
                connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.timeout)
                self._local.connection = connection
                self._connections.add(connection)
            return connection

    def _discard_connection(self, connection=None):
        if connection is None:
            connection = getattr(self._local, 'connection', None)
        if connection is not None:
            with self._state:
                self._connections.discard(connection)
                if getattr(self._local, 'connection', None) is connection:
                    del self._local.connection
            connection.close()

    def _payload(self, tile):
        # Tensor serialization happens only on an explicit prediction call.
        record = self._diagnostic_record()
        started = perf_counter() if record is not None else None
        if self.input_format == 'torch':
            import torch
            started = self._stage_end(record, 'tensor_runtime_import', started)
        with np.errstate(over='ignore', invalid='ignore'):
            values = np.ascontiguousarray(tile, dtype='<f4' if self.transport != 'json' else self.dtype)
        if not np.isfinite(values).all():
            raise ValueError('Normalized input cannot be represented in the selected precision.')
        started = self._stage_end(record, 'input_cast_and_validate', started)
        buffer = io.BytesIO()
        if self.input_format == 'npy':
            # vLLM's ImageEmbeddingMediaIO detects the NPY magic prefix and
            # reconstructs torch.from_numpy(np.load(buffer)); no value transform.
            np.save(buffer, values, allow_pickle=False)
        else:
            torch.save(torch.from_numpy(values), buffer)
        started = self._stage_end(record, 'tensor_serialize', started)
        if self.transport != 'json':
            body = buffer.getvalue()
            self._stage_end(record, 'binary_body_materialize', started)
            if record is not None:
                record['serialized_tensor_bytes'] = len(body)
            return body
        if self.input_format == 'npy' or self.base64_backend == 'pybase64':
            # The encoder consumes the buffer directly without materializing an
            # additional bytes object. Release the view immediately after use.
            with buffer.getbuffer() as serialized:
                encoded = self._base64_encode(serialized).decode('ascii')
        else:
            encoded = self._base64_encode(buffer.getvalue()).decode('ascii')
        self._stage_end(record, 'input_base64_encode', started)
        if record is not None:
            record['serialized_tensor_bytes'] = buffer.tell()
            record['input_base64_characters'] = len(encoded)
        return {
            'model': self.model_name,
            'encoding_format': self.response_format,
            'embed_dtype': 'float32',
            'endianness': 'little',
            'use_activation': False,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_embeds', 'image_embeds': {'S2L2A': encoded}}
            ]}],
        }

    def _post(self, payload, *, group_size=None):
        record = self._diagnostic_record()
        started = perf_counter() if record is not None else None
        if self.transport != 'json':
            if not isinstance(payload, bytes):
                raise ValueError('The binary transport requires an NPY bytes body.')
            body = payload
            path = '/aureon/pooling-binary'
            headers = {'Content-Type': 'application/octet-stream', 'X-Aureon-Model': self.model_name}
            if self.transport in ('binary_group', 'binary_pack'):
                if type(group_size) is not int or not 1 <= group_size <= MAX_GROUP_SIZE:
                    raise ValueError('Grouped transport requires between one and eight tiles.')
                path = '/aureon/pooling-binary-pack' if self.transport == 'binary_pack' else '/aureon/pooling-binary-group'
                headers['X-Aureon-Concurrency'] = str(self.max_in_flight)
            started = self._stage_end(record, 'request_body_prepare', started)
        else:
            body = self._json_dumps(payload)
            path = '/pooling'
            headers = {'Content-Type': 'application/json'}
            started = self._stage_end(record, 'request_json_encode', started)
        if record is not None:
            # stdlib dumps escapes Unicode by default, and orjson returns UTF-8 bytes.
            record['request_body_bytes'] = len(body)
        connection = self._connection()
        keep_connection = False
        try:
            connection.request('POST', path, body, headers)
            started = self._stage_end(record, 'http_send', started)
            response = connection.getresponse()
            started = self._stage_end(record, 'http_wait_headers', started)
            budget = LOGIT_BYTES * (group_size or 1) if self.response_format == 'bytes_only' else MAX_RESPONSE_BYTES
            raw = response.read(budget + 1)
            started = self._stage_end(record, 'http_read_body', started)
            if record is not None:
                record['response_body_bytes'] = len(raw)
                record['http_status'] = response.status
                if self.transport != 'json':
                    request_id = response.getheader('X-Request-Id')
                    if isinstance(request_id, str) and request_id.startswith('aureon-binary-') and len(request_id) <= 128:
                        record['server_external_request_id'] = request_id
                    server_stages = {}
                    stages = ('input-decode', 'group-inference', 'output-join') if self.transport == 'binary_group' else (
                        'input-decode', 'preprocess', 'engine', 'output-encode')
                    for stage in stages:
                        header = 'x-aureon-' + stage + '-ms'
                        value = response.getheader(header)
                        if value is None:
                            continue
                        try:
                            milliseconds = float(value)
                            if not math.isfinite(milliseconds) or milliseconds < 0:
                                raise ValueError('Invalid duration')
                        except (TypeError, ValueError, OverflowError):
                            record.setdefault('invalid_server_stage_headers', []).append(header)
                        else:
                            server_stages[stage] = milliseconds / 1000
                    record['server_stage_wall_seconds'] = server_stages
            if len(raw) > budget:
                raise ValueError('vLLM response exceeds the fixed tile response budget.')
            if response.status != 200:
                raise RuntimeError(f'vLLM returned HTTP {response.status}; no retry was attempted.')
            if self.transport in ('binary_group', 'binary_pack'):
                expected_headers = {'X-Aureon-Model': self.model_name,
                                    'X-Aureon-Tile-Count': str(group_size),
                                    'X-Aureon-Concurrency': str(self.max_in_flight)}
                if any(response.getheader(name) != value for name, value in expected_headers.items()):
                    raise ValueError('Grouped response model, tile count or concurrency differs from the request.')
            if self.response_format == 'bytes_only':
                if response.getheader('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/octet-stream':
                    raise ValueError('Expected application/octet-stream raw logits.')
                result = raw
            else:
                result = self._json_loads(raw)
                self._stage_end(record, 'response_json_decode', started)
            keep_connection = self.reuse_connections and not response.will_close
            return result
        finally:
            if not keep_connection:
                self._discard_connection(connection)

    def _predict_one(self, tile):
        record = self._diagnostic_record()
        try:
            result = self._post(self._payload(tile))
            try:
                started = perf_counter() if record is not None else None
                if self.response_format == 'bytes_only':
                    if not isinstance(result, bytes):
                        raise ValueError('Expected raw float32 bytes.')
                    raw = result
                else:
                    rows = result['data']
                    if not isinstance(rows, list) or len(rows) != 1:
                        raise ValueError('Expected exactly one pooling result.')
                    encoded = rows[0]['data']
                    if not isinstance(encoded, str) or len(encoded) > MAX_RESPONSE_BYTES:
                        raise ValueError('Expected bounded base64 logits.')
                    raw = self._base64_decode(encoded, validate=True)
                    started = self._stage_end(record, 'output_base64_decode', started)
                if len(raw) != LOGIT_BYTES:
                    raise ValueError('Expected float32 logits with shape [1,10,224,224].')
                values = np.frombuffer(raw, dtype='<f4').reshape(OUTPUT_SHAPE)
                if not np.isfinite(values).all():
                    raise ValueError('vLLM returned nonfinite logits.')
                self._stage_end(record, 'logits_view_and_validate', started)
                return values
            except (KeyError, TypeError, binascii.Error) as exc:
                raise ValueError('Invalid vLLM pooling response.') from exc
        except BaseException:
            self._discard_connection()
            raise

    def _predict_diagnostic(self, indexed_tile):
        index, tile, call = indexed_tile
        record = {'tile_index': index, 'stage_wall_seconds': {}, 'success': False}
        call['tiles'][index] = record
        self._local.diagnostic_record = record
        started = perf_counter()
        try:
            result = self._predict_one(tile)
            record['success'] = True
            return result
        except BaseException as exc:
            record['error_type'] = type(exc).__name__
            raise
        finally:
            record['wall_seconds'] = perf_counter() - started
            del self._local.diagnostic_record

    def _predict_group(self, batch, diagnostic):
        record = None
        if diagnostic is not None:
            record = {'request_index': 0, 'tile_indices': list(range(len(batch))),
                      'engine_max_in_flight': self.max_in_flight,
                      'stage_wall_seconds': {}, 'success': False}
            diagnostic['requests'] = [record]
            self._local.diagnostic_record = record
        started = perf_counter() if record is not None else None
        try:
            raw = self._post(self._payload(batch), group_size=len(batch))
            decoding = perf_counter() if record is not None else None
            if not isinstance(raw, bytes) or len(raw) != len(batch) * LOGIT_BYTES:
                raise ValueError('Expected float32 grouped logits [N,10,224,224].')
            values = np.frombuffer(raw, dtype='<f4').reshape((len(batch), *OUTPUT_SHAPE))
            if not np.isfinite(values).all():
                raise ValueError('vLLM returned nonfinite grouped logits.')
            self._stage_end(record, 'logits_view_and_validate', decoding)
            if record is not None:
                record['success'] = True
            return values
        except BaseException as exc:
            self._discard_connection()
            if record is not None:
                record['error_type'] = type(exc).__name__
            raise
        finally:
            if record is not None:
                record['wall_seconds'] = perf_counter() - started
                del self._local.diagnostic_record

    def _predict_batch(self, pool, batch, diagnostic):
        if self.transport in ('binary_group', 'binary_pack'):
            started = perf_counter() if diagnostic is not None else None
            result = pool.submit(self._predict_group, batch, diagnostic).result()
            if diagnostic is not None:
                diagnostic['stage_wall_seconds']['executor_submit_and_wait'] = perf_counter() - started
            return result
        if diagnostic is None:
            return np.stack(list(pool.map(self._predict_one, batch)))
        started = perf_counter()
        futures = [pool.submit(self._predict_diagnostic, (index, tile, diagnostic))
                   for index, tile in enumerate(batch)]
        # Finish all instrumented requests before publishing a completed record,
        # including when one fails while other persistent workers are running.
        wait(futures)
        diagnostic['stage_wall_seconds']['executor_submit_and_wait'] = perf_counter() - started
        values = [future.result() for future in futures]
        started = perf_counter()
        result = np.stack(values)
        diagnostic['stage_wall_seconds']['output_stack'] = perf_counter() - started
        return result

    def __call__(self, batch):
        call_started = perf_counter() if self.collect_diagnostics else None
        multiple = self.transport in ('binary_group', 'binary_pack')
        maximum = MAX_GROUP_SIZE if multiple else 16
        if (not isinstance(batch, np.ndarray) or batch.ndim != 4
                or tuple(batch.shape[1:]) != TILE_SHAPE or not 1 <= batch.shape[0] <= maximum
                or batch.dtype.kind not in 'fiu' or not np.isfinite(batch).all()):
            raise ValueError(f'Expected finite numeric tiles [N,12,224,224], with 1 <= N <= {maximum}.')
        # executor.map preserves tile order even when responses finish out of order.
        # The server schedules concurrent tiles; actual batching is measured there.
        with self._state:
            if self._closing or self._closed:
                raise RuntimeError('The vLLM predictor is closed.')
            if self.reuse_connections and self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1 if multiple else self.max_in_flight)
            pool = self._executor
            self._active_calls += 1
            diagnostic = None
            if self.collect_diagnostics:
                diagnostic = {'call_id': self._diagnostic_next_call, 'batch_size': len(batch),
                              'success': False, 'stage_wall_seconds': {},
                              'tiles': [] if multiple else [None] * len(batch)}
                self._diagnostic_next_call += 1
                diagnostic['stage_wall_seconds']['input_validation_and_dispatch'] = perf_counter() - call_started
        try:
            if pool is not None:
                result = self._predict_batch(pool, batch, diagnostic)
            else:
                with ThreadPoolExecutor(max_workers=1 if multiple else min(self.max_in_flight, len(batch))) as pool:
                    result = self._predict_batch(pool, batch, diagnostic)
            if diagnostic is not None:
                diagnostic['success'] = True
            return result
        except BaseException as exc:
            if diagnostic is not None:
                diagnostic['error_type'] = type(exc).__name__
            raise
        finally:
            with self._state:
                if diagnostic is not None:
                    diagnostic['wall_seconds'] = perf_counter() - call_started
                    if len(self._diagnostics) == self._diagnostics.maxlen:
                        self._diagnostic_dropped_calls += 1
                    self._diagnostics.append(diagnostic)
                self._active_calls -= 1
                self._state.notify_all()
