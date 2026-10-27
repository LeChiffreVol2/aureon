"""Control-only local endpoint; the native vLLM worker owns tensor attachments."""
import asyncio
import json
import logging
import os

from .binary_gateway import BinaryPoolingMiddleware, ClientDisconnected, GatewayError, native_pooling
from .shm_files import require_linux, decode_descriptor, MAX_CONTROL_BYTES, SCHEMA
from .shm_gateway import _drain
from .shm_engine_protocol import ACK_SHAPE, encode_descriptor_tensor_values, validate_ack, WorkerReportedError
from .shm_engine_transport import PATH

MAX_ACTIVE_PAIRS = 4
logger = logging.getLogger(__name__)


class SharedMemoryEnginePoolingMiddleware(BinaryPoolingMiddleware):
    """Single API worker admission; no bulk tensor mapping in the API process."""
    request_path = PATH
    encode_control = staticmethod(encode_descriptor_tensor_values)
    def __init__(self, app):
        super().__init__(app)
        self._active = set()

    async def __call__(self, scope, receive, send):
        if scope.get('type') != 'http' or scope.get('path') != self.request_path:
            return await self.app(scope, receive, send)
        inference = watcher = client_key = None
        try:
            require_linux()
            if scope.get('method') != 'POST':
                raise GatewayError(405, 'Use POST for engine shared-memory pooling.')
            state = scope['app'].state
            args = state.args
            if (getattr(args, 'host', None) != '127.0.0.1' or not scope.get('client')
                    or scope['client'][0] not in ('127.0.0.1', '::1')):
                raise GatewayError(403, 'Engine shared memory requires a controlled loopback service.')
            if getattr(args, 'api_key', None) or os.environ.get('VLLM_API_KEY'):
                raise GatewayError(403, 'Engine shared memory cannot bypass API-key authentication.')
            if (getattr(args, 'dtype', None) != 'float32' or getattr(args, 'max_num_seqs', None) != 1
                    or getattr(args, 'api_server_count', 1) not in (None, 1)
                    or any(os.environ.get(key) != '1' for key in
                           ('AUREON_SHM_ENGINE', 'AUREON_SHARED_MEMORY_API', 'AUREON_PACKED_TILES', 'AUREON_CUDA_GRAPH'))):
                raise GatewayError(403, 'Enable the controlled engine/shared-memory packed FP32 graph service with one API worker and max_num_seqs=1.')
            headers = {}
            for key, value in scope.get('headers', []):
                key = key.lower()
                if key in headers:
                    raise GatewayError(400, 'Duplicate engine shared-memory control header.')
                headers[key] = value
            if b'origin' in headers:
                raise GatewayError(403, 'Browser-origin requests are not admitted to local shared memory.')
            if headers.get(b'content-type', b'').split(b';', 1)[0].strip().lower() != b'application/json':
                raise GatewayError(415, 'Expected an application/json control descriptor.')
            encoded_length = headers.get(b'content-length', b'')
            if not encoded_length.isdigit() or not 0 < int(encoded_length) <= MAX_CONTROL_BYTES:
                raise GatewayError(413, 'Control Content-Length must be between 1 and 2048 bytes.')
            body = await asyncio.wait_for(self._read_body(receive, int(encoded_length), MAX_CONTROL_BYTES), 10)
            try:
                descriptor = decode_descriptor(body)
                values = self.encode_control(descriptor)
            except (ValueError, TypeError, UnicodeError, OverflowError) as error:
                raise GatewayError(400, 'Invalid fixed-eight engine shared-memory descriptor.') from error
            handler = getattr(state, 'serving_pooling', None)
            if handler is None or descriptor['model'] != handler.models.model_name():
                raise GatewayError(400, 'Control model must match the served KAO-TerraMind model.')
            key = descriptor['client']
            if key in self._active:
                raise GatewayError(409, 'This engine shared-memory client already has an active request.')
            if len(self._active) >= MAX_ACTIVE_PAIRS:
                raise GatewayError(503, 'Engine shared-memory active-client budget is full.')
            self._active.add(key)
            client_key = key
            request_id = 'aureon-shm-engine-' + descriptor['request']
            inference = asyncio.create_task(native_pooling(state, values, descriptor['model'], request_id,
                output_shape=ACK_SHAPE, return_array=True))
            watcher = asyncio.create_task(self._disconnect(receive))
            done, _ = await asyncio.wait((inference, watcher), timeout=300, return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                raise ClientDisconnected()
            if inference not in done:
                raise GatewayError(504, 'Engine shared-memory inference timed out.')
            output, _ = inference.result()
            try:
                validate_ack(output, descriptor)
            except (ValueError, WorkerReportedError) as error:
                raise GatewayError(502, 'Worker completion failed or did not match this request.') from error
            # Worker ACK means synchronized output storage and attachment release.
            # Release API admission before success bytes permit immediate reuse.
            await _drain(watcher)
            watcher = None
            self._active.discard(client_key)
            client_key = None
            reply = {'schema': SCHEMA, 'request': descriptor['request'], 'client': descriptor['client'],
                     'model': descriptor['model'], 'count': 8, 'status': 'completed'}
            await self._respond(send, 200, json.dumps(reply, separators=(',', ':')).encode(),
                                [(b'content-type', b'application/json')])
        except ClientDisconnected:
            return
        except asyncio.TimeoutError:
            await self._respond(send, 408, b'{"error":"Engine shared-memory control body timed out."}',
                                [(b'content-type', b'application/json')])
        except GatewayError as error:
            await self._respond(send, error.status, json.dumps({'error': str(error)}).encode(),
                                [(b'content-type', b'application/json')])
        except Exception:
            logger.exception('Engine shared-memory pooling failed.')
            await self._respond(send, 502, b'{"error":"Engine shared-memory pooling failed."}',
                                [(b'content-type', b'application/json')])
        finally:
            await _drain(inference)
            await _drain(watcher)
            if client_key is not None:
                self._active.discard(client_key)
