"""Explicit Linux-only local mmap pooling endpoint; no raw tensor HTTP bodies.

Enable only on the controlled packed FP32 graph service. The Linux UID and
private namespaced files define the local trust boundary; API-key deployments
are rejected because this middleware executes before native authentication.
"""
import asyncio
import json
import logging
import os

import numpy as np

from .binary_gateway import BinaryPoolingMiddleware, ClientDisconnected, GatewayError, native_pooling
from .shm_files import SharedSlots, require_linux, decode_descriptor, MAX_CONTROL_BYTES, OUTPUT_SHAPE, SCHEMA
from .shm_transport import PATH

MAX_ACTIVE_PAIRS = 4
logger = logging.getLogger(__name__)


async def _drain(task):
    """Do not release an attachment while cancellation/engine abort is pending."""
    if task is None:
        return
    if not task.done() and not task.cancelling():
        task.cancel()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if task.done() and not task.cancelled():
        task.exception()  # Retrieve failures; the request already records/reports them.


class SharedMemoryPoolingMiddleware(BinaryPoolingMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._active = set()

    async def __call__(self, scope, receive, send):
        if scope.get('type') != 'http' or scope.get('path') != PATH:
            return await self.app(scope, receive, send)
        slots = inference = watcher = client_key = None
        try:
            require_linux()
            if scope.get('method') != 'POST':
                raise GatewayError(405, 'Use POST for shared-memory pooling.')
            state = scope['app'].state
            args = state.args
            if (getattr(args, 'host', None) != '127.0.0.1' or not scope.get('client')
                    or scope['client'][0] not in ('127.0.0.1', '::1')):
                raise GatewayError(403, 'Shared-memory pooling requires a controlled loopback service.')
            if getattr(args, 'api_key', None) or os.environ.get('VLLM_API_KEY'):
                raise GatewayError(403, 'Shared-memory pooling cannot bypass API-key authentication.')
            if (getattr(args, 'dtype', None) != 'float32' or getattr(args, 'max_num_seqs', None) != 1
                    or os.environ.get('AUREON_SHARED_MEMORY_API') != '1'
                    or os.environ.get('AUREON_PACKED_TILES') != '1'
                    or os.environ.get('AUREON_CUDA_GRAPH') != '1'):
                raise GatewayError(403, 'Enable the controlled shared-memory, packed, FP32 graph service with max_num_seqs=1.')
            headers = {}
            for key, value in scope.get('headers', []):
                key = key.lower()
                if key in headers:
                    raise GatewayError(400, 'Duplicate shared-memory control header.')
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
            except (ValueError, TypeError, UnicodeError) as error:
                raise GatewayError(400, 'Invalid shared-memory descriptor.') from error
            handler = getattr(state, 'serving_pooling', None)
            if handler is None or descriptor['model'] != handler.models.model_name():
                raise GatewayError(400, 'Control model must match the served KAO-TerraMind model.')
            key = descriptor['client']
            if key in self._active:
                raise GatewayError(409, 'This shared-memory client already has an active request.')
            if len(self._active) >= MAX_ACTIVE_PAIRS:
                raise GatewayError(503, 'Shared-memory active-client budget is full.')
            self._active.add(key)
            client_key = key
            try:
                slots = SharedSlots.attach(descriptor)
            except BlockingIOError as error:
                raise GatewayError(409, 'Shared-memory slots are still owned by an active request.') from error
            except (OSError, ValueError) as error:
                raise GatewayError(400, 'Shared-memory file identity or permissions are invalid.') from error
            values = slots.snapshot('input', descriptor['count'])
            if not np.isfinite(values).all():
                raise GatewayError(400, 'Shared-memory input must be finite normalized FP32.')
            request_id = 'aureon-shm-' + descriptor['request']
            inference = asyncio.create_task(native_pooling(state, values, descriptor['model'], request_id,
                output_shape=(len(values), *OUTPUT_SHAPE), return_array=True))
            watcher = asyncio.create_task(self._disconnect(receive))
            done, _ = await asyncio.wait((inference, watcher), timeout=300, return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                raise ClientDisconnected()
            if inference not in done:
                raise GatewayError(504, 'Shared-memory inference timed out.')
            output, _ = inference.result()
            if (not isinstance(output, np.ndarray) or output.dtype != np.dtype('<f4')
                    or output.shape != (descriptor['count'], *OUTPUT_SHAPE) or not np.isfinite(output).all()):
                raise GatewayError(502, 'Native engine returned invalid shared-memory logits.')
            slots.write('output', output)
            reply = {'schema': SCHEMA, 'request': descriptor['request'], 'client': descriptor['client'],
                     'model': descriptor['model'], 'count': descriptor['count'], 'status': 'completed'}
            # A completion reply authorizes immediate slot reuse. Release the
            # old request's watcher/file lock before any success bytes can reach
            # a persistent client, otherwise its next valid call can hit 409.
            await _drain(watcher)
            watcher = None
            slots.close()
            slots = None
            self._active.discard(client_key)
            client_key = None
            await self._respond(send, 200, json.dumps(reply, separators=(',', ':')).encode(),
                                [(b'content-type', b'application/json')])
        except ClientDisconnected:
            return
        except asyncio.TimeoutError:
            await self._respond(send, 408, b'{"error":"Shared-memory control body timed out."}',
                                [(b'content-type', b'application/json')])
        except GatewayError as error:
            await self._respond(send, error.status, json.dumps({'error': str(error)}).encode(),
                                [(b'content-type', b'application/json')])
        except Exception:
            logger.exception('Shared-memory pooling failed.')
            await self._respond(send, 502, b'{"error":"Shared-memory pooling failed."}',
                                [(b'content-type', b'application/json')])
        finally:
            await _drain(inference)
            await _drain(watcher)
            if slots is not None:
                slots.close()
            if client_key is not None:
                self._active.discard(client_key)
