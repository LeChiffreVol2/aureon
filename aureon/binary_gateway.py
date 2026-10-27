"""Opt-in loopback NPY transport through the existing vLLM pooling engine.

Load with --middleware aureon.binary_gateway.BinaryPoolingMiddleware. This
module never constructs a model and imports neural packages only on a request.
"""
import asyncio
import io
import json
import logging
import os
import time
import uuid

import numpy as np


PATH = '/aureon/pooling-binary'
GROUP_PATH = '/aureon/pooling-binary-group'
PACK_PATH = '/aureon/pooling-binary-pack'
MAX_GROUP_SIZE = 8
INPUT_SHAPE = (12, 224, 224)
OUTPUT_SHAPE = (10, 224, 224)
INPUT_BYTES = 12 * 224 * 224 * 4
MAX_BODY_BYTES = INPUT_BYTES + 4096
MAX_GROUP_BODY_BYTES = MAX_GROUP_SIZE * INPUT_BYTES + 4096
BODY_TIMEOUT_SECONDS = 30
INFERENCE_TIMEOUT_SECONDS = 300
MODEL_IDS = frozenset(('kao-terramind-a', 'kao-terramind-b', 'kao-terramind-c'))
logger = logging.getLogger(__name__)


class GatewayError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class ClientDisconnected(Exception):
    pass


def decode_tile(body, *, grouped=False):
    """Validate the bounded header before creating an array; never unpickle."""
    if len(body) > (MAX_GROUP_BODY_BYTES if grouped else MAX_BODY_BYTES):
        raise GatewayError(413, 'NPY tile exceeds the request budget.')
    stream = io.BytesIO(body)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=1024)
        else:
            raise ValueError('Unsupported NPY version.')
    except (ValueError, EOFError, TypeError) as exc:
        raise GatewayError(400, 'Expected a bounded NPY v1 or v2 tile.') from exc
    valid_shape = (len(shape) == 4 and type(shape[0]) is int
                   and 1 <= shape[0] <= MAX_GROUP_SIZE and shape[1:] == INPUT_SHAPE) if grouped else shape == INPUT_SHAPE
    if not valid_shape or fortran or dtype.str != '<f4':
        raise GatewayError(400, 'Expected C-order little-endian float32 ' +
                           ('[N,12,224,224], 1 <= N <= 8.' if grouped else '[12,224,224].'))
    offset = stream.tell()
    if len(body) - offset != INPUT_BYTES * (shape[0] if grouped else 1):
        raise GatewayError(400, 'NPY payload size does not match its tile shape.')
    values = np.frombuffer(body, dtype='<f4', offset=offset).reshape(shape)
    if not np.isfinite(values).all():
        raise GatewayError(400, 'Expected finite normalized input.')
    return values.copy()  # Own writable storage for torch.from_numpy.


async def native_pooling(state, values, model_name, request_id, *, output_shape=OUTPUT_SHAPE, return_array=False):
    """Keep native TerraTorch rendering, scheduling, model, and output copying."""
    import torch
    from vllm.entrypoints.pooling.pooling.protocol import PoolingCompletionRequest

    handler = state.serving_pooling
    request = PoolingCompletionRequest(
        model=model_name, input=[1], task=handler.pooling_task,
        use_activation=False, encoding_format='bytes_only',
        embed_dtype='float32', endianness='little',
    )
    prompt = {'prompt_token_ids': [1],
              'multi_modal_data': {'image': {'S2L2A': torch.from_numpy(values)}}}
    started = time.perf_counter()
    inputs = await handler.openai_serving_render.preprocess_cmpl(request, [prompt])
    if len(inputs) != 1:
        raise RuntimeError('Binary tile must produce exactly one native engine input.')
    rendered = time.perf_counter()
    final = None
    finished = False
    diagnostics = os.environ.get('AUREON_ENGINE_DIAGNOSTICS') == '1'
    if diagnostics:
        from .engine_diagnostics import emit_api_event, install_api_engine_diagnostics
        install_api_engine_diagnostics(state.engine_client)
        emit_api_event('gateway_engine_submit', request_id)
    try:
        async for result in state.engine_client.encode(inputs[0], request.to_pooling_params(), request_id):
            if diagnostics:
                emit_api_event('gateway_engine_yield', request_id, finished=bool(getattr(result, 'finished', False)))
            final = result
        finished = True
    finally:
        if not finished:
            await state.engine_client.abort(request_id)
    returned = time.perf_counter()
    output = None if final is None else final.outputs.data
    if (not isinstance(output, torch.Tensor) or tuple(output.shape) != output_shape
            or output.dtype != torch.float32 or output.device.type != 'cpu'):
        raise RuntimeError('Native engine must return CPU float32 logits with the requested tile shape.')
    array = output.detach().contiguous().numpy()
    if not np.isfinite(array).all():
        raise RuntimeError('Native engine returned nonfinite logits.')
    body = array.astype('<f4', copy=False) if return_array else array.astype('<f4', copy=False).tobytes()
    ended = time.perf_counter()
    return body, {
        'preprocess': (rendered - started) * 1000,
        'engine': (returned - rendered) * 1000,
        'output-encode': (ended - returned) * 1000,
    }


async def native_pooling_group(state, values, model_name, request_id, concurrency):
    """One HTTP group, independent native requests; no model batching or cache.

    A semaphore bounds active render/encode operations. gather preserves input
    order even when completions differ. Every failed/cancelled group drains its
    children, whose native_pooling finally blocks abort unfinished engine IDs.
    """
    semaphore = asyncio.Semaphore(concurrency)
    async def one(index, tile):
        async with semaphore:
            return await native_pooling(state, tile, model_name, f'{request_id}-{index}')
    started = time.perf_counter()
    tasks = [asyncio.create_task(one(index, tile)) for index, tile in enumerate(values)]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            # Do not interrupt an abort already running after gather cancellation.
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    returned = time.perf_counter()
    body = b''.join(result[0] for result in results)
    ended = time.perf_counter()
    return body, {'group-inference': (returned - started) * 1000,
                  'output-join': (ended - returned) * 1000}


class BinaryPoolingMiddleware:
    def __init__(self, app):
        self.app = app

    async def _read_body(self, receive, content_length, maximum=MAX_BODY_BYTES):
        parts = []
        total = 0
        while True:
            event = await receive()
            if event['type'] == 'http.disconnect':
                raise ClientDisconnected()
            if event['type'] != 'http.request':
                raise GatewayError(400, 'Invalid HTTP body event.')
            part = event.get('body', b'')
            total += len(part)
            if total > maximum:
                raise GatewayError(413, 'NPY tile exceeds the request budget.')
            parts.append(part)
            if not event.get('more_body', False):
                break
        if content_length is not None and total != content_length:
            raise GatewayError(400, 'Content-Length does not match the request body.')
        return b''.join(parts)

    async def _disconnect(self, receive):
        while True:
            if (await receive())['type'] == 'http.disconnect':
                return

    async def _respond(self, send, status, body, headers=()):
        await send({'type': 'http.response.start', 'status': status,
                    'headers': [(b'content-length', str(len(body)).encode()), *headers]})
        await send({'type': 'http.response.body', 'body': body})

    async def __call__(self, scope, receive, send):
        if scope.get('type') != 'http' or scope.get('path') not in (PATH, GROUP_PATH, PACK_PATH):
            return await self.app(scope, receive, send)
        grouped = scope['path'] == GROUP_PATH
        packed = scope['path'] == PACK_PATH
        multiple = grouped or packed
        maximum = MAX_GROUP_BODY_BYTES if multiple else MAX_BODY_BYTES
        inference = watcher = None
        try:
            if scope.get('method') != 'POST':
                raise GatewayError(405, 'Use POST for binary pooling.')
            state = scope['app'].state
            args = state.args
            if (getattr(args, 'host', None) != '127.0.0.1'
                    or not scope.get('client') or scope['client'][0] not in ('127.0.0.1', '::1')):
                raise GatewayError(403, 'Binary pooling requires a controlled loopback service.')
            # Custom middleware precedes native auth middleware; do not bypass it.
            if getattr(args, 'api_key', None) or os.environ.get('VLLM_API_KEY'):
                raise GatewayError(403, 'Binary pooling is limited to the local service without API keys.')
            headers = {}
            for key, value in scope.get('headers', []):
                key = key.lower()
                if key in headers and key in (b'content-type', b'content-length', b'x-aureon-model', b'x-aureon-concurrency'):
                    raise GatewayError(400, 'Duplicate binary request header.')
                headers[key] = value
            if headers.get(b'content-type', b'').split(b';', 1)[0].strip() != b'application/octet-stream':
                raise GatewayError(415, 'Expected application/octet-stream.')
            try:
                model_name = headers.get(b'x-aureon-model', b'').decode('ascii')
            except UnicodeDecodeError as exc:
                raise GatewayError(400, 'Invalid model identity.') from exc
            handler = getattr(state, 'serving_pooling', None)
            if (model_name not in MODEL_IDS or handler is None
                    or model_name != handler.models.model_name()):
                raise GatewayError(400, 'X-Aureon-Model must match the served KAO-TerraMind model.')
            concurrency = None
            if multiple:
                if getattr(args, 'max_num_seqs', None) != 1:
                    raise GatewayError(400, 'Grouped pooling requires service max_num_seqs=1.')
                encoded_concurrency = headers.get(b'x-aureon-concurrency', b'')
                if encoded_concurrency not in tuple(str(value).encode() for value in range(1, MAX_GROUP_SIZE + 1)):
                    raise GatewayError(400, 'X-Aureon-Concurrency must be an integer from 1 to 8.')
                concurrency = int(encoded_concurrency)
                if packed and (os.environ.get('AUREON_PACKED_TILES') != '1'
                               or os.environ.get('AUREON_CUDA_GRAPH') != '1' or concurrency != 1):
                    raise GatewayError(400, 'Packed pooling requires the enabled packed graph worker and concurrency one.')
            length = None
            if b'content-length' in headers:
                try:
                    length = int(headers[b'content-length'])
                except ValueError as exc:
                    raise GatewayError(400, 'Invalid Content-Length.') from exc
                if length < 0:
                    raise GatewayError(400, 'Invalid Content-Length.')
                if length > maximum:
                    raise GatewayError(413, 'NPY tile exceeds the request budget.')
            body = await asyncio.wait_for(self._read_body(receive, length, maximum), BODY_TIMEOUT_SECONDS)
            started = time.perf_counter()
            values = decode_tile(body, grouped=multiple)
            decode_ms = (time.perf_counter() - started) * 1000
            request_id = 'aureon-binary-' + uuid.uuid4().hex
            if packed:
                operation = native_pooling(state, values, model_name, request_id,
                                           output_shape=(len(values), *OUTPUT_SHAPE))
            elif grouped:
                operation = native_pooling_group(state, values, model_name, request_id, concurrency)
            else:
                operation = native_pooling(state, values, model_name, request_id)
            inference = asyncio.create_task(operation)
            watcher = asyncio.create_task(self._disconnect(receive))
            done, _ = await asyncio.wait((inference, watcher), timeout=INFERENCE_TIMEOUT_SECONDS,
                                         return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                raise ClientDisconnected()
            if inference not in done:
                raise GatewayError(504, 'Binary pooling timed out.')
            result, timings = inference.result()
            timings['input-decode'] = decode_ms
            response_headers = [(b'content-type', b'application/octet-stream'),
                                (b'x-aureon-model', model_name.encode()),
                                (b'x-request-id', request_id.encode())]
            if multiple:
                response_headers.extend([(b'x-aureon-tile-count', str(len(values)).encode()),
                                         (b'x-aureon-concurrency', str(concurrency).encode())])
            response_headers.extend((('x-aureon-' + key + '-ms').encode(), f'{value:.6f}'.encode())
                                    for key, value in timings.items())
            await self._respond(send, 200, result, response_headers)
        except ClientDisconnected:
            return
        except asyncio.TimeoutError:
            await self._respond(send, 408, b'{"error":"Request body timed out."}',
                                [(b'content-type', b'application/json')])
        except GatewayError as exc:
            await self._respond(send, exc.status, json.dumps({'error': str(exc)}).encode(),
                                [(b'content-type', b'application/json')])
        except Exception:
            logger.exception('Binary pooling failed.')
            await self._respond(send, 502, b'{"error":"Native binary pooling failed."}',
                                [(b'content-type', b'application/json')])
        finally:
            pending = [task for task in (inference, watcher) if task is not None]
            for task in pending:
                if not task.done():
                    task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
