"""Bounded exact-byte FP32 control frames for the opt-in local engine route.

These tensors contain control metadata, never image values or output logits.
Only private fixed-shape /dev/shm descriptors are admitted; no arbitrary paths,
addresses, object arrays, pickle or implicit dtype conversions are accepted.
"""
import json

import numpy as np

from .shm_files import decode_descriptor

DESCRIPTOR_SHAPE = (2048,)
ACK_SHAPE = (512,)
ACK_SCHEMA = 'aureon.shm_engine_ack.v1'
_DESCRIPTOR_MAGIC = b'ASD1'
_ACK_MAGIC = b'ASA1'


class WorkerReportedError(RuntimeError):
    def __init__(self, statuscode):
        self.statuscode = statuscode
        super().__init__('Shared-memory engine worker reported failure status ' + str(statuscode))


def _encode(value, shape, magic):
    body = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')
    if not 0 < len(body) <= shape[0] - 8:
        raise ValueError('Control frame exceeds its fixed tensor bound')
    frame = magic + len(body).to_bytes(4, 'little') + body
    result = np.zeros(shape, dtype=np.float32)
    result[:len(frame)] = np.frombuffer(frame, dtype=np.uint8)
    return result


def _decode(values, shape, magic):
    if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('float32')
            or values.shape != shape or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 255) or np.any(values != np.floor(values))):
        raise ValueError('Control tensor must contain exactly bounded integer bytes in FP32')
    raw = values.astype(np.uint8).tobytes()
    size = int.from_bytes(raw[4:8], 'little')
    if raw[:4] != magic or not 0 < size <= shape[0] - 8 or any(raw[8 + size:]):
        raise ValueError('Invalid control frame magic, length or padding')
    return raw[8:8 + size]


def _descriptor(value):
    body = json.dumps(value, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
    descriptor = decode_descriptor(body)
    if descriptor['count'] != 8 or descriptor['capacity'] != 8:
        raise ValueError('Engine shared memory requires exactly eight FP32 tiles and capacity eight')
    return descriptor


def encode_descriptor_tensor_values(descriptor):
    return _encode(_descriptor(descriptor), DESCRIPTOR_SHAPE, _DESCRIPTOR_MAGIC)


def decode_descriptor_tensor_values(values):
    descriptor = decode_descriptor(_decode(values, DESCRIPTOR_SHAPE, _DESCRIPTOR_MAGIC))
    return _descriptor(descriptor)


def encode_ack(descriptor=None, statuscode=0):
    if type(statuscode) is not int or not 0 <= statuscode <= 255:
        raise ValueError('Worker status must be an integer byte')
    if descriptor is None:
        if statuscode == 0:
            raise ValueError('Success ACK requires a validated descriptor')
        identity = {'client': '', 'request': '', 'model': '', 'count': 0}
    else:
        value = _descriptor(descriptor)
        identity = {key: value[key] for key in ('client', 'request', 'model', 'count')}
    return _encode({'schema': ACK_SCHEMA, **identity, 'statuscode': statuscode}, ACK_SHAPE, _ACK_MAGIC)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate ACK field')
        result[key] = value
    return result


def validate_ack(array, descriptor):
    expected = _descriptor(descriptor)
    body = _decode(array, ACK_SHAPE, _ACK_MAGIC)
    value = json.loads(body, object_pairs_hook=_pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite ACK field')))
    if (not isinstance(value, dict) or set(value) != {'schema', 'client', 'request', 'model', 'count', 'statuscode'}
            or value['schema'] != ACK_SCHEMA or type(value['statuscode']) is not int
            or not 0 <= value['statuscode'] <= 255):
        raise ValueError('Invalid shared-memory engine ACK schema/status')
    if value['statuscode'] != 0:
        raise WorkerReportedError(value['statuscode'])
    if any(type(value[key]) is not type(expected[key]) or value[key] != expected[key]
           for key in ('client', 'request', 'model', 'count')):
        raise ValueError('Shared-memory engine ACK does not match this request identity')
