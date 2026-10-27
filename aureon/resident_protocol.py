"""Distinct fixed frame for the opt-in bounded resident-mapping worker route."""
import numpy as np

from .shm_engine_protocol import encode_descriptor_tensor_values, decode_descriptor_tensor_values

DESCRIPTOR_SHAPE = (2112,)
PATH = '/aureon/pooling-shm-resident'


def encode_descriptor(descriptor):
    frame = np.zeros(DESCRIPTOR_SHAPE, dtype=np.float32)
    frame[:2048] = encode_descriptor_tensor_values(descriptor)
    return frame


def decode_descriptor(values):
    if (not isinstance(values, np.ndarray) or values.shape != DESCRIPTOR_SHAPE
            or values.dtype != np.dtype('float32') or np.any(values[2048:] != 0)):
        raise ValueError('Resident descriptor requires its exact FP32 frame and zero reserved tail')
    return decode_descriptor_tensor_values(values[:2048])
