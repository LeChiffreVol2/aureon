"""Opt-in persistent transfer staging around strict KAO-TerraMind inference.

The arithmetic remains FP32 and batch one. A group shares one host-to-device
copy and one device-to-host copy; CUDA execution and transfer buffers persist.
The existing graph controller still owns its static buffers and clones outputs.
This module neither loads a model nor executes CUDA work on import.

Stream and pinned-memory semantics:
https://docs.pytorch.org/docs/2.10/notes/cuda.html#asynchronous-execution
"""
import math
import threading

import numpy as np

from .cuda_graph import FixedInputCudaGraph, _precision
from .reference import PyTorchPredictor


INPUT_SHAPE = (12, 224, 224)
OUTPUT_SHAPE = (10, 224, 224)


class PersistentPyTorchPredictor:
    """Reuse staging around an already strictly loaded CUDA FP32 predictor.

    ``graph=None`` binds the original eager forward, even when the model already
    has an AUREON graph installed. A supplied graph must belong to this exact
    model. Later changes to ``model.forward`` do not switch this instance's mode.
    CUDA allocation is lazy. Pinning failures raise; there is no silent fallback.

    Calls serialize through a lock and complete their CUDA event before host
    buffers can be read or reused. Returned NumPy logits own independent memory.
    The caller must keep the shared model and global precision settings fixed.
    No speed or numerical-parity claim follows from installing this candidate.
    """

    execution_backend = 'pytorch_persistent_staging'
    dtype = 'float32'

    def __init__(self, reference, *, graph=None, capacity=8, pin_memory=True):
        if not isinstance(reference, PyTorchPredictor):
            raise ValueError('Supply an existing strictly loaded PyTorchPredictor')
        if type(capacity) is not int or not 1 <= capacity <= 16:
            raise ValueError('Staging capacity must be an integer from 1 to 16')
        if type(pin_memory) is not bool:
            raise ValueError('pin_memory must be an explicit boolean')
        if (reference.dtype != 'float32' or reference._device.type != 'cuda'
                or reference._as_dict is not True or reference.model.training):
            raise ValueError('Staging requires the CUDA FP32 S2L2A eval reference')
        self.reference = reference
        self.model = reference.model
        self.torch = reference._torch
        self.model_id = reference.model_id
        self.device = reference.device
        self.capacity = capacity
        self.pin_memory = pin_memory
        self.graph = graph
        installed = getattr(self.model, '_aureon_cuda_graph', None)
        if graph is not None:
            if (not isinstance(graph, FixedInputCudaGraph) or graph.model is not self.model
                    or graph.torch is not self.torch or graph.state == 'failed'):
                raise ValueError('Supply this exact reference model\'s nonfailed AUREON CUDA graph')
            self._forward = graph
        elif installed is not None:
            if not isinstance(installed, FixedInputCudaGraph) or installed.model is not self.model:
                raise ValueError('Unexpected installed graph controller')
            self._forward = installed.original_forward
        else:
            self._forward = self.model.forward
        self.precision = _precision(self.torch)
        if (self.precision['float32_matmul_precision'] != 'highest'
                or any(self.precision[key] for key in ('matmul_allow_tf32', 'cudnn_allow_tf32',
                                                      'cudnn_sdp_enabled', 'cudnn_benchmark'))):
            raise ValueError('Staging requires the highest-FP32 precision contract')
        self._lock = threading.Lock()
        self._device = reference._device
        self._stream = self._completion = None
        self._host_input = self._host_output = None
        self._device_input = self._device_output = None
        self._input_numpy = self._output_numpy = None
        self._allocated = self._closed = False
        self._failure = None
        self._cleanup_error = None
        self.completed_calls = self.completed_tiles = 0

    def _check_contract(self):
        if self.model.training or _precision(self.torch) != self.precision:
            raise RuntimeError('Model eval mode or precision changed after staging installation')
        if self.graph is not None and self.graph.state == 'failed':
            raise RuntimeError('The supplied CUDA graph has failed')
        if (getattr(self.model, '_forward_hooks', None)
                or getattr(self.model, '_forward_pre_hooks', None)):
            raise RuntimeError('Persistent staging requires a model without forward instrumentation hooks')

    def _allocate(self):
        torch = self.torch
        with torch.cuda.device(self._device):
            self._device = torch.device('cuda', torch.cuda.current_device())
            self._stream = torch.cuda.Stream(device=self._device)
            self._completion = torch.cuda.Event(enable_timing=False)
            with torch.cuda.stream(self._stream):
                self._host_input = torch.empty((self.capacity, *INPUT_SHAPE), dtype=torch.float32,
                                               device='cpu', pin_memory=self.pin_memory)
                self._host_output = torch.empty((self.capacity, *OUTPUT_SHAPE), dtype=torch.float32,
                                                device='cpu', pin_memory=self.pin_memory)
                if (self._host_input.is_pinned() != self.pin_memory
                        or self._host_output.is_pinned() != self.pin_memory):
                    raise RuntimeError('Host staging pinning does not match the requested policy')
                self._device_input = torch.empty((self.capacity, *INPUT_SHAPE), dtype=torch.float32, device=self._device)
                self._device_output = torch.empty((self.capacity, *OUTPUT_SHAPE), dtype=torch.float32, device=self._device)
            self._input_numpy = self._host_input.numpy()
            self._output_numpy = self._host_output.numpy()
            self._allocated = True

    def _fail(self, error):
        self._failure = error
        # Preserve buffers if synchronization itself fails. A failed candidate
        # never reuses them or quietly falls back to a different execution path.
        if self._stream is not None:
            try:
                self._stream.synchronize()
            except Exception as cleanup_error:
                self._cleanup_error = str(cleanup_error)

    def __call__(self, normalized_batch):
        if (not isinstance(normalized_batch, np.ndarray) or normalized_batch.dtype != np.float32
                or normalized_batch.ndim != 4 or normalized_batch.shape[1:] != INPUT_SHAPE
                or not 1 <= len(normalized_batch) <= self.capacity):
            raise ValueError(f'Expected float32 normalized tiles [N,12,224,224], 1 <= N <= {self.capacity}')
        count = len(normalized_batch)
        with self._lock, self.torch.inference_mode():
            if self._closed:
                raise RuntimeError('Persistent predictor is closed')
            if self._failure is not None:
                raise RuntimeError('Persistent predictor is disabled after a recorded failure') from self._failure
            try:
                self._check_contract()
                if not self._allocated:
                    self._allocate()
            except Exception as error:
                self._fail(error)
                raise RuntimeError('Persistent staging initialization or precision check failed; no fallback') from error
            # Own the CPU snapshot before validation, including noncontiguous or
            # read-only caller inputs. Async H2D never references caller storage.
            np.copyto(self._input_numpy[:count], normalized_batch, casting='no')
            if not np.isfinite(self._input_numpy[:count]).all():
                raise ValueError('Normalized input must contain only finite FP32 values')
            try:
                with self.torch.cuda.device(self._device):
                    current = self.torch.cuda.current_stream(self._device)
                    if current != self._stream:
                        self._stream.wait_stream(current)
                    with self.torch.cuda.stream(self._stream):
                        self._device_input[:count].copy_(self._host_input[:count], non_blocking=self.pin_memory)
                        for index in range(count):
                            value = self._forward({'S2L2A': self._device_input[index:index + 1]})
                            output = value if isinstance(value, self.torch.Tensor) else getattr(value, 'output', None)
                            if (not isinstance(output, self.torch.Tensor)
                                    or tuple(output.shape) != (1, *OUTPUT_SHAPE)
                                    or output.dtype != self.torch.float32 or not output.is_cuda
                                    or output.device != self._device):
                                raise ValueError('Batch-one forward must return CUDA FP32 logits [1,10,224,224]')
                            self._device_output[index:index + 1].copy_(output, non_blocking=True)
                        self._host_output[:count].copy_(self._device_output[:count], non_blocking=self.pin_memory)
                        self._completion.record(self._stream)
                    # The completed event covers H2D, every batch-one forward,
                    # output staging and D2H. Only then may CPU buffers be read.
                    self._completion.synchronize()
                logits = self._output_numpy[:count].copy(order='C')
                if not np.isfinite(logits).all():
                    raise ValueError('Persistent predictor returned nonfinite FP32 logits')
                self._check_contract()
                self.completed_calls += 1
                self.completed_tiles += count
                return logits
            except Exception as error:
                self._fail(error)
                raise RuntimeError('Persistent staged inference failed; no fallback') from error

    def synchronize(self):
        """Complete this candidate's stream without executing a new inference."""
        with self._lock:
            if self._stream is not None:
                self._stream.synchronize()

    def metadata(self):
        """Describe allocation scope; no peak-memory or acceleration claim."""
        with self._lock:
            tensors = {'host_input': self._host_input, 'host_output': self._host_output,
                       'device_input': self._device_input, 'device_output': self._device_output}
            allocated = {name + '_bytes': int(tensor.numel() * tensor.element_size()) if tensor is not None else 0
                         for name, tensor in tensors.items()}
            return {'schema': 'aureon.pytorch_persistent_staging.v1', 'backend': self.execution_backend,
                'model_id': self.model_id, 'bundle_receipt_sha256': self.reference.bundle['receipt_sha256'],
                'state': 'closed' if self._closed else 'failed' if self._failure else 'allocated' if self._allocated else 'unallocated',
                'mode': 'cuda_graph' if self.graph is not None else 'eager', 'dtype': self.dtype,
                'device': str(self._device), 'model_batch_size': 1, 'capacity': self.capacity,
                'pin_memory_requested': self.pin_memory,
                'pin_memory_actual': self._host_input.is_pinned() if self._host_input is not None else None,
                'allocated_staging': allocated,
                'planned_bytes_per_host_or_device': self.capacity * (math.prod(INPUT_SHAPE) + math.prod(OUTPUT_SHAPE)) * 4,
                'memory_scope': 'Persistent staging only; excludes model, graph pools, per-forward clones, independent returned NumPy outputs and allocator overhead',
                'transfer_scope': 'One H2D group copy and one D2H group copy per call; one completion-event synchronization; ordered batch-one forwards',
                'output_ownership': 'Independent C-contiguous NumPy copy after D2H completion',
                'output_cache': False, 'precision_settings_actual_at_install': dict(self.precision),
                'completed_calls': self.completed_calls, 'completed_tiles': self.completed_tiles,
                'failure': str(self._failure) if self._failure is not None else None,
                'cleanup_error': self._cleanup_error, 'numerical_parity_verified': False,
                'eager_fallback': False}

    def close(self):
        """Finish transfers before releasing this instance's staging allocations."""
        with self._lock:
            if self._closed:
                return
            if self._stream is not None:
                self._stream.synchronize()
            self._input_numpy = self._output_numpy = None
            self._host_input = self._host_output = None
            self._device_input = self._device_output = None
            self._completion = self._stream = None
            self._allocated = False
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Persistent predictor is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
