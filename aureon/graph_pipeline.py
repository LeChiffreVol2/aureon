"""Explicit fixed-eight FP32 graph pipeline; never loads or starts a model.

One graph captures pinned H2D, eight ordered batch-one eager forwards, output
slice copies, and pinned D2H. Construction is CPU-only; capture is lazy. The
caller must serialize all other uses of this model during capture/inference.
This is an unmeasured candidate until separate numerical and timing checks.
"""
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading

import numpy as np

from .cuda_graph import FixedInputCudaGraph, _precision
from .reference import PyTorchPredictor


COUNT = 8
INPUT_SHAPE = (COUNT, 12, 224, 224)
OUTPUT_SHAPE = (COUNT, 10, 224, 224)
MODEL_IDS = {'kao-terramind-a', 'kao-terramind-b', 'kao-terramind-c'}


def _original(model, supplied=None):
    forward = supplied
    if forward is None:
        installed = getattr(model, '_aureon_cuda_graph', None)
        if installed is not None:
            if not isinstance(installed, FixedInputCudaGraph) or installed.model is not model:
                raise ValueError('Unexpected installed graph controller')
            forward = installed.original_forward
        else:
            forward = model.forward
    if (getattr(forward, '__self__', None) is not model
            or getattr(forward, '__func__', None) is not getattr(type(model), 'forward', None)):
        raise ValueError('Supply this model class\'s original bound eager forward')
    return forward


class FullGroupGraphPipeline:
    """Fixed-group candidate with exclusive buffers and explicit provenance.

    Use ``from_reference`` or ``from_verified_worker``. ``predict_into`` accepts
    a caller-owned, writable, C-contiguous output array (including a guarded
    shared-memory view). That caller must retain exclusive output ownership
    through return. ``__call__`` instead returns a newly owned NumPy array.
    Neither path caches results or pads smaller input groups.
    """

    execution_backend = 'pytorch_full_group_cuda_graph'
    dtype = 'float32'
    capacity = COUNT
    pin_memory = True

    @classmethod
    def from_reference(cls, reference, evidence_dir):
        if not isinstance(reference, PyTorchPredictor):
            raise ValueError('Supply an existing strictly loaded PyTorchPredictor')
        if (reference.dtype != 'float32' or reference._device.type != 'cuda'
                or reference._as_dict is not True):
            raise ValueError('Full-group graph requires the strict CUDA FP32 S2L2A reference')
        return cls(reference.model, _original(reference.model), reference.bundle, evidence_dir,
                   torch_module=reference._torch, provenance={'kind': 'strict_pytorch_reference',
                       'scope': 'Inherited strict local reference load; weights are not rehashed by this candidate'}, rank=0)

    @classmethod
    def from_verified_worker(cls, model, worker_record, verified_bundle, evidence_dir, *,
                             original_forward=None, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        if (not isinstance(worker_record, dict) or worker_record.get('schema') != 'aureon.worker_attestation.v1'
                or worker_record.get('pid') != os.getpid()
                or worker_record.get('loaded_tensor_manifest_matches') is not True
                or worker_record.get('model_training') is not False
                or worker_record.get('model_id') != verified_bundle.get('model_id')
                or worker_record.get('bundle_receipt_sha256') != verified_bundle.get('receipt_sha256')
                or worker_record.get('bundle_files_sha256') != verified_bundle.get('files_sha256')
                or worker_record.get('precision_settings_actual') != _precision(torch_module)
                or worker_record.get('loaded_tensor_count') != len(verified_bundle.get('receipt', {}).get('tensor_manifest', {}))):
            raise ValueError('Worker record must match this process, verified bundle, full state and current precision')
        return cls(model, _original(model, original_forward), verified_bundle, evidence_dir,
                   torch_module=torch_module, rank=worker_record.get('rank'),
                   provenance={'kind': 'same_process_verified_worker',
                       'worker_record_sha256': hashlib.sha256(json.dumps(worker_record, sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                       'scope': 'Inherited same-process worker state attestation; not remote attestation or a fresh value hash'})

    def __init__(self, model, original_forward, verified_bundle, evidence_dir, *, torch_module, provenance, rank):
        self.torch, self.model = torch_module, model
        self._forward = _original(model, original_forward)
        self.model_id = verified_bundle.get('model_id')
        self.bundle_receipt_sha256 = verified_bundle.get('receipt_sha256')
        if (self.model_id not in MODEL_IDS or not isinstance(self.bundle_receipt_sha256, str)
                or re.fullmatch('[0-9a-f]{64}', self.bundle_receipt_sha256) is None
                or type(rank) is not int or rank < 0):
            raise ValueError('Full-group graph requires a KAO-TerraMind bundle identity and valid rank')
        self.rank, self.provenance = rank, dict(provenance)
        self.precision = _precision(self.torch)
        if (self.precision['float32_matmul_precision'] != 'highest'
                or any(self.precision[key] for key in ('matmul_allow_tf32', 'cudnn_allow_tf32',
                                                      'cudnn_sdp_enabled', 'cudnn_benchmark'))):
            raise ValueError('Full-group graph requires the highest-FP32 precision contract')
        state = model.state_dict()
        expected = verified_bundle.get('receipt', {}).get('tensor_manifest')
        if not isinstance(expected, dict) or not expected or set(state) != set(expected):
            raise ValueError('Live state names must match the complete verified tensor manifest')
        devices = set()
        for name, value in state.items():
            if (not isinstance(value, self.torch.Tensor) or not value.is_cuda
                    or list(value.shape) != expected[name].get('shape')
                    or str(value.dtype) != expected[name].get('dtype')
                    or (value.is_floating_point() and value.dtype != self.torch.float32)):
                raise ValueError('Live model state must preserve verified shapes/dtypes on CUDA in FP32')
            devices.add(value.device)
        if len(devices) != 1:
            raise ValueError('Full-group graph requires one CUDA device')
        self._device = next(iter(devices))
        self.folder = Path(evidence_dir).resolve(strict=True)
        if not self.folder.is_dir():
            raise ValueError('Full-group graph evidence directory must already exist')
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._lock = threading.Lock()
        self._stream = self._completion = self._graph = None
        self._host_input = self._host_output = self._device_input = self._device_output = None
        self._input_numpy = self._output_numpy = None
        self.state, self._error, self._cleanup_error, self._evidence_error = 'armed', None, None, None
        self.capture_count = self.replay_count = self.completed_calls = self.completed_tiles = self._event_count = 0
        self._check_contract()
        self._event('armed')

    def _check_contract(self):
        if (self.model.training or _precision(self.torch) != self.precision
                or getattr(self.model, '_forward_hooks', None) or getattr(self.model, '_forward_pre_hooks', None)):
            raise RuntimeError('Full-group graph requires unchanged FP32 precision, eval mode and no forward hooks')

    def _event(self, event, **extra):
        self._event_count += 1
        record = {'schema': 'aureon.full_group_graph_event.v1', 'event': event,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'state': self.state,
            'model_id': self.model_id, 'bundle_receipt_sha256': self.bundle_receipt_sha256,
            'provenance': self.provenance, 'source_sha256': self.source_sha256,
            'precision_settings_actual': _precision(self.torch), 'input_shape': list(INPUT_SHAPE),
            'output_shape': list(OUTPUT_SHAPE), 'dtype': self.dtype, 'neural_batch_size': 1,
            'ordered_forwards_per_replay': COUNT, 'captured_host_transfers': True,
            'capture_count': self.capture_count, 'replay_count': self.replay_count,
            'output_cache': False, 'padding': False, 'eager_fallback': False,
            'numerical_parity_verified': False, **extra}
        with (self.folder / f'full-group-graph-rank{self.rank}-{self._event_count:04d}-{event}.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')

    def _allocate(self):
        torch = self.torch
        with torch.cuda.device(self._device):
            self._stream = torch.cuda.Stream(device=self._device)
            self._completion = torch.cuda.Event(enable_timing=False)
            with torch.cuda.stream(self._stream):
                self._host_input = torch.empty(INPUT_SHAPE, dtype=torch.float32, device='cpu', pin_memory=True)
                self._host_output = torch.empty(OUTPUT_SHAPE, dtype=torch.float32, device='cpu', pin_memory=True)
                if not self._host_input.is_pinned() or not self._host_output.is_pinned():
                    raise RuntimeError('Pinned host allocation is mandatory; no pageable fallback')
                self._device_input = torch.empty(INPUT_SHAPE, dtype=torch.float32, device=self._device)
                self._device_output = torch.empty(OUTPUT_SHAPE, dtype=torch.float32, device=self._device)
            self._input_numpy, self._output_numpy = self._host_input.numpy(), self._host_output.numpy()

    def _logits(self, value):
        if isinstance(value, self.torch.Tensor):
            result = value
        elif (is_dataclass(value) and {field.name for field in fields(value)} == {'output', 'auxiliary_heads'}
              and value.auxiliary_heads in (None, {})):
            result = value.output
        else:
            raise ValueError('Expected tensor logits or TerraTorch ModelOutput without auxiliary heads')
        if (not isinstance(result, self.torch.Tensor) or tuple(result.shape) != (1, *OUTPUT_SHAPE[1:])
                or result.dtype != self.torch.float32 or not result.is_cuda or result.device != self._device):
            raise ValueError('Each original forward must return CUDA FP32 [1,10,224,224] logits')
        return result

    def _body(self):
        self._device_input.copy_(self._host_input, non_blocking=True)
        for index in range(COUNT):
            result = self._forward({'S2L2A': self._device_input[index:index + 1]})
            self._device_output[index:index + 1].copy_(self._logits(result), non_blocking=True)
        self._host_output.copy_(self._device_output, non_blocking=True)

    def _capture(self):
        self.state = 'capturing'
        self._event('capture-started', warmup_groups=3, warmup_model_forwards=3 * COUNT)
        with self.torch.cuda.stream(self._stream):
            for _ in range(3):
                self._body()
        self._stream.synchronize()
        self._graph = self.torch.cuda.CUDAGraph()
        with self.torch.cuda.graph(self._graph, stream=self._stream, capture_error_mode='global'):
            self._body()
        self._check_contract()
        self.capture_count += 1
        self.state = 'captured'
        self._event('captured', warmup_groups=3, captured_model_forwards=COUNT)

    def _fail(self, error):
        # Never retain a traceback: its frames can hold caller-owned mmap views
        # that must be releasable by the worker after cancellation/failure.
        self.state, self._error = 'failed', {'type': type(error).__name__, 'message': str(error)}
        if self._stream is not None:
            try:
                self._stream.synchronize()
            except BaseException as cleanup_error:
                self._cleanup_error = str(cleanup_error)
        try:
            self._event('failed', error_type=type(error).__name__, error=str(error), cleanup_error=self._cleanup_error)
        except Exception as evidence_error:
            self._evidence_error = str(evidence_error)

    @staticmethod
    def _validate_input(values):
        if not isinstance(values, np.ndarray) or values.dtype != np.dtype('<f4') or values.shape != INPUT_SHAPE:
            raise ValueError('Full-group graph accepts exactly float32 [8,12,224,224]; no padding')

    def _predict(self, values, destination):
        self._validate_input(values)
        if destination is not None and (not isinstance(destination, np.ndarray)
                or destination.dtype != np.dtype('<f4') or destination.shape != OUTPUT_SHAPE
                or not destination.flags.c_contiguous or not destination.flags.writeable
                or np.shares_memory(values, destination)):
            raise ValueError('Destination must be exclusive writable contiguous float32 [8,10,224,224]')
        with self._lock, self.torch.inference_mode():
            if self.state in ('failed', 'closed'):
                raise RuntimeError(f'Full-group graph is {self.state}; no reuse or fallback')
            try:
                self._check_contract()
                if self._stream is None:
                    self._allocate()
                np.copyto(self._input_numpy, values, casting='no')
            except BaseException as error:
                self._fail(error)
                raise
            # CPU snapshot is owned before validation and H2D. Invalid caller
            # data does not poison a valid graph or modify its output buffers.
            if not np.isfinite(self._input_numpy).all():
                raise ValueError('Normalized input must contain finite FP32 values')
            try:
                with self.torch.cuda.device(self._device):
                    current = self.torch.cuda.current_stream(self._device)
                    if current != self._stream:
                        self._stream.wait_stream(current)
                    if self._graph is None:
                        self._capture()
                    with self.torch.cuda.stream(self._stream):
                        self._graph.replay()
                        self.replay_count += 1
                        self._completion.record(self._stream)
                    self._completion.synchronize()
                self._check_contract()
                if not np.isfinite(self._output_numpy).all():
                    raise ValueError('Full-group graph produced nonfinite FP32 logits')
                if destination is None:
                    result = self._output_numpy.copy(order='C')
                else:
                    np.copyto(destination, self._output_numpy, casting='no')
                    result = destination
                self.completed_calls += 1
                self.completed_tiles += COUNT
                if self.state != 'active':
                    self.state = 'active'
                    self._event('first-replay-completed')
                return result
            except BaseException as error:
                self._fail(error)
                raise

    def __call__(self, normalized_batch):
        return self._predict(normalized_batch, None)

    def predict_into(self, normalized_batch, output_array):
        """Fill the caller's exclusive output only after replay/D2H/finite checks."""
        return self._predict(normalized_batch, output_array)

    def synchronize(self):
        with self._lock:
            if self._stream is not None:
                self._stream.synchronize()

    def metadata(self):
        with self._lock:
            buffers = {'host_input': self._host_input, 'host_output': self._host_output,
                       'device_input': self._device_input, 'device_output': self._device_output}
            return {'schema': 'aureon.full_group_graph_pipeline.v1', 'backend': self.execution_backend,
                'state': self.state, 'model_id': self.model_id, 'bundle_receipt_sha256': self.bundle_receipt_sha256,
                'provenance': dict(self.provenance), 'source_sha256': self.source_sha256,
                'device': str(self._device), 'dtype': self.dtype, 'capacity': COUNT, 'neural_batch_size': 1,
                'input_shape': list(INPUT_SHAPE), 'output_shape': list(OUTPUT_SHAPE),
                'capture_count': self.capture_count, 'replay_count': self.replay_count,
                'completed_calls': self.completed_calls, 'completed_tiles': self.completed_tiles,
                'precision_settings_actual_at_install': dict(self.precision),
                'precision_settings_actual': _precision(self.torch),
                'pin_memory_required': True, 'pin_memory_actual': self._host_input.is_pinned() if self._host_input is not None else None,
                'allocated_staging_bytes': {name: int(value.numel() * value.element_size()) if value is not None else 0
                                           for name, value in buffers.items()},
                'planned_bytes_per_host_or_device': (math.prod(INPUT_SHAPE) + math.prod(OUTPUT_SHAPE)) * 4,
                'memory_scope': 'Staging only; excludes model, graph private pool, allocator overhead and returned arrays',
                'transfer_scope': 'One captured H2D, eight ordered original batch-one forwards with output slice copies, one captured D2H, one completion event per replay',
                'output_ownership': '__call__ owns independent NumPy; predict_into fills caller-owned exclusive destination after completion',
                'output_cache': False, 'padding': False, 'per_tile_clones': False, 'eager_fallback': False,
                'numerical_parity_verified': False, 'failure': dict(self._error) if self._error else None,
                'cleanup_error': self._cleanup_error, 'evidence_error': self._evidence_error}

    def close(self):
        with self._lock:
            if self.state == 'closed':
                return
            if self._stream is not None:
                self._stream.synchronize()  # Retain storage if CUDA cannot confirm completion.
            self._graph = None
            self._input_numpy = self._output_numpy = None
            self._host_input = self._host_output = self._device_input = self._device_output = None
            self._completion = self._stream = None
            self.state = 'closed'

    def __enter__(self):
        if self.state in ('closed', 'failed'):
            raise RuntimeError('Full-group graph is unavailable')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
