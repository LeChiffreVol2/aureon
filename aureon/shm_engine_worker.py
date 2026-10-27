"""Opt-in descriptor transport inside the verified native vLLM worker.

Only S2L2A [1,2048] is a control frame. Ordinary model inputs delegate to the
previous forward unchanged. The native renderer, scheduler, worker and identity
pooler still execute; tensor payloads live in guarded same-UID /dev/shm slots.
Construction/installation performs no model execution or CUDA graph capture.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import traceback

from .cuda_graph import FixedInputCudaGraph, _precision
from .shm_engine_protocol import ACK_SHAPE, DESCRIPTOR_SHAPE, decode_descriptor_tensor_values, encode_ack
from .shm_files import SharedSlots


MAX_FAILURE_EVENTS = 8


def _output(value):
    from terratorch.models.model import ModelOutput
    return ModelOutput(output=value, auxiliary_heads=None)


def _pipeline_factory(model, worker_record, verified_bundle, evidence_dir, **kwargs):
    from .graph_pipeline import FullGroupGraphPipeline
    return FullGroupGraphPipeline.from_verified_worker(
        model, worker_record, verified_bundle, evidence_dir, **kwargs)


class ShmEngineForward:
    """One completed fixed-eight inference per bounded descriptor/ACK pair."""

    descriptor_shape = DESCRIPTOR_SHAPE
    decode_descriptor = staticmethod(decode_descriptor_tensor_values)

    def __init__(self, model, folder, *, rank, worker_record, verified_bundle,
                 torch_module=None, pipeline_factory=None, output_factory=None, slots_factory=None):
        graph = getattr(model, '_aureon_cuda_graph', None)
        if (not isinstance(graph, FixedInputCudaGraph) or graph.model is not model
                or graph.state == 'failed'):
            raise ValueError('Worker descriptor transport requires this model\'s verified graph wrapper')
        self.torch = graph.torch if torch_module is None else torch_module
        if self.torch is not graph.torch:
            raise ValueError('Descriptor transport and verified graph must use the same Torch runtime')
        self.model, self.forward = model, model.forward
        self.original_forward = graph.original_forward
        if (getattr(self.original_forward, '__self__', None) is not model
                or getattr(self.original_forward, '__func__', None) is not getattr(type(model), 'forward', None)):
            raise ValueError('Verified graph must retain this model class\'s original eager forward')
        self.worker_record, self.verified_bundle = deepcopy(worker_record), deepcopy(verified_bundle)
        self.pid = os.getpid()
        self.precision = _precision(self.torch)
        if (type(rank) is not int or rank < 0 or worker_record.get('rank') != rank
                or worker_record.get('schema') != 'aureon.worker_attestation.v1'
                or worker_record.get('pid') != self.pid
                or worker_record.get('loaded_tensor_manifest_matches') is not True
                or worker_record.get('model_training') is not False or model.training
                or worker_record.get('model_id') != verified_bundle.get('model_id')
                or worker_record.get('bundle_receipt_sha256') != verified_bundle.get('receipt_sha256')
                or worker_record.get('bundle_files_sha256') != verified_bundle.get('files_sha256')
                or worker_record.get('precision_settings_actual') != self.precision
                or worker_record.get('loaded_tensor_count') != len(verified_bundle.get('receipt', {}).get('tensor_manifest', {}))
                or not worker_record.get('loaded_tensor_count')
                or self.precision['float32_matmul_precision'] != 'highest'
                or any(self.precision[key] for key in ('matmul_allow_tf32', 'cudnn_allow_tf32',
                                                      'cudnn_sdp_enabled', 'cudnn_benchmark'))):
            raise ValueError('Require same-process verified model/bundle/state and highest FP32 worker evidence')
        self.device = next(model.parameters()).device
        if self.device.type != 'cuda':
            raise ValueError('Descriptor worker requires the resident CUDA model')
        self.folder = Path(folder).resolve(strict=True)
        if not self.folder.is_dir():
            raise ValueError('Use an existing controlled worker evidence directory')
        self.pipeline_folder = self.folder / 'full-group-graph-worker'
        self.pipeline_folder.mkdir(exist_ok=False)
        self.rank, self.model_id = rank, worker_record['model_id']
        self._pipeline_factory = pipeline_factory or _pipeline_factory
        self._output_factory = output_factory or _output
        self._slots_factory = slots_factory or SharedSlots.attach
        self._pipeline = None
        self._lock = threading.Lock()
        self.completed_calls = self.failed_calls = self.failure_events = 0
        self.evidence_error = None
        self._last_error = None
        self.source_sha256 = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in ('shm_engine_worker.py', 'shm_engine_protocol.py',
                                           'shm_files.py', 'graph_pipeline.py')}

    def metadata(self):
        return {'schema': 'aureon.shm_engine_worker.v1', 'pid': self.pid, 'rank': self.rank,
                'model_id': self.model_id, 'bundle_receipt_sha256': self.worker_record['bundle_receipt_sha256'],
                'loaded_tensor_manifest_matches': self.worker_record['loaded_tensor_manifest_matches'],
                'loaded_tensor_count': self.worker_record['loaded_tensor_count'],
                'precision_settings_actual': self.precision,
                'descriptor_shape': [1, *self.descriptor_shape], 'ack_shape': [1, *ACK_SHAPE],
                'dtype': 'float32', 'group_size': 8, 'neural_batch_size': 1,
                'completed_calls': self.completed_calls, 'failed_calls': self.failed_calls,
                'failure_events_written': self.failure_events, 'maximum_failure_events': MAX_FAILURE_EVENTS,
                'source_sha256': self.source_sha256, 'pipeline_directory': self.pipeline_folder.name,
                'pipeline': self._pipeline.metadata() if self._pipeline is not None else None,
                'last_error': self._last_error, 'evidence_error': self.evidence_error,
                'ordinary_inputs': 'delegate_to_previous_forward_unchanged',
                'payload_scope': 'worker-owned guarded mmap input/output; native engine carries descriptor and ACK',
                'completion': 'pipeline synchronization and output validation/write; close maps/flock; then ACK',
                'output_cache': False, 'retry': False, 'numerical_parity_verified': False}

    def _event(self, event, *, request=None, statuscode=0):
        suffix = event if event != 'failed' else f'failed-{self.failure_events:04d}'
        record = {**self.metadata(), 'event': event, 'statuscode': statuscode,
                  'created_utc': datetime.now(timezone.utc).isoformat(),
                  'request': request, 'scope': 'local transport execution evidence; separate numerical comparison required'}
        with (self.folder / f'shm-engine-rank{self.rank}-{suffix}.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')

    def _record_failure(self, error_metadata, descriptor, statuscode):
        self.failed_calls += 1
        # Never retain exception tracebacks: they may own caller mmap views.
        self._last_error = error_metadata
        if self.failure_events < MAX_FAILURE_EVENTS:
            self.failure_events += 1
            try:
                self._event('failed', request=descriptor.get('request') if descriptor else None,
                            statuscode=statuscode)
            except Exception as evidence_error:
                self.evidence_error = str(evidence_error)[:512]

    def _ack(self, descriptor, statuscode):
        values = encode_ack(descriptor, statuscode=statuscode)
        return self._output_factory(self._ack_tensor(values))

    def _ack_tensor(self, values):
        return self.torch.as_tensor(values, dtype=self.torch.float32, device=self.device).unsqueeze(0)

    def _descriptor_device_valid(self, value):
        return value.is_cuda and value.device == self.device

    def __call__(self, *args, **kwargs):
        value = args[0].get('S2L2A') if len(args) == 1 and isinstance(args[0], dict) else None
        # No changes to the previous ordinary/packed model request path.
        if not isinstance(value, self.torch.Tensor) or tuple(value.shape) != (1, *self.descriptor_shape):
            return self.forward(*args, **kwargs)
        descriptor = slots = input_view = output_view = None
        statuscode = 1
        with self._lock, self.torch.inference_mode():
            try:
                if kwargs or set(args[0]) != {'S2L2A'}:
                    raise ValueError('Descriptor request requires exactly S2L2A')
                if (value.dtype != self.torch.float32 or not self._descriptor_device_valid(value)
                        or os.getpid() != self.pid or self.model.training
                        or _precision(self.torch) != self.precision):
                    raise ValueError('Descriptor request changed worker PID/device/eval/FP32 contract')
                descriptor = self.decode_descriptor(value.detach().cpu().numpy()[0])
                statuscode = 2
                if descriptor['model'] != self.model_id:
                    raise ValueError('Descriptor model differs from the verified resident model')
                statuscode = 3
                slots = self._slots_factory(descriptor)
                input_view, output_view = slots._array('input'), slots._array('output')
                statuscode = 4
                if self._pipeline is None:
                    self._pipeline = self._pipeline_factory(
                        self.model, self.worker_record, self.verified_bundle, self.pipeline_folder,
                        original_forward=self.original_forward, torch_module=self.torch)
                self._pipeline.predict_into(input_view, output_view)
                # predict_into returns only after completed D2H, finite checks
                # and a synchronous CPU copy into the mapped output destination.
                input_view = output_view = None
                statuscode = 5
                slots.close()
                slots = None
                if self.completed_calls == 0:
                    statuscode = 6
                    self.completed_calls += 1
                    try:
                        self._event('first-success', request=descriptor['request'])
                    except Exception:
                        self.completed_calls -= 1
                        raise
                else:
                    self.completed_calls += 1
                return self._ack(descriptor, 0)
            except Exception as error:
                # Invalid control data, late attach and recoverable pipeline
                # failures remain failed requests, never stale successful logits.
                error_metadata = {'type': type(error).__name__, 'message': str(error)[:512]}
                # Even a temporarily active exception can retain exported mmap
                # arrays through completed pipeline frames. Clear the complete
                # cause/context chain before releasing mappings; the exception
                # may also be retained by an injected runtime or caller.
                pending, visited = [error], set()
                while pending:
                    current = pending.pop()
                    if id(current) in visited:
                        continue
                    visited.add(id(current))
                    pending.extend(item for item in (current.__cause__, current.__context__)
                                   if item is not None)
                    traceback.clear_frames(current.__traceback__)
                    current.__traceback__ = None
                current = None
            # Leave the exception handler before closing: no active exception
            # traceback owns pipeline-local views of the exported mmap buffers.
            input_view = output_view = None
            if slots is not None:
                try:
                    slots.close()
                except Exception as close_error:
                    self.evidence_error = 'slot close: ' + str(close_error)[:480]
                slots = None
            self._record_failure(error_metadata, descriptor, statuscode)
            return self._ack(descriptor, statuscode)


def install_shm_engine_forward(model, folder, *, rank=0, worker_record, verified_bundle):
    """Install after loaded-state verification and ordinary startup warmup."""
    if hasattr(model, '_aureon_shm_engine_forward'):
        raise ValueError('Worker descriptor transport is already installed')
    controller = ShmEngineForward(model, folder, rank=rank, worker_record=worker_record,
                                 verified_bundle=verified_bundle)
    controller._event('installed')
    model.forward = controller
    model._aureon_shm_engine_forward = controller
    return controller
