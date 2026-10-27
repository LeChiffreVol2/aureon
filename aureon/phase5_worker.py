"""Explicit Phase 5 request instrumentation, separate from clean execution.

Install only in a new diagnostic service before its first descriptor request.
Frozen core sources are imported but never edited. The diagnostic controller
retains existing descriptor/slot guards, original batch-one arithmetic and ACK
ordering. Probe overhead and CUDA events make these runs ineligible for speed.
"""
import os
import traceback

import numpy as np

from .cuda_graph import _precision
from .graph_pipeline import FullGroupGraphPipeline, OUTPUT_SHAPE
from .phase5_trace import DiagnosticSink
from .shm_engine_protocol import DESCRIPTOR_SHAPE, decode_descriptor_tensor_values
from .shm_engine_worker import ShmEngineForward


class DiagnosticFullGroupGraphPipeline(FullGroupGraphPipeline):
    execution_backend = 'diagnostic_full_group_cuda_graph'
    diagnostic_cuda_events = True

    def _event(self, event, **extra):
        return super()._event(event, diagnostic_only=True, clean_speed_eligible=False, **extra)

    def __call__(self, values, *, trace):
        return self._diagnose(values, None, trace)

    def predict_into(self, values, output_array, *, trace):
        return self._diagnose(values, output_array, trace)

    def _diagnose(self, values, destination, trace):
        self._validate_input(values)
        if destination is not None and (not isinstance(destination, np.ndarray)
                or destination.dtype != np.dtype('<f4') or destination.shape != OUTPUT_SHAPE
                or not destination.flags.c_contiguous or not destination.flags.writeable
                or np.shares_memory(values, destination)):
            raise ValueError('Destination must be exclusive writable contiguous FP32 [8,10,224,224]')
        with self._lock, self.torch.inference_mode():
            if self.state in ('failed', 'closed'):
                raise RuntimeError(f'Diagnostic pipeline is {self.state}; no reuse or fallback')
            try:
                self._check_contract()
                if self._stream is None:
                    with trace.span('pipeline_allocate'):
                        self._allocate()
                        self._timing_start = self.torch.cuda.Event(enable_timing=True) if self.diagnostic_cuda_events else None
                        self._timing_end = self.torch.cuda.Event(enable_timing=True) if self.diagnostic_cuda_events else None
                with trace.span('pipeline_input_copy_to_pinned'):
                    np.copyto(self._input_numpy, values, casting='no')
                with trace.span('pipeline_input_finite'):
                    if not np.isfinite(self._input_numpy).all():
                        raise ValueError('Normalized input must be finite FP32')
                with self.torch.cuda.device(self._device):
                    current = self.torch.cuda.current_stream(self._device)
                    if current != self._stream:
                        self._stream.wait_stream(current)
                    if self._graph is None:
                        with trace.span('pipeline_warmup_and_capture'):
                            self._capture()
                    with trace.span('pipeline_cuda_replay_and_completion_host'):
                        with self.torch.cuda.stream(self._stream):
                            if self._timing_start is not None:
                                self._timing_start.record(self._stream)
                            self._graph.replay()
                            self.replay_count += 1
                            if self._timing_end is not None:
                                self._timing_end.record(self._stream)
                            self._completion.record(self._stream)
                        self._completion.synchronize()
                    trace.extra['cuda_replay_elapsed_ms'] = (float(self._timing_start.elapsed_time(self._timing_end))
                                                             if self._timing_start is not None else None)
                    trace.extra['cuda_event_scope'] = 'captured H2D + eight ordered batch-one forwards + D2D staging + D2H; not model compute alone'
                    trace.extra['cuda_event_timing_enabled'] = self.diagnostic_cuda_events
                self._check_contract()
                with trace.span('pipeline_output_finite'):
                    if not np.isfinite(self._output_numpy).all():
                        raise ValueError('Diagnostic pipeline returned nonfinite logits')
                with trace.span('pipeline_output_copy_from_pinned'):
                    if destination is None:
                        result = self._output_numpy.copy(order='C')
                    else:
                        np.copyto(destination, self._output_numpy, casting='no')
                        result = destination
                self.completed_calls += 1
                self.completed_tiles += 8
                if self.state != 'active':
                    self.state = 'active'
                    self._event('first-replay-completed')
                return result
            except BaseException as error:
                self._fail(error)
                raise

    def metadata(self):
        return {**super().metadata(), 'diagnostic_only': True, 'clean_speed_eligible': False,
                'forward_diagnostics': True, 'cuda_event_timing_enabled': self.diagnostic_cuda_events}


def _clear_frames(error):
    pending, visited = [error], set()
    while pending:
        item = pending.pop()
        if id(item) in visited:
            continue
        visited.add(id(item))
        pending.extend(value for value in (item.__cause__, item.__context__) if value is not None)
        traceback.clear_frames(item.__traceback__)
        item.__traceback__ = None


class DiagnosticShmEngineForward(ShmEngineForward):
    def metadata(self):
        return {**super().metadata(), 'diagnostic_only': True, 'clean_speed_eligible': False,
                'forward_diagnostics': True, 'diagnostic_limit': self.diagnostics.limit,
                'diagnostic_requests_started': self.diagnostics.started}

    def __call__(self, *args, **kwargs):
        value = args[0].get('S2L2A') if len(args) == 1 and isinstance(args[0], dict) else None
        if not isinstance(value, self.torch.Tensor) or tuple(value.shape) != (1, *DESCRIPTOR_SHAPE):
            return self.forward(*args, **kwargs)
        with self._lock, self.torch.inference_mode():
            trace = self.diagnostics.begin()
            trace.extra.update(forward_diagnostics=True, bundle_receipt_sha256=self.worker_record['bundle_receipt_sha256'],
                               copy_bytes={'input': 8 * 12 * 224 * 224 * 4, 'output': 8 * 10 * 224 * 224 * 4})
            descriptor = slots = input_view = output_view = None
            statuscode = 1
            error_metadata = None
            try:
                if (kwargs or set(args[0]) != {'S2L2A'} or value.dtype != self.torch.float32 or not value.is_cuda
                        or value.device != self.device or os.getpid() != self.pid or self.model.training
                        or _precision(self.torch) != self.precision):
                    raise ValueError('Descriptor changed worker PID/device/eval/FP32 contract')
                with trace.span('worker_descriptor_d2h_decode'):
                    descriptor = decode_descriptor_tensor_values(value.detach().cpu().numpy()[0])
                    trace.bind(descriptor)
                statuscode = 2
                if descriptor['model'] != self.model_id:
                    raise ValueError('Descriptor model differs from verified worker')
                statuscode = 3
                with trace.span('worker_attach_maps'):
                    slots = self._slots_factory(descriptor)
                    input_view, output_view = slots._array('input'), slots._array('output')
                statuscode = 4
                if self._pipeline is None:
                    with trace.span('worker_pipeline_construct'):
                        self._pipeline = DiagnosticFullGroupGraphPipeline.from_verified_worker(
                            self.model, self.worker_record, self.verified_bundle, self.pipeline_folder,
                            original_forward=self.original_forward, torch_module=self.torch)
                        self._pipeline.diagnostic_cuda_events = self.diagnostic_cuda_events
                self._pipeline.predict_into(input_view, output_view, trace=trace)
                input_view = output_view = None
                statuscode = 5
                with trace.span('worker_close_maps'):
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
                with trace.span('worker_ack_build'):
                    output = self._ack(descriptor, 0)
                trace.finish(status='completed')
                return output
            except Exception as error:
                error_metadata = {'type': type(error).__name__, 'message': str(error)[:512]}
                _clear_frames(error)
            input_view = output_view = None
            if slots is not None:
                with trace.span('worker_failure_close_maps'):
                    try:
                        slots.close()
                    except Exception as close_error:
                        self.evidence_error = 'slot close: ' + str(close_error)[:480]
                slots = None
            self._record_failure(error_metadata, descriptor, statuscode)
            with trace.span('worker_failure_ack_build'):
                output = self._ack(descriptor, statuscode)
            if not trace.finished:
                trace.finish(status='failed', error=error_metadata)
            return output


def install_shm_engine_diagnostics(controller, diagnostics_dir, *, cuda_events=True, limit=64):
    """Replace an unused descriptor controller in an explicitly diagnostic run."""
    if os.environ.get('AUREON_PHASE5_DIAGNOSTICS') != '1':
        raise ValueError('Explicitly enable AUREON_PHASE5_DIAGNOSTICS=1 for diagnostic-only runs')
    if (type(controller) is not ShmEngineForward or type(cuda_events) is not bool
            or controller.model.forward is not controller):
        raise ValueError('Supply the installed, original descriptor controller')
    with controller._lock:
        if controller._pipeline is not None or controller.completed_calls or controller.failed_calls:
            raise ValueError('Install diagnostics before the first descriptor request')
        replacement = object.__new__(DiagnosticShmEngineForward)
        replacement.__dict__.update(controller.__dict__)
        replacement.diagnostics = DiagnosticSink(diagnostics_dir, 'worker', controller.model_id, limit=limit)
        replacement.diagnostic_cuda_events = cuda_events
        replacement._event('diagnostic-installed')
        controller.model.forward = replacement
        controller.model._aureon_shm_engine_forward = replacement
        return replacement
