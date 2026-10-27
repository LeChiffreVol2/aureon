"""Opt-in fixed-input CUDA graph for the verified TerraTorch eval model.

No capture occurs on import or installation. The first explicit forward after
vLLM startup performs side-stream warmup and capture. Every call copies fresh
input, replays the graph and clones its result. This is a performance candidate;
actual KAO-TerraMind parity and timing require separate GPU measurements.

Reference: https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs
"""

from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from time import perf_counter


INPUT_SHAPE = (1, 12, 224, 224)
OUTPUT_SHAPE = (1, 10, 224, 224)


def _precision(torch):
    return {
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
        'cudnn_sdp_enabled': bool(torch.backends.cuda.cudnn_sdp_enabled()),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'flash_sdp_enabled': bool(torch.backends.cuda.flash_sdp_enabled()),
        'mem_efficient_sdp_enabled': bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        'math_sdp_enabled': bool(torch.backends.cuda.math_sdp_enabled()),
    }


class FixedInputCudaGraph:
    """Callable installed as model.forward without changing its state_dict.

    Only FP32 CUDA S2L2A batch-one inference is admitted. Capture/replay errors
    fail closed and remain recorded; there is no silent eager fallback. A CPU
    lock plus stream dependencies protect shared graph buffers. Output tensors
    are cloned after every replay so callers never receive reusable storage.
    """

    def __init__(self, model, folder, *, rank=0, worker_record=None, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        if model.training:
            raise ValueError('CUDA graph requires an eval-mode model')
        self.torch = torch_module
        self.model = model
        self.original_forward = model.forward
        self.folder = Path(folder).resolve(strict=True)
        if not self.folder.is_dir():
            raise ValueError('CUDA graph evidence folder must already exist')
        if type(rank) is not int or rank < 0:
            raise ValueError('CUDA graph rank must be a nonnegative integer')
        self.rank = rank
        self.worker_record = dict(worker_record or {})
        self.precision = _precision(self.torch)
        if (self.precision['float32_matmul_precision'] != 'highest'
                or any(self.precision[key] for key in ('matmul_allow_tf32', 'cudnn_allow_tf32',
                                                     'cudnn_sdp_enabled', 'cudnn_benchmark'))):
            raise ValueError('CUDA graph requires the verified highest-FP32 precision contract')
        self.state = 'armed'
        self.capture_count = 0
        self.replay_count = 0
        self._event_count = 0
        self._lock = threading.Lock()
        self._graph = self._static_input = self._static_result = None
        self._capture_stream = self._last_stream = None
        self._device = None
        self._error = None
        self._event('armed', capture_phase='first_explicit_forward_after_vllm_startup')

    def _event(self, event, **extra):
        self._event_count += 1
        value = {
            'schema': 'aureon.cuda_graph_event.v1', 'event': event,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'rank': self.rank,
            'status': self.state, 'capture_count': self.capture_count, 'replay_count': self.replay_count,
            'model_id': self.worker_record.get('model_id'),
            'bundle_receipt_sha256': self.worker_record.get('bundle_receipt_sha256'),
            'loaded_tensor_manifest_matches': self.worker_record.get('loaded_tensor_manifest_matches'),
            'input_shape': list(INPUT_SHAPE), 'output_shape': list(OUTPUT_SHAPE),
            'input_key': 'S2L2A', 'dtype': 'float32', 'precision_settings_actual': self.precision,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'eager_fallback': False, 'output_storage': 'independent_clone_after_every_replay',
            'output_cache': False, 'numerical_parity_verified': False,
            'scope': 'local capture/replay execution evidence; separate numerical comparison required',
            **extra,
        }
        name = f'cuda-graph-rank{self.rank}-{self._event_count:04d}-{event}.json'
        with (self.folder / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')

    def _input(self, args, kwargs):
        if kwargs or len(args) != 1 or not isinstance(args[0], dict) or set(args[0]) != {'S2L2A'}:
            raise ValueError('CUDA graph accepts one positional dictionary containing S2L2A only')
        value = args[0]['S2L2A']
        if (not isinstance(value, self.torch.Tensor) or tuple(value.shape) != INPUT_SHAPE
                or value.dtype != self.torch.float32 or not value.is_cuda):
            raise ValueError('CUDA graph requires CUDA float32 S2L2A [1,12,224,224]')
        if self._device is not None and value.device != self._device:
            raise ValueError('CUDA graph input device changed')
        return value

    def _logits(self, result):
        if isinstance(result, self.torch.Tensor):
            output = result
        elif (is_dataclass(result) and {item.name for item in fields(result)} == {'output', 'auxiliary_heads'}
              and result.auxiliary_heads in (None, {})):
            output = result.output
        else:
            raise ValueError('CUDA graph supports tensor logits or TerraTorch ModelOutput without auxiliary heads')
        if (not isinstance(output, self.torch.Tensor) or tuple(output.shape) != OUTPUT_SHAPE
                or output.dtype != self.torch.float32 or not output.is_cuda or output.device != self._device):
            raise ValueError('CUDA graph requires CUDA float32 logits [1,10,224,224]')
        return output

    def _capture(self, value):
        torch = self.torch
        self.state = 'capturing'
        self._device = value.device
        self._static_input = value.detach().clone()
        current = torch.cuda.current_stream(self._device)
        self._capture_stream = torch.cuda.Stream(device=self._device)
        self._capture_stream.wait_stream(current)
        began = perf_counter()
        self._event('capture-started', warmup_forwards=3)
        with torch.cuda.stream(self._capture_stream):
            for _ in range(3):
                self._logits(self.original_forward({'S2L2A': self._static_input}))
        # Warmup finishes before capture; this synchronization is outside all
        # graph work and outside subsequent replay-only requests.
        self._capture_stream.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, stream=self._capture_stream, capture_error_mode='global'):
            self._static_result = self.original_forward({'S2L2A': self._static_input})
            self._logits(self._static_result)
        current.wait_stream(self._capture_stream)
        self.capture_count += 1
        self.state = 'captured'
        self._last_stream = current
        self._event('captured', warmup_forwards=3, warmup_and_capture_host_seconds=perf_counter() - began)

    def __call__(self, *args, **kwargs):
        with self._lock, self.torch.inference_mode():
            # Device identity can become fixed during the first capture on
            # another caller. Validate only after taking the same buffer lock.
            value = self._input(args, kwargs)
            if self.state == 'failed':
                raise RuntimeError('CUDA graph is disabled after a recorded failure; restart without the option') from self._error
            try:
                if self.model.training or _precision(self.torch) != self.precision:
                    raise RuntimeError('Model evaluation or precision settings changed after graph installation')
                if self._graph is None:
                    self._capture(value)
                current = self.torch.cuda.current_stream(self._device)
                if self._last_stream is not None and current != self._last_stream:
                    current.wait_stream(self._last_stream)
                self._static_input.copy_(value)
                self._graph.replay()
                cloned = self._logits(self._static_result).clone()
                self._last_stream = current
                self.replay_count += 1
                if self.replay_count == 1:
                    # Confirm the first replay completed before recording active
                    # status. Later calls incur no receipt I/O or synchronization.
                    current.synchronize()
                    self.state = 'active'
                    self._event('first-replay-completed')
                if isinstance(self._static_result, self.torch.Tensor):
                    return cloned
                auxiliary = None if self._static_result.auxiliary_heads is None else {}
                return replace(self._static_result, output=cloned, auxiliary_heads=auxiliary)
            except Exception as exc:
                self.state = 'failed'
                self._error = exc
                self._event('failed', error_type=type(exc).__name__, error=str(exc))
                raise RuntimeError('CUDA graph capture/replay failed; eager fallback is disabled') from exc


def install_cuda_graph(model, folder, *, rank=0, worker_record=None, torch_module=None):
    """Install only after verification and startup, with no immediate GPU work."""
    if hasattr(model, '_aureon_cuda_graph'):
        raise ValueError('CUDA graph wrapper is already installed')
    controller = FixedInputCudaGraph(model, folder, rank=rank, worker_record=worker_record, torch_module=torch_module)
    model.forward = controller
    model._aureon_cuda_graph = controller
    return controller
