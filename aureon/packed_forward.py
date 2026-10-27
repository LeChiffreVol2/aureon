"""Opt-in vLLM request packing with unchanged batch-one TerraTorch arithmetic."""
from dataclasses import is_dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import threading


class PackedTileForward:
    """A single engine request may carry up to eight independently computed tiles.

    The enclosing vLLM sequence batch remains one. Packed shape is [1,N,12,H,W].
    This is transfer/scheduling packing, never a larger neural-network batch.
    """

    def __init__(self, model, *, forward=None, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.model = model
        self.forward = forward if forward is not None else model.forward
        self._lock = threading.Lock()

    def __call__(self, inputs):
        torch = self.torch
        if not isinstance(inputs, dict) or set(inputs) != {'S2L2A'}:
            raise ValueError('Packed forward requires exactly S2L2A')
        value = inputs['S2L2A']
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.device.type != 'cuda' or self.model.training):
            raise ValueError('Packed forward requires eval CUDA FP32')
        if tuple(value.shape) == (1, 12, 224, 224):
            return self.forward(inputs)
        if (value.ndim != 5 or value.shape[0] != 1 or not 1 <= value.shape[1] <= 8
                or tuple(value.shape[2:]) != (12, 224, 224)):
            raise ValueError('Packed forward expects [1,N,12,224,224], N=1..8')
        with self._lock, torch.inference_mode():
            outputs = []
            first = None
            for index in range(value.shape[1]):
                result = self.forward({'S2L2A': value[:, index]})
                logits = result if isinstance(result, torch.Tensor) else getattr(result, 'output', None)
                if (not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 10, 224, 224)
                        or logits.dtype != torch.float32 or logits.device != value.device):
                    raise ValueError('Packed inner forward returned invalid logits')
                if first is None:
                    first = result
                if not isinstance(result, torch.Tensor) and (
                        not is_dataclass(result) or getattr(result, 'auxiliary_heads', None) not in (None, {})):
                    raise ValueError('Packed forward requires plain logits or a dataclass without auxiliary outputs')
                outputs.append(logits)
            joined = torch.stack(outputs, dim=1)
            if isinstance(first, torch.Tensor):
                return joined
            return replace(first, output=joined, auxiliary_heads=None)


def install_packed_forward(model, folder, *, rank=0, worker_record=None):
    if hasattr(model, '_aureon_packed_forward'):
        raise ValueError('Packed forward is already installed')
    # The graph wrapper guarantees independent cloned outputs for every tile.
    from .cuda_graph import FixedInputCudaGraph
    graph = getattr(model, '_aureon_cuda_graph', None)
    if (not isinstance(graph, FixedInputCudaGraph) or graph.model is not model
            or graph.state == 'failed'):
        raise ValueError('Packed service requires verified batch-one CUDA graph wrapper')
    controller = PackedTileForward(model, forward=graph, torch_module=graph.torch)
    model.forward = controller
    model._aureon_packed_forward = controller
    record = {'schema': 'aureon.packed_forward.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'model_id': (worker_record or {}).get('model_id'),
              'bundle_receipt_sha256': (worker_record or {}).get('bundle_receipt_sha256'),
              'neural_batch_size': 1, 'maximum_tiles_per_request': 8,
              'packed_input_shape': '[1,N,12,224,224]', 'packed_output_shape': '[1,N,10,224,224]',
              'scope': 'wrapper installation; execution and numerical equivalence require separate capture'}
    with (Path(folder)/f'packed-forward-rank{rank}.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    return controller
