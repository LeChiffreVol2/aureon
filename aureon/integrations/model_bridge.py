"""Explicit CPU FP32 bridges. Importing this module does not import Torch."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Literal, Protocol, cast

from ..terramind_native.encoder_loader import fp32_policy
from ..terramind_native.rng import rng_capsule

if TYPE_CHECKING:
    from torch import Tensor, nn


class ForwardAdapter(Protocol):
    def __call__(self, model: nn.Module, inputs: Mapping[str, Tensor], /) -> object: ...


class OutputAdapter(Protocol):
    def __call__(self, output: object, /) -> Mapping[str, Tensor]: ...


@dataclass(frozen=True)
class BridgeCapabilities:
    device: str = 'cpu'
    precision: str = 'highest_fp32'
    graph: bool = False
    automatic_batching: bool = False
    qualification: str = 'user_adapter_cpu_only'


class BridgeCancelled(RuntimeError):
    """Cooperative cancellation before execution or before returning results."""


def _validate_model(model: nn.Module) -> None:
    import torch
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Expected a PyTorch module')
    if any(m.training for m in model.modules()):
        raise ValueError('Borrowed model must already be in evaluation mode')
    for tensor in (*model.parameters(), *model.buffers()):
        if (tensor.device.type != 'cpu' or tensor.layout != torch.strided
                or tensor.is_complex() or tensor.is_quantized
                or (tensor.is_floating_point()
                    and (tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())))):
            raise ValueError('This bridge requires finite dense CPU FP32 model tensors; no implicit move or cast')


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Seed must be an integer in [0, 2**63)')


def _validate_tensors(value: Mapping[str, Tensor], *, label: str) -> None:
    import torch
    if (not isinstance(value, Mapping) or not value
            or any(not isinstance(k, str) or not k.strip() or not isinstance(v, torch.Tensor)
                   or v.device.type != 'cpu' or v.dtype != torch.float32
                   or v.layout != torch.strided or not bool(torch.isfinite(v).all())
                   for k, v in value.items())):
        raise ValueError(label + ' must be named finite dense CPU FP32 tensors')


def _copy_model(model: nn.Module) -> nn.Module:
    copied = deepcopy(model)
    if set(map(id, model.modules())).intersection(map(id, copied.modules())):
        raise ValueError('Model copy retained caller modules')
    originals = {t.untyped_storage().data_ptr() for t in (*model.parameters(), *model.buffers()) if t.numel()}
    if any(t.numel() and t.untyped_storage().data_ptr() in originals
           for t in (*copied.parameters(), *copied.buffers())):
        raise ValueError('Model copy retained caller tensor storage')
    return copied


class ModelBridge:
    """Snapshot an eval model without transferring ownership of the caller object.

    Caller must exclude concurrent mutation during construction. Standard
    deepcopy semantics are required; custom copy hooks/adapters are trusted code.
    Each request executes a private copy, so buffer updates and ordinary module
    attribute writes cannot escape to the caller or a subsequent request.
    No optimization eligibility is inferred from accepting an nn.Module.
    """

    def __init__(self, model: nn.Module, *, forward: ForwardAdapter,
                 output: OutputAdapter, seed: int = 17) -> None:
        if not callable(forward) or not callable(output):
            raise TypeError('forward and output must be callable adapters')
        _validate_seed(seed)
        self._lock = threading.Lock()
        self._state: Literal['ready', 'failed', 'closed'] = 'ready'
        self._ownership = 'borrowed_snapshot'
        self._identity: dict[str, object] = {}
        with rng_capsule(seed), fp32_policy():
            _validate_model(model)
            self._model: nn.Module | None = _copy_model(model)
        self._forward, self._output = forward, output

    @property
    def state(self) -> Literal['ready', 'failed', 'closed']:
        return self._state

    def metadata(self) -> dict[str, object]:
        return {'ownership': self._ownership, 'capabilities': asdict(BridgeCapabilities()),
                'checkpoint': deepcopy(self._identity), 'state': self._state}

    def predict(self, inputs: Mapping[str, Tensor], *, seed: int = 17,
                cancel: threading.Event | None = None) -> dict[str, Tensor]:
        import torch
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('ModelBridge is busy')
        try:
            if self._state != 'ready' or self._model is None:
                raise RuntimeError('ModelBridge is ' + self._state)
            if cancel is not None and cancel.is_set():
                raise BridgeCancelled('Cancelled before forward')
            _validate_seed(seed)
            _validate_tensors(inputs, label='Inputs')
            execution_started = False
            try:
                with rng_capsule(seed), fp32_policy(), torch.inference_mode(), torch.autocast('cpu', enabled=False):
                    execution_started = True
                    model = _copy_model(self._model)
                    owned = {k: v.detach().clone() for k, v in inputs.items()}
                    result = self._output(self._forward(model, owned))
                    _validate_tensors(result, label='Output adapter results')
                with torch.inference_mode(False):
                    copied = {k: v.detach().clone() for k, v in result.items()}
                if cancel is not None and cancel.is_set():
                    raise BridgeCancelled('Cancelled after forward; output discarded')
                return copied
            except BridgeCancelled:
                raise
            except BaseException:
                # A rejected global RNG reservation has not touched our model.
                # Actual execution failures can have changed private state.
                if execution_started:
                    self._state = 'failed'
                    self._model = None
                raise
        finally:
            self._lock.release()

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close a busy ModelBridge')
        try:
            self._model = None
            self._state = 'closed'
        finally:
            self._lock.release()

    def __enter__(self) -> ModelBridge:
        if self._state != 'ready':
            raise RuntimeError('ModelBridge is ' + self._state)
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @classmethod
    def from_checkpoint(cls, config: Mapping[str, object], checkpoint: str | Path,
                        *, expected_sha256: str, factory: Callable[[dict[str, object]], nn.Module],
                        forward: ForwardAdapter, output: OutputAdapter, seed: int = 17) -> ModelBridge:
        """Load a local raw state_dict strictly using an explicit trusted factory.

        Config is JSON data, not an import path or executable YAML. Factory must
        construct locally without pretrained acquisition. No remote resolver is
        supplied. Lightning checkpoint extraction must be explicit outside this API.
        """
        import torch
        _validate_seed(seed)
        if not callable(factory) or not callable(forward) or not callable(output):
            raise TypeError('factory, forward and output must be callable adapters')
        config_data = cast(dict[str, object], json.loads(json.dumps(dict(config), allow_nan=False)))
        path = Path(checkpoint)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Checkpoint must be an explicit regular local file')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != expected_sha256:
                raise ValueError('Checkpoint SHA256 mismatch')
            stream.seek(0)
            state = torch.load(stream, map_location='cpu', weights_only=True)
            stream.seek(0)
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('Checkpoint changed during read')
        with rng_capsule(seed), fp32_policy():
            model = factory(deepcopy(config_data))
            model.eval()
            _validate_model(model)
            expected = model.state_dict()
            if not isinstance(state, dict) or set(state) != set(expected):
                raise ValueError('Checkpoint keys differ from model; strict loading required')
            identities: dict[str, object] = {}
            for key, value in state.items():
                if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                        or value.shape != expected[key].shape or value.dtype != expected[key].dtype
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError('Checkpoint tensor shape/dtype mismatch: ' + str(key))
                identities[key] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                    'sha256': hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
            model.load_state_dict(state, strict=True)
            loaded = model.state_dict()
            if any(not torch.equal(loaded[k], v) for k, v in state.items()):
                raise ValueError('Loaded tensor values differ from checkpoint')
        bridge = cls(model, forward=forward, output=output, seed=seed)
        bridge._ownership = 'engine_owned_snapshot'
        bridge._identity = {'sha256': digest, 'config': config_data, 'strict': True,
                            'tensor_identity': identities, 'qualification': 'user_checkpoint_not_pretrained_qualification'}
        return bridge
