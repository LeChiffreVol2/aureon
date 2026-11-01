"""Guarded resident borrowing and device-native FP32 tensors; lazy Torch import."""
from __future__ import annotations
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import math
import re
import threading
from typing import TYPE_CHECKING, Literal, cast

from .model_bridge import BridgeCancelled, ForwardAdapter, OutputAdapter, _validate_seed
from ..terramind_native.rng import rng_capsule
from ..terramind_native.encoder_loader import fp32_policy

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.cuda import Stream


class BorrowedMutation(RuntimeError):
    """Observed model mutation; the caller's state is not silently rolled back."""


@dataclass(frozen=True)
class BorrowedValidationPolicy:
    """Audit by default; serving trusts an explicitly identified adapter contract.

    The contract ID is a caller assertion, not library-issued certification.
    Serving requires exclusive, immutable module ownership until close. It checks
    tensor identity/version and metadata, not tensor bytes; unsafe .data/storage
    writes may evade it. Replace models by closing and constructing a new bridge.
    """
    mode: Literal['audit', 'serving'] = 'audit'
    exclusive_ownership: bool = False
    contract_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ('audit', 'serving') or type(self.exclusive_ownership) is not bool:
            raise ValueError('Declare audit or serving validation and boolean ownership')
        if self.contract_id is not None and (not isinstance(self.contract_id, str)
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}', self.contract_id)):
            raise ValueError('Use a bounded adapter contract identifier')
        if self.mode == 'serving' and (not self.exclusive_ownership or self.contract_id is None):
            raise ValueError('Serving requires exclusive_ownership=True and an explicit adapter contract_id')


@dataclass(frozen=True)
class DeviceTensorPolicy:
    device: str = 'cpu'
    allow_unqualified_cuda: bool = False
    max_state_bytes: int = 2 * 1024**3

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not re.fullmatch(r'cpu|cuda:(0|[1-9][0-9]*)', self.device):
            raise ValueError('Specify cpu or an explicit cuda:N device')
        if type(self.allow_unqualified_cuda) is not bool:
            raise ValueError('CUDA opt-in must be a bool')
        if type(self.max_state_bytes) is not int or self.max_state_bytes < 1:
            raise ValueError('Declare a positive model-state snapshot byte budget')

    def eligibility(self) -> dict[str, object]:
        cuda = self.device.startswith('cuda:')
        return {'device': self.device, 'admitted': not cuda or self.allow_unqualified_cuda,
                'cuda_qualified': False, 'cuda_probed': False,
                'stream_contract': 'explicit_stream_synchronized_return' if cuda else 'synchronous_cpu',
                'input_output': 'owned_clones_on_same_device', 'precision': 'float32',
                'graph': False, 'automatic_batching': False}


def _plain(value: object, depth: int = 0) -> object:
    import torch
    if depth > 16:
        raise ValueError('Model metadata exceeds the supported nesting bound')
    if value is None or type(value) in (bool, int, str):
        return (type(value).__name__, value)
    if type(value) is float and math.isfinite(value):
        return ('float', value.hex())
    if isinstance(value, (torch.dtype, torch.device)):
        return (type(value).__name__, str(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        if len(value) > 4096:
            raise ValueError('Model metadata container exceeds budget')
        entries = tuple(_plain(v, depth + 1) for v in value)
        return (type(value).__name__, tuple(sorted(entries, key=repr)) if isinstance(value, (set, frozenset)) else entries)
    if isinstance(value, dict) and len(value) <= 4096 and all(type(k) is str for k in value):
        return ('dict', tuple((k, _plain(v, depth + 1)) for k, v in sorted(value.items())))
    raise ValueError('Unsupported model attribute; register tensors as state and use plain metadata')


@dataclass
class _TensorState:
    tensor: Tensor
    identity: tuple[object, ...]
    value: Tensor | None


def _tensor_identity(tensor: Tensor) -> tuple[object, ...]:
    # Inference tensors lack a version counter and cannot satisfy this guard.
    return (id(tensor), tensor.untyped_storage().data_ptr(), tensor._version,
            tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device), tensor.requires_grad,
            id(tensor.grad) if tensor.is_leaf else None)


def _inspect(model: nn.Module, device: str, *, check_values: bool = True) -> tuple[dict[str, object], dict[str, Tensor]]:
    import torch
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Expected an eval nn.Module')
    metadata: dict[str, object] = {}
    tensors: dict[str, Tensor] = {}
    for index, (path, module) in enumerate(model.named_modules()):
        if index >= 4096 or module.training:
            raise ValueError('Require a bounded module tree already entirely in eval mode')
        attrs = vars(module)
        plain = {}
        for key, value in attrs.items():
            if key in ('_parameters', '_buffers', '_modules'):
                continue
            if 'hooks' in key and value:
                raise ValueError('Hooked modules are outside the read-only borrowing contract')
            plain[key] = _plain(value)
        metadata[path] = (id(module), id(type(module).forward), id(type(module).__call__), plain,
                          tuple((k, id(v)) for k, v in module._modules.items()),
                          tuple((k, id(v)) for k, v in module._parameters.items()),
                          tuple((k, id(v)) for k, v in module._buffers.items()))
        for category, values in (('parameter', module._parameters), ('buffer', module._buffers)):
            for key, tensor in values.items():
                if tensor is None:
                    continue
                if (str(tensor.device) != device or tensor.layout != torch.strided or tensor.is_complex()
                        or tensor.is_quantized or tensor.is_inference()
                        or (tensor.is_floating_point() and tensor.dtype != torch.float32)
                        or (check_values and not bool(torch.isfinite(tensor).all()))):
                    raise ValueError('State must be finite dense normal tensors on the declared device; floating state must be FP32')
                tensors[path + ':' + category + ':' + key] = tensor
                if category == 'parameter' and tensor.grad is not None:
                    grad = tensor.grad
                    if (str(grad.device) != device or grad.layout != torch.strided or grad.dtype != tensor.dtype
                            or grad.is_inference() or (check_values and not bool(torch.isfinite(grad).all()))):
                        raise ValueError('Parameter gradients must also satisfy the resident state contract')
                    tensors[path + ':gradient:' + key] = grad
    return metadata, tensors


@contextmanager
def _policy(device: str) -> Iterator[None]:
    import torch
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_enabled = torch.backends.cudnn.enabled
    default_device = torch.get_default_device()
    try:
        if str(default_device) != device:
            torch.set_default_device(device)
        with fp32_policy(), torch.no_grad(), torch.autocast(device.split(':')[0], enabled=False):
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.enabled = cudnn_enabled
        if torch.get_default_device() != default_device:
            torch.set_default_device(default_device)


class BorrowedModelBridge:
    """Use the exact caller model across requests under a read-only forward contract.

    Caller excludes concurrent use/mutation and supplies trusted adapters. Supported
    state is registered tensors plus plain module metadata, without hooks/custom
    mutable objects. Audit retains a baseline tensor snapshot; no module deepcopy
    occurs. Audit compares identity/version/bytes and metadata before/after.
    Explicit serving keeps identities only after one load-time finite-state scan.
    Serving does not detect writes bypassing tensor version counters.
    This detects persistent/versioned state changes, not arbitrary external Python
    side effects or transient writes hidden through unsafe tensor storage access.

    Mutation/failure makes this bridge terminal. Model changes are NOT rolled back;
    caller must inspect them. close only drops our references, never moves/casts,
    changes training mode, invokes model.close or frees caller-owned state.
    """
    def __init__(self, model: nn.Module, *, forward: ForwardAdapter, output: OutputAdapter,
                 policy: DeviceTensorPolicy = DeviceTensorPolicy(), stream: Stream | None = None,
                 validation: BorrowedValidationPolicy = BorrowedValidationPolicy()) -> None:
        import torch
        if type(policy) is not DeviceTensorPolicy or not callable(forward) or not callable(output):
            raise TypeError('Explicit device policy and trusted forward/output adapters required')
        if type(validation) is not BorrowedValidationPolicy:
            raise TypeError('Use an explicit BorrowedValidationPolicy')
        self._validation = validation
        if not policy.eligibility()['admitted']:
            raise ValueError('CUDA is unqualified; explicit allow_unqualified_cuda=True is required')
        self._policy = policy
        self._stream = stream
        self._devices: tuple[int, ...] = ()
        if policy.device == 'cpu':
            if stream is not None:
                raise ValueError('CPU bridge does not accept a CUDA stream')
        else:
            # No implicit context creation, device transfer or stream selection.
            if not cast(Callable[[], bool], torch.cuda.is_initialized)():
                raise ValueError('Caller must already own an initialized CUDA context')
            if not isinstance(stream, torch.cuda.Stream) or str(stream.device) != policy.device:
                raise ValueError('Provide a CUDA stream on the exact declared device')
            self._devices = (int(policy.device.split(':')[1]),)
        self._lock = threading.Lock()
        self._state: Literal['ready', 'failed', 'closed'] = 'ready'
        self._model: nn.Module | None = model
        self._forward: ForwardAdapter | None = forward
        self._output: OutputAdapter | None = output
        self._baseline: dict[str, _TensorState] = {}
        with rng_capsule(17, devices=self._devices), _policy(policy.device), self._stream_scope():
            self._metadata, tensors = _inspect(model, policy.device)
            if sum(t.numel() * t.element_size() for t in tensors.values()) > policy.max_state_bytes:
                raise ValueError('Model state exceeds declared baseline snapshot budget')
            with torch.inference_mode(False), torch.no_grad():
                self._baseline = {k: _TensorState(t, _tensor_identity(t),
                    t.detach().clone() if validation.mode == 'audit' else None) for k, t in tensors.items()}

    @contextmanager
    def _stream_scope(self) -> Iterator[None]:
        import torch
        if self._stream is None:
            yield
        else:
            # Caller guarantees producers used its current stream or arranged
            # explicit waits on it; foreign-stream work is outside this contract.
            self._stream.wait_stream(torch.cuda.current_stream(self._policy.device))
            try:
                with torch.cuda.stream(self._stream):
                    yield
            finally:
                self._stream.synchronize()

    def _check(self) -> None:
        import torch
        assert self._model is not None
        metadata, tensors = _inspect(self._model, self._policy.device,
                                     check_values=self._validation.mode == 'audit')
        if metadata != self._metadata or set(tensors) != set(self._baseline):
            raise BorrowedMutation('Borrowed module metadata/tree changed; caller state was not restored')
        for name, tensor in tensors.items():
            baseline = self._baseline[name]
            if (_tensor_identity(tensor) != baseline.identity
                    or (baseline.value is not None and
                        not torch.equal(tensor.detach().contiguous().reshape(-1).view(torch.uint8),
                                        baseline.value.contiguous().reshape(-1).view(torch.uint8)))):
                raise BorrowedMutation('Borrowed tensor state changed; caller state was not restored')

    def _validate(self, values: Mapping[str, Tensor]) -> None:
        import torch
        if not isinstance(values, Mapping) or not values:
            raise ValueError('Supply named tensors')
        for name, tensor in values.items():
            if (not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor)
                    or str(tensor.device) != self._policy.device or tensor.dtype != torch.float32
                    or tensor.layout != torch.strided or not bool(torch.isfinite(tensor).all())):
                raise ValueError('Inputs/outputs require finite dense FP32 tensors on the declared device')

    @property
    def state(self) -> Literal['ready', 'failed', 'closed']:
        return self._state

    def metadata(self) -> dict[str, object]:
        return {'ownership': 'borrowed_resident_read_only', 'state': self.state,
                'capabilities': self._policy.eligibility(), 'module_copies_per_request': 0,
                'baseline_bytes': sum(v.value.numel() * v.value.element_size() for v in self._baseline.values() if v.value is not None),
                'validation_mode': self._validation.mode, 'adapter_contract_id': self._validation.contract_id,
                'weight_value_scans_per_request': 4 if self._validation.mode == 'audit' else 0,
                'unsafe_tensor_writes_detected': False,
                'caller_exclusive_access_required': True, 'model_mutation_rollback': False,
                'cuda_execution_tested': False}

    def predict(self, inputs: Mapping[str, Tensor], *, seed: int = 17,
                cancel: threading.Event | None = None) -> dict[str, Tensor]:
        import torch
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Borrowed bridge is busy')
        try:
            if self._state != 'ready' or self._model is None:
                raise RuntimeError('Borrowed bridge is ' + self._state)
            assert self._forward is not None and self._output is not None
            _validate_seed(seed)
            if cancel is not None and cancel.is_set():
                raise BridgeCancelled('Cancelled before execution')
            started = False
            completed = False
            try:
                with rng_capsule(seed, devices=self._devices), _policy(self._policy.device), self._stream_scope():
                    self._validate(inputs)
                    started = True
                    self._check()
                    with torch.inference_mode(False), torch.no_grad():
                        owned = {k: v.detach().clone() for k, v in inputs.items()}
                        result = self._output(self._forward(self._model, owned))
                        self._check()
                        self._validate(result)
                        copied = {k: v.detach().clone() for k, v in result.items()}
                    completed = True
                if cancel is not None and cancel.is_set():
                    raise BridgeCancelled('Cancelled after synchronized execution; output discarded')
                return copied
            except BridgeCancelled:
                if started and not completed:
                    self._state = 'failed'
                raise
            except BaseException:
                if started:
                    self._state = 'failed'
                raise
        finally:
            self._lock.release()

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close a busy borrowed bridge')
        try:
            self._model = None
            self._forward = self._output = None
            self._stream = None
            self._baseline.clear()
            self._state = 'closed'
        finally:
            self._lock.release()

    def __enter__(self) -> BorrowedModelBridge:
        if self._state != 'ready':
            raise RuntimeError('Borrowed bridge is ' + self._state)
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
