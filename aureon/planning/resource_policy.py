"""Pure workload estimates and explicit lifecycle credits; never an allocator cap."""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Literal

DType = Literal['float32', 'float16', 'bfloat16', 'int64', 'bool']
_SIZES: dict[str, int] = {'float32': 4, 'float16': 2, 'bfloat16': 2, 'int64': 8, 'bool': 1}


def _count(value: int, name: str, *, zero: bool = False) -> None:
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(name + ' must be an explicit nonnegative integer' if zero else name + ' must be positive')


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, ...]
    dtype: DType = 'float32'

    def __post_init__(self) -> None:
        if type(self.shape) is not tuple or not 1 <= len(self.shape) <= 8:
            raise ValueError('Declare 1 to 8 tensor dimensions')
        for dim in self.shape:
            _count(dim, 'shape dimension')
        if self.dtype not in _SIZES:
            raise ValueError('Unsupported estimate dtype')

    @property
    def bytes(self) -> int:
        return math.prod(self.shape) * _SIZES[self.dtype]


@dataclass(frozen=True)
class MemoryEstimate:
    weights_bytes: int
    input_bytes: int
    activation_bytes: int
    state_bytes: int
    graph_bytes: int
    pinned_bytes: int
    output_bytes: int
    reserve_bytes: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _count(getattr(self, name), name, zero=True)

    @property
    def workspace_bytes(self) -> int:
        return self.input_bytes + self.activation_bytes + self.state_bytes + self.graph_bytes + self.output_bytes

    @property
    def device_bytes(self) -> int:
        return self.weights_bytes + self.workspace_bytes + self.reserve_bytes


@dataclass(frozen=True)
class WorkloadMemory:
    """Declared simultaneous tensors per lane, excluding the explicit batch axis.

    Multipliers are planning assumptions, not calibrated peak measurements.
    Graph pools are retained per lane/shape bucket and never assumed shared.
    """
    inputs: TensorLayout = TensorLayout((12, 224, 224))
    outputs: TensorLayout = TensorLayout((2, 224, 224))
    latent: TensorLayout = TensorLayout((32, 28, 28))
    tokens: TensorLayout = TensorLayout((196,), 'int64')
    batch_size: int = 1
    weights_bytes: int = 0
    activation_multiplier: int = 64
    graph_copies: int = 2
    pinned_copies: int = 2
    reserve_bytes: int = 512 * 1024**2

    def __post_init__(self) -> None:
        for name in ('inputs', 'outputs', 'latent', 'tokens'):
            if type(getattr(self, name)) is not TensorLayout:
                raise TypeError('Use explicit TensorLayout values')
        for name in ('batch_size', 'activation_multiplier'):
            _count(getattr(self, name), name)
        for name in ('weights_bytes', 'graph_copies', 'pinned_copies', 'reserve_bytes'):
            _count(getattr(self, name), name, zero=True)

    def estimate(self) -> MemoryEstimate:
        inputs = self.inputs.bytes * self.batch_size
        outputs = self.outputs.bytes * self.batch_size
        state = (self.latent.bytes + self.tokens.bytes) * self.batch_size
        activations = inputs * self.activation_multiplier
        graph = self.graph_copies * (inputs + outputs + state + activations)
        return MemoryEstimate(self.weights_bytes, inputs, activations, state, graph,
                              (inputs + outputs) * self.pinned_copies, outputs, self.reserve_bytes)


@dataclass(frozen=True)
class DeviceCapability:
    backend: Literal['cpu', 'cuda']
    total_bytes: int
    free_bytes: int
    compute_capability: tuple[int, int] | None = None
    dtypes: tuple[DType, ...] = ('float32',)
    graph: bool = False
    visible_devices: int = 1

    def __post_init__(self) -> None:
        if self.backend not in ('cpu', 'cuda') or type(self.graph) is not bool:
            raise ValueError('Declare backend and graph support')
        _count(self.total_bytes, 'total_bytes')
        _count(self.free_bytes, 'free_bytes', zero=True)
        _count(self.visible_devices, 'visible_devices')
        if self.free_bytes > self.total_bytes:
            raise ValueError('Free memory exceeds total memory')
        if type(self.dtypes) is not tuple or not self.dtypes or any(d not in _SIZES for d in self.dtypes):
            raise ValueError('Declare supported dtypes')
        if self.compute_capability is not None:
            if type(self.compute_capability) is not tuple or len(self.compute_capability) != 2:
                raise ValueError('Declare (major, minor) capability')
            for value in self.compute_capability:
                _count(value, 'compute capability', zero=True)

    def lane_rejection(self, *, dtype: DType = 'float32') -> str | None:
        if self.backend != 'cuda' or not self.graph:
            return 'backend_requires_serial_reference'
        if self.visible_devices != 1:
            return 'lane_owner_requires_one_visible_device'
        if self.compute_capability is None or self.compute_capability[0] < 3:
            return 'CUDA_graph_capability_unavailable'
        if dtype not in self.dtypes:
            return 'requested_dtype_unavailable'
        return None


@dataclass(frozen=True)
class MemoryCredits:
    device: int = 0
    pinned: int = 0
    output: int = 0

    def __post_init__(self) -> None:
        for name in ('device', 'pinned', 'output'):
            _count(getattr(self, name), name, zero=True)


class MemoryBudget:
    """Atomic accounting for caller-owned lifetimes; release only after completion.

    Hold graph/model credits until eviction/close, request credits until device
    completion, and output credits until consumption/discard. No memory is freed
    by this ledger. The allocator and other processes remain outside its control.
    """
    def __init__(self, capacity: MemoryCredits, *, max_leases: int = 4096) -> None:
        if type(capacity) is not MemoryCredits:
            raise TypeError('Declare typed memory credits')
        _count(max_leases, 'max_leases')
        self._capacity = capacity
        self._max_leases = max_leases
        self._leases: dict[int, MemoryCredits] = {}
        self._next = 0
        self._lock = threading.Lock()
        self._closed = False

    @property
    def capacity(self) -> MemoryCredits:
        return self._capacity

    def _used(self) -> MemoryCredits:
        return MemoryCredits(*(sum(getattr(v, name) for v in self._leases.values())
                               for name in ('device', 'pinned', 'output')))

    @property
    def used(self) -> MemoryCredits:
        with self._lock:
            return self._used()

    def reserve(self, credits: MemoryCredits) -> int:
        if type(credits) is not MemoryCredits:
            raise TypeError('Declare typed memory credits')
        with self._lock:
            if self._closed:
                raise RuntimeError('Memory budget is closed')
            if len(self._leases) >= self._max_leases:
                raise MemoryError('Memory lease metadata capacity exhausted')
            used = self._used()
            if any(getattr(used, k) + getattr(credits, k) > getattr(self.capacity, k)
                   for k in ('device', 'pinned', 'output')):
                raise MemoryError('Memory lifecycle credits exhausted')
            self._next += 1
            self._leases[self._next] = credits
            return self._next

    def release(self, token: int) -> None:
        with self._lock:
            if type(token) is not int or token not in self._leases:
                raise ValueError('Unknown or already released memory lease')
            del self._leases[token]

    def close(self) -> None:
        with self._lock:
            if self._leases:
                raise RuntimeError('Release live memory owners before closing the budget')
            self._closed = True
