"""Compatibility import for :mod:`aureon.planning.resource_policy`."""
import sys as _sys
from ..planning import resource_policy as _implementation
from ..planning.resource_policy import (
    DType as DType,
    DeviceCapability as DeviceCapability,
    MemoryBudget as MemoryBudget,
    MemoryCredits as MemoryCredits,
    MemoryEstimate as MemoryEstimate,
    TensorLayout as TensorLayout,
    WorkloadMemory as WorkloadMemory,
    _SIZES as _SIZES,
    _count as _count,
)

_sys.modules[__name__] = _implementation
