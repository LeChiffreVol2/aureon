"""Compatibility import for :mod:`aureon.integrations.borrowed_model`."""
import sys as _sys
from ..integrations import borrowed_model as _implementation
from ..integrations.borrowed_model import (
    BorrowedModelBridge as BorrowedModelBridge,
    BorrowedMutation as BorrowedMutation,
    BorrowedValidationPolicy as BorrowedValidationPolicy,
    DeviceTensorPolicy as DeviceTensorPolicy,
    _TensorState as _TensorState,
    _inspect as _inspect,
    _plain as _plain,
    _policy as _policy,
    _tensor_identity as _tensor_identity,
)

_sys.modules[__name__] = _implementation
