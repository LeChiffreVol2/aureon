"""Compatibility import for :mod:`aureon.integrations.model_bridge`."""
import sys as _sys
from ..integrations import model_bridge as _implementation
from ..integrations.model_bridge import (
    BridgeCancelled as BridgeCancelled,
    BridgeCapabilities as BridgeCapabilities,
    ForwardAdapter as ForwardAdapter,
    ModelBridge as ModelBridge,
    OutputAdapter as OutputAdapter,
    _copy_model as _copy_model,
    _validate_model as _validate_model,
    _validate_seed as _validate_seed,
    _validate_tensors as _validate_tensors,
)

_sys.modules[__name__] = _implementation
