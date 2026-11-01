"""Compatibility import for :mod:`aureon.integrations.upstream_task`."""
import sys as _sys
from ..integrations import upstream_task as _implementation
from ..integrations.upstream_task import (
    TaskConstructor as TaskConstructor,
    TaskKind as TaskKind,
    TaskPrediction as TaskPrediction,
    UpstreamTaskBridge as UpstreamTaskBridge,
    _config as _config,
)

_sys.modules[__name__] = _implementation
