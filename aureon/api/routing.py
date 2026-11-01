"""Compatibility import for :mod:`aureon.planning.routing`."""
import sys as _sys
from ..planning import routing as _implementation
from ..planning.routing import (
    ExecutionKey as ExecutionKey,
    PlacementRequest as PlacementRequest,
    WorkerSnapshot as WorkerSnapshot,
    _nonnegative as _nonnegative,
    plan_placement as plan_placement,
)

_sys.modules[__name__] = _implementation
