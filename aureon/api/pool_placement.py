"""Compatibility import for :mod:`aureon.planning.pool_placement`."""
import sys as _sys
from ..planning import pool_placement as _implementation
from ..planning.pool_placement import (
    WorkerCapabilities as WorkerCapabilities,
    WorkerOffer as WorkerOffer,
    rank_worker_offers as rank_worker_offers,
)

_sys.modules[__name__] = _implementation
