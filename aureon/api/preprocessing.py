"""Compatibility import for :mod:`aureon.integrations.preprocessing`."""
import sys as _sys
from ..integrations import preprocessing as _implementation
from ..integrations.preprocessing import (
    PreparedTensor as PreparedTensor,
    PreprocessingRecipe as PreprocessingRecipe,
)

_sys.modules[__name__] = _implementation
