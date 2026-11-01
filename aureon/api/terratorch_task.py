"""Compatibility import for :mod:`aureon.integrations.terratorch_task`."""
import sys as _sys
from ..integrations import terratorch_task as _implementation
from ..integrations.terratorch_task import (
    SegmentationPrediction as SegmentationPrediction,
    TerraTorchSegmentation as TerraTorchSegmentation,
    TerraTorchTaskConfig as TerraTorchTaskConfig,
    build_terratorch_task as build_terratorch_task,
)

_sys.modules[__name__] = _implementation
