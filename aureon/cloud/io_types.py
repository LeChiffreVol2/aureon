"""The untyped rasterio import boundary used for bounded upload inspection.

Only dataset dimensions and context-manager cleanup are available here; image
decoding/normalization belongs to the separately qualified raster pipeline.
"""
from contextlib import AbstractContextManager
from typing import Protocol

from ..serving.contracts import PathInput


class RasterDimensions(Protocol):
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def count(self) -> int: ...


class RasterioInspection(Protocol):
    def open(self, fp: PathInput, *, driver: str) -> AbstractContextManager[RasterDimensions]: ...
