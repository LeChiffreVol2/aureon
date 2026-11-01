"""Small raster interfaces used by streaming delivery; no GDAL import."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Protocol
from ..terramind_native._types import Array, FloatArray


class CRS(Protocol):
    def to_string(self) -> str: ...


class RasterReader(Protocol):
    driver: str
    files: list[str]
    count: int
    width: int
    height: int
    dtypes: tuple[str, ...]
    crs: CRS | None
    transform: Iterable[float]
    scales: tuple[float, ...]
    offsets: tuple[float, ...]
    block_shapes: list[tuple[int, int]]
    def read(self, *indexes: int, window: object = ...) -> Array: ...
    def read_masks(self, *indexes: int, window: object = ...) -> Array: ...
    def window_transform(self, window: object) -> Iterable[float]: ...
    def close(self) -> None: ...


class RasterWriter(Protocol):
    def write(self, data: FloatArray, *, window: object) -> None: ...
    def write_mask(self, data: Array, *, window: object) -> None: ...
