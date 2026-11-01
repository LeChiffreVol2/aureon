"""Shared value boundaries; importing these names never loads a neural runtime.

JsonObject is intentionally dynamic only at JSON decoding/extension metadata
boundaries. Public requests/results and numerical arrays use concrete types.
Torch tensors are annotated at their lazy third-party boundary in each module.
"""
from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypedDict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

Pathish: TypeAlias = str | PathLike[str]
FloatArray: TypeAlias = "NDArray[np.float32]"
BoolArray: TypeAlias = "NDArray[np.bool_]"
ClassArray: TypeAlias = "NDArray[np.uint8]"
Array: TypeAlias = "NDArray[np.generic]"
JsonObject: TypeAlias = dict[str, Any]


class ArrayPredictor(Protocol):
    def __call__(self, normalized_batch: FloatArray, /) -> FloatArray: ...


class HashDigest(Protocol):
    def update(self, data: bytes, /) -> None: ...
    def hexdigest(self) -> str: ...


class Closable(Protocol):
    def close(self) -> object: ...


class _RequiredParityStatistics(TypedDict):
    max_abs_error: float
    mean_abs_error: float
    allclose: bool
    atol: float
    rtol: float
    bitwise_equal: bool
    class_agreement: float
    tile_count: int
    passed: bool


class ParityStatistics(_RequiredParityStatistics, total=False):
    scene_class_map_equal: bool
    scene_mask_equal: bool
    normalization_and_tile_order_equal: bool
    warmup: _RequiredParityStatistics
    reversed_input_probe: _RequiredParityStatistics
    reference_scope: str


class RasterGrid(TypedDict):
    height: int
    width: int
    crs_wkt: str
    crs: str
    transform: list[float]
    bounds: list[float]
    area_or_point: str
    pixel_coordinates_preserved: bool
    resampling: str
    reprojection: str


WindowCoordinates: TypeAlias = tuple[int, int, int, int]


class ProcessMemoryPolicy(TypedDict):
    schema: str
    requested: str
    pid: int
    pr_get_thp_disable_before: int
    pr_get_thp_disable_after: int
    kernel: str
    system_controls_read_only: dict[str, str | dict[str, str]]
    scope: str


class StreamingBudget(TypedDict):
    scratch_estimate_bytes: int
    scratch_budget_bytes: int
    application_estimate_bytes: int
    application_budget_bytes: int
    gdal_cache_bytes: int
    maximum_group_tiles: int
    blend_rows: int
    width: int
    model_memory_included: bool
    scope: str
