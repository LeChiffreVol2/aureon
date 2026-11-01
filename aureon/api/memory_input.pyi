from collections.abc import Mapping
from typing import Any
import numpy as np
from numpy.typing import NDArray

class GenerationInput:
    def __init__(self, image: Any, *, bands: list[str] | tuple[str, ...], units: str,
                 valid_mask: Any, normalization: str = ..., grid: Mapping[str, Any] | None = ...,
                 source_id: str | None = ...) -> None: ...
    @classmethod
    def from_torchgeo(cls, sample: Mapping[str, Any], *, bands: list[str] | tuple[str, ...],
                      units: str, valid_mask: Any, normalization: str = ...,
                      grid: Mapping[str, Any] | None = ..., source_id: str | None = ...) -> GenerationInput: ...
    @property
    def identity(self) -> dict[str, Any]: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
    def snapshot(self) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, Any], dict[str, Any]]: ...
