"""Owned, validated inputs for TerraMind generation; no model or runtime imports.

NumPy is imported on construction only. A CPU Torch tensor is accepted when Torch
is already imported by the caller. This module never downloads or opens imagery.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ..terramind_native._types import Array, Metadata
if TYPE_CHECKING:
    from torch import Tensor

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
import sys
from typing import Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

from ..terramind_native.common import INPUT_BANDS, digest


def _array(value: object, name: str, shape: tuple[int, ...]) -> Array:
    import numpy as np

    if np.ma.isMaskedArray(value):  # type: ignore[no-untyped-call]  # NumPy masked-array helper lacks annotations.
        raise ValueError(f"{name} must not be a masked array; supply plain values and an explicit validity mask")
    if isinstance(value, np.ndarray):
        array = value
    else:
        torch = sys.modules.get("torch")
        if torch is None or not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a NumPy array or CPU torch.Tensor")
        if cast("Tensor", value).device.type != "cpu":
            raise ValueError(f"{name} must be on CPU; explicitly transfer it before admission")
        if tuple(cast("Tensor", value).shape) != shape or cast("Tensor", value).layout != torch.strided:
            raise ValueError(f"{name} must be a dense array with shape {shape}")
        # detach does not modify the caller's tensor or its autograd graph.
        try:
            array = cast("Tensor", value).detach().numpy()
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"{name} must use a NumPy-compatible dtype and layout") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; batches are not admitted by this single-input API")
    return array


def _grid(value: object) -> Metadata | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - {"crs", "transform", "shape"}:
        raise ValueError("grid needs crs and a six-value affine transform, with optional shape")
    if not {"crs", "transform"}.issubset(value):
        raise ValueError("A bounding box alone is insufficient; provide grid crs and transform")
    shape = value.get("shape", (224, 224))
    if not isinstance(shape, (tuple, list)) or list(shape) != [224, 224]:
        raise ValueError("grid shape must match the 224x224 input")
    transform = value["transform"]
    if (not isinstance(transform, (list, tuple)) or len(transform) != 6
            or not all(type(x) in (int, float) and math.isfinite(x) for x in transform)):
        raise ValueError("grid transform must contain six finite numbers in affine a,b,c,d,e,f order")
    determinant = transform[0] * transform[4] - transform[1] * transform[3]
    if not math.isfinite(determinant) or determinant == 0:
        raise ValueError("grid transform is singular or overflows")
    crs = value["crs"]
    if not isinstance(crs, str) or not crs.strip() or len(crs) > 16384:
        raise ValueError("grid crs must be an explicit CRS string")
    from rasterio.crs import CRS  # Optional upstream typing is declared in the implementation-checker boundary policy.
    try:
        crs = CRS.from_user_input(crs).to_string()
    except Exception as exc:
        raise ValueError("grid crs is invalid") from exc
    return {"crs": crs, "transform": [float(x) for x in transform], "shape": [224, 224]}


class GenerationInput:
    """Snapshot one all-valid S2L2A crop without file I/O or model construction.

    ``image`` must be FP32 CHW (12, 224, 224), in surface reflectance x10000,
    in the declared exact band order, before model standardization. A separate
    validity mask is mandatory; training labels are never used as that mask.
    ``grid`` is optional for array output and required for georeferenced export.
    The constructor makes owned immutable backing buffers; ``snapshot`` returns
    independent writable copies for a request. Caller mutation cannot alter an
    already admitted input, nor can request mutation alter a later invocation.
    """
    __slots__ = ("_image", "_mask", "_metadata", "_identity")

    def __init__(self, image: Any, *, bands: list[str] | tuple[str, ...], units: str,
                 valid_mask: Any, normalization: str = "none", grid: Mapping[str, Any] | None = None,
                 source_id: str | None = None) -> None:
        import numpy as np

        if not isinstance(bands, (list, tuple)) or list(bands) != INPUT_BANDS:
            raise ValueError("Declare the exact S2L2A band order: " + " ".join(INPUT_BANDS))
        if units not in ("surface_reflectance_x10000", "reflectance_x10000"):
            raise ValueError("Use surface_reflectance_x10000; stored DN or reflectance 0..1 must be explicitly converted first")
        if normalization != "none":
            raise ValueError("Pass unstandardized input with normalization='none'; the model owns standardization")
        if source_id is not None and (not isinstance(source_id, str) or not source_id.strip()
                                      or len(source_id) > 512 or "\x00" in source_id):
            raise ValueError("source_id must be a nonempty string of at most 512 characters")
        image = _array(image, "image", (12, 224, 224))
        if image.dtype != np.float32:
            raise ValueError("image must be float32; no implicit precision conversion is performed")
        # Own the bytes before value checks, hashing and request admission.
        image_bytes = image.tobytes(order="C")
        image = np.frombuffer(image_bytes, dtype=np.float32).reshape(12, 224, 224)
        if not np.isfinite(image).all():
            raise ValueError("image contains nonfinite values")
        mask = _array(valid_mask, "valid_mask", (224, 224))
        if mask.dtype not in (np.dtype("bool"), np.dtype("uint8")):
            raise ValueError("valid_mask must be bool or uint8 with 0/1 values")
        mask_bytes = mask.tobytes(order="C")
        mask = np.frombuffer(mask_bytes, dtype=mask.dtype).reshape(224, 224)
        if not np.all(mask == 1):
            raise ValueError("This TerraMind profile requires an explicit all-valid mask; nodata/cloud filling is unsupported")
        mask = np.frombuffer(mask.astype(np.bool_).tobytes(), dtype=np.bool_).reshape(224, 224)
        metadata = {
            "schema": "aureon.generation_memory_input.v1", "modality": "S2L2A",
            "bands": list(INPUT_BANDS), "units": "surface_reflectance_x10000",
            "normalization": "none", "standardization_owner": "upstream_model",
            "grid": _grid(grid), "source_id": source_id,
            "image": {"shape": [12, 224, 224], "dtype": "float32", "bytes": image.nbytes,
                      "sha256": hashlib.sha256(image_bytes).hexdigest()},
            "mask": {"shape": [224, 224], "dtype": "bool", "bytes": mask.nbytes,
                     "sha256": hashlib.sha256(mask.tobytes()).hexdigest(), "meaning": "true_is_valid"},
        }
        self._image, self._mask = image, mask
        self._metadata = metadata
        self._identity = {"kind": "memory", "schema": metadata["schema"], "sha256": digest(metadata),
                          "bytes": image.nbytes + mask.nbytes}

    @classmethod
    def from_torchgeo(cls, sample: Mapping[str, Any], *, bands: list[str] | tuple[str, ...],
                      units: str, valid_mask: Any, normalization: str = "none",
                      grid: Mapping[str, Any] | None = None, source_id: str | None = None) -> GenerationInput:
        """Adapt one sample's image with explicit semantics, without a TorchGeo dependency.

        The caller supplies a validity mask separately. ``sample['mask']`` may be
        training labels, and ``bounds``/``crs`` do not necessarily identify an
        exact raster transform, so these fields are never silently interpreted.
        This is not an arbitrary DataLoader/batch or transform compatibility API.
        """
        if not isinstance(sample, Mapping) or "image" not in sample:
            raise ValueError("Provide one TorchGeo-style sample mapping containing 'image'")
        return cls(sample["image"], bands=bands, units=units, valid_mask=valid_mask,
                   normalization=normalization, grid=grid, source_id=source_id)

    @property
    def identity(self) -> dict[str, Any]:
        return deepcopy(self._identity)

    @property
    def metadata(self) -> dict[str, Any]:
        return deepcopy(self._metadata)

    def snapshot(self) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, Any], dict[str, Any]]:
        """Return owned image/mask arrays, metadata and identity for one request."""
        return self._image.copy(), self._mask.copy(), self.metadata, self.identity

