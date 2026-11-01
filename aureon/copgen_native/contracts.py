"""One explicit native route using the shared EO request types."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from ..api.eo import ConditioningSpec, EOGrid, EORequest, UnsupportedRoute

MODEL = 'copgen-base'
ADAPTER = 'copgen-latent-s1rtc-fp32-v1'
INPUT = 'S2L2A_B02_B03_B04_B08'
TARGET = 'S1RTC_vh_vv'
BANDS = ('B04', 'B03', 'B02', 'B08')


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CopgenPlan:
    request: EORequest
    request_sha256: str
    seed: int
    sample_steps: int = 50


def plan_copgen(request: EORequest) -> CopgenPlan:
    if type(request) is not EORequest:
        raise TypeError('Use the shared EORequest')
    if request.model_id != MODEL or len(request.conditioning) != 1 or len(request.targets) != 1:
        raise UnsupportedRoute('COP-GEN adapter supports one S2L2A to S1RTC route')
    observed, target = request.conditioning[0], request.targets[0]
    if (observed.modality != 'S2L2A' or observed.bands != BANDS
            or observed.units != 'surface_reflectance_x10000' or observed.normalization != 'none'
            or observed.mask_policy != 'explicit_all_valid' or observed.grid.shape != (192, 192)):
        raise UnsupportedRoute('Declare raw FP32 B04,B03,B02,B08 on an explicit all-valid 192 grid')
    if (target.modality != 'S1RTC' or target.bands != ('VV', 'VH')
            or target.units != 'linear_backscatter' or target.grid != observed.grid):
        raise UnsupportedRoute('Target must be synthetic S1RTC VV,VH linear backscatter on the same grid')
    if (request.sampling.adapter != ADAPTER or dict(request.sampling.parameters) != {'sample_steps': 50}
            or type(dict(request.sampling.parameters).get('sample_steps')) is not int
            or not 0 <= request.sampling.seed < 2**32 or request.execution_profile != 'reference'):
        raise UnsupportedRoute('Only explicit FP32 reference, 50 native solver evaluations and uint32 seed are admitted')
    return CopgenPlan(request, digest(asdict(request)), request.sampling.seed)


class CopgenInput:
    """Owned immutable raw optical bytes and explicit validity/grid; no resampling."""
    def __init__(self, image: Any, *, valid_mask: Any, bands: tuple[str, ...], units: str,
                 grid: EOGrid, source_id: str) -> None:
        import numpy as np
        if bands != BANDS or units != 'surface_reflectance_x10000':
            raise ValueError('Declare exact COP-GEN raw optical bands and units')
        if type(grid) is not EOGrid or grid.shape != (192, 192):
            raise ValueError('An explicit 192x192 EOGrid is required')
        # Reuse the shared identifier policy rather than inventing another label.
        ConditioningSpec('S2L2A',BANDS,units,'none','explicit_all_valid',grid,source_id)
        import rasterio.crs  # Optional Rasterio has no typing marker.
        import rasterio.errors
        try:rasterio.crs.CRS.from_user_input(grid.crs)
        except rasterio.errors.CRSError as error:
            raise ValueError('Input CRS cannot be resolved') from error
        if not isinstance(image, np.ndarray) or image.shape != (4,192,192) or image.dtype != np.float32:
            raise ValueError('Input must be a (4,192,192) FP32 NumPy array')
        if not isinstance(valid_mask,np.ndarray) or valid_mask.shape != (192,192) or valid_mask.dtype != np.bool_:
            raise ValueError('Supply a separate bool validity mask')
        self._image = image.tobytes(order='C'); self._mask = valid_mask.tobytes(order='C')
        if not np.isfinite(np.frombuffer(self._image,np.float32)).all() or not np.frombuffer(self._mask,np.bool_).all():
            raise ValueError('Input must be finite and explicitly all-valid')
        self._metadata = {'bands':list(BANDS),'units':units,'normalization':'none','grid':grid.native(), 'source_id':source_id,
                          'image_sha256':hashlib.sha256(self._image).hexdigest(),'mask_sha256':hashlib.sha256(self._mask).hexdigest(),
                          'bytes':len(self._image)+len(self._mask)}

    @property
    def identity(self) -> dict[str, Any]:
        return {'sha256':digest(self._metadata), **deepcopy(self._metadata)}

    def snapshot(self) -> tuple[Any, Any]:
        import numpy as np
        return (np.frombuffer(self._image,np.float32).reshape(4,192,192).copy(),
                np.frombuffer(self._mask,np.bool_).reshape(192,192).copy())

    def validate_plan(self, plan: CopgenPlan) -> None:
        observed=plan.request.conditioning[0]
        if self._metadata['grid'] != observed.grid.native() or self._metadata['source_id'] != observed.source_id:
            raise ValueError('Input identity/grid differs from the declared observation')
