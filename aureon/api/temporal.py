"""Explicit temporal tensors and geographic conditioning; no CRS inference."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
import importlib
import importlib.metadata
import math
import os
import threading
from typing import TYPE_CHECKING, Literal, cast

from .borrowed_model import BorrowedModelBridge
from .eo import _utc
from ..terramind_native.encoder_loader import fp32_policy
from ..terramind_native.rng import rng_capsule

if TYPE_CHECKING:
    from torch import Tensor, nn


@dataclass(frozen=True)
class GeographicConditioning:
    """Explicit latitude/longitude degrees; independent of an image's CRS/grid."""
    latitude_longitude: tuple[tuple[float, float], ...]
    encoding: Literal['degrees_lat_lon', 'unit_sphere_xyz']

    def __post_init__(self) -> None:
        if type(self.latitude_longitude) is not tuple or not 1 <= len(self.latitude_longitude) <= 4096:
            raise ValueError('Provide a bounded tuple of latitude/longitude pairs')
        for pair in self.latitude_longitude:
            if (type(pair) is not tuple or len(pair) != 2
                    or any(type(v) not in (float, int) or not math.isfinite(v) for v in pair)
                    or not -90 <= pair[0] <= 90 or not -180 <= pair[1] <= 180):
                raise ValueError('Coordinates must be explicit finite latitude/longitude degrees in range')
        if self.encoding not in ('degrees_lat_lon', 'unit_sphere_xyz'):
            raise ValueError('Declare a supported coordinate encoding')

    def encoded(self) -> tuple[tuple[float, ...], ...]:
        if self.encoding == 'degrees_lat_lon':
            return tuple((float(lat), float(lon)) for lat, lon in self.latitude_longitude)
        return tuple((math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
                      math.cos(math.radians(lat)) * math.sin(math.radians(lon)), math.sin(math.radians(lat)))
                     for lat, lon in self.latitude_longitude)


@dataclass(frozen=True)
class TemporalSpec:
    bands: tuple[str, ...]
    units: str
    timestamps: tuple[str, ...]
    preprocessing: Literal['caller_prepared_fp32'] = 'caller_prepared_fp32'

    def __post_init__(self) -> None:
        if (type(self.bands) is not tuple or not 1 <= len(self.bands) <= 64
                or any(not isinstance(v, str) or not v or len(v) > 128 for v in self.bands)
                or len(set(self.bands)) != len(self.bands)):
            raise ValueError('Declare unique ordered band names')
        if not isinstance(self.units, str) or not self.units or len(self.units) > 128:
            raise ValueError('Declare input units')
        if type(self.timestamps) is not tuple or not 1 <= len(self.timestamps) <= 64:
            raise ValueError('Declare 1..64 ordered UTC timestamps')
        for stamp in self.timestamps:
            _utc(stamp)
        instants = [datetime.fromisoformat(v.replace('Z', '+00:00')) for v in self.timestamps]
        if any(a >= b for a, b in zip(instants, instants[1:])):
            raise ValueError('Timestamps must be unique and strictly chronological')
        if self.preprocessing != 'caller_prepared_fp32':
            raise ValueError('This bridge performs no inferred normalization')


class TemporalInput:
    """Owned CPU B,C,T,H,W values and explicit all-valid B,T,H,W mask."""
    def __init__(self, values: Tensor, valid_mask: Tensor, spec: TemporalSpec) -> None:
        import torch
        if type(spec) is not TemporalSpec:
            raise TypeError('TemporalSpec required')
        if (not isinstance(values, torch.Tensor) or str(values.device) != 'cpu'
                or values.dtype != torch.float32 or values.layout != torch.strided or values.ndim != 5
                or values.shape[1:3] != (len(spec.bands), len(spec.timestamps))
                or any(n < 1 for n in values.shape) or values.numel() * 4 > 256 * 1024**2):
            raise ValueError('Supply a bounded dense CPU FP32 B,C,T,H,W tensor matching the spec')
        if (not isinstance(valid_mask, torch.Tensor) or str(valid_mask.device) != 'cpu'
                or valid_mask.dtype != torch.bool or valid_mask.layout != torch.strided
                or tuple(valid_mask.shape) != (values.shape[0], values.shape[2], values.shape[3], values.shape[4])):
            raise ValueError('Supply an explicit boolean CPU B,T,H,W validity mask')
        with torch.inference_mode(False), torch.no_grad():
            self._values = values.detach().clone()
            mask = valid_mask.detach().clone()
        if not bool(torch.isfinite(self._values).all()) or not bool(mask.all()):
            raise ValueError('Nonfinite or invalid temporal samples are unsupported; no silent filling')
        self._spec = spec

    @property
    def spec(self) -> TemporalSpec:
        return self._spec

    def snapshot(self) -> Tensor:
        import torch
        with torch.inference_mode(False), torch.no_grad():
            return self._values.clone()


Pooling = Literal['mean', 'max', 'diff', 'keep', 'concat']


@dataclass(frozen=True)
class TemporalFeatures:
    values: tuple[Tensor, ...]
    timestamps: tuple[str, ...]
    pooling: Pooling
    coordinate_conditioning_used: bool = False


class TemporalBridge:
    """Actual TerraTorch1.2.13 TemporalWrapper around a borrowed CPU eval encoder.

    Encoder returns a list of spatial/token features. Timestamps label the ordered
    samples; they are not automatically passed as embeddings. GeographicConditioning
    is a separate explicit encoding utility, not secretly derived from CRS or passed
    to this wrapper. Supported pooling delegates to upstream without reimplementation.
    """
    def __init__(self, encoder: nn.Module, *, spec: TemporalSpec, pooling: Pooling,
                 subset_lengths: tuple[int, ...] | None = None) -> None:
        import torch
        if type(spec) is not TemporalSpec or pooling not in ('mean', 'max', 'diff', 'keep', 'concat'):
            raise ValueError('Explicit temporal spec and supported pooling required')
        if subset_lengths is not None and (type(subset_lengths) is not tuple or not subset_lengths
                or any(type(n) is not int or n <= 0 for n in subset_lengths)
                or sum(subset_lengths) != len(spec.timestamps)):
            raise ValueError('Temporal subsets must partition the declared frames')
        count = len(subset_lengths) if subset_lengths is not None else len(spec.timestamps)
        if pooling == 'diff' and count != 2:
            raise ValueError('Difference pooling requires exactly two frames or two subsets')
        if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('LIGHTLY_DID_VERSION_CHECK') != 'True'
                or cast(Callable[[], bool], torch.cuda.is_initialized)()):
            raise ValueError('Set CPU/offline process policy before importing upstream')
        if importlib.metadata.version('terratorch') != '1.2.13':
            raise ValueError('Temporal wrapper compatibility requires TerraTorch1.2.13')
        if any(module.training for module in encoder.modules()):
            raise ValueError('Borrowed encoder must already be entirely in eval mode')
        with rng_capsule(17), fp32_policy():
            module = importlib.import_module('terratorch.models.utils')
            constructor = cast('Callable[..., nn.Module]', module.TemporalWrapper)
            wrapper = constructor(encoder, pooling=pooling, n_timestamps=len(spec.timestamps),
                                  subset_lengths=list(subset_lengths) if subset_lengths is not None else None)
            # Only the newly owned wrapper flag changes; never call train/eval on the borrowed encoder.
            wrapper.training = False
        self._spec, self._pooling = spec, pooling
        self._bridge = BorrowedModelBridge(wrapper, forward=self._forward, output=self._output)

    @staticmethod
    def _forward(model: nn.Module, inputs: Mapping[str, Tensor], /) -> object:
        return model(inputs['image'])

    @staticmethod
    def _output(value: object, /) -> Mapping[str, Tensor]:
        import torch
        if not isinstance(value, list) or not value or any(not isinstance(v, torch.Tensor) for v in value):
            raise ValueError('Temporal encoder must return a nonempty list of tensor feature maps')
        return {str(index): tensor for index, tensor in enumerate(cast(list['Tensor'], value))}

    def predict(self, inputs: TemporalInput, *, seed: int = 17,
                cancel: threading.Event | None = None) -> TemporalFeatures:
        if not isinstance(inputs, TemporalInput) or inputs.spec != self._spec:
            raise ValueError('Temporal input semantics/timestamps differ from this bridge')
        result = self._bridge.predict({'image': inputs.snapshot()}, seed=seed, cancel=cancel)
        return TemporalFeatures(tuple(result[str(i)] for i in range(len(result))), self._spec.timestamps, self._pooling)

    def metadata(self) -> dict[str, object]:
        return {**self._bridge.metadata(), 'adapter': 'terratorch-1.2.13-temporal-wrapper',
                'pooling': self._pooling, 'timestamps': self._spec.timestamps,
                'coordinate_conditioning': False, 'normalization': 'caller_prepared_fp32',
                'pretrained_qualification': False}

    def close(self) -> None:
        self._bridge.close()

    def __enter__(self) -> TemporalBridge:
        self._bridge.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
