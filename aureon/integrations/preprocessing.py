"""Exactly-once explicit tensor preprocessing; no inferred units/statistics."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class PreparedTensor:
    tensor: Tensor
    bands: tuple[str, ...]
    units: str
    transform_id: str


@dataclass(frozen=True)
class PreprocessingRecipe:
    bands: tuple[str, ...]
    input_units: str
    output_units: str
    scale: float = 1.0
    offset: float = 0.0
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    workflow: Literal['custom', 'generation_s2l2a', 'encoder_s2l2a'] = 'custom'

    def __post_init__(self) -> None:
        if not isinstance(self.bands, tuple) or not self.bands or len(set(self.bands)) != len(self.bands):
            raise ValueError('Declare unique immutable ordered bands')
        if (any(not isinstance(b, str) or not b.strip() for b in self.bands)
                or any(not isinstance(u, str) or not u.strip() for u in (self.input_units, self.output_units))):
            raise ValueError('Bands and units must be explicit')
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (self.scale, self.offset)) or self.scale == 0):
            raise ValueError('Scaling must be finite and nonzero')
        if (self.mean is None) != (self.std is None):
            raise ValueError('Declare both mean and std or neither')
        if self.mean is not None and self.std is not None:
            if (not isinstance(self.mean, tuple) or not isinstance(self.std, tuple)
                    or len(self.mean) != len(self.bands) or len(self.std) != len(self.bands)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                           for v in self.mean + self.std) or any(v <= 0 for v in self.std)):
                raise ValueError('Normalization requires one finite mean and positive std per output band')
        if self.input_units == 'standardized' and (self.output_units != 'standardized'
                or self.mean is not None or self.scale != 1 or self.offset != 0):
            raise ValueError('Standardized input cannot be normalized again or relabelled as raw values')
        if self.workflow not in ('custom', 'generation_s2l2a', 'encoder_s2l2a'):
            raise ValueError('Unknown workflow')
        if self.workflow != 'custom':
            full = ('B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12')
            if self.bands != full:
                raise ValueError('TerraMind S2L2A requires the full pretrained band order')
            if self.workflow == 'generation_s2l2a' and (self.output_units != 'surface_reflectance_x10000' or self.mean is not None):
                raise ValueError('Generation receives raw reflectance x10000; upstream owns normalization')
            if self.workflow == 'encoder_s2l2a' and self.output_units != 'standardized':
                raise ValueError('Backbone/TiM inputs must be explicitly standardized')
            if self.workflow == 'encoder_s2l2a' and self.mean is None and self.input_units != 'standardized':
                raise ValueError('Declare normalization statistics for raw encoder inputs')

    def apply(self, tensor: Tensor | PreparedTensor, *, bands: tuple[str, ...], units: str) -> PreparedTensor:
        import torch
        if isinstance(tensor, PreparedTensor):
            raise ValueError('Already prepared; double normalization is forbidden')
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tensor.layout != torch.strided or tensor.device.type != 'cpu'
                or tensor.ndim != 4 or any(size <= 0 for size in tensor.shape)
                or tensor.shape[1] != len(bands) or not bool(torch.isfinite(tensor).all())):
            raise ValueError('Expected finite CPU FP32 B,C,H,W input')
        if units != self.input_units or len(set(bands)) != len(bands) or set(bands) != set(self.bands):
            raise ValueError('Input units/bands differ from explicit recipe')
        if units == 'standardized' and (self.mean is not None or self.scale != 1 or self.offset != 0):
            raise ValueError('Standardized input cannot be normalized again')
        # Results stay usable in downstream code even if the caller is currently
        # inside inference_mode. The recipe never takes over caller device policy.
        with torch.inference_mode(False), torch.no_grad(), torch.autocast('cpu', enabled=False):
            value = tensor.detach()[:, [bands.index(b) for b in self.bands]].clone()
            if self.scale != 1:
                value = value * self.scale
            if self.offset != 0:
                value = value + self.offset
            if self.mean is not None and self.std is not None:
                mean = torch.tensor(self.mean, dtype=torch.float32, device='cpu').view(1,-1,1,1)
                std = torch.tensor(self.std, dtype=torch.float32, device='cpu').view(1,-1,1,1)
                value = (value - mean) / std
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Preprocessing produced nonfinite values')
        return PreparedTensor(value, self.bands, self.output_units, repr(self))
