"""Explicit EO requests and supported-route dispatch; no model or I/O on import."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import re
from typing import Literal

from .contracts import ExecutionProfile, GenerationResult
from .generation_bridges import GenerationRecipe
from .generation_engine import GenerationEngine
from .memory_input import GenerationInput
from ..terramind_native.common import INPUT_BANDS
from ..terramind_native.families import GENERATION_MODELS


class UnsupportedRoute(ValueError):
    """The declared request has no implemented adapter; nothing is executed."""


def _label(value: str, name: str) -> None:
    if (not isinstance(value, str) or '://' in value
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/ -]{0,255}', value)):
        raise ValueError(name + ' must be a bounded plain identifier, not a URL or credential')


def _bands(value: tuple[str, ...]) -> None:
    if type(value) is not tuple or not 1 <= len(value) <= 64 or len(set(value)) != len(value):
        raise ValueError('bands must be an explicit unique tuple of 1..64 names')
    for band in value:
        _label(band, 'band')


def _utc(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z', value):
        raise ValueError('Timestamp must be an explicit UTC RFC3339 instant ending in Z')
    datetime.fromisoformat(value.replace('Z', '+00:00'))


@dataclass(frozen=True)
class EOGrid:
    """Declared pixel grid; no inferred CRS, reprojection or resampling.

    This planner admits explicit EPSG identifiers syntactically. The existing
    GenerationInput validates the CRS using Rasterio before execution.
    """
    crs: str
    transform: tuple[float, float, float, float, float, float]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.crs, str) or not re.fullmatch(r'EPSG:[1-9][0-9]{0,6}', self.crs):
            raise ValueError('Supply an explicit EPSG CRS identifier')
        if (type(self.shape) is not tuple or len(self.shape) != 2
                or any(type(n) is not int or not 1 <= n <= 100000 for n in self.shape)):
            raise ValueError('Grid shape must be two positive integer dimensions <=100000')
        if (type(self.transform) is not tuple or len(self.transform) != 6
                or any(type(n) not in (float, int) or not math.isfinite(n) for n in self.transform)):
            raise ValueError('Grid transform needs six finite affine a,b,c,d,e,f values')
        a, b, _, d, e, _ = self.transform
        determinant = a * e - b * d
        if not math.isfinite(determinant) or determinant == 0:
            raise ValueError('Grid transform is singular or overflows')

    def native(self) -> dict[str, object]:
        return {'crs': self.crs, 'transform': list(self.transform), 'shape': list(self.shape)}


@dataclass(frozen=True)
class ConditioningSpec:
    modality: str
    bands: tuple[str, ...]
    units: str
    normalization: str
    mask_policy: str
    grid: EOGrid
    source_id: str
    acquired_at: str | None = None

    def __post_init__(self) -> None:
        for name in ('modality', 'units', 'normalization', 'mask_policy', 'source_id'):
            _label(getattr(self, name), name)
        _bands(self.bands)
        if type(self.grid) is not EOGrid:
            raise TypeError('Conditioning requires its own EOGrid')
        if self.acquired_at is not None:
            _utc(self.acquired_at)


@dataclass(frozen=True)
class GeneratedTarget:
    modality: str
    bands: tuple[str, ...]
    units: str
    grid: EOGrid
    output_kind: Literal['raster'] = 'raster'

    def __post_init__(self) -> None:
        _label(self.modality, 'modality')
        _label(self.units, 'units')
        _bands(self.bands)
        if type(self.grid) is not EOGrid or self.output_kind != 'raster':
            raise ValueError('This contract describes raster targets with their own explicit grid')


@dataclass(frozen=True)
class SamplingRecipe:
    """Adapter-specific recipe identity and values, never a universal noise schedule."""
    adapter: str
    seed: int
    parameters: tuple[tuple[str, int | float], ...]

    def __post_init__(self) -> None:
        _label(self.adapter, 'adapter')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        if type(self.parameters) is not tuple or len(self.parameters) > 32:
            raise ValueError('parameters must be a bounded tuple of named numeric values')
        names = []
        for pair in self.parameters:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError('Each parameter must be a name/value tuple')
            name, value = pair
            _label(name, 'parameter')
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Sampling values must be finite numbers, not bools')
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError('Duplicate sampling parameter')
        object.__setattr__(self, 'parameters', tuple(sorted(self.parameters)))

    @classmethod
    def terramind(cls, recipe: GenerationRecipe) -> SamplingRecipe:
        recipe.native()
        return cls('terramind-token-s1grd-v1', recipe.seed, (
            ('token_steps', recipe.token_steps), ('diffusion_steps', recipe.diffusion_steps),
            ('temperature', recipe.temperature), ('top_p', recipe.top_p), ('top_k', recipe.top_k)))


@dataclass(frozen=True)
class EORequest:
    model_id: str
    conditioning: tuple[ConditioningSpec, ...]
    targets: tuple[GeneratedTarget, ...]
    sampling: SamplingRecipe
    execution_profile: ExecutionProfile = 'recommended'

    def __post_init__(self) -> None:
        _label(self.model_id, 'model_id')
        for values, cls in ((self.conditioning, ConditioningSpec), (self.targets, GeneratedTarget)):
            if type(values) is not tuple or not 1 <= len(values) <= 16 or any(type(v) is not cls for v in values):
                raise ValueError('Conditioning/targets require bounded typed tuples')
            if len({v.modality for v in values}) != len(values):
                raise ValueError('Duplicate modality; temporal stacks require a distinct adapter')
        if type(self.sampling) is not SamplingRecipe or self.execution_profile not in ('reference', 'recommended'):
            raise ValueError('Explicit sampling recipe and supported execution profile required')


@dataclass(frozen=True)
class EOCapability:
    model_id: str
    adapter: str
    conditioning: tuple[str, ...]
    targets: tuple[str, ...]
    grid_policy: Literal['same_224_grid'] = 'same_224_grid'
    precision: Literal['float32'] = 'float32'
    execution: Literal['existing_sdk_route'] = 'existing_sdk_route'
    qualification: Literal['not_established_by_planner'] = 'not_established_by_planner'


def eo_capabilities() -> tuple[EOCapability, ...]:
    return tuple(EOCapability(model, 'terramind-token-s1grd-v1', ('S2L2A',), ('S1GRD',))
                 for model in sorted(GENERATION_MODELS))


def _native_recipe(recipe: SamplingRecipe) -> GenerationRecipe:
    values = dict(recipe.parameters)
    if recipe.adapter != 'terramind-token-s1grd-v1' or set(values) != {
            'token_steps', 'diffusion_steps', 'temperature', 'top_p', 'top_k'}:
        raise UnsupportedRoute('No adapter for the declared sampling recipe')
    for key in ('token_steps', 'diffusion_steps', 'top_k'):
        if type(values[key]) is not int:
            raise ValueError(key + ' must be an integer')
    return GenerationRecipe(token_steps=int(values['token_steps']), diffusion_steps=int(values['diffusion_steps']),
                            temperature=float(values['temperature']), top_p=float(values['top_p']),
                            top_k=int(values['top_k']), seed=recipe.seed)


@dataclass(frozen=True)
class EORoutePlan:
    request: EORequest
    capability: EOCapability
    request_sha256: str


def plan_eo(request: EORequest) -> EORoutePlan:
    """Resolve an implemented route without reading assets or probing hardware."""
    if type(request) is not EORequest:
        raise TypeError('request must be EORequest')
    capabilities = {entry.model_id: entry for entry in eo_capabilities()}
    if request.model_id not in capabilities:
        raise UnsupportedRoute('No implemented generation adapter for this model')
    capability = capabilities[request.model_id]
    if (tuple(v.modality for v in request.conditioning) != capability.conditioning
            or tuple(v.modality for v in request.targets) != capability.targets):
        raise UnsupportedRoute('Implemented route requires exactly observed S2L2A and generated S1GRD')
    observed, target = request.conditioning[0], request.targets[0]
    if (observed.bands != tuple(INPUT_BANDS) or observed.units != 'surface_reflectance_x10000'
            or observed.normalization != 'none' or observed.mask_policy != 'explicit_all_valid'):
        raise UnsupportedRoute('S2L2A requires exact raw bands/units and an explicit all-valid mask')
    if target.bands != ('VV', 'VH') or target.units != 'dB':
        raise UnsupportedRoute('Generated S1GRD requires VV,VH in upstream unstandardized dB domain')
    if observed.grid.shape != (224, 224) or target.grid != observed.grid:
        raise UnsupportedRoute('Current adapter requires matching 224x224 grids; no implicit resampling')
    _native_recipe(request.sampling)
    identity = hashlib.sha256(json.dumps(asdict(request), sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
    return EORoutePlan(request, capability, identity)


def generate_eo(engine: GenerationEngine, request: EORequest, input_data: GenerationInput, *,
                job_id: str | None = None, attempt_id: str = '1') -> GenerationResult:
    """Borrow a caller-owned engine; delegate execution and lifecycle to its SDK.

    No scheduler, RNG, graph, normalization or grid conversion is implemented here.
    Engine errors (including busy/closed) propagate. This helper never closes it.
    """
    plan = plan_eo(request)
    if not isinstance(engine, GenerationEngine) or engine.model != request.model_id:
        raise ValueError('Engine model must match the planned route')
    if not isinstance(input_data, GenerationInput):
        raise TypeError('Supply an already validated, owned GenerationInput')
    observed = request.conditioning[0]
    metadata = input_data.metadata
    if (metadata['bands'] != list(observed.bands) or metadata['units'] != observed.units
            or metadata['normalization'] != observed.normalization
            or metadata['source_id'] != observed.source_id or metadata['grid'] != observed.grid.native()):
        raise ValueError('Input identity/semantics/grid differ from conditioning declaration')
    result = engine.generate(input_data, recipe=_native_recipe(request.sampling),
                             execution_profile=request.execution_profile, job_id=job_id, attempt_id=attempt_id)
    # Keep SDK-owned arrays and validity semantics unchanged; detach added metadata.
    from copy import deepcopy
    output = result.copy()
    output['provenance'] = deepcopy(result['provenance'])
    output['provenance']['eo_route'] = {
        'schema': 'aureon.eo_route.v1', 'request_sha256': plan.request_sha256,
        'adapter': plan.capability.adapter, 'conditioning': [asdict(v) for v in request.conditioning],
        'generated_targets': [asdict(v) for v in request.targets],
        'input_identity': input_data.identity, 'sampling': asdict(request.sampling),
        'sensor_observation': False, 'scientific_quality': 'not_evaluated',
        'output_validity': 'sdk_result_valid_mask_not_observation_truth'}
    return output
