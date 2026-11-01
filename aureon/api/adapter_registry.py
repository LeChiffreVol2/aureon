"""Explicit trusted adapter registration and fail-closed dependency admission."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import importlib.metadata
import re
import threading
from typing import Generic, Literal, TypeVar

from .borrowed_model import DeviceTensorPolicy

Options = TypeVar('Options')
Adapter = TypeVar('Adapter')


def _name(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}', value):
        raise ValueError('Use a bounded explicit adapter/dependency identifier')


@dataclass(frozen=True)
class DependencyPin:
    distribution: str
    versions: tuple[str, ...]

    def __post_init__(self) -> None:
        _name(self.distribution)
        if type(self.versions) is not tuple or not self.versions or len(set(self.versions)) != len(self.versions):
            raise ValueError('Declare distinct admitted base versions')
        for version in self.versions:
            if not isinstance(version, str) or not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version):
                raise ValueError('Pin exact base versions; local build suffix is recorded separately')


@dataclass(frozen=True)
class AdapterSpec:
    adapter_id: str
    input_kinds: tuple[str, ...]
    output_kinds: tuple[str, ...]
    devices: tuple[Literal['cpu', 'cuda'], ...]
    dependencies: tuple[DependencyPin, ...]
    precision: Literal['float32'] = 'float32'
    max_batch_size: int = 1
    graph: bool = False

    def __post_init__(self) -> None:
        _name(self.adapter_id)
        for values in (self.input_kinds, self.output_kinds):
            if type(values) is not tuple or not values or len(set(values)) != len(values):
                raise ValueError('Declare distinct input/output kinds')
            for value in values:
                _name(value)
        if (type(self.devices) is not tuple or not self.devices or len(set(self.devices)) != len(self.devices)
                or any(v not in ('cpu', 'cuda') for v in self.devices)):
            raise ValueError('Declare CPU/CUDA support explicitly')
        if (type(self.dependencies) is not tuple or any(type(v) is not DependencyPin for v in self.dependencies)
                or len({v.distribution for v in self.dependencies}) != len(self.dependencies)):
            raise ValueError('Declare unique typed dependency pins')
        if self.precision != 'float32' or type(self.graph) is not bool:
            raise ValueError('Only explicit FP32 and boolean graph eligibility are supported')
        if type(self.max_batch_size) is not int or not 1 <= self.max_batch_size <= 1024:
            raise ValueError('Declare a batch limit in [1,1024]')


@dataclass(frozen=True)
class AdapterRequest:
    input_kind: str
    output_kind: str
    device: str = 'cpu'
    precision: str = 'float32'
    batch_size: int = 1
    graph: bool = False
    graph_fallback: Literal['error', 'eager'] = 'error'
    allow_unqualified_cuda: bool = False

    def __post_init__(self) -> None:
        _name(self.input_kind)
        _name(self.output_kind)
        DeviceTensorPolicy(self.device, self.allow_unqualified_cuda)
        if (not isinstance(self.precision, str) or type(self.batch_size) is not int or self.batch_size < 1
                or type(self.graph) is not bool or self.graph_fallback not in ('error', 'eager')):
            raise ValueError('Invalid explicit adapter admission request')


@dataclass(frozen=True)
class AdapterEligibility:
    adapter_id: str
    admitted: bool
    reasons: tuple[str, ...]
    notices: tuple[str, ...]
    resolved: AdapterRequest
    dependency_versions: tuple[tuple[str, str | None], ...]
    qualification: Literal['declarations_not_execution_qualification'] = 'declarations_not_execution_qualification'


class AdapterRegistry(Generic[Options, Adapter]):
    """Instance-local registry, with no discovery/import strings or global plugins.

    Factories are trusted explicit code and receive the admitted request plus caller
    options. They must honor it and perform no implicit acquisition. Registration
    declares capability; it does not sandbox a factory or attest GPU/model support.
    A returned adapter belongs to the caller; registry removal never closes it.
    """
    def __init__(self) -> None:
        self._entries: dict[str, tuple[AdapterSpec, Callable[[AdapterRequest, Options], Adapter]]] = {}
        self._lock = threading.Lock()

    def register(self, spec: AdapterSpec, factory: Callable[[AdapterRequest, Options], Adapter]) -> None:
        if type(spec) is not AdapterSpec or not callable(factory):
            raise TypeError('Typed adapter spec and explicit factory required')
        with self._lock:
            if spec.adapter_id in self._entries:
                raise ValueError('Adapter already registered')
            self._entries[spec.adapter_id] = spec, factory

    def unregister(self, adapter_id: str) -> None:
        with self._lock:
            if adapter_id not in self._entries:
                raise KeyError('Adapter is not registered')
            del self._entries[adapter_id]

    def specifications(self) -> tuple[AdapterSpec, ...]:
        with self._lock:
            return tuple(entry[0] for _, entry in sorted(self._entries.items()))

    @staticmethod
    def _plan(spec: AdapterSpec, request: AdapterRequest) -> AdapterEligibility:
        if type(request) is not AdapterRequest:
            raise TypeError('AdapterRequest required')
        reasons, notices = [], []
        if request.input_kind not in spec.input_kinds or request.output_kind not in spec.output_kinds:
            reasons.append('unsupported_input_or_output_kind')
        if request.device.split(':')[0] not in spec.devices:
            reasons.append('unsupported_device')
        if request.device.startswith('cuda:'):
            notices.append('cuda_execution_unqualified_and_not_probed')
            if not request.allow_unqualified_cuda:
                reasons.append('cuda_requires_explicit_unqualified_opt_in')
        if request.precision != spec.precision:
            reasons.append('unsupported_precision')
        if request.batch_size > spec.max_batch_size:
            reasons.append('batch_limit_exceeded_no_implicit_splitting')
        resolved = request
        if request.graph and not spec.graph:
            if request.graph_fallback == 'eager':
                resolved = replace(request, graph=False)
                notices.append('explicit_eager_fallback')
            else:
                reasons.append('graph_not_supported')
        versions = []
        for dependency in spec.dependencies:
            try:
                version = importlib.metadata.version(dependency.distribution)
            except importlib.metadata.PackageNotFoundError:
                version = None
            versions.append((dependency.distribution, version))
            if version is None or version.split('+')[0] not in dependency.versions:
                reasons.append('dependency_not_admitted:' + dependency.distribution)
        return AdapterEligibility(spec.adapter_id, not reasons, tuple(reasons), tuple(notices), resolved, tuple(versions))

    def plan(self, adapter_id: str, request: AdapterRequest) -> AdapterEligibility:
        with self._lock:
            if adapter_id not in self._entries:
                raise KeyError('Adapter is not registered')
            spec = self._entries[adapter_id][0]
        return self._plan(spec, request)

    def create(self, adapter_id: str, request: AdapterRequest, options: Options) -> Adapter:
        # Snapshot registration atomically. Concurrent unregister affects future
        # admissions, not a factory already admitted from this immutable snapshot.
        with self._lock:
            if adapter_id not in self._entries:
                raise KeyError('Adapter is not registered')
            spec, factory = self._entries[adapter_id]
        admitted = self._plan(spec, request)
        if not admitted.admitted:
            raise ValueError('Adapter admission rejected: ' + ','.join(admitted.reasons))
        return factory(admitted.resolved, options)

    def metadata(self) -> dict[str, object]:
        return {'adapters': [asdict(spec) for spec in self.specifications()],
                'ownership': 'caller_owned_instances', 'automatic_discovery': False,
                'qualification': 'declarations_not_execution_qualification'}
