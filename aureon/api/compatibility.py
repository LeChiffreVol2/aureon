"""Dependency admission is distinct from exact reproducibility and GPU evidence."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
import sys
from typing import Literal

from .profiles import PROFILES

BackendProfile = Literal['borrowed-cpu', 'terramind-generation']


@dataclass(frozen=True)
class BackendCompatibility:
    backend: BackendProfile
    dependency_compatible: bool
    reasons: tuple[str, ...]
    installed: tuple[tuple[str, str | None], ...]
    reproducibility_lock: str | None
    execution_qualified: bool = False
    gpu_probed: bool = False


def inspect_backend_compatibility(backend: BackendProfile, *,
        versions: Mapping[str, str] | None = None,
        python: tuple[int, int] | None = None, platform: str | None = None) -> BackendCompatibility:
    """Read distribution metadata only; never import/install a backend.

    Optional snapshots are useful to deployment planners and CPU tests, and are
    caller assertions. They are not attestations of a subsequently started worker.
    Supported CPU bridge tuples do not relax the package's pinned install recipe.
    """
    if backend == 'borrowed-cpu':
        supported = {'torch': ('2.10.0', '2.11.0'), 'numpy': ('2.2.6',)}
        lock = None
    elif backend == 'terramind-generation':
        supported = {name: (version,) for name, version in PROFILES['generation'].items()}
        lock = 'packaging/lock-generation-linux-py312-cu128.txt'
    else:
        raise ValueError('Unknown backend compatibility profile')
    reasons = []
    py = python if python is not None else sys.version_info[:2]
    system = platform if platform is not None else sys.platform
    if py != (3, 12):
        reasons.append('python_requires_tested_3.12')
    if system not in ('linux', 'win32') or (backend == 'terramind-generation' and system != 'linux'):
        reasons.append('platform_not_supported_by_profile')
    installed: list[tuple[str, str | None]] = []
    for name, admitted in sorted(supported.items()):
        if versions is not None:
            value = versions.get(name)
        else:
            try:
                value = metadata.version(name)
            except metadata.PackageNotFoundError:
                value = None
        installed.append((name, value))
        if value is None or value.split('+')[0] not in admitted:
            reasons.append(name + '_requires_' + '_or_'.join(admitted))
    return BackendCompatibility(backend, not reasons, tuple(reasons), tuple(installed), lock)
