"""Explicit dependency tuples; version labels alone do not qualify arbitrary stacks."""
from __future__ import annotations

import importlib.metadata
from typing import Any

_PROFILES = {
    'torch260': {
        'torch': '2.6.0+cu124', 'torchvision': '0.21.0+cu124',
        'numpy': '1.26.4', 'rasterio': '1.4.3', 'einops': '0.8.1',
        'ml-collections': '1.0.0', 'lmdb': '1.6.2', 'absl-py': '2.1.0',
        'pillow': '11.1.0', 'tqdm': '4.67.1', 'PyYAML': '6.0.3',
    },
    'torch210': {
        'torch': '2.10.0', 'torchvision': '0.25.0',
        'numpy': '2.2.6', 'rasterio': '1.5.2', 'einops': '0.8.2',
        'ml-collections': '1.0.0', 'lmdb': '1.6.2', 'absl-py': '2.5.1',
        'pillow': '13.0.0', 'tqdm': '4.70.1', 'PyYAML': '6.0.3',
    },
}
_BUILDS = {'torch260': ('2.6.0+cu124', '12.4'), 'torch210': ('2.10.0+cu128', '12.8')}


def profile_requirements(profile: str) -> dict[str, str]:
    """Return an independent copy without importing Torch or touching CUDA."""
    if profile not in _PROFILES:
        raise ValueError('Choose explicit COP-GEN dependency_profile torch260 or torch210')
    return dict(_PROFILES[profile])


def validate_profile(profile: str, *, torch_version: str, cuda_build: str | None) -> dict[str, Any]:
    """Fail before CUDA admission unless every direct dependency matches."""
    expected = profile_requirements(profile)
    actual: dict[str, str] = {}
    for name in expected:
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = '<missing>'
    mismatches = {name: {'expected': version, 'actual': actual[name]}
                  for name, version in expected.items() if actual[name] != version}
    if mismatches or (torch_version, cuda_build) != _BUILDS[profile]:
        raise RuntimeError(f'Pinned profile {profile} mismatch: dependencies={mismatches}; '
                           f'Torch build={(torch_version, cuda_build)} expected={_BUILDS[profile]}')
    return {'name': profile, 'dependencies': actual, 'torch_module_version': torch_version,
            'cuda_build': cuda_build, 'exact_direct_tuple': True}
