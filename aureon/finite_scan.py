"""Explicit hash-bound, read-only CPU scan of contiguous little-endian FP32.

No auto-build, global NumPy patch, precision change, skipped validation or
fallback on a missing/changed build. The caller owns storage through return.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build/native/finite-v2'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

class FiniteScanner:
    def __init__(self):
        if sys.platform != 'linux' or platform.machine() != 'x86_64':
            raise ValueError('Native finite scan requires the explicit Linux x86_64 build')
        manifest = json.loads((BUILD / 'manifest.json').read_bytes())
        expected = {'source_sha256': ROOT / 'aureon/native/finite_fp32.c',
                    'library_sha256': BUILD / 'finite_fp32.so',
                    'build_script_sha256': ROOT / 'maintenance/build-finite.py'}
        if (manifest.get('schema') != 'aureon.finite_build.v1' or manifest.get('abi') != 2
                or manifest.get('fast_math') is not False
                or any(sha(path) != manifest.get(key) for key, path in expected.items())):
            raise ValueError('Native finite build identity differs; explicit rebuild required')
        self.lib = ctypes.CDLL(str(BUILD / 'finite_fp32.so'))
        for name in ('aureon_finite_abi', 'aureon_has_avx2'):
            getattr(self.lib, name).argtypes = []
            getattr(self.lib, name).restype = ctypes.c_int
        for name in ('aureon_finite_fp32', 'aureon_finite_scalar'):
            getattr(self.lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            getattr(self.lib, name).restype = ctypes.c_int
        self.lib.aureon_copy_finite_fp32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.aureon_copy_finite_fp32.restype = ctypes.c_int
        if self.lib.aureon_finite_abi() != 2: raise ValueError('Native finite ABI differs')
        self.identity = {'schema': 'aureon.finite_scan.v1', 'implementation': 'integer_exponent_scan',
            'dispatch': 'avx2' if self.lib.aureon_has_avx2() else 'scalar',
            'manifest_sha256': sha(BUILD / 'manifest.json'), **{key: manifest[key] for key in expected},
            'wrapper_sha256': sha(Path(__file__)), 'temporary_array_bytes': 0,
            'all_values_checked': True, 'floating_point_arithmetic': False}

    def __call__(self, values, *, scalar=False):
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('<f4')
                or not values.flags.c_contiguous):
            raise ValueError('Finite scan requires contiguous little-endian FP32 storage')
        function = self.lib.aureon_finite_scalar if scalar else self.lib.aureon_finite_fp32
        return bool(function(int(values.ctypes.data), values.size))

    def metadata(self): return dict(self.identity)

    def copy(self, destination, source):
        for values in (destination, source):
            if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('<f4')
                    or not values.flags.c_contiguous):
                raise ValueError('Fused copy requires contiguous little-endian FP32 arrays')
        if (destination.shape != source.shape or not destination.flags.writeable
                or np.shares_memory(source, destination)):
            raise ValueError('Fused copy requires a writable independent destination of the same shape')
        # A false result may leave a partially copied destination. The client
        # poisons the request and never sends or returns that destination.
        return bool(self.lib.aureon_copy_finite_fp32(int(destination.ctypes.data),
                    int(source.ctypes.data), source.size))
