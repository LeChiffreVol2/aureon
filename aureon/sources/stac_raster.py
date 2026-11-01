"""Bounded STAC raster windows over local GeoTIFFs or policy-controlled ranges.

No download-to-memory fallback, reprojection, implicit scaling or model load.
This reader deliberately accepts only the prepared TerraMind S2L2A recipe.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
from concurrent.futures import CancelledError
import hashlib
import io
from pathlib import Path
from threading import Event
from types import TracebackType
from typing import Self, cast

from ..api._raster_types import RasterReader
from ..api.memory_input import GenerationInput
from ..api.spatial import GenerationWindow, InvalidTilePolicy, PreparedGenerationRaster, TILE_SIZE
from ..api.stac import ResolvedAssetBand, ResolvedObservation
from ..terramind_native._types import BoolArray, FloatArray, Metadata
from ..terramind_native.common import INPUT_BANDS
from .http_range import HTTPRangeSource, RangePolicy


class _RangeStore:
    def __init__(self, source: HTTPRangeSource, cache_bytes: int, event: Event | None) -> None:
        self.source, self.event = source, event
        source.read_range(0, 1, cancel_event=event)
        identity = source.identity
        if identity is None:
            raise ValueError('Range source has no validated identity')
        self.size = identity.size
        self.block_size = min(64*1024, source.policy.max_range_bytes, cache_bytes)
        self.capacity = max(1, cache_bytes // self.block_size)
        self.blocks: OrderedDict[int, bytes] = OrderedDict()

    def check(self) -> None:
        if self.event is not None and self.event.is_set():
            raise CancelledError('Raster range read cancelled')

    def read(self, offset: int, size: int) -> bytes:
        self.check()
        result = bytearray()
        end = min(offset + size, self.size)
        while offset < end:
            self.check()
            start = offset // self.block_size * self.block_size
            block = self.blocks.pop(start, None)
            if block is None:
                block = self.source.read_range(start, min(self.block_size, self.size-start), cancel_event=self.event)
            self.blocks[start] = block
            while len(self.blocks) > self.capacity:
                self.blocks.popitem(last=False)
            count = min(end-offset, len(block)-(offset-start))
            result.extend(block[offset-start:offset-start+count])
            offset += count
        return bytes(result)


class _RangeFile(io.RawIOBase):
    """Seekable VSI cursor. It borrows a shared, bounded range store."""
    def __init__(self, store: _RangeStore, max_read: int) -> None:
        self.store, self.max_read, self.position = store, max_read, 0

    def readable(self) -> bool: return not self.closed
    def seekable(self) -> bool: return not self.closed
    def writable(self) -> bool: return False
    def tell(self) -> int: return self.position

    def seek(self, offset: int, whence: int=0) -> int:
        self._checkClosed()
        if whence not in (0, 1, 2):
            raise ValueError('Invalid seek origin')
        target = offset + (0 if whence == 0 else self.position if whence == 1 else self.store.size)
        if target < 0:
            raise ValueError('Negative raster seek')
        self.position = target
        return target

    def read(self, size: int=-1) -> bytes:
        self._checkClosed()
        if type(size) is not int or size < 0 or size > self.max_read:
            raise ValueError('Unbounded or oversized raster read rejected')
        result = self.store.read(self.position, size)
        self.position += len(result)
        return result


def _contained(path: Path, root: Path) -> Path:
    path = path.absolute()
    if not path.is_relative_to(root):
        raise ValueError('Local raster outside declared root')
    # Reject every reparse/symlink component, not just a final-file symlink.
    for part in (path, *path.parents):
        stat = part.lstat()
        if part.is_symlink() or getattr(stat, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Raster path contains a symlink/reparse point')
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError('Raster path must be a contained regular file')
    return path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class STACGenerationRaster:
    """An explicitly aligned S2L2A window source for ``generate_raster_scene``.

    The caller owns the context and engine. This context owns all raster/network
    handles. Remote assets need an allowlist policy and a strong ETag; supplied
    whole-file SHA expectations require a local file and are never downgraded.
    Local roots must not be writable by an untrusted concurrent process.
    """
    def __init__(self, observation: ResolvedObservation, *, local_root: Path | None=None,
                 remote_policy: RangePolicy | None=None, cancel_event: Event | None=None,
                 max_decode_bytes: int=32*1024**2, max_cache_bytes: int=8*1024**2,
                 max_remote_total_bytes: int=64*1024**2) -> None:
        if type(observation) is not ResolvedObservation:
            raise TypeError('Expected a resolved STAC observation')
        spec = observation.observation
        if (spec.modality != 'S2L2A' or spec.bands != tuple(INPUT_BANDS)
                or spec.units != 'surface_reflectance_x10000' or spec.normalization != 'none'
                or len(observation.bands) != 12
                or observation.mask_meaning != 'one_is_valid_zero_is_invalid'):
            raise ValueError('Reader requires explicit prepared S2L2A twelve-band reflectance and binary validity')
        if min(spec.grid.shape) < TILE_SIZE:
            raise ValueError('Scene dimensions must be at least 224; no implicit padding')
        if any(type(n) is not int or n <= 0 for n in (max_decode_bytes, max_cache_bytes, max_remote_total_bytes)):
            raise ValueError('Reader budgets must be positive integer bytes')
        self.resolved = observation
        self.root = None if local_root is None else Path(local_root).absolute()
        self.policy, self.cancel_event = remote_policy, cancel_event
        self.max_decode_bytes, self.max_cache_bytes = max_decode_bytes, max_cache_bytes
        self.max_remote_total_bytes = max_remote_total_bytes
        self.height, self.width = spec.grid.shape
        self.grid = spec.grid.native()
        self._stack: ExitStack | None = None
        self._datasets: dict[str, RasterReader] = {}
        self._local: dict[str, tuple[Path, str, tuple[int, int, int, int]]] = {}
        self._remote: dict[str, HTTPRangeSource] = {}

    @property
    def is_open(self) -> bool: return self._stack is not None and bool(self._datasets)

    @property
    def identity(self) -> Metadata:
        return {'stac_item_sha256': self.resolved.item_sha256,
                'source_id': self.resolved.observation.source_id,
                'acquired_at': self.resolved.observation.acquired_at,
                'band_order': list(self.resolved.observation.bands),
                'selected_bands': [{'band': band, 'asset_key': ref.asset_key, 'band_index': ref.band_index}
                                   for band, ref in zip(self.resolved.observation.bands, self.resolved.bands)],
                'valid_mask': {'asset_key': self.resolved.valid_mask.asset_key,
                               'band_index': self.resolved.valid_mask.band_index,
                               'meaning': self.resolved.mask_meaning},
                'assets': [{'asset_key': key, 'size': stats[0], 'sha256': sha,
                            'identity_scope': 'whole local file content'}
                           for key, (_, sha, stats) in self._local.items()] +
                          [{'asset_key': key, **source.provenance()} for key, source in self._remote.items()],
                'memory_policy': {'range_cache_total_bytes': self.max_cache_bytes,
                                  'gdal_cache_bytes': 64*1024**2, 'max_decode_bytes': self.max_decode_bytes},
                'radiometry': 'explicit identity scaling; uint16/int16/uint8 to FP32 is lossless',
                'dependencies': 'self-contained GeoTIFF only; no sidecar fetching'}

    @staticmethod
    def _stat(path: Path) -> tuple[int, int, int, int]:
        st = path.stat()
        return st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino

    def _check(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CancelledError('STAC raster operation cancelled')

    def __enter__(self) -> Self:
        import rasterio
        if self._stack is not None:
            raise RuntimeError('Reader already open')
        self._check()
        refs = (*self.resolved.bands, self.resolved.valid_mask)
        grouped: dict[str, ResolvedAssetBand] = {}
        for ref in refs:
            if type(ref) is not ResolvedAssetBand or type(ref.band_index) is not int or not 1 <= ref.band_index <= 64:
                raise ValueError('Invalid resolved band reference')
            existing = grouped.get(ref.asset_key)
            if existing is not None and (existing.href, existing.expected_sha256) != (ref.href, ref.expected_sha256):
                raise ValueError('Inconsistent references to the same asset')
            grouped[ref.asset_key] = ref
        if len({(r.href, r.band_index) for r in self.resolved.bands}) != 12:
            raise ValueError('Repeated source band')
        if self.resolved.valid_mask.href in {r.href for r in self.resolved.bands}:
            raise ValueError('Validity must be a distinct asset')
        remote_count = sum('://' in r.href for r in grouped.values())
        if remote_count and (self.policy is None or remote_count*self.policy.max_total_bytes > self.max_remote_total_bytes):
            raise ValueError('Explicit remote policy and aggregate payload budget required')
        if self.max_cache_bytes < max(1, remote_count):
            raise ValueError('Insufficient per-source cache budget')
        self._stack = ExitStack()
        try:
            self._stack.enter_context(rasterio.Env(GDAL_CACHEMAX=64*1024**2,
                GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_PAM_ENABLED=False))
            for ordinal, (key, ref) in enumerate(grouped.items()):
                self._check()
                if '://' in ref.href:
                    if ref.expected_sha256 is not None:
                        raise ValueError('Whole-file SHA expectation requires a local verified file; range ETag is not SHA-256')
                    source = self._stack.enter_context(HTTPRangeSource(ref.href, policy=cast(RangePolicy, self.policy)))
                    self._remote[key] = source
                    store = _RangeStore(source, self.max_cache_bytes//remote_count, self.cancel_event)
                    name = f'aureon-asset-{ordinal}.tif'
                    def opener(path: str, mode: str='rb', *, _name: str=name, _store: _RangeStore=store) -> _RangeFile:
                        if path != _name or mode not in ('r', 'rb'):
                            raise FileNotFoundError('Only the selected raster asset can be opened')
                        return _RangeFile(_store, self.max_decode_bytes)
                    ds = self._stack.enter_context(rasterio.open(name, driver='GTiff', opener=opener))
                else:
                    if self.root is None:
                        raise ValueError('Local assets require an explicit trusted root')
                    path = _contained(Path(ref.href), self.root)
                    PreparedGenerationRaster._reject_sidecars(path)
                    before = self._stat(path)
                    sha = _digest(path)
                    if ref.expected_sha256 is not None and sha != ref.expected_sha256:
                        raise ValueError('Local raster SHA-256 mismatch')
                    if self._stat(path) != before:
                        raise ValueError('Local raster changed during hashing')
                    self._local[key] = path, sha, before
                    ds = self._stack.enter_context(rasterio.open(path, driver='GTiff'))
                    PreparedGenerationRaster._reject_sidecars(path, ds)
                self._datasets[key] = ds
            for ref in refs:
                ds = self._datasets[ref.asset_key]
                if ds.driver != 'GTiff' or ds.crs is None or (ds.height, ds.width) != (self.height, self.width):
                    raise ValueError('Actual raster dimensions/driver do not match the declared grid')
                if ds.crs.to_string() != self.grid['crs'] or list(ds.transform)[:6] != self.grid['transform']:
                    raise ValueError('Actual raster CRS/transform differs from the declared grid; resample explicitly')
                if ref.band_index > ds.count:
                    raise ValueError('Selected band is absent')
                dtype = ds.dtypes[ref.band_index-1]
                allowed = ('uint8',) if ref == self.resolved.valid_mask else ('float32', 'uint16', 'int16', 'uint8')
                if dtype not in allowed:
                    raise ValueError('Unsupported raster type; cast/scale with an explicit preprocessing recipe')
                if ds.scales[ref.band_index-1] != 1. or ds.offsets[ref.band_index-1] != 0.:
                    raise ValueError('Actual raster has unapplied scales/offsets')
                # Conservative pixel-interleave bound, including unselected bands.
                if max(h*w for h, w in ds.block_shapes)*ds.count*4 > self.max_decode_bytes:
                    raise ValueError('Raster block exceeds decode budget; retile explicitly')
            for band, ref in zip(self.resolved.observation.bands, self.resolved.bands):
                ds = self._datasets[ref.asset_key]
                descriptions = getattr(ds, 'descriptions', ())
                units = getattr(ds, 'units', ())
                if descriptions and descriptions[ref.band_index-1] not in (None, '', band):
                    raise ValueError('Actual raster band description conflicts with selected band')
                if units and units[ref.band_index-1] not in (None, '', 'surface_reflectance_x10000'):
                    raise ValueError('Actual raster units conflict with the declared radiometry')
            return self
        except BaseException as exc:
            from ..api.lifecycle import close_preserving_error
            close_preserving_error(lambda: self.__exit__(type(exc), exc, exc.__traceback__), exc,
                                   context='STAC reader cleanup failed')
            raise

    def _pixels(self, row: int, col: int, height: int, width: int) -> tuple[FloatArray, BoolArray, Metadata]:
        import numpy as np
        from rasterio.windows import Window
        self._check()
        if not self.is_open:
            raise RuntimeError('Use the reader context manager')
        if (min(row, col) < 0 or not 1 <= height <= TILE_SIZE or not 1 <= width <= TILE_SIZE
                or row+height > self.height or col+width > self.width):
            raise ValueError('Window outside source')
        for key, (path, _, before) in self._local.items():
            if self._stat(_contained(path, cast(Path, self.root))) != before:
                raise ValueError('Local raster changed')
            PreparedGenerationRaster._reject_sidecars(path, self._datasets[key])
        window = Window(col, row, width, height)
        values = np.empty((12, height, width), dtype='float32')
        valid = np.ones((height, width), dtype='bool')
        for index, ref in enumerate(self.resolved.bands):
            self._check()
            ds = self._datasets[ref.asset_key]
            values[index] = ds.read(ref.band_index, window=window)
            valid &= ds.read_masks(ref.band_index, window=window) > 0
        mask_ref = self.resolved.valid_mask
        mask_ds = self._datasets[mask_ref.asset_key]
        declared = mask_ds.read(mask_ref.band_index, window=window)
        if not np.isin(declared, (0, 1)).all():
            raise ValueError('Validity raster must contain only 0/1')
        valid &= (declared == 1) & (mask_ds.read_masks(mask_ref.band_index, window=window) > 0) & np.isfinite(values).all(axis=0)
        grid = {'crs': self.grid['crs'], 'transform': list(mask_ds.window_transform(window))[:6], 'shape': [height, width]}
        return values, valid, grid

    def read_if_valid(self, window: GenerationWindow, *, invalid_policy: InvalidTilePolicy='reject') -> GenerationInput | None:
        if not isinstance(window, GenerationWindow) or invalid_policy not in ('reject', 'skip_invalid_tiles'):
            raise ValueError('Explicit generation window and invalid policy required')
        image, valid, grid = self._pixels(window.row, window.col, window.height, window.width)
        if not valid.all():
            if invalid_policy == 'skip_invalid_tiles': return None
            raise ValueError('Tile contains invalid pixels; no silent filling')
        return GenerationInput(image, bands=INPUT_BANDS, units='surface_reflectance_x10000',
                               valid_mask=valid, grid=grid, source_id=window.tile_id)

    def read(self, window: GenerationWindow) -> GenerationInput:
        return cast(GenerationInput, self.read_if_valid(window))

    def inspect_validity(self, *, cancel_event: Event | None=None) -> Metadata:
        count = 0
        for row in range(0, self.height, TILE_SIZE):
            for col in range(0, self.width, TILE_SIZE):
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError('Scene inspection cancelled')
                _, valid, _ = self._pixels(row, col, min(TILE_SIZE, self.height-row), min(TILE_SIZE, self.width-col))
                count += int(valid.sum())
        return {'source_pixels': self.height*self.width, 'source_valid_pixels': count,
                'source_invalid_pixels': self.height*self.width-count,
                'validity_definition': 'explicit mask AND raster masks AND finite values in all twelve bands'}

    def verify_unchanged(self) -> None:
        self._check()
        if not self.is_open:
            raise RuntimeError('Reader is closed')
        for key, (path, sha, before) in self._local.items():
            _contained(path, cast(Path, self.root))
            PreparedGenerationRaster._reject_sidecars(path, self._datasets[key])
            if self._stat(path) != before or _digest(path) != sha or self._stat(path) != before:
                raise ValueError('Local raster changed before delivery')
        for source in self._remote.values():
            source.read_range(0, 1, cancel_event=self.cancel_event)

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        stack, self._stack = self._stack, None
        self._datasets.clear()
        if stack is not None:
            stack.__exit__(exc_type, exc, tb)
