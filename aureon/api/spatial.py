"""Bounded spatial delivery over the existing single-crop generation contract.

The tiled recipe is explicit: independent 224px generations, coordinate-derived
seeds, row-major FP32 uniform blending, no padding or invalid-pixel filling.
This is not full-scene model equivalence or a scientific-quality qualification.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, cast, Self, TypeAlias, Literal, Protocol, runtime_checkable
from collections.abc import Iterable, Iterator, Sequence, Generator
from threading import Event
from ._raster_types import RasterReader, RasterWriter
from .contracts import GenerationResult
from ..terramind_native._types import Metadata, PathInput, FloatArray, BoolArray
if TYPE_CHECKING:
    from .generation_engine import GenerationEngine
    from .generation_bridges import GenerationJob

TileResult: TypeAlias = tuple["GenerationWindow", int, Metadata, GenerationResult | None]
InvalidTilePolicy: TypeAlias = Literal['reject', 'skip_invalid_tiles']
from dataclasses import dataclass
from concurrent.futures import CancelledError
from collections import deque
from contextlib import contextmanager, ExitStack
from types import TracebackType
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from time import perf_counter

from .memory_input import GenerationInput
from .v1 import ExecutionRequired
from ..terramind_native.common import INPUT_BANDS, binding, write_json


TILE_SIZE = 224


@dataclass(frozen=True)
class GenerationWindow:
    row: int
    col: int
    height: int = TILE_SIZE
    width: int = TILE_SIZE

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in (self.row,self.col)):
            raise ValueError("Window offsets must be nonnegative integers")
        if self.height != TILE_SIZE or self.width != TILE_SIZE:
            raise ValueError("Generation windows must be 224x224")

    @property
    def tile_id(self) -> str:
        return f"r{self.row}-c{self.col}"


def _axis(length: int, stride: int) -> Iterator[int]:
    last = length - TILE_SIZE
    yield from range(0, last + 1, stride)
    if last % stride:
        yield last


def generation_windows(height: int, width: int, *, stride: int=TILE_SIZE) -> Iterator[GenerationWindow]:
    """Lazy complete window cover; terminal windows anchor to the source edge."""
    if any(type(n) is not int or n < TILE_SIZE for n in (height, width)):
        raise ValueError("Scene dimensions must be integers >=224; no implicit padding")
    if type(stride) is not int or not 1 <= stride <= TILE_SIZE:
        raise ValueError("stride must be an integer in 1..224")
    for row in _axis(height, stride):
        for col in _axis(width, stride):
            yield GenerationWindow(row, col)


def generation_tile_seed(seed: int, window: GenerationWindow) -> int:
    """Stable under reorder/chunking; origin preserves the single-crop seed."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if not isinstance(window, GenerationWindow):
        raise TypeError("window requires GenerationWindow")
    if window.row == 0 and window.col == 0:
        return seed
    encoded = f"aureon.spatial.seed.v1:{seed}:{window.row}:{window.col}".encode('ascii')
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], 'big') % (2**31)


@runtime_checkable
class GenerationRasterSource(Protocol):
    """Open, aligned source owned by the caller; readers perform their own I/O policy."""
    height: int
    width: int
    grid: Metadata
    @property
    def is_open(self) -> bool: ...
    @property
    def identity(self) -> Metadata: ...
    def read(self, window: GenerationWindow) -> GenerationInput: ...
    def read_if_valid(self, window: GenerationWindow, *, invalid_policy: InvalidTilePolicy='reject') -> GenerationInput | None: ...
    def inspect_validity(self, *, cancel_event: Event | None=None) -> Metadata: ...
    def verify_unchanged(self) -> None: ...


class PreparedGenerationRaster:
    """Persistent handles for explicit aligned FP32 S2L2A and validity raster.

    Source radiometry must already be prepared. Inputs must be self-contained
    GeoTIFFs: external masks, overviews, PAM metadata and world files are rejected
    so the two recorded file identities cover every consumed raster dependency.
    This class does not infer units, apply cloud correction, resample, download
    data or load a model.
    """
    _image: RasterReader | None
    _mask: RasterReader | None
    _environment: ExitStack | None
    grid: Metadata

    @property
    def is_open(self) -> bool:
        return self._image is not None

    def __init__(self, image_path: PathInput, mask_path: PathInput, *, bands: Sequence[str], units: str, max_decode_bytes: int=32*1024**2) -> None:
        if list(bands) != INPUT_BANDS or units != 'surface_reflectance_x10000':
            raise ValueError("Explicit S2L2A band order and surface_reflectance_x10000 required")
        if type(max_decode_bytes) is not int or max_decode_bytes <= 0:
            raise ValueError("max_decode_bytes must be a positive integer")
        self.image_path, self.mask_path = Path(image_path).resolve(strict=True), Path(mask_path).resolve(strict=True)
        self.bands, self.units, self.max_decode_bytes = list(bands), units, max_decode_bytes
        self._image = self._mask = None
        self._environment = None

    def __enter__(self) -> Self:
        import rasterio  # Optional upstream typing is declared in the implementation-checker boundary policy.
        if self._image is not None:
            raise RuntimeError("Source already open")
        try:
            # Own the cache policy before opening any raster. Preflight and
            # standalone inspection need the same bound as reconstruction.
            self._environment = ExitStack()
            self._environment.enter_context(rasterio.Env(GDAL_CACHEMAX=64*1024**2))
            self._reject_sidecars(self.image_path)
            self._reject_sidecars(self.mask_path)
            self.identity = {'image': binding(self.image_path), 'valid_mask': binding(self.mask_path)}
            self._image = self._environment.enter_context(rasterio.open(self.image_path))
            self._mask = self._environment.enter_context(rasterio.open(self.mask_path))
            ds, ms = cast(RasterReader, self._image), cast(RasterReader, self._mask)
            self._reject_sidecars(self.image_path, ds)
            self._reject_sidecars(self.mask_path, ms)
            if ds.count != 12 or ds.dtypes != ('float32',)*12 or ds.crs is None:
                raise ValueError("Expected georeferenced FP32 twelve-band S2L2A")
            if ms.count != 1 or ms.dtypes != ('uint8',):
                raise ValueError("Validity mask must be a separate one-band uint8 0/1 raster")
            if (ds.width, ds.height, ds.crs, ds.transform) != (ms.width, ms.height, ms.crs, ms.transform):
                raise ValueError("Image/mask grids must match exactly")
            if any(x != 1. for x in ds.scales) or any(x != 0. for x in ds.offsets):
                raise ValueError("Input TIFF has unapplied scales/offsets")
            if any(x != 1. for x in ms.scales) or any(x != 0. for x in ms.offsets):
                raise ValueError("Mask TIFF has scales/offsets")
            if max(h*w for h, w in ds.block_shapes)*12*4 > self.max_decode_bytes:
                raise ValueError("Source block exceeds decode budget; prepare tiled storage explicitly")
            if max(h*w for h, w in ms.block_shapes) > self.max_decode_bytes:
                raise ValueError("Mask block exceeds decode budget")
            self.height, self.width = ds.height, ds.width
            next(generation_windows(self.height, self.width))
            self.grid = {'crs': ds.crs.to_string(), 'transform': list(ds.transform)[:6], 'shape':[ds.height,ds.width]}
            self._stats = (self._stat(self.image_path), self._stat(self.mask_path))
            return self
        except BaseException as exc:
            from .lifecycle import close_preserving_error
            close_preserving_error(lambda: self.__exit__(type(exc), exc, exc.__traceback__), exc,
                context='Prepared raster cleanup failed')
            raise

    @staticmethod
    def _stat(path: Path) -> tuple[int, int]:
        st = path.stat()
        return st.st_size, st.st_mtime_ns

    @staticmethod
    def _reject_sidecars(path: Path, dataset: RasterReader | None=None) -> None:
        # Dataset.files catches dependencies reported by the driver. The known
        # companion names also catch files added after a handle has cached its
        # dependency list, before the next read and again before delivery.
        companions = {Path(str(path)+suffix) for suffix in ('.msk','.ovr','.aux.xml','.aux')}
        companions.update(path.with_suffix(suffix) for suffix in ('.tfw','.tifw','.wld','.aux','.aux.xml','.rrd'))
        if any(p != path and (p.exists() or p.is_symlink()) for p in companions):
            raise ValueError('Prepared raster must be a self-contained GeoTIFF; external sidecars are unsupported')
        if dataset is not None:
            if dataset.driver != 'GTiff' or {Path(p).resolve() for p in dataset.files} != {path}:
                raise ValueError('Prepared raster must be a self-contained GeoTIFF with no external dependencies')

    def _pixels(self, row: int, col: int, height: int, width: int) -> tuple[FloatArray, BoolArray, Metadata]:
        import numpy as np
        from rasterio.windows import Window  # Optional upstream typing is declared in the implementation-checker boundary policy.
        if self._image is None:
            raise RuntimeError("Open source with its context manager")
        if (row < 0 or col < 0 or height < 1 or width < 1 or height > TILE_SIZE or width > TILE_SIZE
                or row+height > self.height or col+width > self.width):
            raise ValueError("Window outside source")
        if self._stats != (self._stat(self.image_path), self._stat(self.mask_path)):
            raise ValueError("Input changed during generation")
        self._reject_sidecars(self.image_path, self._image)
        self._reject_sidecars(self.mask_path, self._mask)
        w = Window(col, row, width, height)
        image = self._image.read(window=w)
        declared = cast(RasterReader, self._mask).read(1, window=w)
        if not np.isin(declared, (0,1)).all():
            raise ValueError("Mask values must be 0 or 1")
        valid = ((declared == 1) & (self._image.read_masks(window=w)>0).all(axis=0)
                 & (cast(RasterReader, self._mask).read_masks(1,window=w)>0)
                 & np.isfinite(image).all(axis=0))
        grid = {'crs':self.grid['crs'], 'transform':list(self._image.window_transform(w))[:6], 'shape':[height,width]}
        return cast(FloatArray, image), valid, grid

    def read_if_valid(self, window: GenerationWindow, *, invalid_policy: InvalidTilePolicy='reject') -> GenerationInput | None:
        """Never fill invalid values; opt-in skipping omits the entire window."""
        if invalid_policy not in ('reject', 'skip_invalid_tiles'):
            raise ValueError("invalid_policy must be reject or skip_invalid_tiles")
        if not isinstance(window, GenerationWindow):
            raise TypeError("Expected GenerationWindow")
        image, valid, grid = self._pixels(window.row, window.col, window.height, window.width)
        if not valid.all() and invalid_policy == 'skip_invalid_tiles':
            return None
        if not valid.all():
            raise ValueError("This tiled recipe rejects invalid pixels; no silent cloud/nodata filling")
        return GenerationInput(image, bands=self.bands, units=self.units, valid_mask=valid,
            grid=grid, source_id=window.tile_id)

    def read(self, window: GenerationWindow) -> GenerationInput:
        return cast(GenerationInput, self.read_if_valid(window))

    def inspect_validity(self, *, cancel_event: Event | None=None) -> Metadata:
        """Scan unique source pixels with at most one 224px decoded window live."""
        valid_pixels = 0
        for row in range(0, self.height, TILE_SIZE):
            for col in range(0, self.width, TILE_SIZE):
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError("Scene cancelled during input inspection")
                _, valid, _ = self._pixels(row, col, min(TILE_SIZE, self.height-row), min(TILE_SIZE, self.width-col))
                valid_pixels += int(valid.sum())
        total = self.height*self.width
        return {'source_pixels':total, 'source_valid_pixels':valid_pixels,
                'source_invalid_pixels':total-valid_pixels,
                'validity_definition':'explicit mask AND raster masks AND finite values in all twelve bands'}

    def verify_unchanged(self) -> None:
        self._reject_sidecars(self.image_path, self._image)
        self._reject_sidecars(self.mask_path, self._mask)
        if {'image':binding(self.image_path), 'valid_mask':binding(self.mask_path)} != self.identity:
            raise ValueError("Source identity changed before delivery")

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        stack, self._environment = self._environment, None
        self._image = self._mask = None
        if stack is not None:
            # Dataset contexts close first, then the enclosing Env restores the
            # caller's GDAL cache configuration, including failure unwinding.
            stack.__exit__(exc_type, exc, tb)


class _StripeReconstruction:
    """One 224-row stripe, flushed only when no later tile can contribute."""
    def __init__(self, dataset: RasterWriter, width: int, max_buffer_bytes: int, *, allow_uncovered: bool=False) -> None:
        import numpy as np
        self.dataset, self.width, self.row = dataset, width, 0
        self.allow_uncovered, self.covered_pixels = allow_uncovered, 0
        # Sum/count + maximum division/mask temporary; excludes GDAL and model.
        self.estimated_buffer_bytes = (8 if allow_uncovered else 6)*TILE_SIZE*width*4
        if type(max_buffer_bytes) is not int or self.estimated_buffer_bytes > max_buffer_bytes:
            raise ValueError("Reconstruction stripe exceeds max_buffer_bytes")
        self.sums = np.zeros((2,TILE_SIZE,width), dtype=np.float32)
        self.count = np.zeros((TILE_SIZE,width), dtype=np.float32)
        self.last_key: tuple[int, int] | None = None

    def _flush(self, rows: int) -> None:
        import numpy as np
        from rasterio.windows import Window
        count = self.count[:rows]
        covered = count > 0
        if not self.allow_uncovered and not np.all(covered):
            raise ValueError("Uncovered output pixels")
        if self.allow_uncovered:
            values = np.full_like(self.sums[:,:rows], np.nan)
            np.divide(self.sums[:,:rows], count[None], out=values, where=covered[None])
        else:
            values = self.sums[:,:rows]/count[None]
        finite = np.isfinite(values[:,covered]).all() if self.allow_uncovered else np.isfinite(values).all()
        if not finite:
            raise ValueError("Nonfinite reconstructed pixels")
        output_window = Window(0,self.row,self.width,rows)
        self.dataset.write(values, window=output_window)
        if self.allow_uncovered:
            self.dataset.write_mask(covered.astype('uint8')*255, window=output_window)
        self.covered_pixels += int(covered.sum())
        if rows < TILE_SIZE:
            self.sums[:,:TILE_SIZE-rows] = self.sums[:,rows:]
            self.count[:TILE_SIZE-rows] = self.count[rows:]
        self.sums[:,TILE_SIZE-rows:] = 0
        self.count[TILE_SIZE-rows:] = 0
        self.row += rows

    def _advance(self, window: GenerationWindow) -> None:
        key = (window.row, window.col)
        if self.last_key is not None and key <= self.last_key:
            raise ValueError("Reconstruction requires unique row-major windows")
        if window.row > self.row:
            if window.row - self.row > TILE_SIZE:
                raise ValueError("Gap in reconstruction")
            self._flush(window.row-self.row)
        self.last_key = key

    def skip(self, window: GenerationWindow) -> None:
        if not self.allow_uncovered:
            raise ValueError("Skipping is not enabled for this reconstruction")
        self._advance(window)

    def add(self, window: GenerationWindow, values: FloatArray) -> None:
        import numpy as np
        if values.shape != (2,TILE_SIZE,TILE_SIZE) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Expected finite FP32 VV/VH generation")
        self._advance(window)
        self.sums[:,:,window.col:window.col+TILE_SIZE] += values
        self.count[:,window.col:window.col+TILE_SIZE] += 1

    def finish(self, height: int) -> None:
        if height-self.row != TILE_SIZE:
            raise ValueError("Incomplete final reconstruction stripe")
        self._flush(TILE_SIZE)


@contextmanager
def _scene_results(engine: GenerationEngine, source: GenerationRasterSource, windows: Iterable[GenerationWindow], seed: int, scene_id: str, max_in_flight: int, cancel_event: Event | None, times: dict[str, float], invalid_policy: InvalidTilePolicy='reject') -> Iterator[Iterator[TileResult]]:
    """Join ordered queued results to bounded metadata without retaining inputs."""
    from .generation_bridges import GenerationJob, GenerationRecipe
    if invalid_policy == 'skip_invalid_tiles':
        with _masked_scene_results(engine, source, windows, seed, scene_id, max_in_flight, cancel_event, times) as generated:
            yield generated
        return
    pending: deque[tuple[GenerationWindow, int, Metadata]] = deque()
    def jobs() -> Generator[GenerationJob, None, None]:
        for window in windows:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("Scene cancelled between tile generations")
            tick = perf_counter()
            item = source.read(window)
            times['prepare_seconds'] += perf_counter()-tick
            tile_seed = generation_tile_seed(seed,window)
            pending.append((window,tile_seed,item.metadata))
            yield GenerationJob(item,recipe=GenerationRecipe(seed=tile_seed),job_id=scene_id+'-'+window.tile_id)
    descriptors = jobs()
    def serial() -> Iterator[TileResult]:
        for job in descriptors:
            tick = perf_counter()
            result=engine.generate(job.input_data,seed=job.recipe.seed,job_id=job.job_id)
            times['generate_seconds'] += perf_counter()-tick
            yield (*pending.popleft(),result)
    try:
        if max_in_flight == 1:
            yield serial()
        else:
            with engine.generate_many(descriptors,max_in_flight=max_in_flight) as results:
                def joined() -> Iterator[TileResult]:
                    iterator=iter(results)
                    while True:
                        tick=perf_counter()
                        try: result=next(iterator)
                        except StopIteration: break
                        times['generate_seconds'] += perf_counter()-tick
                        yield (*pending.popleft(),result)
                yield joined()
    finally:
        descriptors.close()
        pending.clear()


@contextmanager
def _masked_scene_results(engine: GenerationEngine, source: GenerationRasterSource, windows: Iterable[GenerationWindow], seed: int, scene_id: str, max_in_flight: int, cancel_event: Event | None, times: dict[str, float]) -> Iterator[Iterator[TileResult]]:
    """Bound descriptors even when an entire large scene has no eligible tile."""
    from itertools import islice
    from .generation_bridges import GenerationJob, GenerationRecipe

    def generate() -> Generator[TileResult, None, None]:
        iterator = iter(windows)
        while True:
            batch = list(islice(iterator, max_in_flight))
            if not batch:
                return
            entries: list[tuple[GenerationWindow, int, Metadata, GenerationJob | None]] = []
            for window in batch:
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError("Scene cancelled before reading another tile")
                tick = perf_counter()
                item = source.read_if_valid(window, invalid_policy='skip_invalid_tiles')
                times['prepare_seconds'] += perf_counter()-tick
                tile_seed = generation_tile_seed(seed, window)
                job = None if item is None else GenerationJob(item, recipe=GenerationRecipe(seed=tile_seed), job_id=scene_id+'-'+window.tile_id)
                entries.append((window, tile_seed, {} if item is None else item.metadata, job))
                item = None
            jobs = [entry[3] for entry in entries if entry[3] is not None]
            if not jobs:
                for window, tile_seed, metadata, _ in entries:
                    yield window, tile_seed, metadata, None
            elif max_in_flight == 1:
                job = jobs[0]
                tick = perf_counter()
                result = engine.generate(job.input_data, seed=job.recipe.seed, job_id=job.job_id)
                times['generate_seconds'] += perf_counter()-tick
                yield entries[0][0], entries[0][1], entries[0][2], result
                del result
            else:
                with engine.generate_many(jobs, max_in_flight=max_in_flight) as results:
                    result_iterator = iter(results)
                    for window, tile_seed, metadata, job in entries:
                        if job is None:
                            yield window, tile_seed, metadata, None
                        else:
                            tick = perf_counter()
                            result = next(result_iterator)
                            times['generate_seconds'] += perf_counter()-tick
                            yield window, tile_seed, metadata, result
                            del result
                    try:
                        next(result_iterator)
                    except StopIteration:
                        pass
                    else:
                        raise RuntimeError("Generation stream returned an unexpected extra result")
            entries.clear()
            jobs.clear()
            job = None
    iterator = generate()
    try:
        yield iterator
    finally:
        iterator.close()


def generate_raster_scene(engine: GenerationEngine, source: GenerationRasterSource, output: PathInput, *, seed: int=17, stride: int=TILE_SIZE,
                          max_buffer_bytes: int=128*1024**2, max_output_bytes: int=2*1024**3,
                          min_free_bytes: int=1024**3, max_in_flight: int=1,
                          execute_model: bool=False, cancel_event: Event | None=None,
                          invalid_policy: InvalidTilePolicy='reject') -> Metadata:
    """Stream a prepared scene through engine.generate into a verified COG.

    Uses a fresh directory; partial evidence is retained on failure. No automatic
    retry, cleanup, download or precision change. Caller owns engine and source.
    The returned receipt reports numerical/scientific qualification unevaluated.
    """
    if execute_model is not True:
        raise ExecutionRequired("Pass execute_model=True explicitly")
    if invalid_policy not in ('reject', 'skip_invalid_tiles'):
        raise ValueError("invalid_policy must be reject or skip_invalid_tiles")
    if type(max_in_flight) is not int or max_in_flight < 1:
        raise ValueError("max_in_flight must be a positive integer")
    if max_in_flight > 1 and not callable(getattr(engine,'generate_many',None)):
        raise TypeError("Concurrent scene preparation requires engine.generate_many")
    if not isinstance(source, GenerationRasterSource) or not source.is_open:
        raise TypeError("source must be an open GenerationRasterSource")
    import numpy as np
    import rasterio
    from affine import Affine  # Optional upstream typing is declared in the implementation-checker boundary policy.
    from rasterio.shutil import copy as raster_copy  # Optional upstream typing is declared in the implementation-checker boundary policy.
    generation_tile_seed(seed, GenerationWindow(0,0))
    windows = generation_windows(source.height, source.width, stride=stride)
    next(generation_windows(source.height,source.width,stride=stride))
    allow_uncovered = invalid_policy == 'skip_invalid_tiles'
    required = (8 if allow_uncovered else 6)*TILE_SIZE*source.width*4
    if type(max_buffer_bytes) is not int or required > max_buffer_bytes:
        raise ValueError("Scene width exceeds reconstruction memory budget")
    output = Path(output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be fresh with an existing parent")
    if any(type(v) is not int or v < 0 for v in (max_output_bytes,min_free_bytes)):
        raise ValueError("Storage budgets require nonnegative integer bytes")
    def axis_count(length: int) -> int:
        last=length-TILE_SIZE
        return last//stride + 1 + int(bool(last%stride))
    tiles = axis_count(source.height)*axis_count(source.width)
    # Two lossless float32 rasters, bounded per-tile JSON, TIFF/block overhead.
    estimated_storage = 2*2*4*source.height*source.width + tiles*256*1024 + 4*1024**2
    if estimated_storage > max_output_bytes:
        raise ValueError("Scene exceeds max_output_bytes admission estimate")
    if shutil.disk_usage(output.parent).free < estimated_storage+min_free_bytes:
        raise ValueError("Insufficient free storage for scene and reserve")
    validity = source.inspect_validity(cancel_event=cancel_event) if allow_uncovered else None
    output.mkdir(exist_ok=False)
    scene_id = 'scene-' + uuid.uuid4().hex[:16]
    staging, cog, receipts = output/'mosaic.partial.tif', output/'synthetic-s1grd.tif', output/'tiles.jsonl'
    completed = skipped = 0
    started = perf_counter()
    times = {'prepare_seconds':0.,'generate_seconds':0.,'reconstruct_seconds':0.,
             'source_verify_seconds':0.,'export_seconds':0.,'audit_seconds':0.}
    try:
        with rasterio.Env(GDAL_CACHEMAX=64*1024**2, GDAL_TIFF_INTERNAL_MASK=True), receipts.open('x',encoding='utf-8') as log:
            with rasterio.open(staging,'w',driver='GTiff',width=source.width,height=source.height,count=2,
                    dtype='float32',crs=source.grid['crs'],transform=Affine(*source.grid['transform']),
                    tiled=True,blockxsize=256,blockysize=256,compress='DEFLATE',BIGTIFF='IF_SAFER') as ds:
                sink = _StripeReconstruction(ds,source.width,max_buffer_bytes,allow_uncovered=allow_uncovered)
                ds.set_band_description(1,'synthetic_S1GRD_VV'); ds.set_band_description(2,'synthetic_S1GRD_VH')
                ds.update_tags(synthetic='true',sensor_observation='false',units='dB',scientific_quality='not_evaluated',
                               spatial_recipe='independent224-uniform-fp32-v1',invalid_policy=invalid_policy)
                with _scene_results(engine,source,windows,seed,scene_id,max_in_flight,cancel_event,times,invalid_policy) as generated:
                    for window,tile_seed,input_metadata,result in generated:
                        if cancel_event is not None and cancel_event.is_set():
                            raise CancelledError("Scene cancelled before tile reconstruction")
                        if shutil.disk_usage(output).free < min_free_bytes:
                            raise OSError("Scene storage reserve exhausted")
                        tick = perf_counter()
                        if result is None:
                            sink.skip(window)
                            log.write(json.dumps({'window':vars(window),'seed':tile_seed,'status':'skipped',
                                'reason':'window_contains_invalid_source_pixels','model_executed':False},sort_keys=True)+'\n')
                            times['reconstruct_seconds'] += perf_counter()-tick
                            skipped += 1
                            continue
                        if result['provenance']['input'] != input_metadata:
                            raise ValueError("Generated input/grid provenance does not match the admitted tile")
                        if (np.asarray(result['valid_mask']).shape != (224,224)
                                or not np.asarray(result['valid_mask']).all()):
                            raise ValueError("Unexpected invalid generated pixels")
                        from ..terramind_native.memory_output import array_identity
                        if array_identity(result['array']) != result['provenance']['array']:
                            raise ValueError("Generated array identity does not match provenance")
                        if array_identity(result['valid_mask']) != result['provenance']['valid_mask']:
                            raise ValueError("Generated validity identity does not match provenance")
                        sink.add(window,result['array'])
                        line=json.dumps({'window':vars(window),'seed':tile_seed,'provenance':result['provenance']},sort_keys=True)+'\n'
                        if len(line.encode('utf-8')) > 256*1024:
                            raise ValueError("Tile provenance exceeds storage reservation")
                        log.write(line)
                        times['reconstruct_seconds'] += perf_counter()-tick
                        completed += 1
                        del result
                sink.finish(source.height)
            tick = perf_counter()
            source.verify_unchanged()
            times['source_verify_seconds'] += perf_counter()-tick
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("Scene cancelled before export")
            tick = perf_counter()
            raster_copy(staging,cog,driver='COG',compress='DEFLATE',blocksize=256,OVERVIEWS='NONE',BIGTIFF='IF_SAFER')
            times['export_seconds'] += perf_counter()-tick
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("Scene cancelled during COG export")
            tick = perf_counter()
            # Bounded independent decoded read-back, not just an encoded hash.
            with rasterio.open(staging) as expected, rasterio.open(cog) as actual:
                if (actual.crs,actual.transform,actual.width,actual.height,actual.dtypes) != (expected.crs,expected.transform,expected.width,expected.height,expected.dtypes):
                    raise RuntimeError("COG grid/type mismatch")
                if actual.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') != 'COG':
                    raise RuntimeError("Output is not a COG")
                for _, window in actual.block_windows(1):
                    if cancel_event is not None and cancel_event.is_set():
                        raise CancelledError("Scene cancelled during COG verification")
                    if not np.array_equal(expected.read(window=window),actual.read(window=window),equal_nan=allow_uncovered):
                        raise RuntimeError("COG decoded parity failed")
                    if not np.array_equal(expected.read_masks(window=window),actual.read_masks(window=window)):
                        raise RuntimeError("COG coverage-mask parity failed")
                if actual.tags().get('synthetic') != 'true' or actual.tags().get('units') != 'dB':
                    raise RuntimeError("COG synthetic/unit tags mismatch")
            times['audit_seconds'] += perf_counter()-tick
        if validity is None:
            validity = {'source_pixels':source.height*source.width,
                        'source_valid_pixels':source.height*source.width,'source_invalid_pixels':0,
                        'validity_definition':'all source pixels admitted through complete all-valid tile coverage'}
        record = {'schema':'aureon.generation_scene.v1','status':'completed','scene_id':scene_id,'source':source.identity,'grid':source.grid,
                  'recipe':{'tile_size':224,'stride':stride,'blend':'uniform_fp32_row_major','seed':seed,
                            'seed_policy':'coordinate_sha256_v1_origin_preserves_seed','invalid_policy':invalid_policy,
                            'uncovered_values':'NaN with output mask 0' if allow_uncovered else 'none'},
                  'tiles':completed,'tiles_generated':completed,'tiles_skipped':skipped,'tiles_planned':tiles,
                  'generated_pixels':sink.covered_pixels,'output_pixels':source.height*source.width,
                  'uncovered_pixels':source.height*source.width-sink.covered_pixels,
                  'source_valid_pixels_uncovered':validity['source_valid_pixels']-sink.covered_pixels,
                  'source_validity':validity,'coverage_semantics':'output mask means at least one all-valid generated tile contributed; it is not the original source-valid mask',
                  'tile_receipts':binding(receipts),'cog':binding(cog),
                  'reconstruction_buffer_bound_bytes':required,'gdal_cache_bytes':64*1024**2,
                  'storage_admission_estimate_bytes':estimated_storage,'max_output_bytes':max_output_bytes,
                  'min_free_bytes':min_free_bytes,
                  'decoded_cog_parity':True,'model_parity':'not_evaluated','scientific_quality':'not_evaluated',
                  'scope':'Independent tiled generation; not equivalent to a full-scene model forward',
                  'input_dependency_policy':'self-contained GeoTIFF; external raster sidecars rejected',
                  'stage_times':times,'elapsed_seconds':perf_counter()-started,
                  'max_in_flight':max_in_flight,
                  'timing_scope':'generate_seconds is caller wait including iterator preparation in queued mode; stages may overlap, are not additive, and are not CUDA kernel timings',
                  'staging_preserved':str(staging)}
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Scene cancelled before provenance delivery")
        write_json(output/'provenance.json',record)
        return record
    except BaseException as exc:
        from .lifecycle import close_preserving_error
        close_preserving_error(lambda: write_json(output/'failure.json',{
                   'status':'cancelled' if isinstance(exc,CancelledError) else 'failed',
                   'completed_tiles':completed,'skipped_tiles':skipped,'error_type':type(exc).__name__,
                   'partial_artifacts':True,'delivered':False}), exc,
                   context='Scene failure receipt could not be written')
        raise
