"""File-oriented, bounded TerraMind scene workflow with explicit ownership.

The request is a pure value. Inspection reads source rasters but never loads a
model. Execution owns its raster handles and can either own an engine or borrow
an explicitly compatible resident engine without taking over its lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator
from contextlib import contextmanager
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event, Lock
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary
import shutil

from .contracts import ExecutionProfile, INPUT_BYTES, OUTPUT_BYTES
from .lifecycle import close_preserving_error
from .spatial import (
    InvalidTilePolicy, PreparedGenerationRaster, TILE_SIZE, GenerationWindow,
    generation_windows, generation_tile_seed, generate_raster_scene,
)
from .v1 import ExecutionRequired
from ..terramind_native._types import Metadata, PathInput
from ..terramind_native.common import INPUT_BANDS, binding, write_json
from ..terramind_native.families import GENERATION_MODELS
from ..terramind_native.session import SessionBusy, SessionClosed

if TYPE_CHECKING:
    from .generation_engine import GenerationEngine


@dataclass
class _BorrowState:
    gate: Lock = field(default_factory=Lock)
    identity: dict[str, object] | None = None
    asset_stats: tuple[tuple[int, int, int, int, int], ...] | None = None


_BORROW_STATES: WeakKeyDictionary[GenerationEngine, _BorrowState] = WeakKeyDictionary()
_BORROW_STATES_GATE = Lock()


@dataclass(frozen=True)
class SceneGenerationRequest:
    """Prepared aligned FP32 S2L2A and a separate uint8 validity raster.

    Image values are surface reflectance x10000 in the fixed twelve-band order;
    no normalization, reprojection, band guessing or cloud filling is implicit.
    Model and work paths are used only during explicit execution.
    """
    image: PathInput
    valid_mask: PathInput
    model_file: PathInput
    tokenizer_file: PathInput
    work: PathInput
    output: PathInput
    model: str = 'terramind-v1-base'
    seed: int = 17
    stride: int = TILE_SIZE
    invalid_policy: InvalidTilePolicy = 'reject'
    execution_profile: ExecutionProfile = 'recommended'
    max_in_flight: int = 1
    max_buffer_bytes: int = 128*1024**2
    max_decode_bytes: int = 32*1024**2
    max_output_bytes: int = 2*1024**3
    min_free_bytes: int = 1024**3

    def __post_init__(self) -> None:
        if self.model not in GENERATION_MODELS:
            raise ValueError('A TerraMind generation model ID is required')
        if self.invalid_policy not in ('reject', 'skip_invalid_tiles'):
            raise ValueError('invalid_policy must be reject or skip_invalid_tiles')
        if self.execution_profile not in ('recommended', 'reference'):
            raise ValueError('execution_profile must be recommended or reference')
        generation_tile_seed(self.seed, GenerationWindow(0, 0))
        if type(self.stride) is not int or not 1 <= self.stride <= TILE_SIZE:
            raise ValueError('stride must be an integer in 1..224')
        for name in ('max_in_flight', 'max_buffer_bytes', 'max_decode_bytes', 'max_output_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name+' must be a positive integer')
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 0:
            raise ValueError('min_free_bytes must be a nonnegative integer')
        for name in ('image', 'valid_mask', 'model_file', 'tokenizer_file', 'work', 'output'):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value) or '://' in str(value) or str(value).startswith('/vsi'):
                raise ValueError(name+' must be an explicit local path')


def _cancel(event: Event | None) -> None:
    if event is not None and event.is_set():
        raise CancelledError('Scene cancelled')


def _destinations(request: SceneGenerationRequest, *, borrowed_work: Path | None=None) -> None:
    work, output = Path(request.work).absolute(), Path(request.output).absolute()
    for name, path in (('work', work), ('output', output)):
        if name == 'work' and borrowed_work is not None:
            if path.resolve() != borrowed_work.resolve() or path.is_symlink() or (path.exists() and not path.is_dir()):
                raise ValueError('request.work must identify the borrowed engine work directory')
            if not path.parent.is_dir():
                raise ValueError('Borrowed engine work parent must exist')
            continue
        if path.exists() or path.is_symlink() or not path.parent.is_dir():
            raise ValueError(name+' must be a fresh directory with an existing parent')
    work = work.parent.resolve()/work.name
    output = output.parent.resolve()/output.name
    if work == output or work.is_relative_to(output) or output.is_relative_to(work):
        raise ValueError('work and output must be separate non-nested directories')


def _source(request: SceneGenerationRequest) -> PreparedGenerationRaster:
    return PreparedGenerationRaster(request.image, request.valid_mask, bands=INPUT_BANDS,
        units='surface_reflectance_x10000', max_decode_bytes=request.max_decode_bytes)


def _inspect(request: SceneGenerationRequest, source: PreparedGenerationRaster, cancel_event: Event | None,
             *, borrowed_work: Path | None=None) -> Metadata:
    _cancel(cancel_event)
    _destinations(request, borrowed_work=borrowed_work)
    reconstruction = (8 if request.invalid_policy == 'skip_invalid_tiles' else 6)*TILE_SIZE*source.width*4
    if reconstruction > request.max_buffer_bytes:
        raise ValueError('Scene width exceeds reconstruction memory budget')
    def axis_count(length: int) -> int:
        last = length-TILE_SIZE
        return last//request.stride+1+int(bool(last%request.stride))
    tiles = axis_count(source.height)*axis_count(source.width)
    storage = 2*2*4*source.height*source.width+tiles*256*1024+4*1024**2
    if storage > request.max_output_bytes:
        raise ValueError('Scene exceeds max_output_bytes admission estimate')
    if shutil.disk_usage(Path(request.output).absolute().parent).free < storage+request.min_free_bytes:
        raise ValueError('Insufficient free storage for scene and reserve')
    validity = source.inspect_validity(cancel_event=cancel_event)
    if validity['source_invalid_pixels'] and request.invalid_policy == 'reject':
        raise ValueError('Scene contains invalid/cloud/nodata/nonfinite pixels; choose skip_invalid_tiles explicitly or provide a valid source')
    eligible = tiles
    if request.invalid_policy == 'skip_invalid_tiles':
        eligible = 0
        for window in generation_windows(source.height, source.width, stride=request.stride):
            _cancel(cancel_event)
            item = source.read_if_valid(window, invalid_policy='skip_invalid_tiles')
            eligible += int(item is not None)
            del item
    source.verify_unchanged()
    return {'schema':'aureon.generation_scene_plan.v1', 'admitted':True, 'model_id':request.model,
        'source':source.identity, 'grid':source.grid, 'source_validity':validity,
        'input_dependency_policy':'self-contained GeoTIFF; external raster sidecars rejected',
        'tiles_planned':tiles, 'tiles_eligible':eligible, 'tiles_skipped':tiles-eligible,
        'recipe':{'tile_size':TILE_SIZE, 'stride':request.stride, 'seed':request.seed,
            'invalid_policy':request.invalid_policy, 'blend':'uniform_fp32_row_major',
            'seed_policy':'coordinate_sha256_v1_origin_preserves_seed', 'execution_profile':request.execution_profile},
        'reconstruction_buffer_bound_bytes':reconstruction,
        'queued_io_reservation_bytes':request.max_in_flight*(INPUT_BYTES+OUTPUT_BYTES),
        'max_in_flight':request.max_in_flight, 'max_decode_bytes':request.max_decode_bytes,
        'gdal_cache_bytes':64*1024**2, 'storage_admission_estimate_bytes':storage,
        'max_output_bytes':request.max_output_bytes, 'min_free_bytes':request.min_free_bytes,
        'memory_scope':'Separate application reconstruction and queue bounds; excludes model, allocator, Python and driver overhead. Source decode blocks have their own guard.',
        'assets_checked':False, 'model_execution':False,
        'asset_scope':'Pinned model/tokenizer identity is verified by the existing engine loader at execution; inspection does not hash or load weights.',
        'scope':'Independent tiled recipe, not equivalent to monolithic full-scene generation; scientific quality and seams unevaluated'}


def inspect_generation_scene(request: SceneGenerationRequest, *, cancel_event: Event | None=None) -> Metadata:
    """Read-only scene/resource preflight. No output, work, model or CUDA is created."""
    if not isinstance(request, SceneGenerationRequest):
        raise TypeError('request must be SceneGenerationRequest')
    _cancel(cancel_event)
    with _source(request) as source:
        return _inspect(request, source, cancel_event)


def _execute_scene(engine: GenerationEngine, request: SceneGenerationRequest,
                   source: PreparedGenerationRaster, cancel_event: Event | None) -> Metadata:
    return generate_raster_scene(engine, source, request.output, seed=request.seed,
        stride=request.stride, max_buffer_bytes=request.max_buffer_bytes,
        max_output_bytes=request.max_output_bytes, min_free_bytes=request.min_free_bytes,
        max_in_flight=request.max_in_flight, execute_model=True,
        cancel_event=cancel_event, invalid_policy=request.invalid_policy)


def _delivery(request: SceneGenerationRequest, plan: Metadata, scene: Metadata, *,
              cleanup: Metadata | None, ownership: str, engine_record: Metadata | None=None,
              cancel_event: Event | None=None) -> Metadata:
    _cancel(cancel_event)
    output = Path(request.output)
    write_json(output/'preflight.json', plan)
    receipt: Metadata = {'schema':'aureon.generation_scene_delivery.v1', 'status':'completed',
        'model_id':request.model, 'scene':scene, 'engine_cleanup':cleanup,
        'engine_ownership':ownership,
        'preflight':binding(output/'preflight.json'), 'provenance':binding(output/'provenance.json'),
        'cog':scene['cog'], 'tile_receipts':scene['tile_receipts'], 'version':'0.1.0'}
    if engine_record is not None:
        receipt['resident_engine'] = engine_record
    # Completion is the delivery commit. Export, read-back and owned engine
    # shutdown may take time; observe cancellation again after those stages.
    _cancel(cancel_event)
    write_json(output/'completion.json', receipt)
    return receipt


def _asset_stats(request: SceneGenerationRequest) -> tuple[tuple[int, int, int, int, int], ...]:
    values = []
    for value in (request.model_file, request.tokenizer_file):
        path = Path(value).resolve(strict=True)
        if not path.is_file():
            raise ValueError('Borrowed model and tokenizer must be existing local files')
        info = path.stat()
        values.append((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(values)


def _borrow_identity(engine: GenerationEngine, request: SceneGenerationRequest) -> dict[str, object]:
    identity = engine.binding()
    if identity.get('schema') != 'aureon.generation_engine_binding.v1':
        raise ValueError('Unsupported borrowed engine binding')
    for name, expected in (('model_id',request.model), ('execution_profile',request.execution_profile)):
        if identity.get(name) != expected:
            raise ValueError('Borrowed engine '+name+' does not match the scene request')
    for name, expected_path in (('model_file',request.model_file), ('tokenizer_file',request.tokenizer_file), ('work',request.work)):
        value = identity.get(name)
        if not isinstance(value,str) or Path(value).resolve() != Path(expected_path).resolve():
            raise ValueError('Borrowed engine '+name+' does not match the scene request')
    return identity


def _require_idle(engine: GenerationEngine, request: SceneGenerationRequest) -> None:
    metadata = engine.metadata()
    if metadata['closing'] or metadata['state'] in ('closed','failed'):
        raise SessionClosed('Borrowed engine is closing or terminal; its owner must replace it')
    session = metadata['session']
    queue = metadata['queue']
    if ((session is not None and session.get('state') == 'running') or
            (queue is not None and (queue['pending'] or queue['queued'] or queue['running'] or queue['reserved_results']))):
        raise SessionBusy('Borrowed scene requires an idle engine; existing caller work is not drained or cancelled')
    if queue is not None and (not queue['accepting'] or queue['failed']):
        raise SessionClosed('Borrowed engine queue is terminal')
    credits = engine.capabilities()['credits']
    if request.max_in_flight > min(credits['max_pending'], credits['max_results']):
        raise ValueError('Scene max_in_flight exceeds borrowed engine queue credits')


@contextmanager
def _borrow(engine: GenerationEngine, request: SceneGenerationRequest) -> Iterator[_BorrowState]:
    from .generation_engine import GenerationEngine
    if not isinstance(engine,GenerationEngine):
        raise TypeError('Borrowed scene requires GenerationEngine')
    with _BORROW_STATES_GATE:
        state = _BORROW_STATES.setdefault(engine,_BorrowState())
    if not state.gate.acquire(blocking=False):
        raise SessionBusy('Another scene owns this borrowed engine')
    try:
        identity = _borrow_identity(engine,request)
        stats = _asset_stats(request)
        if state.identity is not None and (state.identity != identity or state.asset_stats != stats):
            raise ValueError('Borrowed engine assets/configuration changed since its first scene')
        _require_idle(engine,request)
        state.identity, state.asset_stats = dict(identity), stats
        yield state
    finally:
        state.gate.release()


def run_generation_scene_with_engine(engine: GenerationEngine, request: SceneGenerationRequest, *,
                                     execute_model: bool=False, cancel_event: Event | None=None) -> Metadata:
    """Borrow one exclusively held engine; preserve its lifetime and profile.

    The caller must not submit other jobs or close the engine until this call
    returns. Existing queue work and overlapping scene calls are rejected, never
    drained/cancelled. On cancellation, only this scene's stream is discarded;
    an already-running tile may finish before the engine becomes idle again.
    Native execution failures retain the engine's own terminal-state semantics.
    No weight rehash or reload is added per scene: pinned identity is verified by
    the native loader, while this boundary matches paths and detects stat changes.
    """
    if execute_model is not True:
        raise ExecutionRequired('Pass execute_model=True explicitly')
    if not isinstance(request,SceneGenerationRequest):
        raise TypeError('request must be SceneGenerationRequest')
    _cancel(cancel_event)
    with _borrow(engine,request) as borrowed:
        with _source(request) as source:
            plan = _inspect(request,source,cancel_event,borrowed_work=Path(request.work))
            _cancel(cancel_event)
            scene = _execute_scene(engine,request,source,cancel_event)
        # Final identity checks precede the delivery receipt. The caller still
        # owns any resident allocations and all subsequent cleanup responsibility.
        if engine.binding() != borrowed.identity or _asset_stats(request) != borrowed.asset_stats:
            raise ValueError('Borrowed engine assets/configuration changed during scene execution')
        _require_idle(engine,request)
        return _delivery(request,plan,scene,cleanup=None,ownership='caller',cancel_event=cancel_event,engine_record={
            'binding':borrowed.identity, 'state_after_scene':engine.metadata(), 'closed_by_scene':False,
            'asset_check':'configured paths and stable file stats; native loader owns pinned-weight verification',
            'concurrency':'exclusive scene borrowing; owner must not submit or close concurrently'})


def run_generation_scene(request: SceneGenerationRequest, *, execute_model: bool=False,
                         cancel_event: Event | None=None) -> Metadata:
    """Preflight every source pixel, then own the reader and resident Engine.

    Successful completion includes engine cleanup. Partial artifacts are retained
    on failure, never delivered as a completed scene. Cancellation is cooperative
    between tiles; it cannot interrupt a running GPU call.
    """
    if execute_model is not True:
        raise ExecutionRequired('Pass execute_model=True explicitly')
    if not isinstance(request, SceneGenerationRequest):
        raise TypeError('request must be SceneGenerationRequest')
    _cancel(cancel_event)
    with _source(request) as source:
        plan = _inspect(request, source, cancel_event)
        _cancel(cancel_event)
        from .generation_engine import GenerationEngine
        engine = GenerationEngine(request.model, model_file=request.model_file,
            tokenizer_file=request.tokenizer_file, work=request.work, execute_model=True,
            execution_profile=request.execution_profile, max_pending=request.max_in_flight,
            max_results=request.max_in_flight)
        failure: BaseException | None = None
        try:
            scene = _execute_scene(engine,request,source,cancel_event)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            cleanup = close_preserving_error(lambda: engine.close(cancel_pending=failure is not None), failure,
                context='Scene-owned generation engine cleanup failed')
            if failure is None and (cleanup is None or cleanup.get('passed') is not True):
                raise RuntimeError('Scene-owned engine cleanup did not pass; no completed delivery receipt')
    return _delivery(request,plan,scene,cleanup=cleanup,ownership='scene',cancel_event=cancel_event)
