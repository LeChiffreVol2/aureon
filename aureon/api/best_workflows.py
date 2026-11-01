"""Prepared scenes and bounded repeated work around the selected file runtime.

Only BestTerraMindSession.run executes a model. This layer stages explicit tile
manifests and reuses the retained spatial reader/reconstruction with one tile in
flight. It never changes numerical policy, hashes weights, retries or removes
artifacts. A valid completion.json is the whole-scene delivery commit.
"""
from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, RLock, get_ident
from typing import TYPE_CHECKING, Any, Iterator, cast
from weakref import WeakKeyDictionary
import hashlib
import json
import math
import os
import re
import shutil
import time

from .contracts import GenerationResult, positive, validate_timeout
from .memory_input import GenerationInput
from .spatial import (GenerationWindow, InvalidTilePolicy, PreparedGenerationRaster,
                      generate_raster_scene, generation_tile_seed, generation_windows)
from .v1 import ExecutionRequired
from ..terramind_native.common import (INPUT_BANDS, binding, read_json, verify_file,
                                      write_json)
from ..terramind_native.memory_output import array_identity

if TYPE_CHECKING:
    from .best import BestTerraMindSession
    from .generation_engine import GenerationEngine

Metadata = dict[str, Any]
_TILE_DISK_BOUND = 16*1024**2
_TILE_HOST_BOUND = 4*(12*4+1)*224*224 + 4*2*4*224*224
_REFERENCE = '0fa5ef2b9440fcc310a17c76f78df52278cbe2d354015be2046a8e4196715c97'
_CLOCK = 'CLOCK_MONOTONIC_RAW' if hasattr(time, 'CLOCK_MONOTONIC_RAW') else 'perf_counter'


class BestWorkflowBusy(RuntimeError):
    """The resident or a bounded workflow credit is already owned."""


class BestWorkflowError(RuntimeError):
    """A queued workflow failed; partial files are not delivered results."""


def _now() -> float:
    return (time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
            if hasattr(time, 'CLOCK_MONOTONIC_RAW') else time.perf_counter())


def _cancel(event: Event | None) -> None:
    if event is not None and event.is_set():
        raise CancelledError('Scene cancelled at a delivery/tile boundary')


def _identifier(value: str) -> None:
    if (not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value)
            or value.endswith('.') or value.split('.')[0].upper() in
            {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}):
        raise ValueError('job_id/attempt_id must be safe 1..80 character identifiers')


@dataclass(frozen=True)
class BestSceneRequest:
    """Already prepared twelve-band FP32 reflectance x10000 and uint8 0/1 mask.

    No resampling, band inference or invalid filling. Paths are snapshotted
    absolutely when constructed. Limits cover new retained files for this scene,
    not model weights, previously retained work, allocator or Python overhead.
    """
    image: str | Path
    valid_mask: str | Path
    output: str | Path
    job_id: str
    attempt_id: str = '1'
    seed: int = 17
    stride: int = 224
    invalid_policy: InvalidTilePolicy = 'reject'
    max_buffer_bytes: int = 128*1024**2
    max_decode_bytes: int = 32*1024**2
    max_workspace_bytes: int = 512*1024**2
    max_output_bytes: int = 512*1024**2
    min_free_bytes: int = 1024**3
    hourly_rate: float | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.job_id); _identifier(self.attempt_id)
        generation_tile_seed(self.seed, GenerationWindow(0,0))
        if type(self.stride) is not int or not 1 <= self.stride <= 224:
            raise ValueError('stride must be an integer in 1..224')
        if self.invalid_policy not in ('reject','skip_invalid_tiles'):
            raise ValueError('Explicit reject or skip_invalid_tiles policy required')
        for name in ('max_buffer_bytes','max_decode_bytes','max_workspace_bytes','max_output_bytes'):
            positive(getattr(self,name),name)
        if type(self.min_free_bytes) is not int or self.min_free_bytes < 0:
            raise ValueError('min_free_bytes must be nonnegative integer bytes')
        for name in ('image','valid_mask','output'):
            value = getattr(self,name)
            if not isinstance(value,(str,Path)) or not str(value) or '://' in str(value) or str(value).startswith('/vsi'):
                raise ValueError(name+' must be an explicit local path')
            object.__setattr__(self,name,Path(value).absolute())
        if self.hourly_rate is None:
            if self.currency is not None: raise ValueError('currency requires an explicit hourly_rate')
        elif (type(self.hourly_rate) not in (int,float) or not math.isfinite(self.hourly_rate)
              or self.hourly_rate < 0 or not isinstance(self.currency,str) or not self.currency.strip()):
            raise ValueError('Supply a finite nonnegative hourly_rate and currency together')


def _source(request: BestSceneRequest) -> PreparedGenerationRaster:
    return PreparedGenerationRaster(request.image, request.valid_mask, bands=INPUT_BANDS,
                                    units='surface_reflectance_x10000',max_decode_bytes=request.max_decode_bytes)


def _plan(request: BestSceneRequest, source: PreparedGenerationRaster, event: Event | None) -> Metadata:
    _cancel(event)
    output=Path(request.output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError('output must be fresh with an existing parent')
    stripe=(8 if request.invalid_policy=='skip_invalid_tiles' else 6)*224*source.width*4
    if stripe+_TILE_HOST_BOUND > request.max_buffer_bytes:
        raise ValueError('Scene stripe and one-tile adapter exceed max_buffer_bytes')
    validity=source.inspect_validity(cancel_event=event)
    if request.invalid_policy=='reject' and validity['source_invalid_pixels']:
        raise ValueError('Invalid pixels require explicit skip_invalid_tiles; no imputation')
    def count(length: int) -> int:
        last=length-224
        return last//request.stride+1+int(bool(last%request.stride))
    tiles=count(source.height)*count(source.width)
    workspace=tiles*_TILE_DISK_BOUND
    output_bound=16*source.height*source.width+tiles*256*1024+4*1024**2
    if workspace>request.max_workspace_bytes or output_bound>request.max_output_bytes:
        raise ValueError('Scene exceeds workspace/output disk admission limits')
    if shutil.disk_usage(output.parent).free < output_bound+request.min_free_bytes:
        raise ValueError('Insufficient output storage and reserve')
    source.verify_unchanged()
    return {'schema':'aureon.best_scene_plan.v1','source':source.identity,'grid':source.grid,
            'source_validity':validity,'tiles_planned':tiles,'workspace_estimate_bytes':workspace,
            'output_estimate_bytes':output_bound,'host_buffer_bound_bytes':stripe+_TILE_HOST_BOUND,
            'max_decode_bytes':request.max_decode_bytes,'gdal_cache_bytes':64*1024**2,
            'recipe':{'seed':request.seed,'stride':request.stride,'invalid_policy':request.invalid_policy,
                      'blend':'uniform_fp32_row_major','tile_seed':'coordinate_sha256_v1_origin_preserves_seed'},
            'model_execution':False,'scientific_quality':'not_evaluated',
            'scope':'Independent tiled generation; fixed-profile bounds, not a model/GPU memory limit'}


def inspect_best_scene(request: BestSceneRequest, *, cancel_event: Event | None=None) -> Metadata:
    """Read-only source/resource admission; no resident, output or weight access."""
    if not isinstance(request,BestSceneRequest): raise TypeError('BestSceneRequest required')
    with _source(request) as source:
        return _plan(request,source,cancel_event)


@dataclass
class _OwnerState:
    gate: Any = field(default_factory=Lock)
    queue: object | None = None


_OWNERS: WeakKeyDictionary[BestTerraMindSession,_OwnerState] = WeakKeyDictionary()
_OWNERS_LOCK=Lock()


def _state(session: BestTerraMindSession) -> _OwnerState:
    from .best import BestTerraMindSession
    if not isinstance(session,BestTerraMindSession): raise TypeError('BestTerraMindSession required')
    return _OWNERS.setdefault(session,_OwnerState())


def _resident(session: BestTerraMindSession) -> Path:
    metadata=session.metadata()
    if metadata['state'] in ('failed','closed','closing'):
        raise BestWorkflowError('Selected resident is terminal; its owner must close/replace it')
    if metadata['state']=='running': raise BestWorkflowBusy('Selected resident is already running')
    work=Path(cast(str,metadata['work'])).resolve(strict=True)
    if not work.is_dir(): raise ValueError('Resident work must be an existing directory')
    return work


@contextmanager
def _borrow(session: BestTerraMindSession, token: object | None) -> Iterator[Path]:
    with _OWNERS_LOCK:
        state=_state(session)
        if (state.queue is not None and state.queue is not token) or not state.gate.acquire(blocking=False):
            raise BestWorkflowBusy('Another workflow owns the selected resident')
    try: yield _resident(session)
    finally: state.gate.release()


def _tree_bytes(path: Path) -> int:
    """Only a newly owned attempt tree, never recursive weight/environment scans."""
    total=0
    def unreadable(error: OSError) -> None: raise error
    for base,dirs,files in os.walk(path,followlinks=False,onerror=unreadable):
        for name in dirs+files:
            item=Path(base)/name
            if item.is_symlink() or getattr(item.lstat(),'st_file_attributes',0)&0x400:
                raise ValueError('Unexpected link in owned workflow files')
        total+=sum((Path(base)/name).stat().st_size for name in files)
    return total


def _owned_path(work: Path, path: Path) -> None:
    if not path.resolve().is_relative_to(work): raise ValueError('Workflow path escaped resident work')
    for parent in (path,*path.parents):
        if parent.is_symlink() or (parent.exists() and getattr(parent.lstat(),'st_file_attributes',0)&0x400):
            raise ValueError('Workflow attempt paths cannot contain links')
        if parent==work: break


def _publish(ready: Path, completed: Path) -> None:
    # Windows rename is atomic and refuses an existing destination. POSIX rename
    # overwrites, so use exclusive link there. Unsupported filesystems fail closed.
    if os.name=='nt': os.rename(ready,completed)
    else: os.link(ready,completed)


class _TileAdapter:
    def __init__(self, session: BestTerraMindSession, request: BestSceneRequest,
                 work: Path, staged: Path, event: Event | None) -> None:
        self.session,self.request,self.work,self.staged,self.event=session,request,work,staged,event
        self.prefix='scene-'+hashlib.sha256((request.job_id+'\0'+request.attempt_id).encode()).hexdigest()[:32]
        self.retained_bytes=0
        self.times={'manifest_seconds':0.,'selected_run_seconds':0.,'native_verification_seconds':0.}

    def generate(self, data: GenerationInput, *, seed: int, job_id: str) -> GenerationResult:
        import numpy as np
        import rasterio
        _cancel(self.event)
        if _resident(self.session)!=self.work: raise ValueError('Resident work identity changed')
        if self.retained_bytes+_TILE_DISK_BOUND>self.request.max_workspace_bytes:
            raise ValueError('Workspace reservation exhausted before tile admission')
        if shutil.disk_usage(self.work).free < _TILE_DISK_BOUND+self.request.min_free_bytes:
            raise OSError('Workspace storage reserve exhausted')
        tick=_now();image,mask,metadata,_=data.snapshot()
        tile=str(metadata['source_id']);_identifier(tile)
        folder=self.staged/tile;folder.mkdir()
        np.save(folder/'image.npy',image,allow_pickle=False);np.save(folder/'mask.npy',mask,allow_pickle=False)
        manifest={'schema':'aureon.terramind_native_input.v1','bands':INPUT_BANDS,
                  'units':'surface_reflectance_x10000','grid':metadata['grid'],'source_id':tile,
                  'preprocessing':'Prepared source values unchanged; model owns standardization',
                  'mask_policy':'all-valid eligible tile; no filling'}
        for name in ('image','mask'):
            manifest[name]={**binding(folder/(name+'.npy')),'path':name+'.npy'}
        write_json(folder/'manifest.json',manifest)
        recipe={'token_steps':8,'diffusion_steps':10,'temperature':1.0,'top_p':.8,'top_k':0,'seed':seed}
        native_job=self.prefix+'-'+tile;native_dir=self.work/'jobs'/native_job/self.request.attempt_id
        _owned_path(self.work,native_dir)
        self.times['manifest_seconds']+=_now()-tick;_cancel(self.event)
        tick=_now()
        r=cast(Metadata,self.session.run(folder/'manifest.json',job_id=native_job,
                                         attempt_id=self.request.attempt_id,recipe=recipe))
        self.times['selected_run_seconds']+=_now()-tick
        _cancel(self.event);tick=_now()
        _owned_path(self.work,native_dir)
        def artifact(item: Metadata, maximum: int) -> Path:
            if type(item.get('bytes')) is not int or not 0<item['bytes']<=maximum:
                raise ValueError('Native artifact exceeds fixed-profile byte bound')
            p=verify_file(item,native_dir)
            if not p.is_relative_to(native_dir.resolve()): raise ValueError('Native artifact escaped its attempt')
            return p
        raw=artifact(r['raw'],1024**2);provenance=artifact(r['provenance'],2*1024**2)
        cog=artifact(r['cog']['file'],2*1024**2);p=read_json(provenance)
        if (r['status']!='completed' or r['job_id']!=native_job or r['attempt_id']!=self.request.attempt_id
                or r['recipe']!=recipe or p['input']!=manifest or p['recipe']!=recipe
                or p['job_id']!=native_job or p['attempt_id']!=self.request.attempt_id
                or p['input_manifest']['sha256']!=binding(folder/'manifest.json')['sha256']
                or p['raw']['sha256']!=r['raw']['sha256'] or not p['synthetic'] or p['sensor_observation'] is not False
                or p['precision']!={'parameters':'float32','matmul':'highest','tf32':False,'autocast':False}):
            raise ValueError('Native delivery identity/policy differs from admitted tile')
        loaded=np.load(raw,allow_pickle=False,mmap_mode='r')
        if loaded.shape!=(2,224,224) or loaded.dtype!=np.float32 or not np.isfinite(loaded).all():
            raise ValueError('Native output must be finite FP32 VV/VH')
        values=np.array(loaded,copy=True);del loaded
        with rasterio.open(cog) as ds:
            if ((ds.height,ds.width,ds.count)!=(224,224,2) or ds.dtypes!=('float32','float32')
                    or str(ds.crs)!=metadata['grid']['crs'] or list(ds.transform)[:6]!=metadata['grid']['transform']
                    or not np.array_equal(ds.read().view('uint32'),values.view('uint32'))
                    or not np.array_equal(ds.dataset_mask()>0,mask)
                    or ds.tags(ns='IMAGE_STRUCTURE').get('LAYOUT')!='COG'
                    or ds.tags().get('synthetic')!='true' or ds.tags().get('units')!='dB'):
                raise ValueError('Native COG failed decoded/grid/mask verification')
        used=_tree_bytes(folder)+_tree_bytes(native_dir)
        if used>_TILE_DISK_BOUND: raise ValueError('Native tile exceeded its workspace bound')
        self.retained_bytes+=used
        if _resident(self.session)!=self.work: raise ValueError('Resident changed during generation')
        self.times['native_verification_seconds']+=_now()-tick
        return {'array':values,'valid_mask':mask,'provenance':{
            'input':metadata,'array':array_identity(values),'valid_mask':array_identity(mask),
            'native_input_manifest':binding(folder/'manifest.json'),'native_provenance':binding(provenance),
            'native_raw':r['raw'],'native_cog':r['cog']['file'],'recipe':recipe,
            'model':p['model'],'tokenizer':p['tokenizer'],'synthetic':True,'sensor_observation':False,
            'selected_reference_wheel':_REFERENCE,'model_parity':'not_evaluated_by_workflow'}}


def _execute(session: BestTerraMindSession, request: BestSceneRequest, event: Event | None,
             token: object | None) -> Metadata:
    started=_now();owned_output=False
    with _borrow(session,token) as work:
        output=Path(request.output).resolve()
        if output==work or output.is_relative_to(work) or work.is_relative_to(output):
            raise ValueError('Scene output and resident work must be separate non-nested directories')
        staged=work/'prepared-scenes'/request.job_id/request.attempt_id
        _owned_path(work,staged)
        if staged.exists() or staged.is_symlink(): raise FileExistsError('This job/attempt already has staged inputs')
        with _source(request) as source:
            plan=_plan(request,source,event)
            needed=plan['workspace_estimate_bytes']
            if work.stat().st_dev==output.parent.stat().st_dev: needed+=plan['output_estimate_bytes']
            if shutil.disk_usage(work).free<needed+request.min_free_bytes:
                raise ValueError('Insufficient combined workspace/output storage and reserve')
            _cancel(event);staged.mkdir(parents=True,exist_ok=False)
            adapter=_TileAdapter(session,request,work,staged,event)
            preflight_seconds=_now()-started
            try:
                # The retained delivery function calls only generate at this bound.
                # The cast satisfies its older static annotation; no GenerationEngine is constructed.
                scene=generate_raster_scene(cast('GenerationEngine',adapter),source,output,
                    seed=request.seed,stride=request.stride,invalid_policy=request.invalid_policy,
                    max_in_flight=1,max_buffer_bytes=request.max_buffer_bytes-_TILE_HOST_BOUND,
                    max_output_bytes=request.max_output_bytes,min_free_bytes=request.min_free_bytes,
                    execute_model=True,cancel_event=event)
                owned_output=True
                _cancel(event);source.verify_unchanged()
                if _resident(session)!=work: raise ValueError('Resident changed during scene delivery')
                write_json(output/'preflight.json',plan)
                if _tree_bytes(output)>request.max_output_bytes: raise ValueError('Scene output byte bound exceeded')
                elapsed=_now()-started
                receipt={'schema':'aureon.best_scene_delivery.v1','status':'completed','job_id':request.job_id,
                    'attempt_id':request.attempt_id,'scene':scene,'cog':scene['cog'],
                    'preflight':binding(output/'preflight.json'),'provenance':binding(output/'provenance.json'),
                    'staged_inputs':str(staged),'workspace_retained_bytes':adapter.retained_bytes,
                    'session':session.metadata(),'session_closed_by_workflow':False,'version':'0.1.0',
                    'selected_reference_wheel':_REFERENCE,'model_parity':'not_evaluated_by_workflow',
                    'scientific_quality':'not_evaluated','timing':{'clock':_CLOCK,'elapsed_before_commit_seconds':elapsed,
                        'preflight_seconds':preflight_seconds,**adapter.times,
                        'scope':'One host clock; nested stages, not additive; includes file adapter and verified delivery, excludes queue wait and final JSON commit'},
                    'cost':{'hourly_rate':request.hourly_rate,'currency':request.currency,
                        'amount':None if request.hourly_rate is None else elapsed*request.hourly_rate/3600,
                        'kind':'explicit_caller_rate_scenario_not_provider_quote'}}
                # Publish complete JSON atomically without replacing another commit.
                encoded=(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
                if len(encoded)>256*1024 or _tree_bytes(output)+2*len(encoded)>request.max_output_bytes:
                    raise ValueError('Completion receipt exceeds reserved output bound')
                with (output/'completion.ready.json').open('xb') as stream: stream.write(encoded)
                _cancel(event);_publish(output/'completion.ready.json',output/'completion.json')
                return receipt
            except BaseException as exc:
                if owned_output and not (output/'completion.json').exists() and not (output/'failure.json').exists():
                    try: write_json(output/'failure.json',{'status':'cancelled' if isinstance(exc,CancelledError) else 'failed',
                        'delivered':False,'partial_artifacts':True,'job_id':request.job_id,'attempt_id':request.attempt_id,
                        'error_type':type(exc).__name__})
                    except OSError as error: exc.add_note('Could not persist failure receipt: '+str(error))
                raise


def run_best_scene(session: BestTerraMindSession, request: BestSceneRequest, *,
                   execute_model: bool=False, cancel_event: Event | None=None) -> Metadata:
    """Exclusively borrow the caller's best resident; never close or replace it."""
    if execute_model is not True: raise ExecutionRequired('Pass execute_model=True explicitly')
    if not isinstance(request,BestSceneRequest): raise TypeError('BestSceneRequest required')
    _cancel(cancel_event)
    return _execute(session,request,cancel_event,None)


def verify_best_scene(output: str | Path, *, job_id: str | None=None,
                      attempt_id: str | None=None) -> Metadata:
    """Verify a retained completion after restart; never retry or remove files.

    Requires the original source and native attempt artifacts at their retained
    paths. A partial/ready/failure record alone cannot establish delivery. This
    is integrity verification, not authentication or a neural parity test.
    """
    root=Path(output).resolve(strict=True)
    if (root/'failure.json').exists(): raise ValueError('Failed/partial scene is not delivered')
    receipt=read_json(root/'completion.json')
    if (receipt.get('schema')!='aureon.best_scene_delivery.v1' or receipt.get('status')!='completed'
            or receipt.get('selected_reference_wheel')!=_REFERENCE
            or (job_id is not None and receipt.get('job_id')!=job_id)
            or (attempt_id is not None and receipt.get('attempt_id')!=attempt_id)):
        raise ValueError('Completion identity differs from expected scene')
    def local(item: Metadata, name: str) -> Path:
        path=verify_file(item,root)
        if path!=root/name: raise ValueError('Scene artifact path differs from completion')
        return path
    plan=read_json(local(receipt['preflight'],'preflight.json'))
    scene=read_json(local(receipt['provenance'],'provenance.json'))
    if scene!=receipt['scene'] or scene['cog']!=receipt['cog'] or scene['source']!=plan['source']:
        raise ValueError('Completion/source provenance differs')
    local(receipt['cog'],'synthetic-s1grd.tif')
    for item in scene['source'].values():
        if isinstance(item,dict) and 'sha256' in item and 'path' in item: verify_file(item,root)
    log=local(scene['tile_receipts'],'tiles.jsonl')
    generated=skipped=0
    with log.open('rb') as stream:
        while line:=stream.readline(256*1024+1):
            if len(line)>256*1024: raise ValueError('Tile record exceeds bound')
            row=json.loads(line)
            if row.get('status')=='skipped': skipped+=1;continue
            p=row['provenance'];generated+=1
            manifest_path=verify_file(p['native_input_manifest'],root)
            manifest=read_json(manifest_path)
            for name in ('image','mask'): verify_file(manifest[name],manifest_path.parent)
            native=read_json(verify_file(p['native_provenance'],root))
            for name in ('native_raw','native_cog'): verify_file(p[name],root)
            if (native['input']!=manifest or native['recipe']!=p['recipe']
                    or native['raw']!=p['native_raw'] or native['model']!=p['model']
                    or native['tokenizer']!=p['tokenizer']
                    or native['input_manifest']!=p['native_input_manifest']):
                raise ValueError('Retained native/source identity differs')
    if (generated,skipped)!=(scene['tiles_generated'],scene['tiles_skipped']):
        raise ValueError('Tile count differs from completed scene')
    return {'passed':True,'completion':binding(root/'completion.json'),'job_id':receipt['job_id'],
            'attempt_id':receipt['attempt_id'],'tiles_verified':generated,'tiles_skipped':skipped,
            'model_calls':0,'automatic_retries':0,'scope':'Retained content identity; no performance or scientific quality claim'}


@dataclass
class _Outcome:
    result: Metadata | None = None
    error: str | None = None
    cancelled: bool = False


class BestWorkHandle:
    """Bounded receipt ownership; consume or discard once. Files remain retained."""
    def __init__(self, owner: BestWorkQueue, request: BestSceneRequest) -> None:
        self._owner,self._request=owner,request
        self._event=Event()
        self._future: Future[_Outcome] = Future()
        self._consumed=False
        self._discard_requested=False

    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> bool:
        if self.done(): return False
        self._event.set()
        # Do not cancel the Future: its WorkItem would remain in the executor's
        # unbounded internal queue. Credit stays owned until the worker drains it.
        return True

    def result(self, timeout: float | None=None) -> Metadata:
        validate_timeout(timeout)
        with self._owner._lock:
            if self._discard_requested: raise RuntimeError('Result already discarded')
        try: outcome=self._future.result(timeout)
        except CancelledError:
            self._owner._release(self);raise
        with self._owner._lock:
            if self._consumed: raise RuntimeError('Result already consumed or discarded')
            self._consumed=True;self._owner._release(self)
            result,error,cancelled=outcome.result,outcome.error,outcome.cancelled
            outcome.result=outcome.error=None
        if cancelled: raise CancelledError('Scene cancelled; partial files are not delivered')
        if error is not None: raise BestWorkflowError(error)
        assert result is not None
        return result

    def discard(self) -> None:
        """Cancel cooperatively if active, then release receipt credit when done."""
        def release(future: Future[_Outcome]) -> None:
            with self._owner._lock:
                if not future.cancelled():
                    value=future.result();value.result=value.error=None
                self._consumed=True;self._owner._release(self)
        with self._owner._lock:
            if self._consumed or self._discard_requested: return
            self._discard_requested=True
            self.cancel()
            self._future.add_done_callback(release)


class BestWorkQueue:
    """One resident, bounded descriptors/results and conservative lifetime disk admission.

    Caller lends the resident exclusively until close returns. max_jobs counts
    queued/running/unconsumed receipts together. max_admitted_bytes reserves each
    request's workspace+output maximum for this queue's lifetime, even after
    consumption, because files remain retained. It is not disk GC or a durable
    broker. Native calls are never interrupted or retried.
    """
    def __init__(self, session: BestTerraMindSession, *, max_jobs: int=2,
                 max_admitted_bytes: int=8*1024**3, execute_model: bool=False) -> None:
        if execute_model is not True: raise ExecutionRequired('Pass execute_model=True explicitly')
        self.max_jobs=positive(max_jobs,'max_jobs')
        self.max_admitted_bytes=positive(max_admitted_bytes,'max_admitted_bytes')
        self.session=session;self._lock=RLock();self._handles: set[BestWorkHandle]=set()
        self._admitted_bytes=0;self._accepting=True;self._failed=False;self._high_water=0
        self._worker_ident: int | None=None
        with _OWNERS_LOCK:
            state=_state(session);_resident(session)
            if state.queue is not None or state.gate.locked(): raise BestWorkflowBusy('Resident already owned')
            self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='aureon-best-work')
            state.queue=self

    def _release(self, handle: BestWorkHandle) -> None:
        with self._lock: self._handles.discard(handle)

    def _run(self, handle: BestWorkHandle) -> _Outcome:
        self._worker_ident=get_ident()
        try:
            _cancel(handle._event)
            return _Outcome(result=_execute(self.session,handle._request,handle._event,self))
        except BaseException as exc:
            # Store no traceback or native tensor-owning exception in the Future.
            return _Outcome(error=type(exc).__name__+': '+str(exc)[:4096],cancelled=isinstance(exc,CancelledError))
        finally:
            with self._lock:
                if self.session.metadata()['state'] in ('failed','closed','closing'):
                    self._failed=True;self._accepting=False
                self._worker_ident=None

    def submit(self, request: BestSceneRequest) -> BestWorkHandle:
        if not isinstance(request,BestSceneRequest): raise TypeError('BestSceneRequest required')
        with self._lock:
            if not self._accepting: raise BestWorkflowError('Queue is closing or terminal')
            reservation=request.max_workspace_bytes+request.max_output_bytes
            if len(self._handles)>=self.max_jobs or self._admitted_bytes+reservation>self.max_admitted_bytes:
                raise BestWorkflowBusy('Consume/discard results or review retained storage before more admission')
            path=Path(request.output).resolve()
            for handle in self._handles:
                other=Path(handle._request.output).resolve()
                if path==other or path.is_relative_to(other) or other.is_relative_to(path):
                    raise ValueError('Admitted output paths must be distinct and non-nested')
                if (request.job_id,request.attempt_id)==(handle._request.job_id,handle._request.attempt_id):
                    raise ValueError('Job/attempt is already admitted')
            handle=BestWorkHandle(self,request);self._handles.add(handle);self._admitted_bytes+=reservation
            try: handle._future=self._executor.submit(self._run,handle)
            except BaseException:
                self._handles.remove(handle);self._admitted_bytes-=reservation;raise
            self._high_water=max(self._high_water,len(self._handles))
            return handle

    def metadata(self) -> Metadata:
        with self._lock:
            return {'accepting':self._accepting,'failed':self._failed,'reserved_jobs':len(self._handles),
                'max_jobs':self.max_jobs,'high_water_jobs':self._high_water,'admitted_storage_budget_bytes':self._admitted_bytes,
                'max_admitted_bytes':self.max_admitted_bytes,'session_ownership':'caller','automatic_retries':0}

    def close(self, *, cancel_pending: bool=False, timeout: float | None=None) -> Metadata:
        validate_timeout(timeout)
        if get_ident()==self._worker_ident: raise BestWorkflowBusy('Worker cannot wait for its own queue')
        if type(cancel_pending) is not bool: raise TypeError('cancel_pending must be bool')
        with self._lock:
            self._accepting=False;handles=list(self._handles)
            if cancel_pending:
                for handle in handles: handle.cancel()
        _,pending=wait([h._future for h in handles],timeout=timeout)
        if pending: raise TimeoutError('Queue still draining; session remains exclusively borrowed')
        self._executor.shutdown(wait=True)
        with _OWNERS_LOCK:
            state=_state(self.session)
            if state.queue is self: state.queue=None
        return {'passed':True,'session_closed':False,**self.metadata()}

    def __enter__(self) -> BestWorkQueue:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        from .lifecycle import close_preserving_error
        close_preserving_error(lambda:self.close(cancel_pending=exc is not None),exc,context='Best work queue close failed')
        return False
