"""Explicit borrowed-free COP-GEN resident owner and finite local COG delivery."""
from __future__ import annotations

from copy import deepcopy
import gc
from pathlib import Path
import threading
from types import TracebackType
from typing import Any, Self

from ..api.eo import EORequest
from .contracts import CopgenInput, plan_copgen, digest
from .runtime import NativeRuntime, file_identity
from .profiles import profile_requirements

_OWNER=threading.Lock()


class CopgenEngine:
    """One owned CUDA process/profile, explicit local pinned assets, no downloads."""
    def __init__(self, *, source: str | Path, assets: str | Path, execute_model: bool=False,
                 dependency_profile: str='torch260') -> None:
        if execute_model is not True:raise ValueError('Pass execute_model=True explicitly')
        profile_requirements(dependency_profile)
        self.dependency_profile=dependency_profile
        self.source=Path(source);self.assets=Path(assets)
        self._runtime: NativeRuntime | None=None
        self._gate=threading.Lock();self._owns=False;self._closed=False;self._failed=False
        self._thread: int | None=None
        self._cleanup: dict[str,Any] | None=None

    def _check(self) -> None:
        if self._closed or self._failed:raise RuntimeError('COP-GEN engine is terminal')
        if self._thread is not None and self._thread!=threading.get_ident():
            raise RuntimeError('COP-GEN engine belongs to its loading thread')

    def _load(self) -> NativeRuntime:
        self._check()
        if self._runtime is None:
            if not _OWNER.acquire(blocking=False):raise RuntimeError('Another COP-GEN engine owns this process')
            self._owns=True;self._thread=threading.get_ident()
            try:self._runtime=NativeRuntime(self.source,self.assets,dependency_profile=self.dependency_profile)
            except BaseException:
                # A failed constructor cannot return its cleanup measurement.
                # Never turn that uncertainty into a fabricated zero allocator.
                self._cleanup={'construction_failed':True,'cleanup_unknown':True,
                               'detail':'See primary exception notes; runtime construction did not return'}
                self._failed=True;self._release();raise
        return self._runtime

    def load(self) -> dict[str,Any]:
        if not self._gate.acquire(blocking=False):raise RuntimeError('COP-GEN engine is busy')
        try:return deepcopy(self._load().identity)
        finally:self._gate.release()

    def generate(self, request: EORequest, input_data: CopgenInput) -> dict[str,Any]:
        # Reject unsupported route/data before a CUDA import, asset read or load.
        plan=plan_copgen(request)
        if not isinstance(input_data,CopgenInput):raise TypeError('Use validated CopgenInput')
        input_data.validate_plan(plan)
        if not self._gate.acquire(blocking=False):raise RuntimeError('COP-GEN engine is busy')
        runtime=None
        try:
            runtime=self._load()
            image,mask=input_data.snapshot()
            output=runtime.generate(image,plan)
            output['valid_mask']=mask
            output['provenance']={'schema':'aureon.copgen_generation.v1','model_id':'copgen-base',
                'synthetic':True,'sensor_observation':False,'physical_quality':'not_evaluated',
                'request_sha256':plan.request_sha256,'seed':plan.seed,'sample_steps':plan.sample_steps,
                'input':input_data.identity,'output':{'modality':'S1RTC','bands':['VV','VH'],'units':'linear_backscatter',
                'grid':request.targets[0].grid.native(),'validity':'generated coverage on explicit all-valid conditioning; not observation truth'},
                'loaded_identity':deepcopy(runtime.identity),'rng':deepcopy(output['rng']),
                'precision':deepcopy(output['precision']),'native_trace':deepcopy(output['native_trace']),
                'compatibility':'two used VAEs resident; all12 joint transformer modalities retained; native posterior and omitted-modality RNG unchanged',
                'numerical_fidelity':'requires explicit reference comparison; not automatically certified per job'}
            import hashlib
            output['provenance']['arrays']={name:{'shape':list(output[name].shape),'dtype':str(output[name].dtype),
                'sha256':hashlib.sha256(output[name].tobytes(order='C')).hexdigest()} for name in ('array','decoded','latent','valid_mask')}
            return output
        except BaseException as primary:
            self._failed=True
            self._runtime=None
            try:
                if runtime is not None:self._cleanup=runtime.close()
            except BaseException as cleanup:
                self._cleanup={'cleanup_failed':True,'error':repr(cleanup)}
                primary.add_note('COP-GEN cleanup also failed: '+repr(cleanup))
            finally:self._release()
            raise
        finally:self._gate.release()

    def _release(self) -> None:
        if self._owns:self._owns=False;_OWNER.release()

    def close(self) -> dict[str,Any]:
        if not self._gate.acquire(blocking=False):raise RuntimeError('Cannot close an active COP-GEN request')
        try:
            if self._thread is not None and self._thread!=threading.get_ident():raise RuntimeError('Close on the owning thread')
            runtime=self._runtime;self._runtime=None
            try:
                if runtime is not None:self._cleanup=runtime.close()
            except BaseException as cleanup:
                self._failed=True;self._cleanup={'cleanup_failed':True,'error':repr(cleanup)}
                raise
            finally:
                self._closed=True;self._release();gc.collect()
            return dict(self._cleanup or {'model_loaded':False,'cuda_memory':'not_measured'})
        finally:self._gate.release()

    def __enter__(self) -> Self:
        self._check()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        try:self.close()
        except BaseException as cleanup:
            if exc is None:raise
            exc.add_note('COP-GEN context cleanup also failed: '+repr(cleanup))


def export_copgen(result: dict[str,Any], output_dir: str | Path) -> dict[str,Any]:
    """Export one result into a new directory, then independently reopen the COG."""
    import hashlib,json
    import numpy as np
    import rasterio  # Optional Rasterio distribution has no typing marker.
    from rasterio.io import MemoryFile
    from rasterio.shutil import copy
    from ..api.eo import EOGrid
    values,mask=result['array'],result['valid_mask']
    provenance=deepcopy(result['provenance']);grid=provenance['output']['grid']
    checked=EOGrid(grid['crs'],tuple(grid['transform']),tuple(grid['shape']))
    if (values.shape!=(2,192,192) or values.dtype!=np.float32 or not np.isfinite(values).all()
            or mask.shape!=(192,192) or mask.dtype!=np.bool_ or not mask.all() or checked.shape!=(192,192)
            or not provenance['synthetic'] or provenance['sensor_observation'] is not False
            or provenance['output']['modality']!='S1RTC' or provenance['output']['units']!='linear_backscatter'):
        raise ValueError('Invalid explicit generated S1RTC result')
    for name in ('array','decoded','latent','valid_mask'):
        if hashlib.sha256(result[name].tobytes(order='C')).hexdigest()!=provenance['arrays'][name]['sha256']:
            raise ValueError('Result changed after inference')
    path=Path(output_dir)
    if path.exists() or path.is_symlink():raise FileExistsError(path)
    path=path.parent.resolve(strict=True)/path.name
    path.mkdir(exist_ok=False)
    transform=rasterio.Affine(*checked.transform)
    cog=path/'synthetic-s1rtc.tif'
    with rasterio.Env(GDAL_CACHEMAX=64*1024**2,GDAL_TIFF_INTERNAL_MASK=True),MemoryFile() as memory:
        with memory.open(driver='GTiff',width=192,height=192,count=2,dtype='float32',crs=checked.crs,
                         transform=transform,tiled=True,blockxsize=256,blockysize=256) as ds:
            ds.write(values);ds.write_mask(mask.astype('uint8')*255)
            ds.set_band_description(1,'synthetic_S1RTC_VV');ds.set_band_description(2,'synthetic_S1RTC_VH')
            ds.update_tags(synthetic='true',sensor_observation='false',units='linear_backscatter',modality='S1RTC',physical_quality='not_evaluated')
        with memory.open() as ds:copy(ds,cog,driver='COG',compress='DEFLATE',blocksize=256,overview_count=0)
    with rasterio.open(cog) as ds:
        passed=(np.array_equal(ds.read(),values) and np.array_equal(ds.dataset_mask()>0,mask)
                and ds.crs==rasterio.crs.CRS.from_user_input(checked.crs) and ds.transform==transform
                and ds.tags(ns='IMAGE_STRUCTURE').get('LAYOUT')=='COG'
                and ds.tags().get('units')=='linear_backscatter' and ds.tags().get('modality')=='S1RTC')
    if not passed:raise RuntimeError('COG values/grid/mask/semantics round-trip failed')
    for name in ('array','decoded','latent','valid_mask'):np.save(path/(name+'.npy'),result[name],allow_pickle=False)
    (path/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    receipt={'cog':file_identity(cog),'decoded_values_exact':True,'grid_mask_exact':True,'synthetic':True,
             'sensor_observation':False,'modality':'S1RTC','units':'linear_backscatter',
             'provenance':file_identity(path/'provenance.json')}
    (path/'export.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
