"""Real raster plumbing with CPU doubles; no model or performance claims."""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
import json
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine

from aureon.api import scene_generation as workflow
from aureon.api.scene_generation import (SceneGenerationRequest, inspect_generation_scene,
    run_generation_scene, run_generation_scene_with_engine)
from aureon.api.generation_engine import GenerationEngine
from aureon.api.memory_input import GenerationInput
from aureon.api.spatial import PreparedGenerationRaster, generate_raster_scene, generation_windows, generation_tile_seed
from aureon.api.v1 import ExecutionRequired
from aureon.terramind_native.common import INPUT_BANDS
from aureon.terramind_native.memory_output import array_identity
from aureon.terramind_native.session import SessionBusy, SessionClosed


class FakeEngine:
    """Constant-by-seed fixture permits independent overlap reconstruction."""
    def __init__(self, *args, event=None, fail=False, **kwargs):
        self.calls = []
        self.closed = False
        self.event = event
        self.fail = fail
        self.max_live_jobs = 0
        self.cleanup_passed = True

    def generate(self, value, *, seed, job_id):
        if self.fail:
            raise RuntimeError('fixture forward failure')
        image, mask, metadata, identity = value.snapshot()
        self.calls.append((metadata['source_id'], seed))
        pixels = np.full((2,224,224), seed % 97, dtype=np.float32)
        result = {'array':pixels, 'valid_mask':mask, 'provenance':{
            'input':metadata, 'array':array_identity(pixels), 'valid_mask':array_identity(mask),
            'synthetic':True, 'sensor_observation':False, 'fixture':True}}
        if self.event is not None:
            self.event.set()
        return result

    @contextmanager
    def generate_many(self, jobs, *, max_in_flight):
        jobs = list(jobs)
        self.max_live_jobs = max(self.max_live_jobs, len(jobs))
        self.assert_bound = len(jobs) <= max_in_flight
        yield (self.generate(job.input_data, seed=job.recipe.seed, job_id=job.job_id) for job in jobs)

    def close(self, **kwargs):
        self.closed = True
        return {'passed':self.cleanup_passed, 'model_load_count':int(bool(self.calls)), 'fixture':True}


class ResidentDouble(GenerationEngine):
    """Real public facade/queue; replaces only neural execution with CPU values."""
    def __init__(self, root, **kwargs):
        super().__init__(model_file=root/'model.pt',tokenizer_file=root/'tokenizer.pt',
            work=root/'work',execute_model=True,**kwargs)
        self.fixture=FakeEngine()
        self.started=None;self.release=None
        self.opens=0;self.close_calls=0

    def _generate_one(self, value, *, recipe, execution_profile, job_id, attempt_id):
        if not self.work.exists():
            self.work.mkdir();self.opens+=1
        if self.started is not None:
            self.started.set()
            if not self.release.wait(10):
                raise RuntimeError('CPU fixture was not released')
        result=self.fixture.generate(value,seed=recipe['seed'],job_id=job_id)
        self._state='ready'
        return result

    def close(self, **kwargs):
        self.close_calls+=1
        return super().close(**kwargs)


class SceneWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='scene-workflow-')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def rasters(self, height=448, width=449, *, invalid=None, nan=None, mask_value=1):
        values = np.full((12,height,width), 1000, dtype=np.float32)
        validity = np.full((height,width), mask_value, dtype=np.uint8)
        if invalid is not None:
            validity[invalid] = 0
        if nan is not None:
            values[(0,*nan)] = np.nan
        grid = dict(driver='GTiff', width=width, height=height, crs='EPSG:32647',
            transform=Affine(10,0,632260,0,-10,1583830), tiled=True,
            blockxsize=128, blockysize=128)
        image, mask = self.root/'image.tif', self.root/'valid.tif'
        with rasterio.open(image,'w',count=12,dtype='float32',**grid) as ds:
            ds.write(values)
        with rasterio.open(mask,'w',count=1,dtype='uint8',**grid) as ds:
            ds.write(validity,1)
        return image, mask, validity

    def request(self, **kwargs):
        return SceneGenerationRequest(self.root/'image.tif', self.root/'valid.tif',
            self.root/'model.pt', self.root/'tokenizer.pt', self.root/'work', self.root/'out',
            min_free_bytes=0, **kwargs)

    def resident(self, **kwargs):
        for name in ('model.pt','tokenizer.pt'):
            (self.root/name).write_bytes(b'CPU fixture; not a model')
        engine=ResidentDouble(self.root,**kwargs)
        self.addCleanup(engine.close)
        return engine

    def test_borrowed_engine_reuses_residency_without_owned_cleanup(self):
        self.rasters(224,224)
        engine=self.resident(max_pending=3,max_results=3)
        request=self.request()
        first=run_generation_scene_with_engine(engine,request,execute_model=True)
        second=run_generation_scene_with_engine(engine,replace(request,output=self.root/'out2',max_in_flight=3),execute_model=True)
        self.assertEqual(engine.opens,1)
        self.assertEqual(engine.close_calls,0)
        self.assertEqual(first['engine_ownership'],'caller')
        self.assertIsNone(first['engine_cleanup'])
        self.assertFalse(second['resident_engine']['closed_by_scene'])
        self.assertEqual(engine.fixture.calls,[('r0-c0',17),('r0-c0',17)])
        with rasterio.open(first['cog']['path']) as left,rasterio.open(second['cog']['path']) as right:
            np.testing.assert_array_equal(left.read(),right.read())
        self.assertEqual(engine.metadata()['execution_profile'],'recommended')

    def test_borrowed_identity_and_capacity_mismatches_stop_before_source_io(self):
        engine=self.resident()
        request=self.request()
        changes=[{'model':'terramind-v1-tiny'},{'execution_profile':'reference'},
            {'model_file':self.root/'other.pt'},{'tokenizer_file':self.root/'other.pt'},
            {'work':self.root/'other-work'},{'max_in_flight':3}]
        for change in changes:
            with self.subTest(change=change),patch.object(workflow,'_source') as source:
                with self.assertRaises(ValueError):
                    run_generation_scene_with_engine(engine,replace(request,**change),execute_model=True)
                source.assert_not_called()
        self.assertEqual(engine.close_calls,0)
        self.assertEqual(engine.fixture.calls,[])

    def test_borrowed_changed_assets_and_terminal_engine_are_rejected(self):
        self.rasters(224,224)
        engine=self.resident()
        run_generation_scene_with_engine(engine,self.request(),execute_model=True)
        (self.root/'model.pt').write_bytes(b'changed CPU fixture')
        with patch.object(workflow,'_source') as source:
            with self.assertRaisesRegex(ValueError,'changed since its first scene'):
                run_generation_scene_with_engine(engine,replace(self.request(),output=self.root/'out2'),execute_model=True)
            source.assert_not_called()
        self.assertEqual(engine.close_calls,0)

    def test_borrowed_asset_change_during_execution_prevents_completion(self):
        self.rasters(224,224)
        engine=self.resident()
        generate=engine.fixture.generate
        def mutate(*args,**kwargs):
            result=generate(*args,**kwargs)
            (self.root/'tokenizer.pt').write_bytes(b'changed during scene')
            return result
        with patch.object(engine.fixture,'generate',mutate):
            with self.assertRaisesRegex(ValueError,'changed during scene execution'):
                run_generation_scene_with_engine(engine,self.request(),execute_model=True)
        self.assertEqual(engine.close_calls,0)
        self.assertFalse((self.root/'out/completion.json').exists())

    def test_borrowed_close_and_early_cancellation_do_not_touch_caller_lifecycle(self):
        engine=self.resident()
        event=Event();event.set()
        with self.assertRaises(ExecutionRequired):
            run_generation_scene_with_engine(engine,self.request())
        with self.assertRaises(CancelledError):
            run_generation_scene_with_engine(engine,self.request(),execute_model=True,cancel_event=event)
        self.assertEqual(engine.close_calls,0)
        engine.close()
        with self.assertRaises(SessionClosed):
            run_generation_scene_with_engine(engine,self.request(),execute_model=True)
        self.assertEqual(engine.close_calls,1)

    def test_borrowed_failure_and_cancellation_preserve_engine_and_profile(self):
        self.rasters(224,224)
        engine=self.resident(execution_profile='reference')
        request=self.request(execution_profile='reference')
        engine.fixture.fail=True
        with self.assertRaisesRegex(RuntimeError,'fixture forward failure'):
            run_generation_scene_with_engine(engine,request,execute_model=True)
        engine.fixture.fail=False
        event=Event();engine.fixture.event=event
        with self.assertRaises(CancelledError):
            run_generation_scene_with_engine(engine,replace(request,output=self.root/'cancelled'),
                execute_model=True,cancel_event=event)
        self.assertEqual(engine.close_calls,0)
        self.assertEqual(engine.metadata()['execution_profile'],'reference')
        engine.fixture.event=None
        result=run_generation_scene_with_engine(engine,replace(request,output=self.root/'successful'),execute_model=True)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(engine.opens,1)

    def test_overlapping_borrowed_scenes_are_rejected_without_cancelling_owner(self):
        self.rasters(224,224)
        engine=self.resident();engine.started=Event();engine.release=Event()
        results=[];errors=[]
        def first():
            try:results.append(run_generation_scene_with_engine(engine,self.request(),execute_model=True))
            except BaseException as exc:errors.append(exc)
        thread=Thread(target=first);thread.start()
        try:
            self.assertTrue(engine.started.wait(5))
            with self.assertRaisesRegex(SessionBusy,'Another scene'):
                run_generation_scene_with_engine(engine,replace(self.request(),output=self.root/'other'),execute_model=True)
            self.assertEqual(engine.close_calls,0)
        finally:
            engine.release.set();thread.join(10)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
        self.assertEqual(len(results),1)

    def test_existing_queue_work_is_not_drained_or_cancelled_by_borrow(self):
        engine=self.resident();engine.started=Event();engine.release=Event()
        value=GenerationInput(np.ones((12,224,224),np.float32),bands=INPUT_BANDS,
            units='surface_reflectance_x10000',valid_mask=np.ones((224,224),bool))
        handle=engine.submit(value,seed=19)
        try:
            self.assertTrue(engine.started.wait(5))
            with patch.object(workflow,'_source') as source:
                with self.assertRaisesRegex(SessionBusy,'existing caller work'):
                    run_generation_scene_with_engine(engine,self.request(),execute_model=True)
                source.assert_not_called()
            self.assertTrue(handle.running());self.assertEqual(engine.close_calls,0)
        finally:
            engine.release.set()
        self.assertEqual(handle.result(timeout=10)['array'][0,0,0],19)

    def test_queued_borrow_cancellation_retires_only_its_stream_and_reuses_engine(self):
        self.rasters(448,448)
        engine=self.resident(max_pending=3,max_results=3)
        event=Event();engine.fixture.event=event
        with self.assertRaises(CancelledError):
            run_generation_scene_with_engine(engine,self.request(max_in_flight=3),
                execute_model=True,cancel_event=event)
        self.assertEqual(engine.close_calls,0)
        # Discard never preempts the active tile. Wait only for this fixture's
        # already-admitted work; the production helper performs no global drain.
        for _ in range(100):
            queue=engine.metadata()['queue']
            if not queue['pending'] and not queue['reserved_results']:
                break
            Event().wait(.01)
        self.assertEqual(engine.metadata()['queue']['pending'],0)
        self.assertEqual(engine.metadata()['queue']['reserved_results'],0)
        self.assertLessEqual(len(engine.fixture.calls),3)
        engine.fixture.event=None
        receipt=run_generation_scene_with_engine(engine,
            replace(self.request(max_in_flight=3),output=self.root/'after-cancel'),execute_model=True)
        self.assertEqual(receipt['scene']['tiles_generated'],4)
        self.assertEqual(engine.opens,1);self.assertEqual(engine.close_calls,0)

    def test_large_scene_uses_bounded_windows_and_reconstruction(self):
        # Stream-create a 3.6-million-pixel source. Never allocate its 174-MiB
        # full twelve-band tensor; the neural double returns small owned tiles.
        from rasterio.windows import Window
        height,width=1793,2017
        grid=dict(driver='GTiff',width=width,height=height,crs='EPSG:32647',
            transform=Affine(10,0,632260,0,-10,1583830),tiled=True,
            blockxsize=224,blockysize=224,compress='DEFLATE')
        tile=np.full((12,224,224),1000,np.float32);valid=np.ones((224,224),np.uint8)
        with rasterio.open(self.root/'image.tif','w',count=12,dtype='float32',**grid) as image, \
                rasterio.open(self.root/'valid.tif','w',count=1,dtype='uint8',**grid) as mask:
            for row in range(0,height,224):
                for col in range(0,width,224):
                    h,w=min(224,height-row),min(224,width-col)
                    window=Window(col,row,w,h)
                    image.write(tile[:,:h,:w],window=window)
                    mask.write(valid[:h,:w],1,window=window)
        del tile,valid
        engine=self.resident(max_pending=3,max_results=3)
        bounds=[];pixels=PreparedGenerationRaster._pixels
        def bounded(source,row,col,h,w):
            bounds.append((h,w));return pixels(source,row,col,h,w)
        tracemalloc.start()
        try:
            with patch.object(PreparedGenerationRaster,'_pixels',bounded):
                result=run_generation_scene_with_engine(engine,self.request(max_in_flight=3),execute_model=True)
            _,peak=tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak,64*1024**2)  # Python/NumPy tracked payload only, not RSS/GDAL.
        self.assertTrue(all(h<=224 and w<=224 for h,w in bounds))
        self.assertEqual(result['scene']['tiles_generated'],90)
        self.assertEqual(result['scene']['generated_pixels'],height*width)
        self.assertEqual(result['scene']['reconstruction_buffer_bound_bytes'],6*224*width*4)
        # Independent block reconstruction checks all output pixels, including
        # overlap caused by terminal anchoring, without reading the full COG.
        with rasterio.open(result['cog']['path']) as output:
            for _,window in output.block_windows(1):
                r,c,h,w=map(int,(window.row_off,window.col_off,window.height,window.width))
                expected=np.zeros((h,w),np.float32);counts=np.zeros((h,w),np.float32)
                for tile_window in generation_windows(height,width):
                    top,left=max(r,tile_window.row),max(c,tile_window.col)
                    bottom,right=min(r+h,tile_window.row+224),min(c+w,tile_window.col+224)
                    if top<bottom and left<right:
                        expected[top-r:bottom-r,left-c:right-c]+=np.float32(generation_tile_seed(17,tile_window)%97)
                        counts[top-r:bottom-r,left-c:right-c]+=1
                np.testing.assert_array_equal(output.read(1,window=window),expected/counts)

    def test_preflight_cache_is_bounded_and_restored_on_success_and_failure(self):
        from rasterio.env import get_gdal_config
        self.rasters(224,224)
        original=PreparedGenerationRaster.inspect_validity
        initial=get_gdal_config('GDAL_CACHEMAX')
        observed=[];sources=[]
        def inspect(source, **kwargs):
            sources.append(source)
            observed.append(get_gdal_config('GDAL_CACHEMAX'))
            return original(source, **kwargs)
        def fail(source, **kwargs):
            sources.append(source)
            observed.append(get_gdal_config('GDAL_CACHEMAX'))
            raise RuntimeError('inspection failed')
        with rasterio.Env(GDAL_CACHEMAX=7*1024**2):
            with patch.object(PreparedGenerationRaster,'inspect_validity',inspect):
                plan=inspect_generation_scene(self.request())
            self.assertEqual(get_gdal_config('GDAL_CACHEMAX'),7*1024**2)
            self.assertEqual(plan['gdal_cache_bytes'],observed[0])
            with patch.object(PreparedGenerationRaster,'inspect_validity',fail):
                with self.assertRaisesRegex(RuntimeError,'inspection failed'):
                    inspect_generation_scene(self.request())
            self.assertEqual(get_gdal_config('GDAL_CACHEMAX'),7*1024**2)
        self.assertEqual(get_gdal_config('GDAL_CACHEMAX'),initial)
        self.assertEqual(observed,[64*1024**2,64*1024**2])
        self.assertTrue(all(source._image is None and source._mask is None and source._environment is None for source in sources))

    def test_nested_export_cache_restores_source_then_caller(self):
        from rasterio.env import get_gdal_config
        self.rasters(224,224)
        observed=[];engine=FakeEngine()
        generate=engine.generate
        def capture(*args,**kwargs):
            observed.append(get_gdal_config('GDAL_CACHEMAX'))
            return generate(*args,**kwargs)
        with rasterio.Env(GDAL_CACHEMAX=9*1024**2):
            with PreparedGenerationRaster(self.root/'image.tif',self.root/'valid.tif',bands=INPUT_BANDS,
                    units='surface_reflectance_x10000') as source:
                self.assertEqual(get_gdal_config('GDAL_CACHEMAX'),64*1024**2)
                with patch.object(engine,'generate',capture):
                    generate_raster_scene(engine,source,self.root/'direct',execute_model=True,min_free_bytes=0)
                self.assertEqual(get_gdal_config('GDAL_CACHEMAX'),64*1024**2)
            self.assertEqual(get_gdal_config('GDAL_CACHEMAX'),9*1024**2)
        self.assertEqual(observed,[64*1024**2])

    def test_source_open_failure_restores_cache_and_closes_first_handle(self):
        from rasterio.env import get_gdal_config
        self.rasters(224,224)
        # Existing but invalid mask file fails after opening the image handle.
        (self.root/'bad-mask.tif').write_bytes(b'not a raster')
        source=PreparedGenerationRaster(self.root/'image.tif',self.root/'bad-mask.tif',bands=INPUT_BANDS,
            units='surface_reflectance_x10000')
        with rasterio.Env(GDAL_CACHEMAX=11*1024**2):
            with self.assertRaises(rasterio.errors.RasterioIOError):
                with source:
                    pass
            self.assertEqual(get_gdal_config('GDAL_CACHEMAX'),11*1024**2)
        self.assertIsNone(source._image)
        self.assertIsNone(source._mask)
        self.assertIsNone(source._environment)

    def test_external_raster_masks_are_rejected_before_model_load(self):
        self.rasters(224,224)
        # Real GDAL .msk files can override the internal nodata/mask semantics.
        for filename in ('image.tif','valid.tif'):
            with self.subTest(filename=filename):
                path=self.root/filename
                with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
                    with rasterio.open(path,'r+') as dataset:
                        dataset.write_mask(np.ones((224,224),dtype='uint8')*255)
                self.assertTrue(Path(str(path)+'.msk').is_file())
                with patch('aureon.api.generation_engine.GenerationEngine') as factory:
                    with self.assertRaisesRegex(ValueError,'external sidecars'):
                        run_generation_scene(self.request(),execute_model=True)
                    factory.assert_not_called()
                Path(str(path)+'.msk').unlink()

    def test_sidecars_added_after_open_are_rejected_on_read_and_delivery(self):
        self.rasters(224,224)
        for filename in ('image.tif.msk','valid.tif.aux.xml','image.tfw'):
            with self.subTest(filename=filename):
                with PreparedGenerationRaster(self.root/'image.tif',self.root/'valid.tif',
                        bands=INPUT_BANDS,units='surface_reflectance_x10000') as source:
                    companion=self.root/filename
                    companion.write_bytes(b'external dependency appeared')
                    with self.assertRaisesRegex(ValueError,'external sidecars'):
                        source.read(next(generation_windows(224,224)))
                    with self.assertRaisesRegex(ValueError,'external sidecars'):
                        source.verify_unchanged()
                    companion.unlink()
                    # Removing the rejected addition restores the exact admitted
                    # self-contained dependency set; the source bytes never changed.
                    source.verify_unchanged()

    def test_non_self_contained_driver_is_rejected(self):
        from rasterio.shutil import copy as raster_copy
        self.rasters(224,224)
        raster_copy(self.root/'image.tif',self.root/'image.vrt',driver='VRT')
        with self.assertRaisesRegex(ValueError,'external dependencies'):
            inspect_generation_scene(replace(self.request(),image=self.root/'image.vrt'))

    def test_receipt_write_failure_preserves_model_error_and_cancellation(self):
        self.rasters(224,224)
        for cancelled in (False,True):
            with self.subTest(cancelled=cancelled):
                event=Event()
                engine=FakeEngine(event=event if cancelled else None,fail=not cancelled)
                request=replace(self.request(),work=self.root/f'w-{cancelled}',output=self.root/f'o-{cancelled}')
                with patch('aureon.api.generation_engine.GenerationEngine',return_value=engine), \
                        patch('aureon.api.spatial.write_json',side_effect=OSError('disk full')):
                    with self.assertRaises(CancelledError if cancelled else RuntimeError) as raised:
                        run_generation_scene(request,execute_model=True,cancel_event=event)
                self.assertTrue(engine.closed)
                self.assertTrue(any('Scene failure receipt could not be written: disk full' in note
                                    for note in raised.exception.__notes__))
                if not cancelled:
                    self.assertEqual(str(raised.exception),'fixture forward failure')
                self.assertFalse((request.output/'completion.json').exists())

    def test_plan_is_read_only_and_accounts_remainder_before_engine(self):
        self.rasters()
        request = self.request()
        with patch('aureon.api.generation_engine.GenerationEngine') as factory:
            plan = inspect_generation_scene(request)
            factory.assert_not_called()
        self.assertEqual(plan['tiles_planned'], 6)
        self.assertEqual(plan['tiles_eligible'], 6)
        self.assertEqual(plan['source_validity']['source_valid_pixels'],448*449)
        self.assertFalse(plan['model_execution'])
        self.assertFalse((self.root/'work').exists())
        self.assertFalse((self.root/'out').exists())

    def test_default_reject_preflight_stops_before_first_model_or_output(self):
        self.rasters(invalid=(slice(447,448),slice(448,449)))
        with patch('aureon.api.generation_engine.GenerationEngine') as factory:
            with self.assertRaisesRegex(ValueError, 'invalid/cloud/nodata'):
                run_generation_scene(self.request(), execute_model=True)
            factory.assert_not_called()
        self.assertFalse((self.root/'out').exists())

    def test_explicit_execution_and_early_cancel_do_no_file_io(self):
        request = self.request()
        with self.assertRaises(ExecutionRequired):
            run_generation_scene(request)
        event = Event();event.set()
        with self.assertRaises(CancelledError):
            inspect_generation_scene(request,cancel_event=event)
        with self.assertRaises(CancelledError):
            run_generation_scene(request,execute_model=True,cancel_event=event)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_overlap_and_remainder_cog_match_independent_row_major_fp32(self):
        image, mask, validity = self.rasters(449,450)
        engine = FakeEngine()
        request = self.request(stride=160)
        expected_sum=np.zeros((2,449,450),np.float32);counts=np.zeros((449,450),np.float32)
        expected_seeds=[]
        for window in generation_windows(449,450,stride=160):
            seed=generation_tile_seed(17,window);expected_seeds.append((window.tile_id,seed))
            expected_sum[:,window.row:window.row+224,window.col:window.col+224]+=np.float32(seed%97)
            counts[window.row:window.row+224,window.col:window.col+224]+=1
        with patch('aureon.api.generation_engine.GenerationEngine',return_value=engine):
            result=run_generation_scene(request,execute_model=True)
        self.assertTrue(engine.closed)
        self.assertEqual(engine.calls,expected_seeds)
        self.assertTrue(result['engine_cleanup']['passed'])
        with rasterio.open(result['cog']['path']) as ds:
            np.testing.assert_array_equal(ds.read(),expected_sum/counts[None])
            self.assertTrue((ds.dataset_mask()==255).all())
            self.assertEqual(ds.crs.to_string(),'EPSG:32647')
            self.assertEqual(ds.tags(ns='IMAGE_STRUCTURE')['LAYOUT'],'COG')
            self.assertEqual(ds.tags()['synthetic'],'true')
        self.assertEqual(result['scene']['model_parity'],'not_evaluated')
        self.assertTrue((self.root/'out/completion.json').is_file())

    def test_skip_invalid_tiles_masks_valid_but_uncovered_pixels(self):
        # One invalid pixel drops top-left224 tile; other original-valid pixels
        # in that window must remain masked, never relabeled as generated.
        image,mask,validity=self.rasters(448,448,invalid=(0,0))
        engine=FakeEngine()
        with patch('aureon.api.generation_engine.GenerationEngine',return_value=engine):
            result=run_generation_scene(self.request(invalid_policy='skip_invalid_tiles',max_in_flight=3),execute_model=True)
        self.assertEqual(len(engine.calls),3)
        self.assertLessEqual(engine.max_live_jobs,3)
        self.assertEqual(result['scene']['tiles_skipped'],1)
        self.assertEqual(result['scene']['generated_pixels'],3*224*224)
        self.assertEqual(result['scene']['source_validity']['source_valid_pixels'],448*448-1)
        self.assertEqual(result['scene']['source_valid_pixels_uncovered'],224*224-1)
        with rasterio.open(result['cog']['path']) as ds:
            self.assertTrue((ds.dataset_mask()[:224,:224]==0).all())
            self.assertTrue(np.isnan(ds.read()[:,:224,:224]).all())
            self.assertTrue((ds.dataset_mask()[224:]==255).all())
        receipts=[json.loads(line) for line in (self.root/'out/tiles.jsonl').read_text().splitlines()]
        self.assertFalse(receipts[0]['model_executed'])
        self.assertEqual(receipts[0]['seed'],17)

    def test_all_invalid_scene_generates_no_tiles_and_empty_coverage_cog(self):
        self.rasters(448,449,mask_value=0)
        engine=FakeEngine()
        with patch('aureon.api.generation_engine.GenerationEngine',return_value=engine):
            result=run_generation_scene(self.request(invalid_policy='skip_invalid_tiles',max_in_flight=2),execute_model=True)
        self.assertEqual(engine.calls,[])
        self.assertEqual(result['scene']['generated_pixels'],0)
        self.assertEqual(result['scene']['tiles_skipped'],6)
        self.assertEqual(result['engine_cleanup']['model_load_count'],0)
        with rasterio.open(result['cog']['path']) as ds:
            self.assertTrue((ds.dataset_mask()==0).all())

    def test_masked_overlap_queue_and_serial_have_same_values_seeds_and_coverage(self):
        self.rasters(449,450,invalid=(0,0))
        request=self.request(invalid_policy='skip_invalid_tiles',stride=160)
        actual=[];seeds=[]
        for lanes in (1,3):
            engine=FakeEngine()
            case=replace(request,max_in_flight=lanes,work=self.root/('w'+str(lanes)),output=self.root/('o'+str(lanes)))
            with patch('aureon.api.generation_engine.GenerationEngine',return_value=engine):
                result=run_generation_scene(case,execute_model=True)
            with rasterio.open(result['cog']['path']) as ds:
                actual.append((ds.read(),ds.dataset_mask()))
            seeds.append(engine.calls)
        np.testing.assert_array_equal(actual[0][0],actual[1][0])
        np.testing.assert_array_equal(actual[0][1],actual[1][1])
        self.assertEqual(seeds[0],seeds[1])
        expected=np.zeros((2,449,450),np.float32);counts=np.zeros((449,450),np.float32)
        for window in generation_windows(449,450,stride=160):
            if window.row==window.col==0:
                continue
            r,c=window.row,window.col
            expected[:,r:r+224,c:c+224]+=np.float32(generation_tile_seed(17,window)%97)
            counts[r:r+224,c:c+224]+=1
        coverage=counts>0
        expected[:,~coverage]=np.nan
        np.divide(expected,counts[None],out=expected,where=coverage[None])
        np.testing.assert_array_equal(actual[0][0],expected)
        np.testing.assert_array_equal(actual[0][1],coverage.astype('uint8')*255)

    def test_nonfinite_source_is_invalid_not_silently_filled(self):
        self.rasters(448,448,nan=(300,300))
        plan=inspect_generation_scene(self.request(invalid_policy='skip_invalid_tiles'))
        self.assertEqual(plan['source_validity']['source_invalid_pixels'],1)
        self.assertEqual(plan['tiles_eligible'],3)
        with self.assertRaisesRegex(ValueError,'invalid/cloud/nodata'):
            inspect_generation_scene(self.request())

    def test_memory_storage_mask_and_destination_contracts_fail_early(self):
        self.rasters(mask_value=2)
        request=self.request()
        with self.assertRaisesRegex(ValueError,'memory budget'):
            inspect_generation_scene(replace(request,max_buffer_bytes=1))
        with self.assertRaisesRegex(ValueError,'max_output_bytes'):
            inspect_generation_scene(replace(request,max_output_bytes=1))
        with self.assertRaisesRegex(ValueError,'0 or 1'):
            inspect_generation_scene(request)
        with self.assertRaisesRegex(ValueError,'non-nested'):
            inspect_generation_scene(replace(request,output=request.work))

    def test_cancel_after_forward_retains_partial_and_closes_owner(self):
        self.rasters()
        event=Event();engine=FakeEngine(event=event)
        with patch('aureon.api.generation_engine.GenerationEngine',return_value=engine):
            with self.assertRaises(CancelledError):
                run_generation_scene(self.request(),execute_model=True,cancel_event=event)
        self.assertTrue(engine.closed)
        failure=json.loads((self.root/'out/failure.json').read_text())
        self.assertEqual(failure['status'],'cancelled')
        self.assertFalse((self.root/'out/completion.json').exists())

    def test_forward_failure_closes_owner_without_completion(self):
        self.rasters()
        engine=FakeEngine(fail=True)
        with patch('aureon.api.generation_engine.GenerationEngine',return_value=engine):
            with self.assertRaisesRegex(RuntimeError,'fixture forward failure'):
                run_generation_scene(self.request(),execute_model=True)
        self.assertTrue(engine.closed)
        self.assertFalse((self.root/'out/completion.json').exists())

    def test_cleanup_failure_is_not_completed_delivery(self):
        self.rasters(224,224)
        engine=FakeEngine();engine.cleanup_passed=False
        with patch('aureon.api.generation_engine.GenerationEngine',return_value=engine):
            with self.assertRaisesRegex(RuntimeError,'cleanup did not pass'):
                run_generation_scene(self.request(),execute_model=True)
        self.assertFalse((self.root/'out/completion.json').exists())

    def test_cancel_during_cog_export_never_delivers_completed_scene(self):
        from rasterio.shutil import copy as actual_copy
        self.rasters()
        event = Event(); engine = FakeEngine()
        def cancel_after_copy(*args, **kwargs):
            result = actual_copy(*args, **kwargs)
            event.set()
            return result
        with patch('aureon.api.generation_engine.GenerationEngine', return_value=engine), \
             patch('rasterio.shutil.copy', side_effect=cancel_after_copy):
            with self.assertRaises(CancelledError):
                run_generation_scene(self.request(), execute_model=True, cancel_event=event)
        self.assertTrue(engine.closed)
        self.assertFalse((self.root/'out/completion.json').exists())
        self.assertFalse((self.root/'out/provenance.json').exists())
        failure = json.loads((self.root/'out/failure.json').read_text())
        self.assertEqual(failure['status'], 'cancelled')
        self.assertFalse(failure['delivered'])

    def test_cancel_during_cog_readback_stops_before_next_block(self):
        self.rasters()
        event = Event(); engine = FakeEngine(); reads = []
        actual_open = rasterio.open
        class CancelAfterRead:
            def __init__(self, dataset): self.dataset = dataset
            def __enter__(self): self.dataset.__enter__(); return self
            def __exit__(self, *args): return self.dataset.__exit__(*args)
            def __getattr__(self, name): return getattr(self.dataset, name)
            def read(self, *args, **kwargs):
                result = self.dataset.read(*args, **kwargs)
                reads.append(kwargs.get('window')); event.set()
                return result
        def opened(path, *args, **kwargs):
            dataset = actual_open(path, *args, **kwargs)
            return CancelAfterRead(dataset) if Path(path).name == 'synthetic-s1grd.tif' else dataset
        with patch('aureon.api.generation_engine.GenerationEngine', return_value=engine), \
             patch('rasterio.open', side_effect=opened):
            with self.assertRaises(CancelledError):
                run_generation_scene(self.request(), execute_model=True, cancel_event=event)
        self.assertTrue(engine.closed)
        self.assertEqual(len(reads), 1)
        self.assertFalse((self.root/'out/completion.json').exists())
        self.assertEqual(json.loads((self.root/'out/failure.json').read_text())['status'], 'cancelled')

    def test_cancel_during_owned_cleanup_prevents_completion_receipt(self):
        self.rasters()
        event = Event(); engine = FakeEngine(); actual_close = engine.close
        def cancel_after_close(**kwargs):
            result = actual_close(**kwargs); event.set(); return result
        engine.close = cancel_after_close
        with patch('aureon.api.generation_engine.GenerationEngine', return_value=engine):
            with self.assertRaises(CancelledError):
                run_generation_scene(self.request(), execute_model=True, cancel_event=event)
        self.assertTrue(engine.closed)
        self.assertFalse((self.root/'out/completion.json').exists())


if __name__=='__main__':
    unittest.main()
