"""Selected-session CPU contract doubles + real Rasterio, never a model benchmark."""
from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path
from threading import Event
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine

from aureon.api.best import BestTerraMindSession
from aureon.api import best_workflows as wf
from aureon.api.spatial import generation_windows, generation_tile_seed
from aureon.terramind_native.common import binding, read_json, verify_file, write_json
from aureon.terramind_native.geo import write_cog


class ResidentDouble(BestTerraMindSession):
    def __init__(self, root):
        self.work=root/'resident';self.work.mkdir()
        self.calls=[];self.state='new';self.close_calls=0
        self.started=self.release=self.cancel_event=None
        self.corrupt=None;self.fail=False

    def metadata(self):
        return {'work':str(self.work),'state':self.state,'model_load_count':int(bool(self.calls)),
                'jobs_completed':len(self.calls),'fixture':True}

    def close(self):
        self.close_calls+=1;self.state='closed';return {'passed':True}

    def run(self, input_manifest, *, job_id, attempt_id, recipe):
        if self.started is not None:
            self.started.set()
            if not self.release.wait(10): raise TimeoutError('Release CPU fixture')
        if self.fail: self.state='failed';raise RuntimeError('CPU terminal fixture')
        path=Path(input_manifest);m=read_json(path)
        image=np.load(verify_file(m['image'],path.parent),allow_pickle=False)
        mask=np.load(verify_file(m['mask'],path.parent),allow_pickle=False)
        assert image.shape==(12,224,224) and image.dtype==np.float32 and np.all(image==1000)
        assert mask.shape==(224,224) and mask.dtype==np.bool_ and mask.all()
        self.calls.append((m['source_id'],dict(recipe),job_id,attempt_id))
        target=self.work/'jobs'/job_id/attempt_id;target.mkdir(parents=True)
        values=np.full((2,224,224),recipe['seed']%97,dtype=np.float32)
        np.save(target/'raw.npy',values,allow_pickle=False)
        p={'job_id':job_id,'attempt_id':attempt_id,'input':m,'input_manifest':binding(path),
           'raw':binding(target/'raw.npy'),'recipe':recipe,'synthetic':True,'sensor_observation':False,
           'precision':{'parameters':'float32','matmul':'highest','tf32':False,'autocast':False},
           'model':{'fixture':'not a model'},'tokenizer':{'fixture':'not a tokenizer'}}
        cog=write_cog(target/'synthetic-s1grd.tif',values,mask,m['grid'],p)
        if self.corrupt=='cog_values':
            cog=write_cog(target/'wrong-cog.tif',values+np.float32(1),mask,m['grid'],p)
        if self.corrupt=='identity': p['attempt_id']='wrong'
        if self.corrupt=='policy': p['precision']['tf32']=True
        write_json(target/'provenance.json',p)
        result={'status':'completed','job_id':job_id,'attempt_id':attempt_id,'recipe':recipe,
                'raw':binding(target/'raw.npy'),'provenance':binding(target/'provenance.json'),'cog':cog}
        if self.corrupt=='hash': (target/'raw.npy').write_bytes(b'corrupt')
        if self.corrupt=='bound': result['raw']['bytes']=100*1024**2
        self.state='ready'
        if self.cancel_event is not None:self.cancel_event.set()
        return result


class BestWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='best-workflow-',dir=os.getenv('AUREON_TEST_SCRATCH')))
        self.resident=ResidentDouble(self.root)
        self.image=self.root/'image.tif';self.mask=self.root/'mask.tif'

    def raster(self,height=224,width=224,invalid=None):
        grid=dict(driver='GTiff',height=height,width=width,crs='EPSG:32647',
                  transform=Affine(10,0,632260,0,-10,1583830),tiled=True,blockxsize=128,blockysize=128)
        with rasterio.open(self.image,'w',count=12,dtype='float32',**grid) as ds:
            ds.write(np.full((12,height,width),1000,dtype=np.float32))
        mask=np.ones((height,width),dtype=np.uint8)
        if invalid is not None:mask[invalid]=0
        with rasterio.open(self.mask,'w',count=1,dtype='uint8',**grid) as ds:ds.write(mask,1)
        return mask

    def request(self,name='one',**kwargs):
        return wf.BestSceneRequest(self.image,self.mask,self.root/name,job_id=name,min_free_bytes=0,**kwargs)

    def run_scene(self,request=None,**kwargs):
        return wf.run_best_scene(self.resident,request or self.request(),execute_model=True,**kwargs)

    def queue(self,**kwargs):
        q=wf.BestWorkQueue(self.resident,execute_model=True,**kwargs)
        self.addCleanup(q.close,cancel_pending=True,timeout=15)
        return q

    def test_opt_in_and_invalid_contracts(self):
        with self.assertRaises(wf.ExecutionRequired):wf.run_best_scene(self.resident,self.request())
        with self.assertRaises(wf.ExecutionRequired):wf.BestWorkQueue(self.resident)
        for values in ({'stride':0},{'seed':True},{'hourly_rate':1},{'currency':'THB'},
                       {'hourly_rate':float('inf'),'currency':'THB'},{'attempt_id':'../x'},
                       {'max_workspace_bytes':True},{'min_free_bytes':-1}):
            with self.subTest(values=values),self.assertRaises((ValueError,TypeError)):
                self.request(**values)
        self.assertEqual(self.resident.calls,[])

    def test_plan_no_output_or_model_and_disk_admission(self):
        self.raster(449,289)
        plan=wf.inspect_best_scene(self.request())
        self.assertEqual(plan['tiles_planned'],6)
        self.assertFalse(Path(self.request().output).exists());self.assertEqual(self.resident.calls,[])
        for field in ('max_workspace_bytes','max_output_bytes','max_buffer_bytes'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.run_scene(replace(self.request(),**{field:1}))
        with patch.object(wf.shutil,'disk_usage',return_value=type('Disk',(),{'free':0})()):
            with self.assertRaises(ValueError):self.run_scene()
        self.assertEqual(self.resident.calls,[])

    def test_odd_scene_independent_reconstruction_seeds_and_recovery_verify(self):
        h,w=449,289;self.raster(h,w)
        request=self.request(hourly_rate=1,currency='THB')
        result=self.run_scene(request)
        sums=np.zeros((2,h,w),dtype=np.float32);counts=np.zeros((h,w),dtype=np.float32)
        expected=[]
        for tile in generation_windows(h,w):
            seed=generation_tile_seed(17,tile);expected.append((tile.tile_id,seed))
            sums[:,tile.row:tile.row+224,tile.col:tile.col+224]+=np.float32(seed%97)
            counts[tile.row:tile.row+224,tile.col:tile.col+224]+=1
        with rasterio.open(result['cog']['path']) as ds:
            np.testing.assert_array_equal(ds.read().view('uint32'),(sums/counts).view('uint32'))
            self.assertTrue((ds.dataset_mask()>0).all());self.assertEqual(ds.tags(ns='IMAGE_STRUCTURE')['LAYOUT'],'COG')
        self.assertEqual([(row[0],row[1]['seed']) for row in self.resident.calls],expected)
        self.assertTrue(wf.verify_best_scene(request.output,job_id='one',attempt_id='1')['passed'])
        self.assertEqual(result['cost']['hourly_rate'],1)
        self.assertAlmostEqual(result['cost']['amount'],result['timing']['elapsed_before_commit_seconds']/3600)
        self.assertEqual(self.resident.close_calls,0)
        with self.assertRaises(ValueError):wf.verify_best_scene(request.output,attempt_id='stale')

    def test_recovery_detects_source_and_native_corruption(self):
        self.raster();result=self.run_scene()
        source=self.image.read_bytes();self.image.write_bytes(source+b'changed')
        with self.assertRaises(ValueError):wf.verify_best_scene(self.request().output)
        self.image.write_bytes(source)
        tile=json.loads((Path(self.request().output)/'tiles.jsonl').read_text().splitlines()[0])
        Path(tile['provenance']['native_raw']['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):wf.verify_best_scene(self.request().output)
        self.assertEqual(result['status'],'completed')

    def test_invalid_default_reject_and_explicit_skipping_preserve_coverage(self):
        mask=self.raster(224,448,invalid=(0,0))
        with self.assertRaises(ValueError):self.run_scene()
        self.assertEqual(self.resident.calls,[])
        result=self.run_scene(self.request(invalid_policy='skip_invalid_tiles'))
        self.assertEqual((result['scene']['tiles_generated'],result['scene']['tiles_skipped']),(1,1))
        with rasterio.open(result['cog']['path']) as ds:
            self.assertTrue(np.isnan(ds.read()[:,:,:224]).all())
            self.assertFalse((ds.dataset_mask()[:,:224]>0).any())
            self.assertTrue((ds.dataset_mask()[:,224:]>0).all())
        self.assertTrue(wf.verify_best_scene(self.request().output)['passed'])
        self.assertTrue(mask[1,1])

    def test_all_invalid_no_model_calls(self):
        self.raster(invalid=(slice(None),slice(None)))
        result=self.run_scene(self.request(invalid_policy='skip_invalid_tiles'))
        self.assertEqual(self.resident.calls,[]);self.assertEqual(result['scene']['tiles_generated'],0)
        self.assertTrue(wf.verify_best_scene(self.request().output)['passed'])

    def test_native_hash_policy_identity_and_size_fail_without_completion(self):
        self.raster()
        for corruption in ('hash','identity','policy','bound','cog_values'):
            self.resident.corrupt=corruption;request=self.request(corruption)
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):self.run_scene(request)
            self.assertFalse((Path(request.output)/'completion.json').exists())
            with self.assertRaises(ValueError):wf.verify_best_scene(request.output)
        self.resident.corrupt=None
        self.assertEqual(self.run_scene(self.request('recovered'))['status'],'completed')

    def test_cancel_before_and_after_tile_fresh_attempt_recovers(self):
        self.raster(224,448);event=Event();event.set()
        with self.assertRaises(CancelledError):self.run_scene(cancel_event=event)
        self.assertFalse(Path(self.request().output).exists())
        event.clear();self.resident.cancel_event=event
        with self.assertRaises(CancelledError):self.run_scene(cancel_event=event)
        self.assertEqual(len(self.resident.calls),1)
        self.assertFalse((Path(self.request().output)/'completion.json').exists())
        event.clear();self.resident.cancel_event=None
        with self.assertRaises(FileExistsError):self.run_scene(cancel_event=event)
        next_request=replace(self.request(),attempt_id='2',output=self.root/'second-attempt')
        self.assertEqual(self.run_scene(next_request)['status'],'completed')
        self.assertEqual(self.resident.close_calls,0)

    def test_atomic_completion_failure_retains_output_and_failed_receipt(self):
        self.raster()
        with patch.object(wf,'_publish',side_effect=OSError('fixture filesystem cannot commit')):
            with self.assertRaises(OSError):self.run_scene()
        output=Path(self.request().output)
        self.assertTrue((output/'completion.ready.json').exists());self.assertTrue((output/'synthetic-s1grd.tif').exists())
        self.assertTrue((output/'failure.json').exists());self.assertFalse((output/'completion.json').exists())
        with self.assertRaises(ValueError):wf.verify_best_scene(output)

    def test_existing_and_nested_output_never_touched(self):
        self.raster();path=Path(self.request().output);path.mkdir();(path/'owned.txt').write_text('other owner')
        with self.assertRaises(ValueError):self.run_scene()
        self.assertEqual(sorted(p.name for p in path.iterdir()),['owned.txt'])
        with self.assertRaises(ValueError):self.run_scene(replace(self.request(),output=self.resident.work/'out'))
        self.assertEqual(self.resident.calls,[])

    def test_cancel_after_receipt_preparation_before_commit(self):
        self.raster();event=Event();original=wf.json.dumps
        def encode(value,*args,**kwargs):
            result=original(value,*args,**kwargs)
            if isinstance(value,dict) and value.get('schema')=='aureon.best_scene_delivery.v1':event.set()
            return result
        with patch.object(wf.json,'dumps',side_effect=encode):
            with self.assertRaises(CancelledError):self.run_scene(cancel_event=event)
        output=Path(self.request().output)
        self.assertTrue((output/'completion.ready.json').exists())
        self.assertFalse((output/'completion.json').exists())
        self.assertEqual(read_json(output/'failure.json')['status'],'cancelled')

    def test_publish_refuses_existing_destination_and_preserves_both(self):
        ready=self.root/'ready.json';ready.write_text('ready')
        completed=self.root/'complete.json';completed.write_text('other owner')
        with self.assertRaises(FileExistsError):wf._publish(ready,completed)
        self.assertEqual(completed.read_text(),'other owner');self.assertEqual(ready.read_text(),'ready')

    def test_output_creation_race_does_not_claim_foreign_directory(self):
        self.raster();output=Path(self.request().output)
        def race(*args,**kwargs):
            output.mkdir();(output/'foreign.txt').write_text('other owner')
            raise FileExistsError('Output lost creation race')
        with patch.object(wf,'generate_raster_scene',side_effect=race):
            with self.assertRaises(FileExistsError):self.run_scene()
        self.assertEqual([p.name for p in output.iterdir()],['foreign.txt'])

    def test_source_change_during_tile_rejected_before_delivery(self):
        self.raster();original=self.resident.run
        def changed(*args,**kwargs):
            result=original(*args,**kwargs)
            with self.image.open('ab') as stream:stream.write(b'source changed')
            return result
        with patch.object(self.resident,'run',side_effect=changed):
            with self.assertRaises(ValueError):self.run_scene()
        self.assertFalse((Path(self.request().output)/'completion.json').exists())

    def test_cancellation_does_not_release_credit_before_executor_drains(self):
        self.raster();self.resident.started=Event();self.resident.release=Event()
        q=self.queue(max_jobs=2);self.addCleanup(self.resident.release.set)
        first=q.submit(self.request());self.assertTrue(self.resident.started.wait(5))
        second=q.submit(self.request('two'))
        for i in range(100):
            second.cancel();second.discard()
            with self.assertRaises(wf.BestWorkflowBusy):q.submit(self.request('overflow'+str(i)))
        self.assertEqual(q.metadata()['reserved_jobs'],2);self.assertFalse(second.done())
        self.assertEqual(len(second._future._done_callbacks),1)
        with self.assertRaises(TimeoutError):q.close(cancel_pending=True,timeout=0)
        with self.assertRaises(wf.BestWorkflowBusy):wf.BestWorkQueue(self.resident,execute_model=True)
        self.resident.release.set()
        with self.assertRaises(CancelledError):first.result(10)
        q.close(timeout=10)
        self.assertEqual(q.metadata()['reserved_jobs'],0);self.assertEqual(self.resident.close_calls,0)
        self.assertEqual(len(self.resident.calls),1)

    def test_completed_result_credit_and_lifetime_storage_budget(self):
        self.raster();request=self.request();budget=request.max_workspace_bytes+request.max_output_bytes
        q=self.queue(max_jobs=1,max_admitted_bytes=budget)
        handle=q.submit(request);handle._future.result(10)
        with self.assertRaises(wf.BestWorkflowBusy):q.submit(self.request('two'))
        self.assertEqual(handle.result()['status'],'completed')
        with self.assertRaises(RuntimeError):handle.result()
        self.assertEqual(q.metadata()['reserved_jobs'],0)
        with self.assertRaises(wf.BestWorkflowBusy):q.submit(self.request('two'))
        q.close();self.assertTrue(wf.verify_best_scene(request.output)['passed'])

    def test_queue_exclusive_ownership_and_terminal_failure(self):
        self.raster();q=self.queue()
        with self.assertRaises(wf.BestWorkflowBusy):self.run_scene()
        with self.assertRaises(wf.BestWorkflowBusy):wf.BestWorkQueue(self.resident,execute_model=True)
        self.resident.fail=True;handle=q.submit(self.request())
        with self.assertRaises(wf.BestWorkflowError):handle.result(10)
        self.assertTrue(q.metadata()['failed'])
        with self.assertRaises(wf.BestWorkflowError):q.submit(self.request('two'))

    def test_expected_workflow_failure_does_not_poison_resident(self):
        self.raster();q=self.queue();bad=replace(self.request(),max_buffer_bytes=1)
        with self.assertRaises(wf.BestWorkflowError):q.submit(bad).result(10)
        self.assertFalse(q.metadata()['failed'])
        self.assertEqual(q.submit(self.request('two')).result(10)['status'],'completed')
        self.assertEqual(len(self.resident.calls),1)


if __name__=='__main__':unittest.main()
