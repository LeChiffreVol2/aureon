"""Real numeric/raster parsing in bounded CPU children. No model execution."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid

from aureon.cloud.control import Error, encoded, file_hash
from aureon.serving.parser import run
from aureon.serving.workflow_config import normalize, binding


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy profile required')
class WorkflowInputTests(unittest.TestCase):
    def setUp(self):
        parent=Path(os.environ.get('AUREON_SERVICE_TEST_ROOT',tempfile.gettempdir()))
        self.root=parent/('workflow-input-'+uuid.uuid4().hex);self.root.mkdir(parents=True)

    def config(self,task,options=None,limits=None):
        return binding(normalize({'schema':'aureon.terramind_service_workflow.v1',
            'task':task,'model_id':'terramind-v1-base' if task=='generation_scene' else 'terramind-v1-base-backbone',
            'model_file':'unavailable-model.safetensors','tokenizer_files':{'S1GRD':'unavailable-tokenizer.safetensors'} if task=='generation_scene' else {},
            'options':options or {},'limits':limits or {}},self.root))

    def stage(self,task,manifest,members):
        folder=self.root/uuid.uuid4().hex;folder.mkdir();source=folder/'source';source.mkdir()
        mapping={'manifest.json':'part0.json'};files=[]
        for index,(name,path) in enumerate(members.items(),1):
            uploaded='part'+str(index)+path.suffix;mapping[name]=uploaded
            # Small real test fixtures only; service itself streams copies.
            (source/uploaded).write_bytes(path.read_bytes())
        (source/'part0.json').write_bytes(encoded(manifest))
        for path in source.iterdir():files.append({'name':path.name,'bytes':path.stat().st_size,'sha256':file_hash(path)})
        return folder,{'files':files,'spec':{'schema':'aureon.prepared_upload.v1','task':task,'manifest':'manifest.json','members':mapping}}

    def features(self,*,modality='S2L2A',batch=1,dtype='float32'):
        import numpy as np
        payload=self.root/(uuid.uuid4().hex+'.npz');np.savez(payload,**{modality:np.zeros((batch,12,224,224),dtype=dtype)})
        manifest={'schema':'aureon.terramind_features_input.v1','inputs':{'path':'inputs.npz','sha256':file_hash(payload),'bytes':payload.stat().st_size}}
        return self.stage('features',manifest,{'inputs.npz':payload})

    def test_real_feature_payload_child_is_cpu_only_and_preserves_bytes(self):
        folder,body=self.features();path=run(body,folder,'accepted','features',workflow=self.config('features'))
        self.assertEqual(file_hash(path),body['files'][0]['sha256'] if body['files'][0]['name']=='part0.json' else file_hash(folder/'source'/'part0.json'))
        self.assertEqual(file_hash(path.parent/'inputs.npz'),file_hash(folder/'source'/'part1.npz'))
        result=json.loads((folder/'parse-accepted-result.json').read_bytes())
        self.assertTrue(result['passed']);self.assertFalse(result['model_imported'])
        self.assertEqual(result['limits']['address_space_bytes'],4*1024**3)

    def test_feature_dtype_modalities_and_batch_fail_before_model(self):
        for options,kwargs in (({}, {'dtype':'float16'}), ({'modalities':['S1GRD']},{}), ({},{'batch':2})):
            with self.subTest(options=options,kwargs=kwargs):
                folder,body=self.features(**kwargs)
                with self.assertRaises(Error):run(body,folder,'rejected','features',workflow=self.config('features',options,{'max_batch':1}))
                result=json.loads((folder/'parse-rejected-result.json').read_bytes());self.assertFalse(result['passed']);self.assertFalse(result['model_imported'])

    def scene(self,invalid=False,grid=None):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from aureon.terramind_native.common import INPUT_BANDS
        image=self.root/(uuid.uuid4().hex+'.tif');mask=self.root/(uuid.uuid4().hex+'.tif')
        base={'driver':'GTiff','width':225,'height':224,'crs':'EPSG:32647','transform':from_origin(500000,1500000,10,10),'tiled':True,'blockxsize':224,'blockysize':224}
        with rasterio.open(image,'w',**base,count=12,dtype='float32') as ds:ds.write(np.ones((12,224,225),dtype='float32'))
        valid=np.ones((224,225),dtype='uint8')
        if invalid:valid[0,0]=0
        with rasterio.open(mask,'w',**base,count=1,dtype='uint8') as ds:ds.write(valid,1)
        manifest={'schema':'aureon.terramind_scene_input.v1','bands':INPUT_BANDS,'units':'surface_reflectance_x10000',
            'image':{'path':'image.tif','bytes':image.stat().st_size,'sha256':file_hash(image)},
            'mask':{'path':'mask.tif','bytes':mask.stat().st_size,'sha256':file_hash(mask)}}
        if grid is not None:manifest['grid']=grid
        return self.stage('generation_scene',manifest,{'image.tif':image,'mask.tif':mask})

    @unittest.skipUnless(importlib.util.find_spec('rasterio'),'Raster profile required')
    def test_real_multitile_raster_child_preserves_grid_and_bytes(self):
        folder,body=self.scene();path=run(body,folder,'accepted','generation_scene',workflow=self.config('generation_scene'))
        self.assertEqual(file_hash(path.parent/'image.tif'),file_hash(folder/'source'/'part1.tif'))
        self.assertEqual(file_hash(path.parent/'mask.tif'),file_hash(folder/'source'/'part2.tif'))
        self.assertFalse(json.loads((folder/'parse-accepted-result.json').read_bytes())['model_imported'])

    @unittest.skipUnless(importlib.util.find_spec('rasterio'),'Raster profile required')
    def test_scene_invalid_pixels_require_explicit_policy(self):
        folder,body=self.scene(invalid=True)
        with self.assertRaises(Error):run(body,folder,'reject','generation_scene',workflow=self.config('generation_scene'))
        path=run(body,folder,'skip','generation_scene',workflow=self.config('generation_scene',{'invalid_policy':'skip_invalid_tiles'}))
        self.assertTrue(path.is_file())

    @unittest.skipUnless(importlib.util.find_spec('rasterio'),'Raster profile required')
    def test_scene_forged_grid_and_width_budget_fail_before_model(self):
        folder,body=self.scene(grid={'crs':'EPSG:4326'})
        with self.assertRaises(Error):run(body,folder,'grid','generation_scene',workflow=self.config('generation_scene'))
        folder,body=self.scene()
        with self.assertRaises(Error):run(body,folder,'size','generation_scene',workflow=self.config('generation_scene',limits={'max_input_side':224}))


if __name__=='__main__':unittest.main()
