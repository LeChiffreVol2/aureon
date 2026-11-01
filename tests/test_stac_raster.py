"""Real Rasterio plumbing; deterministic CPU engine double, no model claims."""
from concurrent.futures import CancelledError
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
import hashlib
import tempfile
import unittest

import numpy as np
import rasterio
from affine import Affine

from aureon.api.eo import ConditioningSpec, EOGrid
from aureon.api.stac import AssetBand, STACSelection, resolve_stac_item
from aureon.api.spatial import PreparedGenerationRaster, GenerationWindow, generate_raster_scene
from aureon.sources.stac_raster import STACGenerationRaster, _RangeFile
from aureon.sources.http_range import RangePolicy, RangeSourceError
from aureon.terramind_native.common import INPUT_BANDS
from aureon.terramind_native.memory_output import array_identity


class EngineDouble:
    def generate(self, value, *, seed, job_id):
        image, mask, meta, identity = value.snapshot()
        pixels = image[:2].copy()
        return {'array': pixels, 'valid_mask': mask, 'provenance': {'input': meta,
            'array': array_identity(pixels), 'valid_mask': array_identity(mask), 'fixture': True}}


class RasterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='stac-window-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.grid = EOGrid('EPSG:32647', (10, 0, 632260, 0, -10, 1583830), (448, 448))
        values = np.arange(12*448*448, dtype='float32').reshape(12, 448, 448) % 10000
        profile = dict(driver='GTiff', width=448, height=448, crs=self.grid.crs,
                       transform=Affine(*self.grid.transform), tiled=True, blockxsize=128, blockysize=128)
        with rasterio.open(self.root/'image.tif', 'w', count=12, dtype='float32', **profile) as ds:
            ds.write(values)
        with rasterio.open(self.root/'valid.tif', 'w', count=1, dtype='uint8', **profile) as ds:
            ds.write(np.ones((448,448), dtype='uint8'), 1)
        spec = ConditioningSpec('S2L2A', tuple(INPUT_BANDS), 'surface_reflectance_x10000',
                                'none', 'all_valid', self.grid, 'fixture', '2026-10-01T00:00:00Z')
        selection = STACSelection(spec, tuple(AssetBand('image', n+1) for n in range(12)),
                                  AssetBand('valid', 1), 'one_is_valid_zero_is_invalid')
        item = {'type':'Feature', 'stac_version':'1.1.0', 'id':'fixture', 'geometry':None, 'links':[],
                'properties':{'datetime':spec.acquired_at},
                'assets':{'image':{'href':'image.tif','type':'image/tiff',
                                   'eo:bands':[{'name':n} for n in INPUT_BANDS]},
                          'valid':{'href':'valid.tif','type':'image/tiff'}}}
        self.observation, = resolve_stac_item(item, selections=(selection,), local_root=self.root)

    def test_windows_equal_prepared_reader_and_scene_cog(self):
        with STACGenerationRaster(self.observation, local_root=self.root) as source, \
             PreparedGenerationRaster(self.root/'image.tif', self.root/'valid.tif',
                                      bands=INPUT_BANDS, units='surface_reflectance_x10000') as old:
            for row, col in ((0,0), (224,224)):
                actual, expected = source.read(GenerationWindow(row,col)).snapshot(), old.read(GenerationWindow(row,col)).snapshot()
                for a,b in zip(actual[:2],expected[:2]): np.testing.assert_array_equal(a,b)
                self.assertEqual(actual[2], expected[2])
            record = generate_raster_scene(EngineDouble(),source,self.root/'out',execute_model=True,min_free_bytes=0)
            self.assertEqual(record['tiles'],4)
            self.assertTrue(record['decoded_cog_parity'])
            with rasterio.open(record['cog']['path']) as ds, rasterio.open(self.root/'image.tif') as original:
                np.testing.assert_array_equal(ds.read(), original.read()[:2])
                self.assertEqual(ds.crs,original.crs)
                self.assertEqual(ds.transform,original.transform)
        self.assertFalse(source.is_open)

    def test_actual_grid_scale_and_hash_validation(self):
        bad = replace(self.observation, observation=replace(self.observation.observation,
                    grid=replace(self.grid, crs='EPSG:32648')))
        with self.assertRaisesRegex(ValueError,'CRS'):
            with STACGenerationRaster(bad,local_root=self.root): pass
        bad = replace(self.observation,bands=tuple(replace(b,expected_sha256='a'*64) for b in self.observation.bands))
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            with STACGenerationRaster(bad,local_root=self.root): pass
        with rasterio.open(self.root/'image.tif','r+') as ds: ds.scales=(.0001,)*12
        with self.assertRaisesRegex(ValueError,'scales'):
            with STACGenerationRaster(self.observation,local_root=self.root): pass

    def test_invalid_mask_reject_or_explicit_skip(self):
        with rasterio.open(self.root/'valid.tif','r+') as ds:
            mask=ds.read(1);mask[0,0]=0;ds.write(mask,1)
        with STACGenerationRaster(self.observation,local_root=self.root) as source:
            with self.assertRaisesRegex(ValueError,'invalid'): source.read(GenerationWindow(0,0))
            self.assertIsNone(source.read_if_valid(GenerationWindow(0,0),invalid_policy='skip_invalid_tiles'))
            record=generate_raster_scene(EngineDouble(),source,self.root/'skip',execute_model=True,
                                         min_free_bytes=0,invalid_policy='skip_invalid_tiles')
            self.assertEqual(record['tiles_skipped'],1)
            self.assertEqual(record['uncovered_pixels'],224*224)

    def test_bad_binary_mask_and_oversized_blocks(self):
        with rasterio.open(self.root/'valid.tif','r+') as ds:
            mask=ds.read(1);mask[0,0]=255;ds.write(mask,1)
        with STACGenerationRaster(self.observation,local_root=self.root) as source:
            with self.assertRaisesRegex(ValueError,'0/1'): source.read(GenerationWindow(0,0))
        with self.assertRaisesRegex(ValueError,'decode budget'):
            with STACGenerationRaster(self.observation,local_root=self.root,max_decode_bytes=1024): pass

    def test_actual_band_names_and_units_must_not_conflict(self):
        with rasterio.open(self.root/'image.tif','r+') as ds:
            ds.set_band_description(1,'B99')
        with self.assertRaisesRegex(ValueError,'band description'):
            with STACGenerationRaster(self.observation,local_root=self.root): pass
        with rasterio.open(self.root/'image.tif','r+') as ds:
            ds.set_band_description(1,'B01');ds.set_band_unit(1,'kelvin')
        with self.assertRaisesRegex(ValueError,'units'):
            with STACGenerationRaster(self.observation,local_root=self.root): pass

    def test_integer_cast_is_lossless_and_selected_mapping_is_recorded(self):
        with rasterio.open(self.root/'image.tif') as ds:
            values=ds.read().astype('uint16');profile=ds.profile
        profile['dtype']='uint16'
        with rasterio.open(self.root/'integer.tif','w',**profile) as ds: ds.write(values)
        obs=replace(self.observation,bands=tuple(replace(b,href=str(self.root/'integer.tif')) for b in self.observation.bands))
        with STACGenerationRaster(obs,local_root=self.root) as source:
            array=source.read(GenerationWindow(224,224)).snapshot()[0]
            self.assertEqual(array.dtype,np.float32)
            np.testing.assert_array_equal(array,values[:,224:448,224:448])
            self.assertEqual(source.identity['selected_bands'][11],{'band':'B12','asset_key':'image','band_index':12})

    def test_containment_sidecars_mutation_and_cancel(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            with STACGenerationRaster(self.observation,local_root=self.root/'other'): pass
        with STACGenerationRaster(self.observation,local_root=self.root) as source:
            with (self.root/'image.tif').open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'): source.verify_unchanged()
        (self.root/'image.tif.aux.xml').write_text('<metadata/>')
        with self.assertRaisesRegex(ValueError,'sidecar'):
            with STACGenerationRaster(self.observation,local_root=self.root): pass
        event=Event();event.set()
        with self.assertRaises(CancelledError):
            with STACGenerationRaster(self.observation,local_root=self.root,cancel_event=event): pass


class RangeHandler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        data=self.server.files[self.path]
        start,end=map(int,self.headers['Range'][6:].split('-'))
        self.server.requests.append((self.path,start,end))
        self.send_response(206)
        self.send_header('Content-Range',f'bytes {start}-{end}/{len(data)}')
        self.send_header('Content-Length',str(end-start+1))
        self.send_header('ETag',self.server.etag)
        self.end_headers();self.wfile.write(data[start:end+1])


class RemoteTests(unittest.TestCase):
    """Controlled loopback transport, not a live HTTPS/provider qualification."""
    def setUp(self):
        RasterTests.setUp(self)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),RangeHandler)
        self.server.files={'/'+n:(self.root/n).read_bytes() for n in ('image.tif','valid.tif')}
        self.server.requests=[];self.server.etag='"fixture-v1"'
        self.thread=Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.stop_server)
        port=self.server.server_address[1]
        # Resolver only admits HTTPS; construct an explicit HTTP test transport
        # from the already-resolved observation under allow_http loopback policy.
        def remote(ref): return replace(ref,href=f'http://127.0.0.1:{port}/{Path(ref.href).name}')
        self.remote=replace(self.observation,bands=tuple(map(remote,self.observation.bands)),valid_mask=remote(self.observation.valid_mask))
        self.policy=RangePolicy(frozenset({'127.0.0.1'}),allowed_ports=frozenset({port}),allow_http=True,
            allowed_private_networks=('127.0.0.1/32',),max_range_bytes=65536,max_total_bytes=16*1024**2,max_requests=1000)

    def stop_server(self):
        self.server.shutdown();self.server.server_close();self.thread.join(2)

    def test_remote_window_and_cog_flow(self):
        with STACGenerationRaster(self.remote,remote_policy=self.policy) as source:
            with STACGenerationRaster(self.observation,local_root=self.root) as local:
                np.testing.assert_array_equal(source.read(GenerationWindow(0,0)).snapshot()[0],local.read(GenerationWindow(0,0)).snapshot()[0])
            transferred=sum(end-start+1 for _,start,end in self.server.requests)
            self.assertLess(transferred,len(self.server.files['/image.tif']))
            record=generate_raster_scene(EngineDouble(),source,self.root/'remote',execute_model=True,min_free_bytes=0)
            self.assertTrue(record['decoded_cog_parity'])
            self.assertEqual(record['tiles'],4)
            self.assertTrue(all(r['identity_scope'].startswith('server strong') for r in record['source']['assets']))
            self.assertTrue(all(end-start+1<=65536 for _,start,end in self.server.requests))
            self.assertTrue(all(path in ('/image.tif','/valid.tif') for path,_,_ in self.server.requests))

    def test_remote_etag_change_prevents_delivery(self):
        with STACGenerationRaster(self.remote,remote_policy=self.policy) as source:
            source.read(GenerationWindow(0,0));self.server.etag='"changed"'
            with self.assertRaises(RangeSourceError): source.verify_unchanged()

    def test_remote_sha_not_downgraded_and_budget_admission(self):
        wrong=replace(self.remote,bands=tuple(replace(b,expected_sha256='a'*64) for b in self.remote.bands))
        with self.assertRaisesRegex(ValueError,'Whole-file SHA'):
            with STACGenerationRaster(wrong,remote_policy=self.policy): pass
        with self.assertRaisesRegex(ValueError,'aggregate'):
            with STACGenerationRaster(self.remote,remote_policy=self.policy,max_remote_total_bytes=100): pass
        self.assertFalse(self.server.requests)


if __name__ == '__main__': unittest.main()
