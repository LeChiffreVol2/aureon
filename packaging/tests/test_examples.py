"""CPU-only regressions for review examples; no model, acquisition or install."""
import contextlib
from email.message import Message
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'examples' / 'packaged_scene'


def load(name):
    spec = importlib.util.spec_from_file_location('review_example_' + name, EXAMPLES / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


api = load('api_job')
offline = load('offline_job')


def temporary():
    parent = (ROOT / 'packaging' / 'tests').resolve(strict=True)
    if not parent.is_relative_to(ROOT.resolve(strict=True)):
        raise ValueError('Test scratch must remain inside this review workspace')
    return tempfile.TemporaryDirectory(prefix='example-test-', dir=parent)


class ClientBoundaryTests(unittest.TestCase):
    def test_only_actual_listener_origins_are_accepted(self):
        for url in ('http://127.0.0.1:8031', 'http://localhost:8031/'):
            self.assertEqual(api.origin(url), ('127.0.0.1', 8031))
        for url in ('https://127.0.0.1:8031', 'http://example.org:8031',
                    'http://127.0.0.1.evil:8031', 'http://127.0.0.2:8031',
                    'http://u:p@localhost:8031', 'http://localhost:8031/path',
                    'http://localhost:8031?token=x', 'http://localhost:8031#x',
                    'http://localhost', 'http://localhost:0', 'http://localhost:65536'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                api.origin(url)

    def test_missing_execute_flag_cannot_connect_or_create_output(self):
        with temporary() as tmp:
            output = Path(tmp) / 'new'
            with patch.object(api.http.client, 'HTTPConnection') as connection:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    api.main(['--url', 'http://127.0.0.1:8031', '--token-file', 'absent',
                              '--manifest', 'absent', '--bundle', 'absent', '--output', str(output)])
                connection.assert_not_called()
            self.assertFalse(output.exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                offline.main(['--manifest', 'absent', '--bundle', 'absent', '--output', str(output),
                              '--audit-output', str(output) + '.json', '--hourly-rate', '1', '--currency', 'THB'])
            self.assertFalse(output.exists())

    def test_help_works_from_unrelated_cwd_without_runtime_import(self):
        with temporary() as tmp:
            for name in ('api_job.py', 'offline_job.py'):
                result = subprocess.run([sys.executable, '-I', '-B', str(EXAMPLES / name), '--help'],
                                        cwd=tmp, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--execute-model', result.stdout)

    def test_token_file_rejects_header_injection_and_nonregular_input(self):
        with temporary() as tmp:
            path = Path(tmp) / 'token.txt'
            token = 'a' * 43
            path.write_text(token + '\n', encoding='ascii')
            self.assertEqual(api.token_from_file(path), token)
            for value in ('short', token + '\nX-Injected: bad', 'x' * 4097):
                path.write_text(value, encoding='ascii')
                with self.assertRaises(ValueError):
                    api.token_from_file(path)
            with self.assertRaises(ValueError):
                api.token_from_file(Path(tmp))

    def test_token_symlink_is_refused_where_supported(self):
        with temporary() as tmp:
            path, link = Path(tmp) / 'token', Path(tmp) / 'link'
            path.write_text('a' * 43, encoding='ascii')
            try:
                link.symlink_to(path)
            except OSError:
                self.skipTest('Creating symlinks is unavailable on this host')
            with self.assertRaises(ValueError):
                api.token_from_file(link)

    def test_authorization_stays_in_header_and_redirect_is_not_followed(self):
        token = 'a' * 43
        message = Message()
        message['Content-Length'] = '2'
        response = MagicMock(status=302, headers=message)
        response.read.return_value = b'{}'
        response.getheader.return_value = None
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch.object(api.http.client, 'HTTPConnection', return_value=connection) as constructor:
            client = api.Client('http://localhost:8031', token, 30)
            status, _, _ = client.request('GET', '/v1/info')
            self.assertEqual(status, 302)
            constructor.assert_called_once()
            arguments = connection.request.call_args
            self.assertEqual(arguments.args, ('GET', '/v1/info'))
            self.assertEqual(arguments.kwargs['headers']['Authorization'], 'Bearer ' + token)
            self.assertNotIn(token, arguments.args[1])
            connection.close.assert_called_once()

    def test_existing_output_and_secret_evidence_are_preserved(self):
        with temporary() as tmp:
            target = api.fresh_directory(Path(tmp) / 'output')
            sentinel = target / 'keep.txt'
            sentinel.write_text('keep', encoding='utf-8')
            with self.assertRaises(ValueError):
                api.fresh_directory(target)
            self.assertEqual(sentinel.read_text(), 'keep')
            token = 'a' * 43
            with self.assertRaises(ValueError):
                api.write_json(target / 'receipt.json', {'bad': token}, token=token)
            self.assertFalse((target / 'receipt.json').exists())


class DownloadIntegrityTests(unittest.TestCase):
    def descriptor(self, name, data):
        return {'url': '/v1/jobs/' + 'f' * 32 + '/artifacts/' + name,
                'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

    def test_download_rejects_bad_hash_etag_url_and_overwrite(self):
        with temporary() as tmp:
            data = b'example byte identity'
            original = self.descriptor('class_map.tif', data)
            headers = {'etag': '"' + original['sha256'] + '"'}
            for descriptor, supplied_headers, content in (
                (original, headers, data + b'changed'),
                (original, {'etag': '"wrong"'}, data),
                ({**original, 'url': 'http://external.invalid/file'}, headers, data),
                ({**original, 'bytes': True}, headers, data),
            ):
                with self.assertRaises(ValueError):
                    api.save_download(tmp, 'f' * 32, 'class_map.tif', descriptor, 200, supplied_headers, content)
                self.assertFalse((Path(tmp) / 'class_map.tif').exists())
            api.save_download(tmp, 'f' * 32, 'class_map.tif', original, 200, headers, data)
            with self.assertRaises(FileExistsError):
                api.save_download(tmp, 'f' * 32, 'class_map.tif', original, 200, headers, data)
            self.assertEqual((Path(tmp) / 'class_map.tif').read_bytes(), data)

    def test_post_download_same_length_mutation_is_detected(self):
        with temporary() as tmp:
            descriptors = {}
            for name in api.ARTIFACTS:
                values = ('payload-' + name).encode()
                (Path(tmp) / name).write_bytes(values)
                descriptors[name] = self.descriptor(name, values)
            api.verify_saved_downloads(tmp, descriptors)
            target = Path(tmp) / 'job.json'
            target.write_bytes(b'X' * target.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'changed'):
                api.verify_saved_downloads(tmp, descriptors)


@unittest.skipUnless(importlib.util.find_spec('rasterio'), 'Optional rasterio is unavailable')
class GeospatialDeliveryTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import rasterio
        from rasterio.shutil import copy as copy_raster
        from aureon.catalog import CLASS_NAMES
        self.scratch = temporary()
        self.addCleanup(self.scratch.cleanup)
        self.folder = Path(self.scratch.name)
        transform = rasterio.Affine(10, 0, 600000, 0, -10, 1500000)
        valid = np.ones((17, 23), dtype=np.uint8)
        valid[2:5, 7:11] = 0
        classes = np.full(valid.shape, 3, dtype=np.uint8)
        classes[valid == 0] = 255
        records = {}
        for name, values, nodata in (('class_map.tif', classes, 255), ('valid_mask.tif', valid, 0)):
            staging, target = self.folder / ('source-' + name), self.folder / name
            with rasterio.open(staging, 'w', driver='GTiff', height=17, width=23, count=1,
                               dtype='uint8', crs='EPSG:32647', transform=transform, nodata=nodata) as ds:
                ds.write(values, 1)
            copy_raster(staging, target, driver='COG', BLOCKSIZE=256, RESAMPLING='NEAREST')
            records[name] = {'sha256': api.sha256(target.read_bytes()), 'bytes': target.stat().st_size}
        self.worker = {'worker_id': 'worker-test', 'model_load_count': 1, 'startup_receipt_sha256': 'c' * 64}
        self.arguments = dict(scene_id='synthetic-scene', manifest_hash='a' * 64, bundle_hash='b' * 64,
                              worker=self.worker)
        self.job = {'schema': 'aureon.scene_job.v1', 'status': 'completed', 'job_id': 'f' * 32,
                    'scene_id': 'synthetic-scene', 'source_manifest_sha256': 'a' * 64,
                    'bundle_receipt_sha256': 'b' * 64, 'model_name': 'KAO-TerraMind B',
                    'precision': {'float32_matmul_precision': 'highest', 'matmul_allow_tf32': False,
                                  'cudnn_allow_tf32': False, 'cudnn_sdp_enabled': False, 'cudnn_benchmark': False},
                    'parity': {'passed': True, 'allclose': True, 'atol': 1e-5, 'rtol': 1e-5,
                               'class_agreement': 1.0, 'scene_class_map_equal': True,
                               'scene_mask_equal': True, 'bitwise_equal': False},
                    'worker': {'worker_id': 'worker-test', 'model_load_count': 1,
                               'worker_startup_receipt_sha256': 'c' * 64}, 'files_sha256': {}}
        self.delivery = {'schema': 'aureon.georaster_delivery.v1', 'status': 'completed', 'format': 'COG',
                         'class_names': list(CLASS_NAMES), 'files': records,
                         'grid': {'crs': 'EPSG:32647', 'transform': list(transform)[:6], 'height': 17, 'width': 23},
                         'provenance': {key: self.job[key] for key in
                                        ('job_id', 'scene_id', 'source_manifest_sha256', 'bundle_receipt_sha256')}}
        self.persist()

    def persist(self):
        (self.folder / 'provenance.json').write_text(json.dumps(self.delivery), encoding='utf-8')
        self.job['files_sha256'] = {key: api.sha256((self.folder / name).read_bytes())
                                   for name, key in api.FILE_KEYS.items()}
        (self.folder / 'job.json').write_text(json.dumps(self.job), encoding='utf-8')

    def test_valid_cogs_are_checked_without_claiming_bitwise_or_model_execution(self):
        result = api.verify_delivery(self.folder, **self.arguments)
        self.assertTrue(result['cog_values_grid_mask_verified'])
        self.assertTrue(result['recorded_parity_only'])
        self.assertFalse(result['parity']['bitwise_equal'])
        self.assertFalse(result['task_accuracy_evaluated'])

    def test_matching_file_hashes_do_not_hide_wrong_declared_grid(self):
        self.delivery['grid']['transform'][2] += 10
        self.persist()
        with self.assertRaisesRegex(ValueError, 'grid'):
            api.verify_delivery(self.folder, **self.arguments)

    def test_relaxed_parity_and_precision_are_refused(self):
        self.job['parity']['atol'] = 1e-4
        self.persist()
        with self.assertRaisesRegex(ValueError, 'parity'):
            api.verify_delivery(self.folder, **self.arguments)
        self.job['parity']['atol'] = 1e-5
        self.job['precision']['matmul_allow_tf32'] = True
        self.persist()
        with self.assertRaisesRegex(ValueError, 'precision'):
            api.verify_delivery(self.folder, **self.arguments)

    def test_other_bundle_or_retained_worker_is_refused(self):
        self.job['bundle_receipt_sha256'] = 'd' * 64
        self.persist()
        with self.assertRaisesRegex(ValueError, 'identity'):
            api.verify_delivery(self.folder, **self.arguments)
        self.job['bundle_receipt_sha256'] = 'b' * 64
        self.job['worker']['worker_startup_receipt_sha256'] = 'e' * 64
        self.persist()
        with self.assertRaisesRegex(ValueError, 'worker'):
            api.verify_delivery(self.folder, **self.arguments)


if __name__ == '__main__':
    unittest.main()
