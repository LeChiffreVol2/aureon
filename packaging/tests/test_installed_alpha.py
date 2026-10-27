"""Installed public-alpha CPU workflow; synthetic input, never model execution."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from importlib import metadata
import importlib.abc
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

OPTIONS = None


class BlockNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'terratorch', 'vllm', 'diffusers'}:
            raise AssertionError('CPU acceptance attempted neural import: ' + fullname)
        return None


def cli(name, *arguments):
    entry = next(e for e in metadata.distribution('aureon-inference').entry_points if e.name == name)
    stdout, stderr = io.StringIO(), io.StringIO()
    previous = sys.argv
    try:
        sys.argv = [name, *map(str, arguments)]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = entry.load()()
            except SystemExit as error:
                code = error.code
    finally:
        sys.argv = previous
    return code or 0, stdout.getvalue(), stderr.getvalue()


class InstalledAlpha(unittest.TestCase):
    def setUp(self):
        import aureon
        self.assertTrue(Path(aureon.__file__).resolve().is_relative_to(OPTIONS.prefix))
        self.temp = tempfile.TemporaryDirectory(prefix='alpha-input-', dir=OPTIONS.scratch)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def success(self, *args):
        code, text, error = cli(*args)
        self.assertEqual(code, 0, error)
        return json.loads(text)

    def scene(self, invalid=False):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        self.pixels = np.arange(12 * 248 * 250, dtype=np.float32).reshape(12, 248, 250) / 100
        self.pixels[:, 13, 11] = -10
        transform = from_origin(500000, 1500000, 10, 10)
        base = dict(driver='GTiff', width=250, height=248, crs='EPSG:32647', transform=transform)
        with rasterio.open(self.work / 'image.tif', 'w', count=12, dtype='float32', **base) as dataset:
            dataset.write(self.pixels)
        mask = np.ones((248, 250), dtype=np.uint8)
        if invalid:
            mask[13, 11] = 0
        with rasterio.open(self.work / 'mask.tif', 'w', count=1, dtype='uint8', **base) as dataset:
            dataset.write(mask, 1)
        from aureon.recipe import S2_BANDS
        spec = {'schema': 'aureon.local_scene_spec.v1', 'kind': 'aligned',
                'scene_id': 'user-scene', 'expected_crs': 'EPSG:32647',
                'provenance': {'source_product_id': 'synthetic-CPU-test-not-imagery'},
                'units': 'surface_reflectance_x10000', 'bands': list(S2_BANDS),
                'raster': 'image.tif', 'valid_mask': 'mask.tif',
                'validity_policy': 'explicit binary fixture mask', 'window': [7, 5, 240, 240]}
        (self.work / 'input.json').write_text(json.dumps(spec))
        self.success('aureon-prepare', 'scene', '--spec', self.work / 'input.json', '--output', self.work / 'scene')
        manifest = self.work / 'scene/manifest.json'
        self.success('aureon-prepare', 'check', '--manifest', manifest)
        return manifest

    def test_asset_discovery_and_public_help_are_cpu_only(self):
        b = self.success('aureon-engine', 'assets', '--model', 'kao-terramind-b')
        self.assertFalse(b['downloaded'])
        self.assertEqual(b['assets'][0]['sha256'], 'd5cd2c97ce853971c847eb1a423c6d5ba996eff8ebc0e1e64d116bed64e5e23a')
        tm = self.success('aureon-engine', 'assets', '--model', 'terramind-v1-base')['assets']
        self.assertEqual(len(tm), 2)
        for row in b['assets'] + tm:
            self.assertIn('/resolve/' + row['revision'] + '/', row['source_url'])
        for command, args in [('aureon', ('prepare-vllm', '--help')),
                              ('aureon-prepare', ('generation', '--help')),
                              ('aureon-engine', ('generation-config', '--help')),
                              ('aureon-service', ('init', '--help')),
                              ('aureon-service-client', ('--help',))]:
            self.assertEqual(cli(command, *args)[0], 0)

    def test_user_scene_to_generation_cli_preserves_asymmetric_crop(self):
        import numpy as np
        manifest = self.scene()
        crop = self.work / 'crop'
        result = self.success('aureon-prepare', 'generation', '--manifest', manifest,
                              '--output', crop, '--row0', 8, '--col0', 4)
        self.assertEqual(Path(result['path']).resolve(), (crop / 'manifest.json').resolve())
        from aureon.terramind_native.geo import load_input, prepare
        image, mask, record = load_input(crop / 'manifest.json')
        self.assertTrue(np.array_equal(image, self.pixels[:, 13:237, 11:235]))
        self.assertTrue(mask.all())
        self.assertEqual(record['grid']['transform'], [10.0, 0.0, 500110.0, 0.0, -10.0, 1499870.0])
        prepare(manifest, self.work / 'direct', row0=8, col0=4)
        for name in ('image.npy', 'mask.npy'):
            self.assertEqual((crop / name).read_bytes(), (self.work / 'direct' / name).read_bytes())
        before = (crop / 'manifest.json').read_bytes()
        self.assertEqual(cli('aureon-prepare', 'generation', '--manifest', manifest,
                             '--output', crop, '--row0', 8, '--col0', 4)[0], 2)
        self.assertEqual((crop / 'manifest.json').read_bytes(), before)

    def test_invalid_crop_and_missing_input_fail_without_outputs(self):
        manifest = self.scene(invalid=True)
        for row, column, expected in ((8, 4, 'all-valid'), (-1, 0, 'crop offset'), (50, 50, 'outside source')):
            target = self.work / f'bad-{row}-{column}'
            code, text, error = cli('aureon-prepare', 'generation', '--manifest', manifest,
                                    '--output', target, '--row0', row, '--col0', column)
            self.assertEqual(code, 2)
            self.assertIn(expected, error)
            self.assertNotIn('Traceback', error)
            self.assertFalse(target.exists())
        code, _, error = cli('aureon-prepare', 'generation', '--manifest', self.work / 'missing.json',
                             '--output', self.work / 'missing-output')
        self.assertEqual(code, 2)
        self.assertIn('Input preparation failed', error)

    def test_offline_commands_require_explicit_execution(self):
        code, _, error = cli('aureon-engine', 'run', '--model', 'kao-terramind-b', '--environment', 'scene',
            '--profile', 'fp32-full-reference-v1', '--manifest', 'missing.json', '--bundle', 'missing', '--output', 'new')
        self.assertEqual(code, 2)
        self.assertIn('explicit', error)
        code, _, error = cli('aureon-engine', 'generate', '--model', 'terramind-v1-base',
            '--config', 'missing.json', '--output', 'new', '--job-id', 'one', '--attempt-id', 'first')
        self.assertEqual(code, 2)
        self.assertIn('explicit', error)


def main():
    global OPTIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    OPTIONS = parser.parse_args()
    OPTIONS.prefix = OPTIONS.prefix.resolve(strict=True)
    assert sys.flags.isolated and not OPTIONS.report.exists()
    sys.meta_path.insert(0, BlockNeural())
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InstalledAlpha))
    receipt = {'passed': result.wasSuccessful(), 'tests': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors), 'skips': len(result.skipped),
               'model_executed': False, 'input': 'synthetic local CPU raster', 'benchmark_run': False}
    OPTIONS.report.write_text(json.dumps(receipt, indent=2) + '\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
