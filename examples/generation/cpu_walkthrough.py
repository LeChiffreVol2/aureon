"""Exercise installed input preparation with artificial pixels; no model or GPU."""
import argparse
from contextlib import redirect_stdout
import importlib.abc
import io
import json
from pathlib import Path
import sys


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'terratorch', 'diffusers', 'vllm'}:
            raise RuntimeError('This CPU walkthrough must not import a neural runtime: ' + fullname)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory for all demo files')
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error('Output already exists; choose a new directory')
    sys.meta_path.insert(0, NoModelImports())
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from aureon.prepare.__main__ import main as prepare
        from aureon.terramind_native.geo import load_input
    except ImportError as error:
        parser.error('Install and activate the generation dependency profile first: ' + str(error))

    output.mkdir(parents=True, exist_ok=False)
    rows, columns = np.indices((256, 256), dtype=np.float32)
    pixels = np.stack([1000 + band * 100 + rows * 2 + columns for band in range(12)])
    metadata = dict(driver='GTiff', width=256, height=256, crs='EPSG:32647',
                    transform=from_origin(500000, 1500000, 10, 10), tiled=True,
                    blockxsize=128, blockysize=128)
    with rasterio.open(output / 'artificial-image.tif', 'w', count=12, dtype='float32', **metadata) as ds:
        ds.write(pixels)
    with rasterio.open(output / 'artificial-mask.tif', 'w', count=1, dtype='uint8', **metadata) as ds:
        ds.write(np.ones((256, 256), dtype=np.uint8), 1)
    spec = {
        'schema': 'aureon.local_scene_spec.v1', 'kind': 'aligned',
        'scene_id': 'artificial-cpu-demo', 'expected_crs': 'EPSG:32647',
        'provenance': {'source_product_id': 'ARTIFICIAL-CPU-FIXTURE-NOT-SATELLITE-DATA'},
        'units': 'surface_reflectance_x10000',
        'bands': ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'],
        'raster': 'artificial-image.tif', 'valid_mask': 'artificial-mask.tif',
        'validity_policy': 'All artificial fixture pixels are valid; not a real-scene policy',
        'window': [0, 0, 256, 256],
    }
    (output / 'aligned-input.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    steps = [
        ['scene', '--spec', str(output / 'aligned-input.json'), '--output', str(output / 'prepared')],
        ['check', '--manifest', str(output / 'prepared/manifest.json')],
        ['generation', '--manifest', str(output / 'prepared/manifest.json'),
         '--row0', '8', '--col0', '16', '--output', str(output / 'generation-input')],
    ]
    for arguments in steps:
        with redirect_stdout(io.StringIO()):
            code = prepare(arguments)
        if code:
            raise RuntimeError('Preparation failed: ' + arguments[0])
    image, mask, manifest = load_input(output / 'generation-input/manifest.json')
    if not (np.array_equal(image, pixels[:, 8:232, 16:240]) and mask.all()
            and manifest['grid']['transform'] == [10.0, 0.0, 500160.0, 0.0, -10.0, 1499920.0]):
        raise RuntimeError('Prepared pixel, mask or grid verification failed')
    result = {
        'status': 'passed', 'input': 'artificial CPU fixture, not satellite imagery',
        'prepared_shape': list(image.shape), 'all_pixels_valid': bool(mask.all()),
        'pixels_and_grid_verified': True, 'model_executed': False, 'benchmark_run': False,
        'manifest': str(output / 'generation-input/manifest.json'),
    }
    (output / 'walkthrough-result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
