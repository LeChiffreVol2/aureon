"""Inspect an existing prepared user input on CPU; never execute a model."""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys


class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'torchgeo', 'terratorch', 'diffusers'}:
            raise RuntimeError('Input inspection must not import a neural runtime: ' + fullname)
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    sys.meta_path.insert(0,NoNeuralImports())
    try:
        from aureon.terramind_native.geo import load_input
        from aureon.api import GenerationInput
        image,mask,manifest=load_input(args.manifest)
        prepared=GenerationInput(image,bands=manifest['bands'],units=manifest['units'],valid_mask=mask,
            grid=manifest['grid'],source_id=manifest.get('source_product_id'))
    except ImportError as error:
        parser.exit(2,'Input inspection requires NumPy and Rasterio in the selected environment: '+str(error)+'\n')
    except (ValueError,OSError,KeyError,RuntimeError) as error:
        parser.exit(2,'Input inspection failed: '+str(error)+'\n')
    print(json.dumps({'status':'passed','shape':list(image.shape),'bands':manifest['bands'],
        'units':manifest['units'],'grid':manifest['grid'],'all_pixels_valid':bool(mask.all()),
        'identity':prepared.identity,'model_executed':False,'benchmark_run':False,
        'scope':'Prepared input contract and file hashes, not imagery authenticity or model quality'},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
