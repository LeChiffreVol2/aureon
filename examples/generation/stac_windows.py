"""Explicit local prepared S2 STAC input; inspect by default, opt in to model execution."""
import argparse
import json
from pathlib import Path

from aureon.api import GenerationEngine
from aureon.api.eo import ConditioningSpec, EOGrid
from aureon.api.stac import AssetBand, STACSelection, resolve_stac_item
from aureon.api.spatial import generate_raster_scene
from aureon.sources.stac_raster import STACGenerationRaster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--item', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--image-key', default='image')
    parser.add_argument('--mask-key', default='valid')
    parser.add_argument('--crs', required=True)
    parser.add_argument('--transform', type=float, nargs=6, required=True, metavar=('A','B','C','D','E','F'))
    parser.add_argument('--shape', type=int, nargs=2, required=True, metavar=('HEIGHT','WIDTH'))
    parser.add_argument('--model-file', type=Path)
    parser.add_argument('--tokenizer-file', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute-model', action='store_true')
    args = parser.parse_args()
    if args.execute_model and any(v is None for v in (args.model_file,args.tokenizer_file,args.work,args.output)):
        parser.error('Execution requires explicit model-file, tokenizer-file, work and output')
    # Bound the metadata file before decoding; resolver also bounds its structure.
    with args.item.open('rb') as stream:
        raw = stream.read(4*1024**2+1)
    if len(raw) > 4*1024**2: parser.error('STAC Item exceeds 4 MiB')
    item = json.loads(raw)
    bands = ('B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12')
    grid = EOGrid(args.crs, tuple(args.transform), tuple(args.shape))
    spec = ConditioningSpec('S2L2A', bands, 'surface_reflectance_x10000', 'none',
                            'explicit_all_valid', grid, item['id'], item['properties']['datetime'])
    selection = STACSelection(spec, tuple(AssetBand(args.image_key,n+1) for n in range(12)),
                              AssetBand(args.mask_key,1), 'one_is_valid_zero_is_invalid')
    observation, = resolve_stac_item(item,selections=(selection,),local_root=args.asset_root)
    with STACGenerationRaster(observation,local_root=args.asset_root) as source:
        validity = source.inspect_validity()
        if validity['source_invalid_pixels']:
            parser.error('This example rejects invalid pixels; prepare a mask/recipe explicitly')
        if not args.execute_model:
            print(json.dumps({'source':source.identity,'grid':source.grid,'validity':validity,'model_executed':False},indent=2))
            return
        with GenerationEngine(model_file=args.model_file,tokenizer_file=args.tokenizer_file,
                              work=args.work,execute_model=True) as engine:
            receipt = generate_raster_scene(engine,source,args.output,execute_model=True)
        print(json.dumps(receipt,indent=2))


if __name__ == '__main__': main()
