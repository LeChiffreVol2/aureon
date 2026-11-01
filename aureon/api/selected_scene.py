"""Prepare one admission tile for the original file-based model session."""
from __future__ import annotations
from pathlib import Path
from typing import cast

from ..terramind_native.common import INPUT_BANDS, binding, write_json
from ..terramind_native.offline import create_config
from ..serving.workflow_config import WorkflowConfig
from .spatial import InvalidTilePolicy, PreparedGenerationRaster, generation_windows


def native_config(image: Path, mask: Path, workflow: WorkflowConfig, output: Path) -> Path:
    """Bind existing assets and the first valid tile; do not generate anything."""
    import numpy as np
    output.mkdir(exist_ok=False)
    options, limits = workflow['options'], workflow['limits']
    if workflow['model_id'] != 'terramind-v1-base' or options['execution_profile'] != 'recommended':
        raise ValueError('Selected original scene route requires TerraMind base recommended FP32')
    with PreparedGenerationRaster(image, mask, bands=INPUT_BANDS, units='surface_reflectance_x10000',
                                  max_decode_bytes=limits['max_decode_bytes']) as source:
        for window in generation_windows(source.height, source.width, stride=cast(int, options['stride'])):
            tile = source.read_if_valid(window, invalid_policy=cast(InvalidTilePolicy, options['invalid_policy']))
            if tile is None:
                continue
            values, valid, metadata, _ = tile.snapshot()
            np.save(output/'image.npy', values, allow_pickle=False)
            np.save(output/'mask.npy', valid, allow_pickle=False)
            manifest = {'schema':'aureon.terramind_native_input.v1','bands':INPUT_BANDS,
                        'units':'surface_reflectance_x10000','grid':metadata['grid'],
                        'source_id':metadata['source_id'],'preprocessing':'Prepared values unchanged',
                        'mask_policy':'all-valid admission tile; no filling'}
            for name in ('image','mask'):
                manifest[name] = {**binding(output/(name+'.npy')), 'path':name+'.npy'}
            write_json(output/'manifest.json', manifest)
            source.verify_unchanged()
            break
        else:
            raise ValueError('No valid native tile; original model session is not started')
    create_config(model=workflow['model_file'],tokenizer=workflow['tokenizer_files']['S1GRD'],
                  input_manifest=output/'manifest.json',output=output/'config.json')
    return output/'config.json'
