"""Candidate source: accepted window math, one content-bound streaming lifetime."""
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np

from ..recipe import S2_BANDS, INPUT_UNITS
from ..scene_pipeline.job import load_manifest, canonical_hash
from ..scene_pipeline.georaster import _grid, _supporting_files, _verify_supporting_files
from ..scene_pipeline.stream_io import WindowSource, anchors, MAX_SIDE, GDAL_CACHE_BYTES


class SinglePassSource(WindowSource):
    """No preliminary raster scan; group normalization and blend enforce coverage.

    Construction mirrors WindowSource's immutable identity/grid contract. Only
    timing boundaries differ. Both pre/post reads still hash actual file bytes.
    """
    def __init__(self, manifest_path, *, manage_env=True):
        if type(manage_env) is not bool:
            raise ValueError('manage_env must be an explicit boolean')
        import rasterio as rio
        self.timings = {}
        self.path = Path(manifest_path).resolve()
        tick = perf_counter()
        self.manifest, self.members, self.manifest_hash = load_manifest(self.path, max_evidence_bytes=2*1024**3)
        self.timings['manifest_file_hash_before'] = perf_counter()-tick
        self.stack = ExitStack()
        try:
            tick = perf_counter()
            if manage_env:
                self.stack.enter_context(rio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES, GDAL_NUM_THREADS='1', VSI_CACHE=False))
            self.dataset = self.stack.enter_context(rio.open(self.members['raster']))
            self.mask = self.stack.enter_context(rio.open(self.members['valid_mask']))
            self.grid = _grid(self.dataset, self.manifest['raster']['crs'], rio, max_side=MAX_SIDE)
            mask_grid = _grid(self.mask, self.grid['crs'], rio, max_side=MAX_SIDE)
            d, declaration = self.dataset, self.manifest['raster']
            if (not d.crs.is_projected or d.count!=12 or d.dtypes!=('float32',)*12 or mask_grid!=self.grid
                    or self.mask.count!=1 or np.dtype(self.mask.dtypes[0]).kind not in 'biu'
                    or declaration.get('shape')!=[12,d.height,d.width]
                    or declaration.get('dtype')!='float32' or declaration.get('transform')!=self.grid['transform']
                    or (any(d.descriptions) and d.descriptions!=S2_BANDS)):
                raise ValueError('Streaming raster/mask projected grid, dtype or bands differ')
            for dataset in (d,self.mask):
                if any(h>512 or w>512 for h,w in dataset.block_shapes):
                    raise ValueError('Streaming requires raster decoder blocks at most 512 square')
            self.timings['raster_open_grid_before'] = perf_counter()-tick
            tick = perf_counter()
            self.supporting = {key:_supporting_files(dataset,self.members[key])
                for key,dataset in [('raster',d),('valid_mask',self.mask)]}
            self.timings['supporting_file_hash_before'] = perf_counter()-tick
            self.height,self.width = d.height,d.width
            self.rows,self.columns = anchors(d.height),anchors(d.width)
            self.total = len(self.rows)*len(self.columns)
            self.valid_pixels = 0
            self.identity = {'schema':'aureon.streaming_input.v1','manifest_sha256':self.manifest_hash,
                'source':self.manifest['source'],'raster':declaration,'valid_mask':self.manifest['valid_mask'],
                'supporting_files':self.supporting,'grid':self.grid,'grid_sha256':canonical_hash(self.grid),
                'band_order':list(S2_BANDS),'units':INPUT_UNITS,'resampling_performed':False,
                'normalization':'unchanged recipe per window; invalid normalized zero',
                'gdal_cache_bytes':GDAL_CACHE_BYTES,'maximum_decoder_block_side':512}
            self.identity['identity_sha256'] = canonical_hash(self.identity)
        except BaseException:
            self.stack.close()
            raise

    def verify(self):
        import rasterio as rio
        tick = perf_counter()
        _,_,digest = load_manifest(self.path,max_evidence_bytes=2*1024**3)
        if digest!=self.manifest_hash:
            raise ValueError('Source manifest changed')
        self.timings['manifest_file_hash_after'] = perf_counter()-tick
        tick = perf_counter()
        for key,dataset in [('raster',self.dataset),('valid_mask',self.mask)]:
            _verify_supporting_files(dataset,self.members[key],self.supporting[key])
            # Reopen metadata to discover newly attached sidecars. An old GDAL
            # handle alone is not proof that the path's interpretation survived.
            with rio.open(self.members[key]) as current:
                _verify_supporting_files(current,self.members[key],self.supporting[key])
                if _grid(current,self.grid['crs'],rio,max_side=MAX_SIDE)!=self.grid:
                    raise ValueError('Source grid changed during reading')
        self.timings['supporting_hash_and_reopen_after'] = perf_counter()-tick
