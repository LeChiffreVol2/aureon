"""Window-only source, rolling FP32 blend and disk categorical delivery.

No model construction, downloads, mmap or whole-scene arrays. Maximum scene
width is explicit; blend RAM scales with width, never height or tile count.
"""
from contextlib import ExitStack
from pathlib import Path
import json
import math

import numpy as np

from ..recipe import S2_BANDS, INPUT_UNITS, WINDOW_SIZE, STRIDE, normalize_s2, RecipeCancelled
from ..catalog import CLASS_NAMES
from .job import load_manifest, canonical_hash, file_hash, write_json
from .georaster import _grid, _supporting_files, _verify_supporting_files

MAX_SIDE = 4096
GDAL_CACHE_BYTES = 64*1024**2
AUDIT_SIDE = 256


def check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise RecipeCancelled('Streaming scene cancelled between bounded operations')


def anchors(length):
    if type(length) is not int or not 1 <= length <= MAX_SIDE:
        raise ValueError('Streaming side must be 1..4096')
    if length <= WINDOW_SIZE: return (0,)
    edge = length-WINDOW_SIZE
    result = tuple(range(0,edge+1,STRIDE))
    return result if result[-1]==edge else (*result,edge)


def windows(height,width,side=AUDIT_SIDE):
    for row in range(0,height,side):
        for col in range(0,width,side):
            yield row,col,min(row+side,height),min(col+side,width)


def rio_window(window):
    from rasterio.windows import Window
    row,col,bottom,right=window
    return Window(col,row,right-col,bottom-row)


class WindowSource:
    def __init__(self, manifest_path, *, manage_env=True):
        if type(manage_env) is not bool:
            raise ValueError('manage_env must be an explicit boolean')
        import rasterio as rio
        self.path=Path(manifest_path).resolve()
        self.manifest,self.members,self.manifest_hash=load_manifest(self.path,max_evidence_bytes=2*1024**3)
        self.stack=ExitStack()
        try:
            if manage_env:
                self.stack.enter_context(rio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES,GDAL_NUM_THREADS='1',VSI_CACHE=False))
            self.dataset=self.stack.enter_context(rio.open(self.members['raster']))
            self.mask=self.stack.enter_context(rio.open(self.members['valid_mask']))
            self.grid=_grid(self.dataset,self.manifest['raster']['crs'],rio,max_side=MAX_SIDE)
            mask_grid=_grid(self.mask,self.grid['crs'],rio,max_side=MAX_SIDE)
            d=self.dataset; declaration=self.manifest['raster']
            if (d.count!=12 or d.dtypes!=('float32',)*12 or mask_grid!=self.grid
                    or self.mask.count!=1 or np.dtype(self.mask.dtypes[0]).kind not in 'biu'
                    or declaration.get('shape')!=[12,d.height,d.width]
                    or declaration.get('dtype')!='float32' or declaration.get('transform')!=self.grid['transform']
                    or (any(d.descriptions) and d.descriptions!=S2_BANDS)):
                raise ValueError('Streaming raster/mask identity, grid, dtype or bands differ')
            for dataset in (d,self.mask):
                if any(h>512 or w>512 for h,w in dataset.block_shapes):
                    raise ValueError('Streaming requires raster decoder blocks at most 512 square')
            self.supporting={key:_supporting_files(dataset,self.members[key])
                             for key,dataset in [('raster',d),('valid_mask',self.mask)]}
            self.height,self.width=d.height,d.width
            self.rows,self.columns=anchors(d.height),anchors(d.width)
            self.total=len(self.rows)*len(self.columns)
            self.valid_pixels=0
            self.identity={'schema':'aureon.streaming_input.v1','manifest_sha256':self.manifest_hash,
                'source':self.manifest['source'],'raster':declaration,'valid_mask':self.manifest['valid_mask'],
                'supporting_files':self.supporting,'grid':self.grid,'grid_sha256':canonical_hash(self.grid),
                'band_order':list(S2_BANDS),'units':INPUT_UNITS,'resampling_performed':False,
                'normalization':'unchanged recipe per window; invalid normalized zero',
                'gdal_cache_bytes':GDAL_CACHE_BYTES,'maximum_decoder_block_side':512}
            self.identity['identity_sha256']=canonical_hash(self.identity)
        except BaseException:
            self.stack.close(); raise

    def valid(self,window):
        w=rio_window(window)
        # Twelve band masks are small bounded windows, not whole-scene masks.
        valid=np.all(self.dataset.read_masks(window=w)>0,axis=0)
        declared=self.mask.read(1,window=w)
        if not np.isin(declared,(0,1)).all(): raise ValueError('Validity values must be 0 or 1')
        valid &= (declared==1) & (self.mask.read_masks(1,window=w)>0)
        return valid

    def normalized(self,window,valid=None):
        valid=self.valid(window) if valid is None else valid
        values=self.dataset.read(window=rio_window(window))
        # normalize_s2 validates finite valid pixels; invalid NaN/Inf are ignored.
        normalized=normalize_s2(values,valid,bands=S2_BANDS,units=INPUT_UNITS)
        tile=np.zeros((12,224,224),dtype=np.float32)
        tile[:,:normalized.shape[1],:normalized.shape[2]]=normalized
        return tile

    def groups(self,cancelled=None):
        buffer=np.empty((8,12,224,224),dtype=np.float32)
        selected=[]
        self.executed=self.skipped=0
        for row in self.rows:
            for col in self.columns:
                check_cancelled(cancelled)
                window=(row,col,min(row+224,self.height),min(col+224,self.width))
                valid=self.valid(window)
                if not valid.any():
                    self.skipped+=1; continue
                buffer[len(selected)]=self.normalized(window,valid)
                selected.append(window); self.executed+=1
                if len(selected)==8:
                    yield buffer,tuple(selected)
                    selected=[]
        if selected: yield buffer[:len(selected)],tuple(selected)

    def verify(self):
        _,_,digest=load_manifest(self.path,max_evidence_bytes=2*1024**3)
        if digest!=self.manifest_hash: raise ValueError('Source manifest changed')
        for key,dataset in [('raster',self.dataset),('valid_mask',self.mask)]:
            _verify_supporting_files(dataset,self.members[key],self.supporting[key])

    def close(self): self.stack.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()


def profile(source,count=1,dtype='uint8',nodata=None):
    import rasterio
    return dict(driver='GTiff',width=source.width,height=source.height,count=count,dtype=dtype,
        nodata=nodata,crs=source.grid['crs_wkt'],transform=rasterio.Affine(*source.grid['transform']),
        tiled=True,blockxsize=256,blockysize=256,compress='deflate',NUM_THREADS='1',BIGTIFF='IF_SAFER')


class CategorySink:
    """Bounded GTiff writes; final publication is a separate COG copy/check."""
    def __init__(self,folder,source):
        import rasterio
        self.folder=Path(folder); self.folder.mkdir(exist_ok=False)
        self.source=source; self.next_row=0; self.valid_pixels=0
        self.stack=ExitStack()
        try:
            self.classes=self.stack.enter_context(rasterio.open(self.folder/'classes.tif','w',**profile(source,nodata=255)))
            self.mask=self.stack.enter_context(rasterio.open(self.folder/'mask.tif','w',**profile(source,nodata=0)))
        except BaseException:
            self.stack.close(); raise

    def write(self,window,classes,valid):
        expected=(window[2]-window[0],window[3]-window[1])
        if classes.shape!=expected or valid.shape!=expected or classes.dtype!=np.uint8 or valid.dtype!=bool:
            raise ValueError('Exact output window shapes/dtypes required; no implicit decimation')
        if np.any(classes[valid]>=10) or np.any(classes[~valid]!=255): raise ValueError('Invalid class output')
        self.classes.write(classes,1,window=rio_window(window))
        self.mask.write(valid.astype(np.uint8),1,window=rio_window(window))
        self.valid_pixels+=int(valid.sum())

    def close(self): self.stack.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()


class StripeBlend:
    """Only 224 rows resident; additions are in global row-major tile order."""
    def __init__(self,source,sink,cancelled=None):
        self.source,self.sink,self.cancelled=source,sink,cancelled
        self.acc=np.zeros((10,224,source.width),dtype=np.float32)
        self.coverage=np.zeros((224,source.width),dtype=np.uint16)
        self.top=0; self.previous=None
        self.allocated_bytes=self.acc.nbytes+self.coverage.nbytes

    def flush(self,stop):
        while self.top<stop:
            check_cancelled(self.cancelled)
            count=min(224,stop-self.top)
            # Bound argmax/division/validity temporaries by width <=4096.
            values=self.acc[:,:count]; coverage=self.coverage[:count]
            np.divide(values,coverage[None],out=values,where=coverage[None]>0)
            if not np.isfinite(values).all(): raise ValueError('FP32 blend overflow')
            for col in range(0,self.source.width,256):
                right=min(col+256,self.source.width)
                window=(self.top,col,self.top+count,right)
                valid=self.source.valid(window)
                if np.any(valid & (coverage[:,col:right]==0)): raise ValueError('Uncovered valid pixels')
                classes=values[:,:,col:right].argmax(axis=0).astype(np.uint8)
                classes[~valid]=255
                self.sink.write(window,classes,valid)
            # Explicit copy bounds overlap-temporary memory to one stripe.
            if count<224:
                self.acc[:,:224-count]=self.acc[:,count:].copy()
                self.coverage[:224-count]=self.coverage[count:].copy()
            self.acc[:,224-count:]=0; self.coverage[224-count:]=0
            self.top+=count

    def add(self,window,logits):
        row,col,bottom,right=window
        if self.previous is not None and (row,col)<=self.previous: raise ValueError('Tile order changed')
        if logits.shape!=(10,224,224) or logits.dtype!=np.float32 or not np.isfinite(logits).all():
            raise ValueError('Expected finite FP32 tile logits')
        self.flush(row)
        with np.errstate(over='ignore',invalid='ignore'):
            self.acc[:,:bottom-row,col:right]+=logits[:,:bottom-row,:right-col]
        self.coverage[:bottom-row,col:right]+=1
        self.previous=(row,col)

    def finish(self): self.flush(self.source.height)


class DiskBlend:
    """Independent reference accumulation: disk raster windows, no rolling code."""
    def __init__(self,folder,source):
        import rasterio
        self.folder=Path(folder); self.source=source; self.stack=ExitStack()
        try:
            p=profile(source,count=10,dtype='float32'); p.update(compress='none',SPARSE_OK='YES')
            self.acc=self.stack.enter_context(rasterio.open(self.folder/'accumulator.tif','w+',**p))
            p=profile(source,dtype='uint16'); p.update(compress='none',SPARSE_OK='YES')
            self.coverage=self.stack.enter_context(rasterio.open(self.folder/'coverage.tif','w+',**p))
        except BaseException:
            self.stack.close(); raise

    def add(self,window,logits):
        row,col,bottom,right=window; w=rio_window(window)
        acc=self.acc.read(window=w); coverage=self.coverage.read(1,window=w)
        with np.errstate(over='ignore',invalid='ignore'): acc+=logits[:,:bottom-row,:right-col]
        if not np.isfinite(acc).all(): raise ValueError('Reference blend overflow')
        coverage+=1
        self.acc.write(acc,window=w); self.coverage.write(coverage,1,window=w)

    def finish(self,sink,cancelled=None):
        for window in windows(self.source.height,self.source.width):
            check_cancelled(cancelled); w=rio_window(window)
            values=self.acc.read(window=w); coverage=self.coverage.read(1,window=w)
            valid=self.source.valid(window)
            if np.any(valid & (coverage==0)): raise ValueError('Reference uncovered valid pixels')
            np.divide(values,coverage[None],out=values,where=coverage[None]>0)
            classes=values.argmax(axis=0).astype(np.uint8); classes[~valid]=255
            sink.write(window,classes,valid)

    def close(self): self.stack.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()


def deliver(folder,source,categories,reference,provenance,cancelled=None):
    """Create disk-backed COGs and independently reopen in fixed 256px windows."""
    import rasterio as rio
    from rasterio.shutil import copy as copy_raster
    destination=Path(folder); destination.mkdir(exist_ok=False)
    if len(json.dumps(provenance).encode())>256*1024: raise ValueError('Provenance exceeds budget')
    digest=canonical_hash(provenance); records={}
    for name,member,nodata in [('class_map.tif','classes.tif',255),('valid_mask.tif','mask.tif',0)]:
        check_cancelled(cancelled)
        staging=Path(categories)/member
        with rio.open(staging,'r+') as dst:
            dst.update_tags(AUREON_SCHEMA='aureon.streaming_delivery.v1',
                AUREON_INPUT_ID=source.identity['identity_sha256'],AUREON_PROVENANCE_SHA256=digest,
                AUREON_CLASS_NAMES=json.dumps(CLASS_NAMES),AREA_OR_POINT=source.grid['area_or_point'])
            dst.set_band_description(1,'KAO-TerraMind class index' if nodata==255 else 'valid=1; invalid=0')
        final=destination/name
        copy_raster(staging,final,driver='COG',COMPRESS='DEFLATE',BLOCKSIZE=256,
                    RESAMPLING='NEAREST',OVERVIEW_RESAMPLING='NEAREST',NUM_THREADS='1',BIGTIFF='IF_SAFER')
        with rio.open(final) as dst,rio.open(Path(reference)/member) as expected:
            if (_grid(dst,source.grid['crs'],rio,max_side=MAX_SIDE)!=source.grid or dst.dtypes!=('uint8',)
                    or dst.nodata!=nodata or dst.tags(ns='IMAGE_STRUCTURE').get('LAYOUT')!='COG'):
                raise ValueError('COG grid/type/layout mismatch')
            for window in windows(source.height,source.width):
                check_cancelled(cancelled); w=rio_window(window)
                if (not np.array_equal(dst.read(1,window=w),expected.read(1,window=w))
                        or not np.array_equal(dst.read_masks(1,window=w)>0,source.valid(window))):
                    raise ValueError('COG pixels or masks differ from independent disk reference')
            records[name]={'sha256':file_hash(final),'bytes':final.stat().st_size,'layout':'COG',
                'reopen_verified':True,'shape':[source.height,source.width],'dtype':'uint8','nodata':nodata,
                'overviews':dst.overviews(1)}
    receipt={'schema':'aureon.streaming_delivery.v1','status':'completed','format':'COG',
             'source':source.identity,'grid':source.grid,'provenance':provenance,
             'provenance_sha256':digest,'files':records,'reopen_validation':'passed',
             'class_names':list(CLASS_NAMES),'quality_evaluation':'not_evaluated'}
    write_json(destination/'receipt.json',receipt)
    return receipt
