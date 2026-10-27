"""Prepared upload transport, with unchanged payloads and no cropping or resampling."""
import json
from pathlib import Path, PurePosixPath
import shutil
from ..cloud.control import require, NAME, SHA, CONTRACT, file_hash
from ..scene_api.durable import _plain_path, _sync_dir

# Prepared model inputs retain the 4096-pixel contract. Bound single-band
# provenance rasters separately: native source windows can include a halo
# beyond the prepared grid (for example 4100 pixels for a 4096-pixel scene).
MAX_PREPARED_SIDE = 4096
MAX_PROVENANCE_SIDE = 8192

def member(name):
    require(isinstance(name,str) and len(name)<=240 and '\\' not in name and ':' not in name,'invalid_member')
    p=PurePosixPath(name)
    require(not p.is_absolute() and p.parts and all(NAME.fullmatch(x) for x in p.parts)
            and all(x not in ('.','..') for x in p.parts) and p.as_posix()==name,'invalid_member')
    return name

def declarations(manifest, task):
    records={}
    def collect(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                name=member(value['path']);require(SHA.fullmatch(value['sha256']) is not None,'invalid_hash')
                require(name not in records or records[name]==value,'conflicting_member')
                records[name]=value
            for item in value.values():collect(item)
        elif isinstance(value,list):
            for item in value:collect(item)
    if task=='generation':
        require(manifest.get('schema')=='aureon.terramind_native_input.v1','wrong_input_schema')
        for key in ('image','mask'):collect(manifest[key])
    else:
        require(manifest.get('schema')=='aureon.sentinel2_scene.v1','wrong_input_schema')
        require(all(manifest[k]['path'].endswith('.tif') for k in ('raster','valid_mask')),'rasters_must_be_tiff');collect(manifest)
    require(1<=len(records)<=63 and 'manifest.json' not in records,'invalid_member_count')
    return records

def validate(body,task):
    require(isinstance(body,dict) and set(body)=={'files','spec'})
    files,spec=body['files'],body['spec']
    require(isinstance(files,list) and 2<=len(files)<=64 and isinstance(spec,dict))
    require(set(spec)=={'schema','task','manifest','members'} and spec['schema']=='aureon.prepared_upload.v1'
            and spec['task']==task and spec['manifest']=='manifest.json','unsupported_prepared_input')
    require(isinstance(spec['members'],dict) and len(spec['members'])==len(files),'invalid_members')
    names=[]
    for f in files:
        require(isinstance(f,dict) and set(f)=={'name','bytes','sha256'})
        require(isinstance(f['name'],str) and NAME.fullmatch(f['name']) and f['name'].endswith(('.json','.npy','.tif','.xml')),'invalid_filename')
        require(type(f['bytes']) is int and 0<f['bytes']<=CONTRACT['max_file_bytes'],'invalid_size')
        require(not f['name'].endswith(('.json','.xml')) or f['bytes']<=4*1024**2,'metadata_too_large')
        require(isinstance(f['sha256'],str) and SHA.fullmatch(f['sha256']),'invalid_hash');names.append(f['name'])
    require(len(set(names))==len(names) and set(spec['members'].values())==set(names),'duplicate_or_missing_member')
    for n in spec['members']:member(n)
    require('manifest.json' in spec['members'],'manifest_required')
    return files

def prepare(body,folder,token,task):
    files={f['name']:f for f in body['files']};mapping=body['spec']['members'];source=folder/'source'
    raw=(source/mapping['manifest.json']).read_bytes();manifest=json.loads(raw)
    records=declarations(manifest,task)
    require(set(mapping)==set(records)|{'manifest.json'},'undeclared_or_missing_members')
    for name,record in records.items():
        f=files[mapping[name]]
        require(f['sha256']==record['sha256'] and ('bytes' not in record or f['bytes']==record['bytes']),'manifest_file_identity')
    prepared_rasters={manifest[k]['path'] for k in ('raster','valid_mask')} if task=='scene' else set()
    for name,uploaded in mapping.items():
        if not uploaded.endswith('.tif'):continue
        path=_plain_path(source/uploaded)
        with path.open('rb') as stream:require(stream.read(4) in (b'II*\x00',b'MM\x00*',b'II+\x00',b'MM\x00+'),'tiff_header_required')
        import rasterio
        with rasterio.open(path,driver='GTiff') as ds:
            primary=name in prepared_rasters
            side=MAX_PREPARED_SIDE if primary else MAX_PROVENANCE_SIDE
            require(0<ds.width<=side and 0<ds.height<=side
                    and (1<=ds.count<=12 if primary else ds.count==1),'raster_dimensions_unsupported')
    target=folder/('prepared-'+token);target.mkdir()
    for name,uploaded in mapping.items():
        path=_plain_path(target/name);path.parent.mkdir(parents=True,exist_ok=True)
        with (source/uploaded).open('rb') as incoming,path.open('xb') as outgoing:
            shutil.copyfileobj(incoming,outgoing,1024**2);outgoing.flush()
            import os
            os.fsync(outgoing.fileno())
        require(file_hash(path)==files[uploaded]['sha256'],'prepared_file_changed');_sync_dir(path.parent)
    path=target/'manifest.json'
    if task=='generation':
        import numpy as np
        for key,shape,dtype in [('image',(12,224,224),'float32'),('mask',(224,224),'bool')]:
            values=np.load(target/manifest[key]['path'],mmap_mode='r',allow_pickle=False)
            require(values.shape==shape and values.dtype==dtype,'unsupported_generation_payload');del values
        # Existing Engine input validator: CPU-only, before any session/model.
        from ..terramind_native.session import _input
        _input(path)
    else:
        from ..prepare import check_scene
        check_scene(path)
    return path
