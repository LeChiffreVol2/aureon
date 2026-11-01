"""Prepared upload transport, with unchanged payloads and no cropping or resampling."""
from __future__ import annotations
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from .workflow_config import WorkflowBinding
from ..cloud.contracts import UploadBody, UploadFile
from ..cloud.io_types import RasterioInspection
from .contracts import ManifestMember, PreparedManifest, PreparedSpec
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

def member(name: str) -> str:
    require(isinstance(name,str) and len(name)<=240 and '\\' not in name and ':' not in name,'invalid_member')
    p=PurePosixPath(name)
    require(not p.is_absolute() and p.parts and all(NAME.fullmatch(x) for x in p.parts)
            and all(x not in ('.','..') for x in p.parts) and p.as_posix()==name,'invalid_member')
    return name

def declarations(manifest: PreparedManifest, task: str) -> dict[str, ManifestMember]:
    records: dict[str, ManifestMember]={}
    def collect(value: object) -> None:
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                name=member(value['path']);require(SHA.fullmatch(value['sha256']) is not None,'invalid_hash')
                require(name not in records or records[name]==value,'conflicting_member')
                records[name]=cast(ManifestMember,value)
            for item in value.values():collect(item)
        elif isinstance(value,list):
            for item in value:collect(item)
    if task in ('generation','generation_scene'):
        schema='aureon.terramind_native_input.v1' if task=='generation' else 'aureon.terramind_scene_input.v1'
        require(manifest.get('schema')==schema,'wrong_input_schema')
        for key in ('image','mask'):collect(cast(Mapping[str, ManifestMember],manifest)[key])
    elif task=='features':
        require(manifest.get('schema')=='aureon.terramind_features_input.v1','wrong_input_schema')
        collect(cast(Mapping[str,ManifestMember],manifest)['inputs'])
    else:
        require(manifest.get('schema')=='aureon.sentinel2_scene.v1','wrong_input_schema')
        require(all(cast(Mapping[str, ManifestMember],manifest)[k]['path'].endswith('.tif') for k in ('raster','valid_mask')),'rasters_must_be_tiff');collect(manifest)
    require(1<=len(records)<=63 and 'manifest.json' not in records,'invalid_member_count')
    return records

def validate(body: UploadBody, task: str) -> list[UploadFile]:
    require(isinstance(body,dict) and set(body)=={'files','spec'})
    files=body['files'];spec=cast(PreparedSpec,body['spec'])
    require(isinstance(files,list) and 2<=len(files)<=64 and isinstance(spec,dict))
    require(set(spec)=={'schema','task','manifest','members'} and spec['schema']=='aureon.prepared_upload.v1'
            and spec['task']==task and spec['manifest']=='manifest.json','unsupported_prepared_input')
    require(isinstance(spec['members'],dict) and len(spec['members'])==len(files),'invalid_members')
    names=[]
    for f in files:
        require(isinstance(f,dict) and set(f)=={'name','bytes','sha256'})
        extensions=('.json','.npy','.tif','.xml','.npz') if task=='features' else ('.json','.npy','.tif','.xml')
        require(isinstance(f['name'],str) and NAME.fullmatch(f['name']) and f['name'].endswith(extensions),'invalid_filename')
        require(type(f['bytes']) is int and 0<f['bytes']<=CONTRACT['max_file_bytes'],'invalid_size')
        require(not f['name'].endswith(('.json','.xml')) or f['bytes']<=4*1024**2,'metadata_too_large')
        require(isinstance(f['sha256'],str) and SHA.fullmatch(f['sha256']),'invalid_hash');names.append(f['name'])
    require(len(set(names))==len(names) and set(spec['members'].values())==set(names),'duplicate_or_missing_member')
    for n in spec['members']:member(n)
    require('manifest.json' in spec['members'],'manifest_required')
    return files

def prepare(body: UploadBody, folder: Path, token: str, task: str, *, workflow: WorkflowBinding | None=None) -> Path:
    files={f['name']:f for f in body['files']};mapping=cast(PreparedSpec,body['spec'])['members'];source=folder/'source'
    raw=(source/mapping['manifest.json']).read_bytes();manifest: PreparedManifest=json.loads(raw)
    records=declarations(manifest,task)
    require(set(mapping)==set(records)|{'manifest.json'},'undeclared_or_missing_members')
    for name,record in records.items():
        f=files[mapping[name]]
        require(f['sha256']==record['sha256'] and ('bytes' not in record or f['bytes']==record['bytes']),'manifest_file_identity')
    raster_keys=('raster','valid_mask') if task=='scene' else ('image','mask') if task=='generation_scene' else ()
    prepared_rasters={cast(Mapping[str, ManifestMember],manifest)[k]['path'] for k in raster_keys}
    if task=='generation_scene':
        require(workflow is not None and all(name.endswith('.tif') for name in prepared_rasters), 'scene_tiff_required')
    if task=='features':
        require(workflow is not None and cast(Mapping[str,ManifestMember],manifest)['inputs']['path'].endswith('.npz'),'feature_npz_required')
    for name,uploaded in mapping.items():
        if not uploaded.endswith('.tif'):continue
        path=_plain_path(source/uploaded)
        with path.open('rb') as stream:require(stream.read(4) in (b'II*\x00',b'MM\x00*',b'II+\x00',b'MM\x00+'),'tiff_header_required')
        import rasterio as untyped_rasterio
        rasterio=cast(RasterioInspection,untyped_rasterio)  # Narrow the upstream module immediately.
        with rasterio.open(path,driver='GTiff') as ds:
            primary=name in prepared_rasters
            side=workflow['limits']['max_input_side'] if task=='generation_scene' and workflow is not None else MAX_PREPARED_SIDE if primary else MAX_PROVENANCE_SIDE
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
            values=np.load(target/cast(Mapping[str, ManifestMember],manifest)[key]['path'],mmap_mode='r',allow_pickle=False)
            require(values.shape==shape and values.dtype==dtype,'unsupported_generation_payload');del values
        # Existing Engine input validator: CPU-only, before any session/model.
        from ..terramind_native.session import _input
        _input(path)
    elif task=='generation_scene':
        require(workflow is not None,'workflow_binding_required')
        checked=cast('WorkflowBinding',workflow)
        from ..api.scene_generation import SceneGenerationRequest, inspect_generation_scene
        from ..terramind_native.common import INPUT_BANDS
        metadata=cast(Mapping[str,object],manifest)
        require(metadata.get('bands')==INPUT_BANDS and metadata.get('units')=='surface_reflectance_x10000','scene_semantics_required')
        options=checked['options'];limits=checked['limits']
        from ..api.spatial import InvalidTilePolicy
        from ..api.contracts import ExecutionProfile
        request=SceneGenerationRequest(image=target/cast(Mapping[str,ManifestMember],manifest)['image']['path'],
            valid_mask=target/cast(Mapping[str,ManifestMember],manifest)['mask']['path'],
            model_file=target/'_model_not_read',tokenizer_file=target/'_tokenizer_not_read',work=target/'_inspect_work',output=target/'_inspect_output',
            model=checked['model_id'],stride=cast(int,options['stride']),invalid_policy=cast(InvalidTilePolicy,options['invalid_policy']),
            execution_profile=cast(ExecutionProfile,options['execution_profile']),max_in_flight=cast(int,options['max_in_flight']),
            max_buffer_bytes=limits['max_buffer_bytes'],max_decode_bytes=limits['max_decode_bytes'],
            max_output_bytes=limits['max_output_bytes'],min_free_bytes=limits['min_free_bytes'])
        plan=inspect_generation_scene(request)
        if 'grid' in metadata:require(metadata['grid']==plan['grid'],'scene_grid_mismatch')
    elif task=='features':
        require(workflow is not None,'workflow_binding_required')
        checked=cast('WorkflowBinding',workflow)
        from ..api.encoder_io import inspect_encoder_input
        inspection=inspect_encoder_input(target/cast(Mapping[str,ManifestMember],manifest)['inputs']['path'],
                                     modalities=cast(list[str],checked['options']['modalities']))
        require(cast(int,inspection['batch'])<=checked['limits']['max_batch']
                and cast(int,inspection['array_bytes'])<=checked['limits']['max_input_bytes'],'feature_input_budget')
    else:
        from ..prepare import check_scene
        check_scene(path)
    return path
