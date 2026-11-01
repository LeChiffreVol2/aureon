"""Create service manifests for existing local inputs, without copying pixels."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Literal, TypedDict

from .contracts import PathInput
from .inputs import member
from ..cloud.control import encoded
from ..scene_api.durable import _plain_path, _sync_dir
from ..terramind_native.common import INPUT_BANDS

_CHUNK=1024**2
_MAX_TOTAL=1024**3


class _Binding(TypedDict):
    path: str
    bytes: int
    sha256: str


def _stamp(info: os.stat_result) -> tuple[int,int,int,int,int]:
    return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


def _source(value: PathInput, parent: Path, suffix: str) -> Path:
    path=_plain_path(value).resolve(strict=True)
    if not path.is_relative_to(parent) or path.suffix!=suffix or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('Input must be a regular '+suffix+' file inside the manifest parent directory')
    member(path.relative_to(parent).as_posix())
    return path


def _hash_source(path: Path, parent: Path, max_bytes: int) -> tuple[_Binding,tuple[int,int,int,int,int]]:
    before=path.stat();identity=_stamp(before)
    if not 0<before.st_size<=max_bytes:raise ValueError('Input exceeds its finite service byte bound')
    hasher=hashlib.sha256();count=0
    flags=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_BINARY',0)
    with os.fdopen(os.open(path,flags),'rb') as stream:
        # Windows path-stat and descriptor-stat expose ctime differently.
        # Match descriptor device/inode/size/mtime; retain path ctime checks
        # before/after the entire hash and immediately before publication.
        if _stamp(os.fstat(stream.fileno()))[:4]!=identity[:4]:raise ValueError('Input changed before hashing')
        while True:
            block=stream.read(_CHUNK)
            if not block:break
            count+=len(block)
            if count>before.st_size:raise ValueError('Input grew during hashing')
            hasher.update(block)
        if _stamp(os.fstat(stream.fileno()))[:4]!=identity[:4]:raise ValueError('Input changed while hashing')
    if count!=before.st_size or _stamp(_plain_path(path).stat())!=identity:
        raise ValueError('Input changed while hashing')
    return {'path':path.relative_to(parent).as_posix(),'bytes':count,'sha256':hasher.hexdigest()},identity


def prepare_service_input(output: PathInput, *, task: Literal['generation_scene','features'],
                          image: PathInput | None=None, valid_mask: PathInput | None=None,
                          inputs: PathInput | None=None) -> dict[str,object]:
    """Bind existing files under the manifest's parent; create one fresh JSON.

    Paths may name nested files within that parent, but never links or outside
    files. Source files remain unchanged and are not copied. This only proves
    file identity; the bounded service parser validates pixels/grids/arrays.
    """
    destination=_plain_path(output)
    if destination.exists() or not destination.parent.is_dir() or destination.suffix!='.json':
        raise ValueError('Output must be a fresh .json file with an existing parent')
    parent=_plain_path(destination.parent).resolve(strict=True);destination=parent/destination.name
    member(destination.name)
    if task=='generation_scene':
        if image is None or valid_mask is None or inputs is not None:
            raise ValueError('generation_scene requires only image and valid_mask')
        sources={'image':_source(image,parent,'.tif'),'mask':_source(valid_mask,parent,'.tif')}
        schema='aureon.terramind_scene_input.v1';limit=_MAX_TOTAL
    elif task=='features':
        if inputs is None or image is not None or valid_mask is not None:
            raise ValueError('features requires only inputs')
        sources={'inputs':_source(inputs,parent,'.npz')};schema='aureon.terramind_features_input.v1';limit=512*1024**2
    else:raise ValueError('Expected generation_scene or features task')
    if len(set(sources.values()))!=len(sources) or destination in sources.values():
        raise ValueError('Sources and output must be distinct paths')
    bindings: dict[str,_Binding]={};identities: dict[str,tuple[int,int,int,int,int]]={}
    for name,path in sources.items():bindings[name],identities[name]=_hash_source(path,parent,limit)
    manifest: dict[str,object]={'schema':schema,**bindings}
    if task=='generation_scene':manifest.update(bands=list(INPUT_BANDS),units='surface_reflectance_x10000')
    payload=encoded(manifest)
    if len(payload)+sum(item['bytes'] for item in bindings.values())>_MAX_TOTAL:
        raise ValueError('Combined prepared input exceeds the service byte bound')
    # Recheck every source after all hashing, immediately before publication.
    for name,path in sources.items():
        if _stamp(_plain_path(path).stat())!=identities[name]:raise ValueError('Input changed before manifest publication')
    _plain_path(destination)
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_BINARY',0)
    with os.fdopen(os.open(destination,flags,0o600),'wb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    _sync_dir(parent)
    return {'schema':'aureon.prepared_service_input.v1','task':task,
        'manifest':{'path':str(destination),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()},
        'members':bindings,'pixel_validation':'deferred_to_service_parser','files_copied':False,'model_executed':False}
