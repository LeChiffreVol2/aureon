"""Exclusive local ownership and safe filesystem inventories; no model imports."""
from __future__ import annotations
from collections.abc import Collection, Iterator
from .contracts import PathInput, ProcessIdentity, FileIdentity
from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
from ..cloud.control import require, file_hash
from .errors import safe_error as safe_error  # Compatibility re-export; redaction has no I/O dependencies.
from ..scene_api.durable import _plain_path, _FileLock

@contextmanager
def exclusive(root: PathInput, name: str = 'worker.lock') -> Iterator[None]:
    lock=_FileLock(_plain_path(Path(root)/name))
    try:yield
    finally:lock.close()

def process_identity(pid: int | None = None) -> ProcessIdentity | None:
    pid=os.getpid() if pid is None else pid
    if os.name=='nt':return {'pid':pid,'platform':'windows','start_ticks':None}
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'start_ticks':fields[19],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'state':fields[0]}
    except (FileNotFoundError,ProcessLookupError):return None

def alive(identity: ProcessIdentity | None) -> bool:
    if not identity:return False
    require(os.name!='nt','linux_process_attestation_required',409)
    current=process_identity(identity['pid'])
    return bool(current and current['state']!='Z' and current['start_ticks']==identity['start_ticks'] and current['boot_id']==identity['boot_id'])

def inventory(root: PathInput, *, skip: Collection[str] = (), max_files: int = 100000, max_bytes: int = 16*1024**3) -> list[FileIdentity]:
    root=_plain_path(root).resolve(strict=True);device=root.stat().st_dev;rows: list[FileIdentity]=[];total=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            p=_plain_path(Path(directory)/name)
            require(p.stat().st_dev==device,'nested_mount_not_owned',409)
        for name in sorted(files):
            p=Path(directory)/name;rel=p.relative_to(root).as_posix()
            if rel in skip:continue
            require(stat.S_ISREG(p.stat().st_mode),'regular_file_required',409)
            size=p.stat().st_size;total+=size
            require(len(rows)<max_files and total<=max_bytes,'inventory_limit',413)
            rows.append({'path':rel,'bytes':size,'sha256':file_hash(p)})
    return sorted(rows,key=lambda x:x['path'])

def safe_child(root: PathInput, name: PathInput) -> Path:
    root=_plain_path(root).resolve(strict=True);relative=Path(name)
    require(not relative.is_absolute() and not relative.drive and relative.parts and '..' not in relative.parts,'unsafe_owned_path',409)
    path=_plain_path(root/relative).resolve()
    require(path.is_relative_to(root) and path!=root,'unsafe_owned_path',409)
    return path

