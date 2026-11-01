"""Linux local-volume admission without requiring findmnt in minimal images."""
from __future__ import annotations
from .contracts import PathInput
from pathlib import Path, PurePosixPath
import re

def mount_type(path: PathInput, text: str) -> dict[str, str]:
    path=PurePosixPath(str(path));matches=[]
    for line in text.splitlines():
        head,tail=line.split(' - ',1);fields=head.split();mount=re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),fields[4])
        root=PurePosixPath(mount)
        if path==root or path.is_relative_to(root):matches.append((len(root.parts),tail.split()[0],mount))
    if not matches:raise ValueError('No mounted filesystem for service path')
    _,kind,mount=max(matches);return {'fstype':kind,'mount':mount,'path':str(path)}

def require_local(path: PathInput, *, mountinfo: str | None = None) -> dict[str, str]:
    path=Path(path).resolve(strict=True)
    result=mount_type(path,Path('/proc/self/mountinfo').read_text() if mountinfo is None else mountinfo)
    if result['fstype'] not in {'ext4','xfs','btrfs','overlay'} or re.match(r'^/mnt/[a-z](?:/|$)',str(path)):
        raise ValueError('Use a Linux local filesystem/overlay; DrvFS and shared network SQLite/scratch are unsupported')
    return result
