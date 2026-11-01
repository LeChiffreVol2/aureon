"""Bounded result persistence with one read-back identity per file.

No model imports. Byte identities are computed from unbuffered descriptor reads of saved files, not
from incoming buffers. A receipt is request-local and never a cross-job cache.
The caller owns a private fresh destination; concurrent writers are unsupported.
No fsync/power-loss durability guarantee is implied by successful read-back.
"""
from __future__ import annotations
from collections.abc import Mapping
from .contracts import FileIdentity, PathInput
import hashlib
import os
from pathlib import Path
import stat
from typing import BinaryIO


def member_name(name: object) -> str:
    if (not isinstance(name,str) or not name or name in {'.','..'}
            or any(c in name for c in ('/','\\','\x00',':'))):
        raise ValueError('Expected one safe artifact filename')
    return name


class VerifiedFiles:
    def __init__(self, destination: PathInput, *, max_total_bytes: int, max_files: int = 32) -> None:
        if type(max_total_bytes) is not int or max_total_bytes<0 or type(max_files) is not int or max_files<1:
            raise ValueError('Explicit positive bounds required')
        destination=Path(destination).absolute()
        # Resolve the parent once. All child names are checked single components,
        # the private directory is new, and every member uses exclusive creation.
        member_name(destination.name)
        self.root=destination.parent.resolve(strict=True)/destination.name
        self.root.mkdir(mode=0o700 if os.name=='posix' else 0o777)
        self.max_total_bytes=max_total_bytes;self.max_files=max_files
        self.total_bytes=0;self.bindings: dict[str, FileIdentity]={};self.data: dict[str, bytes]={};self.failed=False

    def save(self, name: str, data: bytes, *, expected: Mapping[str, object] | None = None) -> FileIdentity:
        if self.failed:raise RuntimeError('Failed artifact batch cannot continue')
        try:
            name=member_name(name)
            if name in self.bindings or len(self.bindings)>=self.max_files:
                raise ValueError('Duplicate artifact or file-count bound')
            if not isinstance(data,bytes):raise TypeError('Expected immutable artifact bytes')
            size=len(data)
            if self.total_bytes+size>self.max_total_bytes:raise ValueError('Artifact byte budget exceeded')
            if expected is not None:
                if type(expected.get('bytes')) is not int or expected['bytes']!=size:
                    raise ValueError('Artifact size differs from declared identity')
                digest=expected.get('sha256')
                if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                    raise ValueError('Invalid artifact hash declaration')
            path=self.root/name
            # One exclusive read/write descriptor. buffering=0 is essential:
            # the read must return kernel file bytes, never a Python write buffer.
            flags=os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_BINARY',0)
            with os.fdopen(os.open(path,flags,0o600),'w+b',buffering=0) as f:
                if f.write(data)!=size:raise OSError('Short artifact write')
                f.seek(0)
                before=os.fstat(f.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size!=size:raise ValueError('Saved artifact size/type mismatch')
                saved=f.read(size+1)
                after=os.fstat(f.fileno())
                # Bind descriptor bytes to the final name, without following links.
                # This catches rename/replacement and preserves path identity.
                named=os.stat(path,follow_symlinks=False)
                def stamp(s: os.stat_result) -> tuple[int, int, int, int]:return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
                if (stamp(before)!=stamp(after) or stamp(after)!=stamp(named)
                        or not stat.S_ISREG(named.st_mode)):
                    raise ValueError('Artifact changed during read-back or path identity differs')
            if len(saved)!=size:raise ValueError('Truncated saved artifact')
            identity: FileIdentity={'path':str(path),'bytes':size,'sha256':hashlib.sha256(saved).hexdigest()}
            if expected is not None and identity['sha256']!=expected['sha256']:
                raise ValueError('Saved artifact hash mismatch')
            self.total_bytes+=size;self.bindings[name]=identity;self.data[name]=saved
            return identity
        except BaseException:
            self.failed=True
            self.data.clear()
            raise

    def release(self) -> None:
        """Release saved-byte buffers; persistent files and identities remain."""
        self.data.clear()

    def save_stream(self, name: str, stream: BinaryIO, *, expected: Mapping[str, object],
                    retain_bytes: bool = False, max_retained_bytes: int = 4*1024**2) -> FileIdentity:
        """Persist bounded chunks, then hash actual saved descriptor bytes.

        Large artifacts are never retained in Python memory. Only explicitly
        bounded receipt files may request a retained copy for binding checks.
        """
        if self.failed:raise RuntimeError('Failed artifact batch cannot continue')
        try:
            name=member_name(name);size=expected.get('bytes');digest=expected.get('sha256')
            if name in self.bindings or len(self.bindings)>=self.max_files:
                raise ValueError('Duplicate artifact or file-count bound')
            if type(size) is not int or size<0 or self.total_bytes+size>self.max_total_bytes:
                raise ValueError('Artifact byte budget exceeded')
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid artifact hash declaration')
            if retain_bytes and size>max_retained_bytes:raise ValueError('Receipt retention budget exceeded')
            path=self.root/name
            flags=os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_BINARY',0)
            with os.fdopen(os.open(path,flags,0o600),'w+b',buffering=0) as target:
                remaining=size
                while remaining:
                    chunk=stream.read(min(1024**2,remaining))
                    if not chunk or len(chunk)>remaining:raise ValueError('Truncated or oversized artifact')
                    if target.write(chunk)!=len(chunk):raise OSError('Short artifact write')
                    remaining-=len(chunk)
                if stream.read(1):raise ValueError('Artifact exceeds declared size')
                target.seek(0);before=os.fstat(target.fileno());hasher=hashlib.sha256();saved=bytearray()
                count=0
                while True:
                    chunk=target.read(1024**2)
                    if not chunk:break
                    count+=len(chunk);hasher.update(chunk)
                    if retain_bytes:saved.extend(chunk)
                after=os.fstat(target.fileno());named=os.stat(path,follow_symlinks=False)
                def stamp(s: os.stat_result) -> tuple[int,int,int,int]:
                    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns
                if count!=size or not stat.S_ISREG(named.st_mode) or stamp(before)!=stamp(after) or stamp(after)!=stamp(named):
                    raise ValueError('Saved artifact changed or path identity differs')
            if hasher.hexdigest()!=digest:raise ValueError('Saved artifact hash mismatch')
            identity: FileIdentity={'path':str(path),'bytes':size,'sha256':digest}
            self.total_bytes+=size;self.bindings[name]=identity
            if retain_bytes:self.data[name]=bytes(saved)
            return identity
        except BaseException:
            self.failed=True;self.data.clear();raise
