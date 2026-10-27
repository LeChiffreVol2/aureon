"""Bounded result persistence with one read-back identity per file.

No model imports. Byte identities are computed from unbuffered descriptor reads of saved files, not
from incoming buffers. A receipt is request-local and never a cross-job cache.
The caller owns a private fresh destination; concurrent writers are unsupported.
No fsync/power-loss durability guarantee is implied by successful read-back.
"""
import hashlib
import os
from pathlib import Path
import stat


def member_name(name):
    if (not isinstance(name,str) or not name or name in {'.','..'}
            or any(c in name for c in ('/','\\','\x00',':'))):
        raise ValueError('Expected one safe artifact filename')
    return name


class VerifiedFiles:
    def __init__(self, destination, *, max_total_bytes, max_files=32):
        if type(max_total_bytes) is not int or max_total_bytes<0 or type(max_files) is not int or max_files<1:
            raise ValueError('Explicit positive bounds required')
        destination=Path(destination).absolute()
        # Resolve the parent once. All child names are checked single components,
        # the private directory is new, and every member uses exclusive creation.
        member_name(destination.name)
        self.root=destination.parent.resolve(strict=True)/destination.name
        self.root.mkdir(mode=0o700 if os.name=='posix' else 0o777)
        self.max_total_bytes=max_total_bytes;self.max_files=max_files
        self.total_bytes=0;self.bindings={};self.data={};self.failed=False

    def save(self,name,data,*,expected=None):
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
                def stamp(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
                if (stamp(before)!=stamp(after) or stamp(after)!=stamp(named)
                        or not stat.S_ISREG(named.st_mode)):
                    raise ValueError('Artifact changed during read-back or path identity differs')
            if len(saved)!=size:raise ValueError('Truncated saved artifact')
            identity={'path':str(path),'bytes':size,'sha256':hashlib.sha256(saved).hexdigest()}
            if expected is not None and identity['sha256']!=expected['sha256']:
                raise ValueError('Saved artifact hash mismatch')
            self.total_bytes+=size;self.bindings[name]=identity;self.data[name]=saved
            return identity
        except BaseException:
            self.failed=True
            self.data.clear()
            raise

    def release(self):
        """Release saved-byte buffers; persistent files and identities remain."""
        self.data.clear()
