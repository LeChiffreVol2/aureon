"""Bounded, hash-addressed blobs for trusted replica operators; no remote paths."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import BinaryIO


def digest_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected SHA256 content key")
    return value


class ArtifactStore:
    """One-process writer, local durable directory. Operators own explicit retention.

    No automatic deletion, URL fetching, object deserialization or symlink support.
    Disk quota includes incomplete uploads from interrupted processes. Use a separate
    directory from evidence/model assets. Cross-host clients use the bounded wire API.
    """
    def __init__(self, root: str | Path, *, max_bytes: int, max_artifact_bytes: int) -> None:
        if any(type(n) is not int or n < 1 for n in (max_bytes, max_artifact_bytes)) or max_artifact_bytes > max_bytes:
            raise ValueError("Positive artifact and disk budgets required")
        self.root = Path(root).absolute()
        if self.root.is_symlink():
            raise ValueError("Artifact root must not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.max_artifact_bytes = max_bytes, max_artifact_bytes
        self._lock = threading.Lock()

    def _path(self, digest: str) -> Path:
        path = self.root / digest_name(digest)
        if path.is_symlink():
            raise ValueError("Artifact symlinks are not admitted")
        return path

    def put(self, source: BinaryIO, *, size: int, digest: str) -> str:
        if type(size) is not int or not 0 <= size <= self.max_artifact_bytes:
            raise ValueError("Artifact size exceeds configured bound")
        target = self._path(digest)
        with self._lock:
            if target.exists():
                checksum, total = hashlib.sha256(), 0
                while total < size:
                    chunk = source.read(min(65536, size - total))
                    if not chunk:
                        raise ValueError("Truncated artifact")
                    total += len(chunk)
                    checksum.update(chunk)
                if checksum.hexdigest() != digest:
                    raise ValueError("Artifact checksum mismatch")
                with self.open(digest):
                    return digest
            usage = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
            if usage + size > self.max_bytes:
                raise BufferError("Artifact disk quota exhausted")
            descriptor, name = tempfile.mkstemp(prefix=".upload-", dir=self.root)
            temporary = Path(name)
            try:
                total = 0
                checksum = hashlib.sha256()
                with os.fdopen(descriptor, "wb") as output:
                    while total < size:
                        chunk = source.read(min(65536, size - total))
                        if not chunk:
                            raise ValueError("Truncated artifact")
                        total += len(chunk)
                        checksum.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if checksum.hexdigest() != digest:
                    raise ValueError("Artifact checksum mismatch")
                os.replace(temporary, target)
                if os.name == "posix":
                    directory = os.open(self.root, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return digest

    @contextmanager
    def open(self, digest: str) -> Iterator[BinaryIO]:
        path = self._path(digest)
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size > self.max_artifact_bytes:
                raise ValueError("Stored artifact exceeds read budget")
            checksum = hashlib.sha256()
            while chunk := stream.read(65536):
                checksum.update(chunk)
            if checksum.hexdigest() != digest:
                raise ValueError("Stored artifact is corrupt")
            stream.seek(0)
            yield stream
