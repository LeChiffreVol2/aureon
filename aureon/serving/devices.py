"""Cooperative single-host device ownership. No CUDA/model import or expiring GPU lease.

Each worker process owns a separate durable state directory. A shared legacy
gate makes UUID-scoped workers conflict with old host-wide exclusive workers.
Kernel locks last until close/process death; never unlink a live lock inode.
Separate OS users/containers require an operator-managed common lock namespace.
"""
from __future__ import annotations
from collections.abc import Iterator
from typing import cast
from .contracts import DeviceAttestation, PathInput, TorchProcess
from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
import re
import stat
import sys
import threading

from ..cloud.control import require
from ..scene_api.durable import _plain_path

_UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z')
_PROCESS = threading.Lock()
_QUARANTINED_DESCRIPTORS: list[int] = []


class DeviceLease(dict[str, str | bool | None]):
    def quarantine(self) -> None:
        """Uncertain CUDA cleanup must not hand a live process's device to another."""
        self['quarantined_until_process_exit'] = True


def device_uuid(value: str) -> str:
    require(isinstance(value, str) and _UUID.fullmatch(value) is not None,
            'full_gpu_uuid_required')
    return 'GPU-' + value[4:].lower()


@contextmanager
def _lock(path: PathInput, shared: bool = False, lease: DeviceLease | None = None) -> Iterator[None]:
    import fcntl
    path = _plain_path(path)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.getuid(),
                'invalid_gpu_lock', 409)
        require(info.st_mode & 0o022 == 0, 'writable_gpu_lock', 409)
        try:
            fcntl.flock(descriptor, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            require(False, 'gpu_owned', 409)
        named = path.stat()
        require((info.st_dev, info.st_ino) == (named.st_dev, named.st_ino), 'gpu_lock_replaced', 409)
        yield
    finally:
        if lease is not None and lease.get('quarantined_until_process_exit'):
            _QUARANTINED_DESCRIPTORS.append(descriptor)
        else: os.close(descriptor)


@contextmanager
def device_lease(device: str | None = None, *, root: PathInput | None = None) -> Iterator[DeviceLease]:
    """Keep this context around backend creation, every job and successful close.

    This is process-lifetime ownership, not a time lease that may be stolen
    while GPU work continues. Ordinals, UUID prefixes and MIG aliases are rejected.
    """
    require(sys.platform == 'linux', 'linux_device_ownership_required')
    selected = None if device is None else device_uuid(device)
    parent = _plain_path(cast(PathInput,root or os.environ.get('AUREON_GPU_LOCK_DIR', '/tmp'))).resolve(strict=True)
    require(parent.is_dir(), 'gpu_lock_directory_required')
    from .storage import require_local
    require_local(parent)
    owner=DeviceLease(scope='device' if selected else 'service_user_host',uuid=selected,
                      mechanism='kernel_flock_until_owned_close_or_process_death',
                      legacy_exclusion=True,lock_namespace=str(parent),quarantined_until_process_exit=False)
    with ExitStack() as stack:
        stack.enter_context(_lock(parent / ('aureon-gpu-' + str(os.getuid()) + '.lock'), shared=selected is not None,lease=owner))
        if selected:
            stack.enter_context(_lock(parent / ('aureon-device-' + str(os.getuid()) + '-' + selected + '.lock'),lease=owner))
        yield owner


@contextmanager
def device_binding(device: str | None = None) -> Iterator[None]:
    """Bind before CUDA initialization; separate GPUs require separate processes."""
    selected = None if device is None else device_uuid(device)
    require(_PROCESS.acquire(blocking=False), 'worker_process_already_owned', 409)
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        if selected:
            torch = cast(TorchProcess | None,sys.modules.get('torch'))
            require(torch is None or not torch.cuda.is_initialized(), 'cuda_already_initialized', 409)
            if previous is not None:
                allowed = [device_uuid(item.strip()) for item in previous.split(',') if item.strip()]
                require(selected in allowed, 'device_outside_visible_allowlist', 409)
            os.environ['CUDA_VISIBLE_DEVICES'] = selected
        yield
    finally:
        if selected:
            if previous is None: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else: os.environ['CUDA_VISIBLE_DEVICES'] = previous
        _PROCESS.release()


def attest_device(device: str) -> DeviceAttestation:
    """Called only by the explicitly executing backend, before loading weights."""
    selected = device_uuid(device)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'single_bound_gpu_required')
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if not actual.startswith('GPU-'): actual = 'GPU-' + actual
    require(device_uuid(actual) == selected, 'gpu_identity_mismatch', 409)
    return {'uuid': selected, 'cuda_index': 0, 'attested': True}
