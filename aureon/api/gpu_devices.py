"""Explicit physical NVIDIA inventory and snapshot-based ModelPool admission.

No Torch import, CUDA allocation, model execution or automatic device selection.
Inventory is host/NVML scope, not CUDA process ordinal scope. Workers select full
UUIDs, avoiding ordinal reordering. A snapshot is not a reservation: ModelPool's
live worker checks and the caller's measured workspace budget remain required.
"""
from __future__ import annotations

from collections.abc import Sequence
import csv
from dataclasses import dataclass
import io
import math
import re
import shutil
import subprocess
from time import monotonic

from .model_pool import DeviceSlot

_UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
_MIB = 1024**2


def _integer(value: int, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))


@dataclass(frozen=True)
class NvidiaDevice:
    """Physical GPU identity and point-in-time driver memory, in bytes."""
    uuid: str
    name: str
    compute_capability: tuple[int, int]
    total_bytes: int
    free_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.uuid, str) or _UUID.fullmatch(self.uuid) is None:
            raise ValueError('Supply a full physical GPU UUID; MIG selectors are not supported by this inventory')
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 256:
            raise ValueError('A bounded GPU name is required')
        if not isinstance(self.compute_capability, tuple) or len(self.compute_capability) != 2:
            raise ValueError('compute_capability must be an immutable (major, minor) pair')
        for value in self.compute_capability:
            _integer(value, 'compute_capability')
        _integer(self.total_bytes, 'total_bytes', minimum=1)
        _integer(self.free_bytes, 'free_bytes')
        if self.free_bytes > self.total_bytes:
            raise ValueError('free_bytes exceeds physical memory')


@dataclass(frozen=True)
class NvidiaInventory:
    """One host snapshot. Constructed snapshots are useful for capacity simulation.

    They never attest hardware execution or performance. ``observed_at`` uses
    the process's monotonic clock; do not serialize it across host clock domains.
    """
    devices: tuple[NvidiaDevice, ...]
    observed_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.devices, tuple) or any(not isinstance(d, NvidiaDevice) for d in self.devices):
            raise TypeError('devices must be an immutable tuple of NvidiaDevice values')
        if len({d.uuid.lower() for d in self.devices}) != len(self.devices):
            raise ValueError('Duplicate physical GPU UUID')
        if isinstance(self.observed_at, bool) or not math.isfinite(self.observed_at) or self.observed_at < 0:
            raise ValueError('observed_at must be a finite nonnegative monotonic time')


def _parse_inventory(output: str, observed_at: float) -> NvidiaInventory:
    if len(output) > 1024**2:
        raise ValueError('NVIDIA inventory output exceeds 1 MiB')
    devices = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 5:
            raise ValueError('NVIDIA driver returned an unsupported inventory format')
        uuid, name, total, free, capability = (item.strip() for item in row)
        if re.fullmatch(r'[0-9]+\.[0-9]+', capability) is None:
            raise ValueError('Driver did not report a numeric compute capability')
        major, minor = capability.split('.')
        if not total.isdecimal() or not free.isdecimal():
            raise ValueError('Driver did not report numeric memory in MiB')
        devices.append(NvidiaDevice(uuid, name, (int(major), int(minor)), int(total)*_MIB, int(free)*_MIB))
    return NvidiaInventory(tuple(devices), observed_at)


def probe_nvidia_gpus(*, timeout_seconds: float = 10.0) -> NvidiaInventory:
    """Read driver inventory explicitly, without creating a CUDA context.

    Queries physical GPUs visible to nvidia-smi; CUDA_VISIBLE_DEVICES is not
    interpreted as an NVML ordinal filter. No GPU is chosen for execution here.
    Unsupported driver/query fields fail explicitly, never imply zero capacity.
    """
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
        raise ValueError('timeout_seconds must be finite in (0, 60]')
    executable = shutil.which('nvidia-smi')
    if executable is None:
        raise RuntimeError('nvidia-smi is unavailable; supply an explicit measured device configuration')
    observed_at = monotonic()  # Conservatively include query time in snapshot age.
    result = subprocess.run([executable, '--query-gpu=uuid,name,memory.total,memory.free,compute_cap',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            encoding='utf-8', errors='strict', timeout=timeout_seconds, check=False)
    if result.returncode != 0:
        raise RuntimeError('NVIDIA inventory query failed: ' + result.stderr[:2000])
    return _parse_inventory(result.stdout, observed_at)


def cuda_device_slots(inventory: NvidiaInventory, *, allowed_uuids: Sequence[str],
                      memory_bytes: int, reserve_bytes: int,
                      max_residents: int = 1, max_age_seconds: float = 5.0,
                      now: float | None = None) -> tuple[DeviceSlot, ...]:
    """Create explicit UUID-bound ModelPool slots if the snapshot has headroom.

    Select devices intentionally with ``allowed_uuids``; no implicit all-device
    claim or CUDA mask inheritance. ``memory_bytes`` is the per-device pool
    envelope including resident/transient/output workspaces. ``reserve_bytes``
    stays outside that envelope. This does not estimate model size or choose N
    from SM count, and never overrides a precision or parallel qualification.
    """
    if not isinstance(inventory, NvidiaInventory):
        raise TypeError('inventory requires NvidiaInventory')
    if isinstance(allowed_uuids, (str, bytes)) or not allowed_uuids:
        raise ValueError('Explicitly select one or more physical GPU UUIDs')
    if any(not isinstance(value, str) for value in allowed_uuids):
        raise TypeError('GPU selectors must be strings')
    selectors = tuple(value.lower() for value in allowed_uuids)
    if len(set(selectors)) != len(selectors):
        raise ValueError('Duplicate physical device selection')
    _integer(memory_bytes, 'memory_bytes', minimum=1)
    _integer(reserve_bytes, 'reserve_bytes')
    _integer(max_residents, 'max_residents', minimum=1)
    current = monotonic() if now is None else now
    if (isinstance(current, bool) or not math.isfinite(current) or
            isinstance(max_age_seconds, bool) or not math.isfinite(max_age_seconds) or max_age_seconds < 0 or
            not 0 <= current - inventory.observed_at <= max_age_seconds):
        raise ValueError('GPU inventory is stale, from another clock domain, or has an invalid age limit')
    lookup = {device.uuid.lower(): device for device in inventory.devices}
    slots = []
    for selector in selectors:
        device = lookup.get(selector)
        if device is None:
            raise ValueError('Selected full GPU UUID is not in this inventory: ' + selector)
        if memory_bytes + reserve_bytes > device.free_bytes:
            raise ValueError('Insufficient observed free memory plus reserve on ' + device.uuid)
        slots.append(DeviceSlot(device.uuid, memory_bytes, device.uuid, max_residents))
    return tuple(slots)
