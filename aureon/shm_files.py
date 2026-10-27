"""Private Linux mmap slots and a strict, bounded local control protocol.

Files are visible only in /dev/shm and are never selected by arbitrary paths.
The same Linux UID is the trust boundary; this is not remote authentication.
No multiprocessing resource tracker is involved and files are never resized
after creation. Unlinking a poisoned client leaves existing server mappings
valid until the server drains and closes its original file descriptors.
"""
import json
import mmap
import os
import re
import stat
import sys
import uuid

import numpy as np

ROOT = '/dev/shm'
MAX_CONTROL_BYTES = 2048
MAX_TILES = 8
INPUT_SHAPE = (12, 224, 224)
OUTPUT_SHAPE = (10, 224, 224)
MODEL_IDS = frozenset(('kao-terramind-a', 'kao-terramind-b', 'kao-terramind-c'))
SCHEMA = 'aureon.shm_pooling.v1'
TOKEN = re.compile(r'[0-9a-f]{32}\Z')


def require_linux():
    if sys.platform != 'linux':
        raise RuntimeError('AUREON shared-memory transport requires Linux; choose binary_pack explicitly elsewhere')


def slot_name(client, kind):
    if not isinstance(client, str) or not TOKEN.fullmatch(client) or kind not in ('input', 'output'):
        raise ValueError('Invalid shared-memory slot identity')
    return f'aureon-shm-v1-{client}-{kind}'


def slot_bytes(kind, capacity):
    return capacity * (12 if kind == 'input' else 10) * 224 * 224 * 4


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate control field')
        value[key] = item
    return value


def decode_descriptor(body):
    if not isinstance(body, bytes) or not 0 < len(body) <= MAX_CONTROL_BYTES:
        raise ValueError('Shared-memory control body must be between 1 and 2048 bytes')
    value = json.loads(body, object_pairs_hook=_unique_object)
    if (not isinstance(value, dict) or set(value) != {'schema', 'client', 'request', 'model', 'count', 'capacity', 'dtype', 'slots'}
            or value['schema'] != SCHEMA or value['model'] not in MODEL_IDS or value['dtype'] != '<f4'):
        raise ValueError('Invalid shared-memory control schema/model/precision')
    for key in ('client', 'request'):
        if not isinstance(value[key], str) or not TOKEN.fullmatch(value[key]):
            raise ValueError('Invalid shared-memory control token')
    if (type(value['capacity']) is not int or not 1 <= value['capacity'] <= MAX_TILES
            or type(value['count']) is not int or not 1 <= value['count'] <= value['capacity']):
        raise ValueError('Shared-memory input count must be bounded by capacity 1..8')
    slots = value['slots']
    if not isinstance(slots, dict) or set(slots) != {'input', 'output'}:
        raise ValueError('Exactly one input/output slot pair is required')
    for kind, item in slots.items():
        if (not isinstance(item, dict) or set(item) != {'name', 'bytes', 'device', 'inode'}
                or item['name'] != slot_name(value['client'], kind)
                or type(item['bytes']) is not int or item['bytes'] != slot_bytes(kind, value['capacity'])
                or type(item['device']) is not int or item['device'] < 0
                or type(item['inode']) is not int or item['inode'] <= 0):
            raise ValueError('Invalid shared-memory slot name, size or inode identity')
    if (slots['input']['device'], slots['input']['inode']) == (slots['output']['device'], slots['output']['inode']):
        raise ValueError('Input and output must be different files')
    return value


def decode_completion(body, expected):
    if not isinstance(body, bytes) or not 0 < len(body) <= MAX_CONTROL_BYTES:
        raise ValueError('Invalid shared-memory completion size')
    value = json.loads(body, object_pairs_hook=_unique_object)
    if (not isinstance(value, dict) or set(value) != set(expected)
            or any(type(value[key]) is not type(item) or value[key] != item for key, item in expected.items())):
        raise ValueError('Shared-memory completion identity does not match this request')
    return value


def _check_stat(info, expected_bytes, *, expected=None):
    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_size != expected_bytes):
        raise ValueError('Shared-memory slots require private 0600 regular files owned by this UID, one link and exact bytes')
    if expected is not None and (info.st_dev != expected['device'] or info.st_ino != expected['inode']):
        raise ValueError('Shared-memory slot inode/device identity changed')


class SharedSlots:
    """Own descriptors/mappings; never retain exported NumPy buffer views."""

    def __init__(self):
        self.directory = None
        self.fds, self.maps, self.identities = {}, {}, {}
        self.created = set()
        self.client = None
        self.capacity = None
        self.closed = False

    @classmethod
    def create(cls, capacity):
        require_linux()
        if type(capacity) is not int or not 1 <= capacity <= MAX_TILES:
            raise ValueError('Shared-memory capacity must be an integer from 1 to 8')
        self = cls()
        self.capacity, self.client = capacity, uuid.uuid4().hex
        try:
            self.directory = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            for kind in ('input', 'output'):
                name, size = slot_name(self.client, kind), slot_bytes(kind, capacity)
                fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=self.directory)
                self.fds[kind] = fd
                self.created.add(kind)
                os.ftruncate(fd, size)
                # Reserve tmpfs pages before NumPy writes: a sparse ftruncate
                # alone can otherwise produce SIGBUS when /dev/shm is full.
                os.posix_fallocate(fd, 0, size)
                info = os.fstat(fd)
                self.identities[kind] = {'name': name, 'bytes': size, 'device': info.st_dev, 'inode': info.st_ino}
                _check_stat(info, size)
                self.maps[kind] = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE if kind == 'input' else mmap.ACCESS_READ)
            return self
        except BaseException:
            self.close(unlink=True)
            raise

    @classmethod
    def attach(cls, descriptor):
        require_linux()
        import fcntl

        # Validate even direct callers; the middleware decodes before attaching.
        descriptor = decode_descriptor(json.dumps(descriptor, separators=(',', ':')).encode())
        self = cls()
        self.capacity, self.client = descriptor['capacity'], descriptor['client']
        try:
            self.directory = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            for kind in ('input', 'output'):
                expected = descriptor['slots'][kind]
                fd = os.open(expected['name'], (os.O_RDONLY if kind == 'input' else os.O_RDWR)
                             | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=self.directory)
                self.fds[kind] = fd
                _check_stat(os.fstat(fd), expected['bytes'], expected=expected)
                if kind == 'input':
                    # Kernel lock also excludes concurrent duplicate requests
                    # handled by a different API worker in the same UID.
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.identities[kind] = dict(expected)
                self.maps[kind] = mmap.mmap(fd, expected['bytes'], access=mmap.ACCESS_READ if kind == 'input' else mmap.ACCESS_WRITE)
            return self
        except BaseException:
            self.close()
            raise

    def _array(self, kind):
        if self.closed:
            raise RuntimeError('Shared-memory slots are closed')
        shape = INPUT_SHAPE if kind == 'input' else OUTPUT_SHAPE
        return np.ndarray((self.capacity, *shape), dtype='<f4', buffer=self.maps[kind])

    def write(self, kind, values):
        view = self._array(kind)
        try:
            np.copyto(view[:len(values)], values, casting='no')
        finally:
            del view

    def snapshot(self, kind, count):
        view = self._array(kind)
        try:
            return view[:count].copy(order='C')
        finally:
            del view

    def descriptor(self, model, count, request):
        value = {'schema': SCHEMA, 'client': self.client, 'request': request, 'model': model,
                 'count': count, 'capacity': self.capacity, 'dtype': '<f4',
                 'slots': {kind: dict(item) for kind, item in self.identities.items()}}
        return decode_descriptor(json.dumps(value, separators=(',', ':')).encode())

    def close(self, *, unlink=False):
        if self.closed:
            return
        for mapping in self.maps.values():
            mapping.close()
        self.maps.clear()
        if unlink and self.directory is not None:
            for kind in self.created:
                item = self.identities.get(kind)
                # Identity from an open descriptor also handles a failed
                # allocation before the normal metadata assignment.
                if item is None and kind in self.fds:
                    info = os.fstat(self.fds[kind])
                    item = {'name': slot_name(self.client, kind), 'device': info.st_dev, 'inode': info.st_ino}
                if item is not None:
                    try:
                        info = os.stat(item['name'], dir_fd=self.directory, follow_symlinks=False)
                        if info.st_dev == item['device'] and info.st_ino == item['inode'] and info.st_uid == os.getuid():
                            os.unlink(item['name'], dir_fd=self.directory)
                    except FileNotFoundError:
                        pass
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
        if self.directory is not None:
            os.close(self.directory)
            self.directory = None
        self.closed = True
