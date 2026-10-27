"""One resident private Linux slot pair, with an exclusive lock per lease.

Only controlled same-UID clients honoring the slot protocol are supported.
No files are created, resized or unlinked here. Inactive mappings remain open
until eviction/close, including after a client unlinks its names. Reuse always
revalidates both named files and open descriptors before exposing any views.
"""
import json
import os
import threading
import weakref

from .shm_files import MODEL_IDS, SharedSlots, _check_stat, decode_descriptor, require_linux


class ResidentSlotLease:
    """Borrowed arrays must be released before closing this exclusive lease."""

    def __init__(self, cache):
        self._cache = cache
        self._views = []
        self.closed = False

    def _array(self, kind):
        if kind not in ('input', 'output'):
            raise ValueError('Exactly input or output is required')
        with self._cache._lock:
            if self.closed or self._cache._active is not self:
                raise RuntimeError('Resident slot lease is inactive')
            self._views = [item for item in self._views if item() is not None]
            if len(self._views) >= 16:
                raise BufferError('Release exported slot views before requesting more')
            value = self._cache._entry._array(kind)
            self._views.append(weakref.ref(value))
            return value

    def close(self):
        self._cache._release(self)

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Resident slot lease is closed')
        return self

    def __exit__(self, *unused):
        self.close()


class ResidentSlotCache:
    """A single mapped pair; a lease owns the flock, never the cache lifetime."""

    def __init__(self, model_id=None):
        require_linux()
        if model_id is not None and model_id not in MODEL_IDS:
            raise ValueError('Expected a KAO-TerraMind model identifier')
        self.model_id = model_id
        self._lock = threading.Lock()
        self._entry = self._descriptor = self._active = None
        self._invalid = False
        self.attachments = self.reuses = self.evictions = self.failures = 0
        self.closed = False

    @staticmethod
    def _identity(descriptor):
        return {key: descriptor[key] for key in ('model', 'client', 'capacity', 'dtype', 'slots')}

    def _discard(self):
        if self._entry is not None:
            # A buffer-export error deliberately retains ownership for close()
            # retry. Never lose an fd or reuse a partly closed mapping pair.
            self._invalid = True
            import fcntl
            # Failed validation may happen after acquiring this request's
            # lock. Release it even if a foreign buffer export prevents unmap.
            if 'input' in self._entry.fds:
                fcntl.flock(self._entry.fds['input'], fcntl.LOCK_UN)
            self._entry.close()
            self._entry = self._descriptor = None
        self._invalid = False

    def _validate_entry(self, descriptor):
        entry = self._entry
        if (entry.closed or entry.client != descriptor['client']
                or entry.capacity != descriptor['capacity']
                or entry.identities != descriptor['slots']):
            raise ValueError('Resident slot descriptor identity changed')
        for kind in ('input', 'output'):
            expected = descriptor['slots'][kind]
            _check_stat(os.fstat(entry.fds[kind]), expected['bytes'], expected=expected)
            # No symlink traversal, including a replacement created after the
            # original attach. _check_stat also checks UID, 0600, size and nlink.
            named = os.stat(expected['name'], dir_fd=entry.directory, follow_symlinks=False)
            _check_stat(named, expected['bytes'], expected=expected)

    def acquire(self, descriptor):
        require_linux()
        import fcntl

        with self._lock:
            if self.closed:
                raise RuntimeError('Resident slot cache is closed')
            if self._active is not None:
                # A rejected duplicate must not unlock the active owner's fd.
                raise RuntimeError('Resident slot cache already has an active lease')
            try:
                if self._invalid:
                    raise RuntimeError('Invalid resident entry requires close before reuse')
                spec = decode_descriptor(json.dumps(descriptor, separators=(',', ':'), allow_nan=False).encode())
                if self.model_id is None:
                    self.model_id = spec['model']
                if spec['model'] != self.model_id:
                    raise ValueError('Resident cache is bound to another served model')
                reuse = self._entry is not None and self._identity(spec) == self._identity(self._descriptor)
                if self._entry is not None and not reuse:
                    if spec['client'] == self._descriptor['client']:
                        raise ValueError('An existing client changed its slot identity or capacity')
                    self._discard()
                    self.evictions += 1
                if reuse:
                    # Each use takes a fresh kernel lease even though fd/maps
                    # remain resident. Closing an inactive cache holds no lock.
                    fcntl.flock(self._entry.fds['input'], fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    self._entry = SharedSlots.attach(spec)
                    self._descriptor = spec
                    self.attachments += 1
                self._validate_entry(spec)
                lease = ResidentSlotLease(self)
                self._active = lease
                if reuse:
                    self.reuses += 1
                return lease
            except Exception:
                self.failures += 1
                self._discard()
                raise

    def _release(self, lease):
        import fcntl

        with self._lock:
            if lease.closed:
                return
            if self._active is not lease:
                raise RuntimeError('Lease does not own this resident entry')
            if any(reference() is not None for reference in lease._views):
                raise BufferError('Release exported NumPy views before closing the resident lease')
            # Keep maps/fds, release only this request's exclusion lock.
            fcntl.flock(self._entry.fds['input'], fcntl.LOCK_UN)
            lease._views.clear()
            lease.closed = True
            self._active = None

    def metadata(self):
        with self._lock:
            return {'schema': 'aureon.resident_slot_cache.v1', 'model_id': self.model_id,
                    'maximum_entries': 1, 'resident_entries': int(self._entry is not None),
                    'active': self._active is not None, 'attachments': self.attachments,
                    'reuses': self.reuses, 'evictions': self.evictions,
                    'failures': self.failures, 'invalid': self._invalid, 'closed': self.closed,
                    'scope': 'Same Linux UID; private mappings persist until eviction/close, locks only during leases',
                    'creates_files': False, 'unlinks_files': False}

    def close(self):
        with self._lock:
            if self.closed:
                return
            if self._active is not None:
                raise RuntimeError('Close the active lease before closing the resident cache')
            self._discard()
            self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Resident slot cache is closed')
        return self

    def __exit__(self, *unused):
        self.close()
