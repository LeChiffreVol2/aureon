"""Single-host crash recovery and bounded retention; no model math changes.

SQLite is authoritative, using one serialized connection, DELETE journaling and
synchronous=EXTRA. A process lock fences a second model owner. Payloads are
fsynced before publication. Interrupted computation is at-least-once; logical
admission/publication is idempotent within the declared key retention window.
"""
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import threading
import time
import uuid

from .api import (APIError, LocalJobService, ARTIFACT_NAMES, _artifact_bytes,
                  _atomic_json, _json_bytes, _snapshot_json, _utc, _JOB_ID)


def _plain_path(path):
    """Reject symlinks/junctions in any existing ancestor, including on Windows."""
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Managed state paths must not contain links/reparse points')
    return path


def _sync_dir(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _sync_tree(folder):
    """All evidence, not only public COGs, precedes the completed DB commit."""
    _plain_path(folder)
    for root, dirs, files in os.walk(folder, followlinks=False):
        for name in dirs + files:
            _plain_path(Path(root) / name)
        for name in files:
            with (Path(root) / name).open('r+b') as stream:
                os.fsync(stream.fileno())
    for root, _, _ in os.walk(folder, topdown=False):
        _sync_dir(root)
    for parent in (folder.parent, folder.parent.parent, folder.parent.parent.parent):
        _sync_dir(parent)


class _FileLock:
    def __init__(self, path):
        self.stream = _plain_path(path).open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                if self.stream.tell() == 0:
                    self.stream.write(b'0'); self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            raise ValueError('Durable state is already owned by another process') from None

    def close(self):
        if not self.stream.closed:
            if os.name == 'nt':
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()


class _Rows(Mapping):
    """Read through SQL; completed jobs are never accumulated in Python RAM."""
    def __init__(self, db):
        self.db = db

    def __getitem__(self, identifier):
        row = self.db.execute('SELECT body FROM jobs WHERE id=?', (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return json.loads(row[0])

    def __len__(self):
        return self.db.execute('SELECT count(*) FROM jobs').fetchone()[0]

    def __iter__(self):
        return (row[0] for row in self.db.execute('SELECT id FROM jobs'))


class DurableJobService(LocalJobService):
    def __init__(self, output_dir, worker_factory, scene_ids, *, state_dir, contract,
                 queue_capacity=2, max_jobs=100000, max_attempts=3,
                 payload_ttl_seconds=3600, key_ttl_seconds=86400,
                 min_free_bytes=2 * 2**30):
        scenes = tuple(scene_ids) if not isinstance(scene_ids, str) else ()
        if (not callable(worker_factory) or not scenes or len(scenes) > 64
                or any(not isinstance(s, str) or not 1 <= len(s) <= 256 for s in scenes)
                or len(set(scenes)) != len(scenes)):
            raise ValueError('Register a callable worker and 1..64 unique scene IDs')
        if (type(queue_capacity) is not int or not 1 <= queue_capacity <= 32
                or type(max_jobs) is not int or not queue_capacity + 1 <= max_jobs <= 1000000
                or type(max_attempts) is not int or not 1 <= max_attempts <= 10
                or type(min_free_bytes) is not int or min_free_bytes < 0):
            raise ValueError('Invalid bounded queue, registry, attempt or free-space policy')
        if (not all(type(v) in (int, float) and math.isfinite(v)
                    for v in (payload_ttl_seconds, key_ttl_seconds))
                or not 1 <= payload_ttl_seconds < key_ttl_seconds <= 365 * 86400):
            raise ValueError('Require 1 <= payload TTL < key TTL <= 365 days')
        if not isinstance(contract, dict) or not contract:
            raise ValueError('A pinned execution contract is required')
        self.output_dir = _plain_path(output_dir)
        self.state_dir = _plain_path(state_dir)
        if (self.output_dir == self.state_dir or self.output_dir.is_relative_to(self.state_dir)
                or self.state_dir.is_relative_to(self.output_dir)):
            raise ValueError('Fresh lifecycle evidence and managed state must be separate')
        if self.output_dir.exists() or not self.output_dir.parent.is_dir() or not self.state_dir.parent.is_dir():
            raise ValueError('New lifecycle output and existing parents required')
        self.policy = dict(payload_ttl_seconds=payload_ttl_seconds, key_ttl_seconds=key_ttl_seconds,
                           max_attempts=max_attempts, min_free_bytes=min_free_bytes)
        identity = _snapshot_json({'schema': 'aureon.durable_store.v1', 'contract': contract,
            'scenes': list(scenes), 'queue_capacity': queue_capacity, 'max_jobs': max_jobs,
            'policy': self.policy})
        self.state_dir.mkdir(mode=0o700, exist_ok=True)
        _sync_dir(self.state_dir.parent)
        self._file_lock = _FileLock(self.state_dir / 'owner.lock')
        self._db = None
        try:
            marker = self.state_dir / 'store.json'
            if marker.exists():
                if json.loads(_plain_path(marker).read_text()) != identity:
                    raise ValueError('Durable store execution contract or retention policy mismatch')
                if not (self.state_dir / 'jobs.sqlite3').is_file():
                    raise ValueError('Existing store is missing its database; refusing to forget jobs')
            else:
                if set(p.name for p in self.state_dir.iterdir()) != {'owner.lock'}:
                    raise ValueError('Refuse to adopt an unowned/nonempty state directory')
                _atomic_json(marker, identity)
                _sync_dir(self.state_dir)
            _plain_path(self.state_dir / 'jobs.sqlite3')
            _plain_path(self.state_dir / 'jobs.sqlite3-journal')
            self._db = sqlite3.connect(self.state_dir / 'jobs.sqlite3', timeout=5, check_same_thread=False)
            if self._db.execute('PRAGMA journal_mode=DELETE').fetchone()[0] != 'delete':
                raise ValueError('Required SQLite journal mode unavailable')
            self._db.execute('PRAGMA synchronous=EXTRA')
            self._db.execute('PRAGMA cache_size=-2048')
            self._db.execute('PRAGMA journal_size_limit=1048576')
            self._db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, key_hash TEXT NOT NULL UNIQUE, digest TEXT NOT NULL,
                    state TEXT NOT NULL, created REAL NOT NULL, finished REAL,
                    payload_deleted INTEGER NOT NULL DEFAULT 0, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS jobs_retention ON jobs(state, finished);
            ''')
            with self._db:
                self._db.execute('INSERT OR IGNORE INTO metadata VALUES(1,?)', (_json_bytes(identity).decode(),))
            if json.loads(self._db.execute('SELECT identity FROM metadata WHERE id=1').fetchone()[0]) != identity:
                raise ValueError('Database identity mismatch')
            if self._db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('SQLite integrity check failed')
            _plain_path(self.state_dir / 'jobs').mkdir(exist_ok=True)
            _sync_dir(self.state_dir)
            self.output_dir.mkdir()
            self.scene_ids, self.queue_capacity, self.max_jobs = scenes, queue_capacity, max_jobs
            self._factory, self._worker_metadata, self._worker_shutdown = worker_factory, {}, None
            self._lock = threading.RLock()
            self._condition = threading.Condition(self._lock)
            self._jobs, self._queue = _Rows(self._db), deque()
            self._stop = False
            self._state, self._error, self._running_id = 'initializing', None, None
            self._init_count, self._next_retention = 0, 0.0
            self._retired, self._forgotten, self._recovered = 0, 0, []
            pending = self._db.execute("SELECT id FROM jobs WHERE state IN ('queued','running') ORDER BY created,id").fetchall()
            if len(pending) > queue_capacity + 1:
                raise ValueError('Stored pending queue exceeds its declared bound')
            for (identifier,) in pending:
                job = self._jobs[identifier]
                if job['status'] == 'running':
                    if job['attempts'] >= max_attempts:
                        self._set_status(job, 'failed', error={'type': 'AttemptLimit', 'message': 'Crash retry budget exhausted'})
                        continue
                    self._set_status(job, 'queued', recovered_after_crash=True)
                self._queue.append(identifier)
                self._recovered.append(identifier)
            self._maintenance()
            self._persist_service()
            self._thread = threading.Thread(target=self._owner, name='aureon-scene-owner', daemon=False)
            self._thread.start()
        except BaseException:
            if self._db is not None:
                self._db.close()
            self._file_lock.close()
            raise

    def _service_snapshot(self):
        result = super()._service_snapshot()
        result.update(schema='aureon.durable_scene_service.v1', durability={
            'journal_mode': 'delete', 'synchronous': 'extra', 'sqlite_version': sqlite3.sqlite_version,
            'policy': self.policy, 'recovered_jobs': list(self._recovered),
            'expired_this_lifecycle': self._retired, 'forgotten_this_lifecycle': self._forgotten,
            'restart': 'Explicit restart with same state directory and pinned contract',
            'execution': 'At-least-once after crash; at most one published attempt per logical job',
            'graceful_stop': 'Preserve queued work; finish running work',
            'idempotency_scope': 'Until key TTL after terminal completion; expired payload returns HTTP 410'})
        return result

    def _persist_job(self, job):
        finished = datetime.fromisoformat(job['finished_utc']).timestamp() if 'finished_utc' in job else None
        if finished is not None:
            job['payload_expires_utc'] = datetime.fromtimestamp(finished + self.policy['payload_ttl_seconds'], timezone.utc).isoformat()
            job['key_expires_utc'] = datetime.fromtimestamp(finished + self.policy['key_ttl_seconds'], timezone.utc).isoformat()
        with self._db:
            self._db.execute('''INSERT INTO jobs(id,key_hash,digest,state,created,finished,body) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET state=excluded.state,finished=excluded.finished,body=excluded.body''',
                (job['job_id'], job['idempotency_key_sha256'], job['request_digest'], job['status'],
                 datetime.fromisoformat(job['created_utc']).timestamp(), finished, _json_bytes(job).decode()))

    def _set_status(self, job, state, **extra):
        if state == 'running':
            job['attempts'] += 1
        super()._set_status(job, state, **extra)

    def _cancel_waiting(self, reason):
        # A graceful stop or an unavailable owner preserves admitted durable work.
        # Explicit DELETE /jobs/:id still cancels through the inherited cancel().
        return None

    def _job_folder(self, identifier):
        if not _JOB_ID.fullmatch(identifier):
            raise ValueError('Invalid managed job identity')
        with self._lock:
            attempts = self._jobs[identifier]['attempts']
        return _plain_path(self.state_dir / 'jobs' / identifier / f'attempt-{attempts:02d}' / identifier)

    def _run_worker_job(self, worker, job, identifier):
        folder = self._job_folder(identifier)
        folder.parent.mkdir(parents=True, exist_ok=False)
        return worker.run_job(job['scene_id'], identifier, folder)

    def _completed_result(self, result, identifier):
        result = super()._completed_result(result, identifier)
        _sync_tree(self._job_folder(identifier))
        return result

    def submit(self, request, idempotency_key):
        if (not isinstance(request, dict) or set(request) != {'scene_id'}
                or not isinstance(request['scene_id'], str) or not 1 <= len(request['scene_id']) <= 256):
            raise APIError(400, 'invalid_request', 'Supply exactly one registered scene_id string')
        if not isinstance(idempotency_key, str) or not re.fullmatch(r'[\x21-\x7e]{1,128}', idempotency_key):
            raise APIError(400, 'invalid_idempotency_key', 'Use 1..128 visible ASCII characters')
        digest = hashlib.sha256(_json_bytes(request)).hexdigest()
        key_hash = hashlib.sha256(idempotency_key.encode('ascii')).hexdigest()
        with self._condition:
            row = self._db.execute('SELECT id,digest FROM jobs WHERE key_hash=?', (key_hash,)).fetchone()
            if row:
                if row[1] != digest:
                    raise APIError(409, 'idempotency_conflict', 'This key is bound to a different request')
                job = self._jobs[row[0]]
                if job['status'] == 'expired':
                    raise APIError(410, 'job_expired', 'Payload expired; the retained key prevents duplicate execution')
                return {'job': self._public_job(job), 'reused': True}
            if request['scene_id'] not in self.scene_ids:
                raise APIError(400, 'unknown_scene', 'Use a registered scene ID from /v1/info')
            if self._state != 'ready' or self._stop:
                raise APIError(503, 'unavailable', 'The model worker is not accepting jobs', retry_after=1)
            if len(self._queue) >= self.queue_capacity:
                raise APIError(429, 'queue_full', 'The bounded waiting queue is full', retry_after=1)
            if len(self._jobs) >= self.max_jobs:
                raise APIError(429, 'job_limit_reached', 'Registry is full until retained keys expire', retry_after=1)
            if shutil.disk_usage(self.state_dir).free < self.policy['min_free_bytes']:
                raise APIError(507, 'storage_pressure', 'Managed state free space is below the configured floor', retry_after=1)
            identifier, now = uuid.uuid4().hex, _utc()
            job = {'schema': 'aureon.durable_scene_job_status.v1', 'job_id': identifier,
                   'scene_id': request['scene_id'], 'status': 'queued', 'created_utc': now,
                   'updated_utc': now, 'request_digest': digest, 'idempotency_key_sha256': key_hash,
                   'attempts': 0}
            # Commit the key and job atomically BEFORE acknowledging or enqueueing.
            try:
                self._persist_job(job)
            except (OSError, sqlite3.Error) as error:
                raise APIError(503, 'durable_admission_failed', 'Admission commit failed; retry with the same key') from error
            self._queue.append(identifier)
            self._condition.notify_all()
            return {'job': self._public_job(job), 'reused': False}

    def artifact(self, identifier, name):
        # Keep retention excluded until the bounded, hash-checked byte snapshot is
        # in memory. HTTP transmission no longer depends on the file afterwards.
        with self._lock:
            if identifier not in self._jobs or name not in ARTIFACT_NAMES:
                raise APIError(404, 'artifact_not_found', 'Unknown job or artifact')
            job = self._jobs[identifier]
            if job['status'] == 'expired':
                raise APIError(410, 'job_expired', 'Managed payload retention expired')
            if job['status'] != 'completed':
                raise APIError(409, 'job_not_completed', 'Job is not completed')
            declaration = deepcopy(job['download_artifacts'][name])
            try:
                return _artifact_bytes(self._job_folder(identifier), declaration), declaration['sha256']
            except (OSError, ValueError) as error:
                raise APIError(409, 'artifact_identity_failed', 'Artifact identity validation failed') from error

    def cancel(self, identifier):
        with self._condition:
            if identifier not in self._jobs:
                raise APIError(404, 'job_not_found', 'Unknown job ID')
            job = self._jobs[identifier]
            if job['status'] == 'cancelled':
                return self._public_job(job)
            if job['status'] != 'queued':
                raise APIError(409, 'not_cancellable', 'Only queued jobs can be cancelled')
            # Persist before changing the in-memory dispatch queue.
            self._set_status(job, 'cancelled', reason='client_requested')
            self._queue.remove(identifier)
            self._persist_service()
            return self._public_job(job)

    def _delete_payload(self, identifier):
        if not _JOB_ID.fullmatch(identifier):
            raise ValueError('Invalid retention target')
        root = _plain_path(self.state_dir / 'jobs')
        target = _plain_path(root / identifier)
        if target.parent != root or not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('Retention target escapes its managed root')
        if target.exists():
            for current, dirs, files in os.walk(target, followlinks=False):
                for name in dirs + files:
                    _plain_path(Path(current) / name)
            shutil.rmtree(target)
        # A prior owner may have removed the payload and crashed before syncing
        # its parent or recording payload_deleted. Finish that durable boundary.
        _sync_dir(root)

    def _maintenance(self):
        if time.monotonic() < self._next_retention:
            return
        self._next_retention = time.monotonic() + 1
        now = time.time()
        rows = self._db.execute("SELECT id FROM jobs WHERE state IN ('completed','failed','cancelled') AND finished<=? LIMIT 16",
                                (now - self.policy['payload_ttl_seconds'],)).fetchall()
        for (identifier,) in rows:
            job = self._jobs[identifier]
            job = {key: value for key, value in job.items() if key not in ('download_artifacts', 'summary', 'error')}
            job.update(previous_status=job['status'], status='expired', updated_utc=_utc())
            self._persist_job(job)  # Tombstone first; crash during delete is resumable.
            self._retired += 1
        for (identifier,) in self._db.execute("SELECT id FROM jobs WHERE state='expired' AND payload_deleted=0 LIMIT 16").fetchall():
            self._delete_payload(identifier)
            with self._db:
                self._db.execute('UPDATE jobs SET payload_deleted=1 WHERE id=?', (identifier,))
        with self._db:
            count = self._db.execute("DELETE FROM jobs WHERE id IN (SELECT id FROM jobs WHERE state='expired' AND payload_deleted=1 AND finished<=? LIMIT 128)",
                                     (now - self.policy['key_ttl_seconds'],)).rowcount
        self._forgotten += count

    def close_storage(self):
        """Call after HTTP shutdown, worker join and final lifecycle receipt."""
        if self._thread.is_alive():
            raise ValueError('Cannot release storage ownership while worker is alive')
        with self._lock:
            self._db.close()
            self._file_lock.close()
