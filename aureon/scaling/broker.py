"""Durable replica work queue with fenced leases and one committed result per job.

Execution is at least once after worker loss. Commit is idempotent; model callbacks
must not perform billing or arbitrary external effects. SQLite belongs to one
coordinator on a local durable disk, never to multiple hosts over a shared mount.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from typing import cast


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Expected a bounded identifier")
    return value


def canonical(value: object, limit: int = 65536) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > limit:
        raise ValueError("Metadata exceeds budget; transfer data through artifacts")
    return encoded


def object_json(text: str) -> dict[str, object]:
    value: object = json.loads(text)
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError("Expected JSON object")
    canonical(value)
    return cast(dict[str, object], value)


@dataclass(frozen=True)
class Job:
    job_id: str
    tenant: str
    route: str
    payload_json: str
    seed: int
    required_bytes: int
    max_attempts: int = 3

    def __post_init__(self) -> None:
        for name in (self.job_id, self.tenant, self.route):
            identifier(name)
        object_json(self.payload_json)
        for name, low, high in (("seed", 0, 2**63 - 1), ("required_bytes", 1, 2**63 - 1),
                                ("max_attempts", 1, 100)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid " + name)

    @classmethod
    def create(cls, job_id: str, *, tenant: str, route: str, payload: Mapping[str, object],
               seed: int, required_bytes: int, max_attempts: int = 3) -> Job:
        return cls(job_id, tenant, route, canonical(dict(payload)), seed, required_bytes, max_attempts)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    routes: tuple[str, ...]
    capacity_bytes: int
    slots: int = 1

    def __post_init__(self) -> None:
        identifier(self.worker_id)
        if type(self.routes) is not tuple or not self.routes or len(set(self.routes)) != len(self.routes):
            raise ValueError("Declare distinct supported routes")
        for route in self.routes:
            identifier(route)
        if type(self.capacity_bytes) is not int or not 1 <= self.capacity_bytes < 2**63:
            raise ValueError("Declare bounded worker capacity")
        if type(self.slots) is not int or not 1 <= self.slots <= 1024:
            raise ValueError("Declare 1..1024 independent execution slots")


@dataclass(frozen=True)
class Lease:
    job: Job
    worker_id: str
    token: str
    attempt: int
    expires_at: float


class Broker:
    def __init__(self, path: str | Path, *, max_jobs: int = 10000,
                 lease_seconds: float = 60., clock: Callable[[], float] = time.time) -> None:
        if type(max_jobs) is not int or max_jobs < 1:
            raise ValueError("Positive retained-job bound required")
        if not math.isfinite(lease_seconds) or not 1 <= lease_seconds <= 86400:
            raise ValueError("Lease duration must be 1..86400 seconds")
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._max_jobs, self._ttl, self._clock = max_jobs, lease_seconds, clock
        self._closed = False
        self._db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS jobs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                tenant TEXT NOT NULL, route TEXT NOT NULL, spec TEXT NOT NULL,
                digest TEXT NOT NULL, bytes INTEGER NOT NULL, max_attempts INTEGER NOT NULL,
                state TEXT NOT NULL, worker TEXT, token TEXT, attempt INTEGER NOT NULL DEFAULT 0,
                expires REAL, result TEXT, error TEXT);
            CREATE INDEX IF NOT EXISTS pending ON jobs(state, seq);
            CREATE TABLE IF NOT EXISTS workers (
                id TEXT PRIMARY KEY, spec TEXT NOT NULL, draining INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS tenants (id TEXT PRIMARY KEY, served INTEGER NOT NULL DEFAULT 0);
        """)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Broker closed")
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.rollback()
                raise
            else:
                self._db.commit()

    def register(self, spec: WorkerSpec) -> None:
        """Operator-only capability declaration, not a remote attestation."""
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM jobs WHERE state='running' AND worker=?", (spec.worker_id,)).fetchone():
                raise RuntimeError("Drain active worker before changing registration")
            db.execute("INSERT INTO workers VALUES (?,?,0) ON CONFLICT(id) DO UPDATE SET spec=excluded.spec",
                       (spec.worker_id, canonical(asdict(spec))))

    def drain(self, worker_id: str, draining: bool = True) -> None:
        with self._transaction() as db:
            if db.execute("UPDATE workers SET draining=? WHERE id=?", (int(draining), worker_id)).rowcount != 1:
                raise KeyError("Unknown worker")

    def submit(self, job: Job) -> dict[str, object]:
        spec = canonical(asdict(job))
        digest = hashlib.sha256(spec.encode()).hexdigest()
        with self._transaction() as db:
            previous = db.execute("SELECT digest FROM jobs WHERE id=?", (job.job_id,)).fetchone()
            if previous:
                if previous["digest"] != digest:
                    raise ValueError("Idempotency conflict: job id already binds another request")
            else:
                if db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] >= self._max_jobs:
                    raise BufferError("Retained job limit reached; explicit terminal retention is required")
                db.execute("INSERT INTO jobs(id,tenant,route,spec,digest,bytes,max_attempts,state) VALUES (?,?,?,?,?,?,?,'queued')",
                           (job.job_id, job.tenant, job.route, spec, digest, job.required_bytes, job.max_attempts))
                db.execute("INSERT OR IGNORE INTO tenants VALUES (?,0)", (job.tenant,))
        return self.status(job.job_id)

    def _expire(self, db: sqlite3.Connection, now: float) -> None:
        db.execute("""UPDATE jobs SET state=CASE WHEN attempt>=max_attempts THEN 'failed' ELSE 'queued' END,
                   error='lease_expired', worker=NULL, token=NULL, expires=NULL
                   WHERE state='running' AND expires<=?""", (now,))

    def claim(self, worker_id: str) -> Lease | None:
        with self._transaction() as db:
            now = self._clock()
            self._expire(db, now)
            worker = db.execute("SELECT * FROM workers WHERE id=?", (worker_id,)).fetchone()
            if worker is None:
                raise KeyError("Unregistered worker")
            if worker["draining"]:
                return None
            data = object_json(worker["spec"])
            routes = cast(list[str], data["routes"])
            active = db.execute("SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM jobs WHERE state='running' AND worker=?",
                                (worker_id,)).fetchone()
            if active[0] >= cast(int, data["slots"]):
                return None
            capacity = cast(int, data["capacity_bytes"]) - active[1]
            marks = ",".join("?" for _ in routes)
            row = db.execute(f"""SELECT j.* FROM jobs j JOIN tenants t ON t.id=j.tenant
                WHERE j.state='queued' AND j.bytes<=? AND j.route IN ({marks})
                ORDER BY t.served,j.seq LIMIT 1""", (capacity, *routes)).fetchone()
            if row is None:
                return None
            token, expires, attempt = secrets.token_urlsafe(32), now + self._ttl, row["attempt"] + 1
            db.execute("UPDATE jobs SET state='running',worker=?,token=?,expires=?,attempt=? WHERE id=?",
                       (worker_id, token, expires, attempt, row["id"]))
            db.execute("UPDATE tenants SET served=(SELECT COALESCE(MAX(served),0)+1 FROM tenants) WHERE id=?", (row["tenant"],))
            spec = object_json(row["spec"])
            job = Job(cast(str, spec["job_id"]), cast(str, spec["tenant"]), cast(str, spec["route"]),
                      cast(str, spec["payload_json"]), cast(int, spec["seed"]), cast(int, spec["required_bytes"]),
                      cast(int, spec["max_attempts"]))
            return Lease(job, worker_id, token, attempt, expires)

    def _owned(self, db: sqlite3.Connection, lease: Lease) -> sqlite3.Row:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (lease.job.job_id,)).fetchone()
        if (row is None or row["state"] != "running" or row["worker"] != lease.worker_id
                or row["token"] != lease.token or row["attempt"] != lease.attempt
                or row["expires"] <= self._clock()):
            raise RuntimeError("Stale or cancelled lease; result cannot commit")
        return cast(sqlite3.Row, row)

    def heartbeat(self, lease: Lease) -> float:
        with self._transaction() as db:
            self._owned(db, lease)
            expires = self._clock() + self._ttl
            db.execute("UPDATE jobs SET expires=? WHERE id=?", (expires, lease.job.job_id))
            return expires

    def complete(self, lease: Lease, result: Mapping[str, object]) -> None:
        encoded = canonical(dict(result))
        with self._transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (lease.job.job_id,)).fetchone()
            if row and row["state"] == "succeeded" and row["token"] == lease.token and row["attempt"] == lease.attempt:
                if row["result"] != encoded:
                    raise ValueError("A committed result cannot change")
                return
            self._owned(db, lease)
            db.execute("UPDATE jobs SET state='succeeded',result=?,expires=NULL WHERE id=?", (encoded, lease.job.job_id))

    def fail(self, lease: Lease, error: str, *, retry: bool = False) -> None:
        if not isinstance(error, str) or len(error) > 2048:
            raise ValueError("Bounded error code/text required")
        with self._transaction() as db:
            row = self._owned(db, lease)
            state = "queued" if retry and row["attempt"] < row["max_attempts"] else "failed"
            db.execute("UPDATE jobs SET state=?,error=?,worker=NULL,token=NULL,expires=NULL WHERE id=?",
                       (state, error, lease.job.job_id))

    def cancel(self, job_id: str) -> None:
        with self._transaction() as db:
            db.execute("UPDATE jobs SET state='cancelled',token=NULL,expires=NULL WHERE id=? AND state IN ('queued','running')", (job_id,))

    def status(self, job_id: str) -> dict[str, object]:
        with self._transaction() as db:
            self._expire(db, self._clock())
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("Unknown job")
            return {"job_id": row["id"], "state": row["state"], "attempt": row["attempt"],
                    "result": object_json(row["result"]) if row["result"] else None,
                    "error": row["error"], "delivery": "at_least_once_execution_single_fenced_commit"}

    def forget(self, job_id: str) -> None:
        """Operator retention; removes idempotency history, never any artifacts."""
        with self._transaction() as db:
            row = db.execute("SELECT state,tenant FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("Unknown job")
            if row["state"] not in ("succeeded", "failed", "cancelled"):
                raise RuntimeError("Only terminal jobs may be forgotten")
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            db.execute("DELETE FROM tenants WHERE id=? AND NOT EXISTS (SELECT 1 FROM jobs WHERE tenant=?)", (row["tenant"], row["tenant"]))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True
