"""Bounded operator aggregates from durable service state, never model telemetry."""
from __future__ import annotations

from typing import TYPE_CHECKING
import json
import sqlite3

from .contracts import ServiceMetrics

if TYPE_CHECKING:
    from .control import Control


def _counts(db: sqlite3.Connection, table: str, states: tuple[str, ...]) -> dict[str, int]:
    # Identifiers come only from the fixed calls below; never from HTTP input.
    assert table in {'jobs', 'inputs', 'attempts', 'service_gc', 'service_scratch'}
    result = {state: 0 for state in states}
    for row in db.execute(f'SELECT state, COUNT(*) AS n FROM {table} GROUP BY state'):
        key = row['state'] if row['state'] in result else 'unknown'
        result[key] = result.get(key, 0) + int(row['n'])
    return result


def snapshot(control: Control) -> ServiceMetrics:
    """Read scalar counts in one transaction without claiming or repairing jobs.

    No per-tenant/job labels, credentials, URLs, local paths or device identifiers
    are returned. Counts describe retained records, not process-lifetime counters.
    Ages use the durable wall clock and are not inference benchmark measurements.
    """
    now = control.clock()
    with control.tx() as db:
        jobs = _counts(db, 'jobs', ('queued', 'running', 'completed', 'failed', 'cancelled', 'expired'))
        inputs = _counts(db, 'inputs', ('uploading', 'verifying', 'validating', 'ready', 'rejected', 'expired'))
        attempts = _counts(db, 'attempts', ('running', 'completed', 'failed', 'lease_lost'))
        expired_leases = int(db.execute('SELECT COUNT(*) FROM jobs WHERE state="running" AND lease<=?', (now,)).fetchone()[0])
        retry_jobs = int(db.execute('SELECT COUNT(*) FROM jobs WHERE attempts>1').fetchone()[0])
        oldest = db.execute('SELECT MIN(created) FROM jobs WHERE state="queued"').fetchone()[0]
        future_jobs = int(db.execute('SELECT COUNT(*) FROM jobs WHERE created>?', (now,)).fetchone()[0])
        reservations = int(db.execute('SELECT COALESCE(SUM(bytes),0) FROM inputs WHERE state!="expired"').fetchone()[0])
        expired_inputs = int(db.execute('SELECT COUNT(*) FROM inputs WHERE state!="expired" AND expires<=?', (now,)).fetchone()[0])
        expired_jobs = int(db.execute('SELECT COUNT(*) FROM jobs WHERE state!="expired" AND expires IS NOT NULL AND expires<=?', (now,)).fetchone()[0])
        usage = {state: {'jobs': 0, 'units': 0} for state in ('reserved', 'settled', 'released')}
        for row in db.execute('SELECT state,COUNT(*) AS n,COALESCE(SUM(units),0) AS units FROM service_usage GROUP BY state'):
            state = row['state'] if row['state'] in usage else 'unknown'
            usage[state] = {'jobs': int(row['n']), 'units': int(row['units'])}
        actual_usage_rows = int(db.execute('SELECT COUNT(*) FROM usage').fetchone()[0])
        meta = {r['key']: json.loads(r['value']) for r in db.execute('SELECT key,value FROM service_meta WHERE key IN ("worker","drain")')}
        is_replica = db.execute('SELECT 1 FROM sqlite_master WHERE type="table" AND name="replica_workers"').fetchone() is not None
        if is_replica:
            worker_counts = db.execute('''SELECT COUNT(*),
              COALESCE(SUM(CASE WHEN state IN ("available","running") AND heartbeat+20>? THEN 1 ELSE 0 END),0),
              COALESCE(SUM(CASE WHEN state IN ("available","running") AND heartbeat+20<=? THEN 1 ELSE 0 END),0)
              FROM replica_workers''', (now, now)).fetchone()
            workers = {'retained_generations': int(worker_counts[0]), 'live': int(worker_counts[1]), 'stale_active': int(worker_counts[2])}
        else:
            worker = meta.get('worker', {})
            active = worker.get('state') in ('available', 'running')
            live = active and worker.get('heartbeat', 0) + 20 > now
            workers = {'retained_generations': int(bool(worker)), 'live': int(live), 'stale_active': int(active and not live)}
        gc = _counts(db, 'service_gc', ('pending', 'removed'))
        scratch = _counts(db, 'service_scratch', ('active', 'closed', 'failed', 'foreign_restore', 'retired'))
    return {
        'schema': 'aureon.service_metrics.v1', 'scope': 'authenticated_operator_aggregates',
        'observed_at_epoch': now, 'execution': control.execution,
        'model_id': control.selected['model_id'], 'draining': bool(meta.get('drain', False)),
        'jobs': jobs, 'inputs': inputs, 'attempts': attempts, 'workers': workers,
        'queue': {'active': jobs['queued'] + jobs['running'], 'max_active': control.contract['max_active'],
                  'max_running': control.contract['max_running'], 'expired_running_leases': expired_leases,
                  'jobs_retried': retry_jobs, 'future_created_records': future_jobs},
        'oldest_queued_age_seconds': None if oldest is None else max(0.0, now - float(oldest)),
        'input_reserved_bytes': reservations,
        'input_reservation_limit_bytes': control.contract['max_global_storage_reservation_bytes'],
        'record_capacity': {'jobs': sum(jobs.values()), 'inputs': sum(inputs.values()),
                            'max_rows_per_table': control.contract['max_rows']},
        'retention': {'expired_inputs_awaiting_gc': expired_inputs, 'expired_jobs_awaiting_gc': expired_jobs,
                      'gc_pending': gc['pending'], 'scratch_retained_records': sum(scratch.values()) - scratch['retired']},
        'usage': usage, 'committed_usage_rows': actual_usage_rows, 'billable': False,
        'semantics': 'Retained durable-state gauges; no per-tenant labels, GPU probes, cleanup or recovery. Ages use wall clock, not benchmark latency.',
    }
