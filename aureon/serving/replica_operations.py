"""Explicit offline replica maintenance; never automatic deletion or model work.

Restore is a same-host handover, not a fork of an executable queue. The source
is durably retired before the new state becomes visible, so both cannot resume.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
from typing import cast
import uuid

from .contracts import FileIdentity, PathInput, ProcessIdentity
from .ownership import alive, exclusive, inventory, safe_child
from .operations import copy_file, verify_database_files
from .replica_control import ReplicaControl
from ..cloud.control import CONTRACT, digest, encoded, file_hash, require, write_file
from ..scene_api.durable import _plain_path, _sync_dir

SCHEMA = 'aureon.replica_backup.v1'
RETIRED = 'replica_restore_retired'
PENDING = 'replica_restore_pending'


def _object(value: object) -> dict[str, object]:
    require(isinstance(value, dict), 'invalid_maintenance_record', 409)
    return cast(dict[str, object], value)


def _load(path: Path) -> dict[str, object]:
    return _object(json.loads(_plain_path(path).read_bytes()))


def _signature(control: ReplicaControl) -> str:
    with control.lock:
        return digest('\n'.join(control.db.iterdump()).encode())


def _fence_copy(database: Path, marker: Mapping[str, object]) -> None:
    """Make a copied database non-executable before publishing its contract."""
    require(not (database.parent/'contract.json').exists(), 'copy_contract_published_before_fence', 409)
    db = sqlite3.connect(database)
    try:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('INSERT INTO service_meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                   (PENDING, encoded(marker).decode()))
        db.commit(); db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        db.close()
    with database.open('r+b') as stream: os.fsync(stream.fileno())


def _skip(control: ReplicaControl) -> set[str]:
    return {'control.sqlite', 'control.sqlite-wal', 'control.sqlite-shm',
            'service.lock', 'supervisor.lock', 'worker.lock', 'retention.lock',
            *('replica-'+w['id']+'.lock' for w in control.resources['workers'])}


@contextmanager
def _offline(control: ReplicaControl) -> Iterator[None]:
    """Hold all cooperative controller/worker/device owners; never steal locks."""
    with ExitStack() as stack:
        for name in ('service.lock', 'supervisor.lock', 'worker.lock', 'retention.lock'):
            stack.enter_context(exclusive(control.root, name))
        for worker in sorted(control.resources['workers'], key=lambda w: w['id']):
            stack.enter_context(exclusive(control.root, 'replica-'+worker['id']+'.lock'))
        if control.execution != 'fake':
            from .devices import device_lease
            for device in sorted(w['device'] for w in control.resources['workers']):
                stack.enter_context(device_lease(device))
        yield


def _dead_owners(control: ReplicaControl) -> None:
    """Called with every kernel owner held, before snapshotting a drained queue."""
    with control.tx() as db:
        rows = db.execute('SELECT * FROM replica_workers WHERE state IN ("starting","available","running","quarantined")').fetchall()
        for row in rows:
            owner = cast(ProcessIdentity, json.loads(row['process']))
            require(not alive(owner), 'old_worker_still_alive', 409)
        for row in rows:
            control._retire(db, row['worker'], row['generation'],
                            {'reason': 'attested_process_dead_and_kernel_leases_reacquired',
                             'process': json.loads(row['process'])})


def reclaim(root: PathInput, worker: str, generation: str, *, work_root: PathInput) -> dict[str, object]:
    """Retire one registered closed scratch directory under an explicit parent.

    Credits are released only after removal succeeds. A frozen inventory makes
    interrupted removal replayable; changed or replaced scratch fails closed.
    """
    require(isinstance(generation, str) and re.fullmatch('[0-9a-f]{32}', generation), 'invalid_worker_generation')
    control = ReplicaControl(root)
    try:
        control.spec(worker)
        parent = _plain_path(work_root).resolve(strict=True)
        require(parent.is_dir(), 'scratch_parent_required')
        with _offline(control):
            with control.tx() as db:
                row = control._worker(db, worker, generation)
                require(row['state'] == 'closed', 'scratch_worker_not_closed', 409)
                require(not alive(cast(ProcessIdentity, json.loads(row['process']))), 'scratch_owner_still_alive', 409)
                require(db.execute('SELECT COUNT(*) FROM replica_assignments WHERE worker=? AND generation=? AND (state IN ("active","quarantined") OR host_bytes>0 OR vram_bytes>0)',
                                   (worker, generation)).fetchone()[0] == 0, 'scratch_assignment_active', 409)
                folder = _plain_path(row['work']).resolve()
                require(folder.parent == parent and folder != control.root and not folder.is_relative_to(control.root)
                        and not control.root.is_relative_to(folder), 'scratch_parent_or_overlap', 409)
                closure = _object(json.loads(row['closure']))
                require(closure.get('reason') in ('verified_owned_close', 'attested_process_dead_and_kernel_leases_reacquired'),
                        'scratch_closure_not_owned', 409)
                instance = control.meta('instance_id')
                intent = control.root/'maintenance'/('replica-scratch-'+worker+'-'+generation+'.json')
                expected = {'worker': worker, 'generation': generation, 'state_instance': instance,
                            'work': str(folder), 'work_root': str(parent), 'process': json.loads(row['process'])}
            if intent.exists():
                record = _load(intent)
                require(all(record.get(k) == v for k, v in expected.items()), 'scratch_intent_owner', 409)
            else:
                require(folder.is_dir(), 'scratch_missing_without_intent', 409)
                ownership = _load(folder/'ownership.json')
                require(ownership == {'worker': worker, 'generation': generation, 'state': str(control.root),
                                      'pid': json.loads(row['process'])['pid'], 'device': row['device']},
                        'scratch_owner_mismatch', 409)
                info = folder.stat()
                files = inventory(folder, max_bytes=control.spec(worker)['scratch_bytes'])
                record = {**expected, 'schema': 'aureon.replica_scratch_retirement.v1', 'files': files,
                          'root_identity': [info.st_dev, info.st_ino], 'bytes': sum(f['bytes'] for f in files)}
                intent.parent.mkdir(exist_ok=True, mode=0o700)
                write_file(intent, encoded(record))
            if folder.exists():
                info = folder.stat()
                require([info.st_dev, info.st_ino] == record['root_identity'], 'scratch_root_replaced', 409)
                known = {f['path']: f for f in cast(list[FileIdentity], record['files'])}
                require(all(f == known.get(f['path']) for f in inventory(folder, max_bytes=control.spec(worker)['scratch_bytes'])),
                        'scratch_changed_after_intent', 409)
                require(shutil.rmtree.avoids_symlink_attacks, 'safe_recursive_removal_required', 409)
                shutil.rmtree(_plain_path(folder)); _sync_dir(parent)
            with control.tx() as db:
                current = control._worker(db, worker, generation)
                require(current['state'] == 'closed' and current['work'] == str(folder), 'scratch_owner_changed', 409)
                released = int(current['scratch_bytes'])
                closure['scratch_retired'] = {'intent': intent.relative_to(control.root).as_posix(),
                                             'sha256': file_hash(intent)}
                db.execute('UPDATE replica_workers SET scratch_bytes=0,closure=? WHERE worker=? AND generation=?',
                           (encoded(closure).decode(), worker, generation))
            return {'retired': True, 'reused': released == 0, 'released_reserved_bytes': released,
                    'logical_removed_bytes': record['bytes'], 'model_executed': False}
    finally:
        control.close()


def backup(root: PathInput, destination: PathInput, credentials: PathInput) -> dict[str, object]:
    """Snapshot a quiescent same-host replica authority; preserve its payloads."""
    control = ReplicaControl(root)
    try:
        target = _plain_path(destination).resolve()
        require(not target.exists() and target.parent.is_dir() and not target.is_relative_to(control.root)
                and not control.root.is_relative_to(target), 'fresh_external_backup_required')
        with _offline(control):
            control._require_active_authority()
            keys = _load(_plain_path(credentials))
            require('operator' in keys and all(isinstance(key, str) and control.authenticate(key) == tenant
                    for tenant, key in keys.items()), 'backup_credentials_invalid', 409)
            control.set_meta('drain', True); control.set_meta('shutdown_requested', True)
            _dead_owners(control)
            before = inventory(control.root, skip=_skip(control))
            signature = _signature(control)
            database_bytes = control.db.execute('PRAGMA page_count').fetchone()[0]*control.db.execute('PRAGMA page_size').fetchone()[0]
            needed = sum(f['bytes'] for f in before) + database_bytes + 2*1024**2
            require(shutil.disk_usage(target.parent).free >= CONTRACT['min_free_bytes']+needed, 'backup_disk_capacity', 507)
            target.mkdir(mode=0o700); state = target/'state'; state.mkdir(mode=0o700)
            for item in before:
                if item['path'] != 'contract.json':
                    copy_file(safe_child(control.root, item['path']), state/item['path'])
            db = sqlite3.connect(state/'control.sqlite')
            try:
                with control.lock:
                    control.db.backup(db)
                verify_database_files(state, db)
            finally:
                db.close()
            os.chmod(state/'control.sqlite', 0o600)
            _fence_copy(state/'control.sqlite', {'schema': 'aureon.replica_snapshot_fence.v1',
                        'source_instance': control.meta('instance_id'), 'restore_required': True})
            copy_file(control.root/'contract.json', state/'contract.json')
            copy_file(credentials, target/'credentials.json')
            require(before == inventory(control.root, skip=_skip(control)) and signature == _signature(control), 'backup_source_changed', 409)
            files = inventory(target)
            record = {'schema': SCHEMA, 'source_root': str(control.root), 'source_instance': control.meta('instance_id'),
                      'state_signature': signature, 'files': files, 'model_executed': False,
                      'contains_secrets': True, 'scope': 'offline same-host replica handover; not a queue clone'}
            write_file(target/'backup.json', encoded(record)); _sync_dir(target); _sync_dir(target.parent)
            return {'backup': str(target), 'manifest_sha256': file_hash(target/'backup.json'),
                    'files': len(files), 'contains_secrets': True, 'drained': True, 'model_executed': False}
    finally:
        control.close()


def _finish_restore(control: ReplicaControl, marker: Mapping[str, object]) -> dict[str, object]:
    require(control.meta(RETIRED) == marker, 'restore_source_not_retired', 409)
    target = _plain_path(cast(str, marker['destination'])).resolve()
    stage = _plain_path(cast(str, marker['stage'])).resolve()
    require(stage.parent == target.parent and stage.name.startswith(target.name+'.restoring-'), 'restore_staging_identity', 409)
    selected = target if target.exists() else stage
    receipt = _load(selected/'restore.json')
    require(receipt.get('source_instance') == control.meta('instance_id')
            and receipt.get('new_instance') == marker['new_instance']
            and receipt.get('backup_manifest_sha256') == marker['backup_sha256']
            and receipt.get('destination') == str(target), 'restore_receipt_identity', 409)
    check = ReplicaControl(selected)
    try:
        require(check.meta('instance_id') == marker['new_instance'], 'restore_instance_identity', 409)
        pending = check.meta(PENDING)
        require(pending == marker or selected == target and pending is False, 'restore_pending_identity', 409)
        if selected == stage:
            verify_database_files(stage, check.db)
    finally:
        check.close()
    reused = selected == target
    if not reused:
        os.rename(stage, target); _sync_dir(target.parent)
    # The published directory is still fenced. Only the validated handover
    # clears pending, after durable source retirement and directory publication.
    active = ReplicaControl(target)
    try:
        require(active.meta('instance_id') == marker['new_instance'], 'restore_instance_identity', 409)
        if active.meta(PENDING):
            require(active.meta(PENDING) == marker, 'restore_pending_identity', 409)
            active.set_meta(PENDING, False)
    finally:
        active.close()
    return {**receipt, 'state': str(target), 'credentials': str(target/'restored-credentials.json'), 'reused': reused}


def restore(source: PathInput, destination: PathInput, expected_sha256: str) -> dict[str, object]:
    """Restore a verified snapshot, retiring its original authority atomically.

    The original same-host state must still be accessible and unchanged. This
    intentionally rejects cross-host/disaster-recovery clones without an
    external ownership authority. No source payload or historical scratch is
    deleted. Repeating a publication interrupted after retirement is safe.
    """
    source = _plain_path(source).resolve(strict=True)
    target = _plain_path(destination).resolve()
    require(target.parent.is_dir() and not target.is_relative_to(source) and not source.is_relative_to(target), 'fresh_restore_required')
    require(file_hash(source/'backup.json') == expected_sha256, 'backup_manifest_identity', 409)
    record = _load(source/'backup.json')
    require(record.get('schema') == SCHEMA and inventory(source, skip={'backup.json'}) == record['files'], 'backup_content_changed', 409)
    control = ReplicaControl(cast(str, record['source_root']))
    try:
        require(target != control.root and not target.is_relative_to(control.root) and not control.root.is_relative_to(target), 'restore_source_overlap', 409)
        with _offline(control):
            require(control.meta('instance_id') == record['source_instance'], 'restore_source_instance', 409)
            retired = control.meta(RETIRED)
            if retired:
                marker = _object(retired)
                require(marker.get('destination') == str(target) and marker.get('backup_sha256') == expected_sha256,
                        'replica_state_retired', 409)
                return _finish_restore(control, marker)
            require(not target.exists(), 'fresh_restore_required')
            require(control.meta('drain', False) and _signature(control) == record['state_signature'], 'restore_source_changed', 409)
            with control.lock:
                require(control.db.execute('SELECT COUNT(*) FROM replica_workers WHERE state IN ("starting","available","running","quarantined")').fetchone()[0] == 0,
                        'restore_source_has_live_generations', 409)
            files = cast(list[FileIdentity], record['files'])
            needed = sum(f['bytes'] for f in files)+2*1024**2
            require(shutil.disk_usage(target.parent).free >= CONTRACT['min_free_bytes']+needed, 'restore_disk_capacity', 507)
            stage = target.parent/(target.name+'.restoring-'+uuid.uuid4().hex)
            stage.mkdir(mode=0o700)
            new_instance = uuid.uuid4().hex
            marker = {'backup_sha256': expected_sha256, 'destination': str(target), 'stage': str(stage), 'new_instance': new_instance}
            for item in files:
                if item['path'].startswith('state/') and item['path'] != 'state/contract.json':
                    copy_file(safe_child(source, item['path']), stage/item['path'][6:])
            # A directory must never become an executable ReplicaControl before
            # the pending fence is durable. Publish its required contract LAST.
            _fence_copy(stage/'control.sqlite', marker)
            copy_file(source/'state/contract.json', stage/'contract.json')
            _sync_dir(stage); _sync_dir(stage.parent)
            restored = ReplicaControl(stage)
            try:
                with restored.tx() as db:
                    verify_database_files(stage, db)
                    require(db.execute('SELECT COUNT(*) FROM replica_assignments WHERE state IN ("active","quarantined") OR host_bytes>0 OR vram_bytes>0').fetchone()[0] == 0,
                            'restore_active_reservations', 409)
                    # Old scratch belongs to the preserved retired source. Never
                    # import its paths into the new authority's reclaim scope.
                    for row in db.execute('SELECT worker,generation FROM replica_workers').fetchall():
                        detached = target/'detached-scratch'/(row['worker']+'-'+row['generation'])
                        db.execute('UPDATE replica_workers SET work=?,scratch_bytes=0,state="closed",closure=? WHERE worker=? AND generation=?',
                                   (str(detached), encoded({'reason': 'restored_detached', 'source_instance': record['source_instance']}).decode(), row['worker'], row['generation']))
                    db.execute('UPDATE files SET state="pending",token=NULL,lease=NULL WHERE state="uploading"')
                    db.execute('UPDATE inputs SET state="verifying",token=NULL,lease=NULL WHERE state="validating"')
                    db.execute('UPDATE service_scratch SET state="foreign_restore"')
                    db.execute('DELETE FROM service_meta WHERE key=?', (RETIRED,))
                for key, value in [('drain', True), ('shutdown_requested', True), ('worker', {}), ('instance_id', new_instance)]:
                    restored.set_meta(key, value)
                # Contract execution/model/profile binding remains identical.
                # Placement authority gets a new identity for a v3 handover.
                if restored.limits is not None:
                    from .replica_capacity import authority
                    restored.contract['replicas']['authority'] = authority(restored.contract['replicas']['host'], uuid.uuid4().hex)
                    write_file(stage/'contract.restore', encoded(restored.contract))
                    os.replace(stage/'contract.restore', stage/'contract.json'); _sync_dir(stage)
                write_file(stage/'secret.restore', secrets.token_bytes(32)); os.chmod(stage/'secret.restore', 0o600)
                os.replace(stage/'secret.restore', stage/'secret.key'); _sync_dir(stage)
                copy_file(source/'credentials.json', stage/'restored-credentials.json')
                result = {'schema': 'aureon.replica_restore.v1', 'backup_manifest_sha256': expected_sha256,
                          'source_instance': record['source_instance'], 'new_instance': new_instance,
                          'destination': str(target), 'drained': True, 'tickets_invalidated': True,
                          'old_scratch_reclamation_disabled': True, 'source_retired': True,
                          'model_executed': False, 'same_host_only': True}
                write_file(stage/'restore.json', encoded(result))
            finally:
                restored.close()
            # The parent entry and full stage precede the source's irreversible
            # retirement marker; retain both on interruption for explicit replay.
            _sync_dir(stage); _sync_dir(stage.parent)
            # Fencing precedes publication. On a rename failure the preserved
            # stage and marker support the same explicit restore command again.
            control.set_meta(RETIRED, marker)
            return _finish_restore(control, marker)
    finally:
        control.close()
