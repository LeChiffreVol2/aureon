"""Offline state backup/restore and explicit owned retention; never starts a model."""
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import uuid
from ..cloud.control import require,encoded,write_file,file_hash,CONTRACT,digest
from ..scene_api.durable import _plain_path,_sync_dir
from .ownership import exclusive,inventory,safe_child

SKIP={'control.sqlite','control.sqlite-wal','control.sqlite-shm','service.lock','worker.lock','supervisor.lock'}

def copy_file(source,target):
    source=_plain_path(source);target=_plain_path(target);target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with source.open('rb') as incoming,target.open('xb') as output:
        os.chmod(target,0o600);shutil.copyfileobj(incoming,output,1024**2);output.flush();os.fsync(output.fileno())
    _sync_dir(target.parent)

def verify_database_files(root,db):
    require(db.execute('PRAGMA integrity_check').fetchone()[0]=='ok','backup_database_integrity',409)
    for row in db.execute('SELECT manifest,declaration FROM inputs WHERE state="ready"'):
        require(row[0] and safe_child(root,row[0]).is_file(),'backup_input_missing',409)
        body=json.loads(row[1]);files={r['name']:r for r in body['files']};prepared=safe_child(root,row[0]).parent
        for member,uploaded in body['spec']['members'].items():
            path=safe_child(prepared,member);item=files[uploaded]
            require(path.stat().st_size==item['bytes'] and file_hash(path)==item['sha256'],'backup_prepared_input_integrity',409)
    for row in db.execute('SELECT id,result FROM jobs WHERE state="completed"'):
        for item in json.loads(row[1]).values():
            p=safe_child(root,item['path'])
            require(p.is_relative_to(root/'results'/row[0]) and p.stat().st_size==item['bytes'] and file_hash(p)==item['sha256'],'backup_artifact_integrity',409)

def backup(root,destination,credentials):
    from .control import Control
    root=_plain_path(root).resolve(strict=True);destination=_plain_path(destination)
    require(not destination.exists() and destination.parent.is_dir() and not destination.is_relative_to(root),'fresh_external_backup_required')
    with exclusive(root,'service.lock'),exclusive(root),exclusive(root,'supervisor.lock'):
        control=Control(root)
        try:
            # Acquired locks attest no current CLI service, worker or supervisor.
            # Draining is durable; callers explicitly resume after maintenance.
            control.set_meta('drain',True);control.set_meta('shutdown_requested',True)
            keys=json.loads(_plain_path(credentials).read_bytes())
            require(isinstance(keys,dict) and 'operator' in keys,'operator_credentials_required')
            require(all(control.authenticate(key)==tenant for tenant,key in keys.items()),'backup_credentials_invalid',409)
            before=inventory(root,skip=SKIP)
            database_bytes=control.db.execute('PRAGMA page_count').fetchone()[0]*control.db.execute('PRAGMA page_size').fetchone()[0]
            needed=sum(r['bytes'] for r in before)+database_bytes+Path(credentials).stat().st_size+2*1024**2
            free_paths=[destination.parent]+([Path('/mnt/c')] if Path('/mnt/c').is_dir() else [])
            require(min(shutil.disk_usage(p).free for p in free_paths)>=CONTRACT['min_free_bytes']+needed,'backup_disk_capacity',507)
            destination.mkdir(mode=0o700);state=destination/'state';state.mkdir(mode=0o700)
            for row in before:copy_file(safe_child(root,row['path']),state/row['path'])
            target=sqlite3.connect(state/'control.sqlite')
            try:
                with control.lock:control.db.backup(target)
                verify_database_files(state,target)
            finally:target.close()
            os.chmod(state/'control.sqlite',0o600)
            with (state/'control.sqlite').open('r+b') as stream:os.fsync(stream.fileno())
            copy_file(credentials,destination/'credentials.json')
            require(before==inventory(root,skip=SKIP),'backup_source_changed',409)
            files=inventory(destination)
            record={'schema':'aureon.service_backup.v1','source_instance':control.meta('instance_id'),'source_root':str(root),
              'files':files,'protected_secrets':True,'database_snapshot':'SQLite backup API with offline owner/worker locks',
              'drained':True,'credentials':'credentials.json','model_executed':False}
            write_file(destination/'backup.json',encoded(record));_sync_dir(destination)
            return {'backup':str(destination),'manifest_sha256':file_hash(destination/'backup.json'),'files':len(files),
              'bytes':sum(r['bytes'] for r in files),'contains_secrets':True,'resume_required':True}
        finally:control.close()

def restore(source,destination,expected_sha256):
    from .control import Control
    source=_plain_path(source).resolve(strict=True);destination=_plain_path(destination)
    require(not destination.exists() and destination.parent.is_dir() and not destination.is_relative_to(source),'fresh_restore_required')
    require(file_hash(source/'backup.json')==expected_sha256,'backup_manifest_identity',409)
    manifest=json.loads((source/'backup.json').read_bytes());require(manifest['schema']=='aureon.service_backup.v1','backup_schema')
    require(inventory(source,skip={'backup.json'})==manifest['files'],'backup_content_changed',409)
    needed=sum(r['bytes'] for r in manifest['files'])+2*1024**2
    free_paths=[destination.parent]+([Path('/mnt/c')] if Path('/mnt/c').is_dir() else [])
    require(min(shutil.disk_usage(p).free for p in free_paths)>=CONTRACT['min_free_bytes']+needed,'restore_disk_capacity',507)
    stage=destination.parent/(destination.name+'.restoring-'+uuid.uuid4().hex);stage.mkdir(mode=0o700)
    try:
        for row in manifest['files']:
            if row['path'].startswith('state/'):copy_file(safe_child(source,row['path']),stage/row['path'][6:])
        control=Control(stage)
        try:
            with control.tx() as db:
                verify_database_files(stage,db)
                now=control.clock()
                for row in db.execute('SELECT * FROM jobs WHERE state="running"').fetchall():
                    state='failed' if row['attempts']>=CONTRACT['max_attempts'] else 'queued'
                    db.execute('UPDATE attempts SET state="lease_lost",finished=? WHERE job=? AND number=?',(now,row['id'],row['attempts']))
                    db.execute('UPDATE jobs SET state=?,token=NULL,lease=NULL,expires=?,error=? WHERE id=?',
                      (state,now+CONTRACT['output_ttl_seconds'] if state=='failed' else None,'attempt_budget_exhausted' if state=='failed' else None,row['id']))
                    db.execute('UPDATE outbox SET state=? WHERE job=?',('finished' if state=='failed' else 'pending',row['id']))
                    if state=='failed':control._terminal(db,row['id'],'failed')
                db.execute('UPDATE inputs SET state="verifying",token=NULL,lease=NULL WHERE state="validating"')
                db.execute('UPDATE files SET state="pending",token=NULL,lease=NULL WHERE state="uploading"')
                db.execute('UPDATE service_scratch SET state="foreign_restore"')
            for key,value in [('drain',True),('shutdown_requested',True),('worker',{}),('instance_id',uuid.uuid4().hex)]:control.set_meta(key,value)
            # Rotating the ticket signing secret invalidates URLs from the old instance.
            write_file(stage/'secret.restore',secrets.token_bytes(32));os.chmod(stage/'secret.restore',0o600)
            os.replace(stage/'secret.restore',stage/'secret.key');_sync_dir(stage)
            copy_file(source/'credentials.json',stage/'restored-credentials.json')
            result={'schema':'aureon.service_restore.v1','backup_manifest_sha256':expected_sha256,'new_instance':control.meta('instance_id'),
              'drained':True,'tickets_invalidated':True,'credential_hashes_preserved':True,'fresh_scratch_required':True,
              'old_scratch_reclamation_disabled':True,'model_executed':False}
            write_file(stage/'restore.json',encoded(result))
        finally:control.close()
        os.rename(stage,destination);_sync_dir(destination.parent)
        return {**result,'state':str(destination),'credentials':str(destination/'restored-credentials.json')}
    except BaseException:
        # Partial staging is retained for diagnosis; never publish it as restored state.
        raise

def expire(control):
    """Commit inaccessible tombstones first; deletion is replayable after failures."""
    with control.tx() as db:
        now=control.clock()
        db.execute('UPDATE inputs SET state="rejected",error="validation_interrupted",token=NULL,lease=NULL WHERE state="validating" AND lease<=?',(now,))
        candidates=[('results',r[0]) for r in db.execute('SELECT id FROM jobs WHERE expires<=? AND state IN ("completed","cancelled","failed")',(now,))]
        candidates += [('inputs',r[0]) for r in db.execute('''SELECT id FROM inputs WHERE expires<=? AND state NOT IN ('expired','validating')
          AND NOT EXISTS(SELECT 1 FROM jobs WHERE jobs.input=inputs.id AND jobs.state IN ('queued','running'))
          AND NOT EXISTS(SELECT 1 FROM files WHERE files.input=inputs.id AND files.state='uploading' AND files.lease>?)''',(now,now))]
        for kind,identifier in candidates:
            db.execute('INSERT OR IGNORE INTO service_gc VALUES (?, ?, "pending", NULL)',(kind,identifier))
            if kind=='inputs':db.execute('UPDATE inputs SET state="expired",manifest=NULL,token=NULL,lease=NULL WHERE id=?',(identifier,))
            else:db.execute('UPDATE jobs SET state="expired",result=NULL,token=NULL,lease=NULL WHERE id=?',(identifier,))
        pending=[dict(r) for r in db.execute('SELECT * FROM service_gc WHERE state="pending"')]
    removed=[]
    for row in pending:
        kind,identifier=row['kind'],row['id'];require(kind in ('inputs','results'),'invalid_gc_kind',500)
        folder=control._folder(kind,identifier);require(folder.parent==control.root/kind,'unsafe_retention_target',500)
        if folder.exists():
            files=inventory(folder);receipt={'files':len(files),'bytes':sum(r['bytes'] for r in files),'inventory_sha256':digest(encoded(files))}
            # Preserve an intent before removal; rerun can complete a partial delete.
            with control.tx() as db:db.execute('UPDATE service_gc SET receipt=COALESCE(receipt,?) WHERE kind=? AND id=?',(encoded(receipt).decode(),kind,identifier))
            shutil.rmtree(_plain_path(folder));_sync_dir(folder.parent)
        with control.tx() as db:db.execute('UPDATE service_gc SET state="removed" WHERE kind=? AND id=?',(kind,identifier))
        removed.append({'kind':kind,'id':identifier})
    return removed

def reclaim(control,folder):
    folder=_plain_path(folder).resolve()
    with exclusive(control.root):
        with control.lock:row=control.db.execute('SELECT * FROM service_scratch WHERE path=?',(str(folder),)).fetchone()
        require(row is not None and row['state'] in ('closed','retired'),'scratch_not_closed',409)
        if row['state']=='retired':
            require(not folder.exists(),'retired_scratch_reappeared',409)
            return {'retired':True,'reused':True}
        intent=control.root/'maintenance'/('scratch-'+row['id']+'.json')
        if not folder.exists():
            require(intent.is_file(),'scratch_missing_without_intent',409)
            record=json.loads(intent.read_bytes());require(record['owner']==json.loads(row['owner']),'scratch_intent_owner',409)
            with control.tx() as db:db.execute('UPDATE service_scratch SET state="retired",receipt=? WHERE id=?',(encoded({'manifest':str(intent.relative_to(control.root)),'sha256':file_hash(intent)}).decode(),row['id']))
            return {'retired':True,'recovered_after_delete':True,'manifest':str(intent)}
        owner=json.loads(row['owner'])
        require(owner['state_instance']==control.meta('instance_id'),'scratch_owner_mismatch',409)
        require(folder!=control.root and not folder.is_relative_to(control.root) and not control.root.is_relative_to(folder),'scratch_overlaps_state',409)
        files=inventory(folder,max_bytes=4*1024**3)
        if intent.exists():
            record=json.loads(intent.read_bytes());require(record['owner']==owner,'scratch_intent_owner',409)
            known={r['path']:r for r in record['files']}
            require(all(r==known.get(r['path']) for r in files),'scratch_changed_after_intent',409)
            receipt=record['worker_result']
        else:
            require(owner==json.loads((folder/'ownership.json').read_bytes()),'scratch_owner_mismatch',409)
            receipt=json.loads((folder/'worker-result.json').read_bytes())
            record={'schema':'aureon.scratch_retirement.v1','owner':owner,'files':files,'worker_result':receipt,'bytes':sum(r['bytes'] for r in files)}
        require(receipt['passed'],'failed_scratch_retained',409)
        if not receipt['fake']:require(receipt['cleanup']['passed'] and receipt['cleanup']['cuda_allocated_bytes']==receipt['cleanup']['cuda_reserved_bytes']==0,'scratch_cleanup_failed',409)
        retained=control.root/'maintenance';retained.mkdir(exist_ok=True,mode=0o700)
        if not intent.exists():write_file(intent,encoded(record))
        shutil.rmtree(_plain_path(folder));_sync_dir(folder.parent)
        with control.tx() as db:db.execute('UPDATE service_scratch SET state="retired",receipt=? WHERE id=?',(encoded({'manifest':str(intent.relative_to(control.root)),'sha256':file_hash(intent)}).decode(),row['id']))
        return {'retired':True,'logical_bytes':record['bytes'],'manifest':str(intent),'evidence_policy':'Owned scratch only; public artifacts remain in durable state. Frozen research directories are never registered scratch.'}
