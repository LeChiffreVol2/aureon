"""Bounded file publication and authorized HTTP delivery for large workflows."""
from __future__ import annotations
from typing import TYPE_CHECKING
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .contracts import ArtifactFiles, AttemptLease
from .workflow_config import artifacts
from ..cloud.control import require, file_hash, encoded
from ..cloud.contracts import StreamBody
from ..scene_api.durable import _plain_path, _sync_dir
if TYPE_CHECKING:
    from .control import Control

CHUNK = 1024**2
METADATA_LIMIT = 4*1024**2


def _stamp(path: Path) -> tuple[int,int,int,int,int]:
    stat = path.stat()
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns


def publish(control: Control, lease: AttemptLease, files: ArtifactFiles) -> dict[str, object]:
    require(control.execution=='engine','engine_not_allowed',409)
    expected=artifacts(control.selected['task'])
    require(set(files)==expected,'incomplete_artifacts',409)
    with control.tx() as db:control._fenced(db,lease)
    paths={name:_plain_path(path) for name,path in files.items()}
    stamps={name:_stamp(path) for name,path in paths.items()}
    sizes={name:stamp[2] for name,stamp in stamps.items()}
    require(all(0<=size<=control.contract['max_artifact_bytes'] for size in sizes.values())
            and sum(sizes.values())<=control.contract['max_result_bytes'],'result_too_large',413)
    require(sizes['job.json']<=METADATA_LIMIT,'result_metadata_too_large',413)
    with paths['job.json'].open('rb') as stream:raw=stream.read(METADATA_LIMIT+1)
    require(len(raw)==sizes['job.json'],'artifact_size_changed',409)
    receipt=json.loads(raw)
    require(receipt.get('schema')=='aureon.async_result.v1' and receipt.get('job_id')==lease['job_id']
            and receipt.get('attempt')==lease['attempt'] and receipt.get('model_id')==control.selected['model_id']
            and receipt.get('task')==control.selected['task'] and receipt.get('profile_id')==control.profile
            and receipt.get('input_manifest_sha256')==file_hash(Path(lease['manifest']))
            and receipt.get('workflow_config_sha256')==control.selected['workflow']['config_sha256']
            and receipt.get('parameters')==lease['request']['parameters'],'result_binding_mismatch',409)
    require(set(receipt['artifacts'])==expected-{'job.json'},'result_artifact_binding',409)
    require(shutil.disk_usage(control.root).free>=control.contract['min_free_bytes']+sum(sizes.values()),'result_storage_capacity',507)
    folder=control._folder('results',lease['job_id']);folder.mkdir(exist_ok=True)
    attempt=_plain_path(folder/('attempt-'+str(lease['attempt'])));attempt.mkdir(exist_ok=True)
    record={};last_touch=time.monotonic()
    def touch() -> None:
        nonlocal last_touch
        if time.monotonic()-last_touch>=4:
            control.heartbeat(lease)
            if 'worker_owner' in lease:control.worker_status(lease['worker_owner'])
            last_touch=time.monotonic()
    def saved_hash(path: Path) -> str:
        hasher=hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                chunk=stream.read(CHUNK)
                if not chunk:break
                hasher.update(chunk);touch()
        return hasher.hexdigest()
    for name,source in paths.items():
        declared=receipt['artifacts'].get(name) if name!='job.json' else {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        require(declared['bytes']==sizes[name],'artifact_size_changed',409)
        target=_plain_path(attempt/name)
        if target.exists():
            require(target.stat().st_size==sizes[name] and saved_hash(target)==declared['sha256'],'artifact_conflict',409)
        else:
            hasher=hashlib.sha256();count=0
            with source.open('rb') as incoming,target.open('xb') as outgoing:
                while True:
                    chunk=incoming.read(CHUNK)
                    if not chunk:break
                    count+=len(chunk);require(count<=sizes[name],'artifact_size_changed',409)
                    outgoing.write(chunk);hasher.update(chunk);touch()
                outgoing.flush();os.fsync(outgoing.fileno())
            require(count==sizes[name] and hasher.hexdigest()==declared['sha256'] and stamps[name]==_stamp(source),'artifact_integrity',409)
            require(saved_hash(target)==declared['sha256'],'saved_artifact_integrity',409)
            _sync_dir(attempt)
        record[name]={'path':target.relative_to(control.root).as_posix(),'bytes':sizes[name],'sha256':declared['sha256']}
        touch()
    with control.tx() as db:
        row=control._fenced(db,lease);now=control.clock()
        db.execute('UPDATE jobs SET state="completed",result=?,expires=?,lease=NULL WHERE id=?',(encoded(record).decode(),now+control.contract['output_ttl_seconds'],row['id']))
        db.execute('UPDATE attempts SET state="completed",finished=? WHERE job=? AND number=?',(now,row['id'],lease['attempt']))
        db.execute('UPDATE outbox SET state="finished" WHERE job=?',(row['id'],))
        db.execute('INSERT INTO usage VALUES (?,?,1,?,0)',(row['id'],row['tenant'],now))
        control._terminal(db,row['id'],'completed')
    return {'job_id':row['id'],'state':'completed','model_executed':True}


def download(control: Control, tenant: str, identifier: str, name: str, ticket: str) -> StreamBody:
    control._check_ticket(ticket,tenant,'download',identifier,name)
    with control.tx() as db:
        row=control._owned(db,'jobs',tenant,identifier)
        require(row['state']=='completed' and row['expires']>control.clock(),'artifacts_unavailable',409)
        record=json.loads(row['result']);require(name in record,'not_found',404)
        declaration=record[name];path=_plain_path(control.root/declaration['path'])
        require(path.is_relative_to(control.root/'results'/identifier),'artifact_path_invalid',500)
        require(0<=declaration['bytes']<=control.contract['max_artifact_bytes'],'artifact_too_large',413)
        stream=path.open('rb')
    try:
        before=os.fstat(stream.fileno());hasher=hashlib.sha256();count=0
        while True:
            chunk=stream.read(CHUNK)
            if not chunk:break
            count+=len(chunk);require(count<=declaration['bytes'],'artifact_integrity',409);hasher.update(chunk)
        after=os.fstat(stream.fileno())
        require(count==declaration['bytes'] and hasher.hexdigest()==declaration['sha256']
                and (before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns),'artifact_integrity',409)
        stream.seek(0)
        return StreamBody(stream,count,declaration['sha256'])
    except BaseException:
        stream.close();raise
