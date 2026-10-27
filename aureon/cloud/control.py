"""Single-host transactional control plane with tenant objects and fenced attempts.

SQLite/local files implement the LOCAL adapter only. They must not be mounted
across serverless replicas. No provider network access or GPU imports occur here.
"""
from contextlib import contextmanager
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import threading
import time
import uuid

from ..client import _decode, ClientError
from ..scene_api.durable import _plain_path, _sync_dir

MODEL = 'kao-terramind-b'
PROFILE = 'fake-control-v1'
IDENTIFIER = re.compile(r'[0-9a-f]{32}\Z')
NAME = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}\Z')
SHA = re.compile(r'[0-9a-f]{64}\Z')
CONTRACT = {'schema': 'aureon.cloud.local.v1', 'model': MODEL, 'profile': PROFILE,
            'execution': 'fake', 'max_active': 3, 'max_running': 1, 'max_attempts': 3,
            'max_file_bytes': 1024**3, 'max_source_bytes': 1024**3, 'prepared_reserve_bytes': 1024**3,
            'max_tenant_storage_reservation_bytes': 2*1024**3,
            'max_global_storage_reservation_bytes': 8*1024**3, 'max_rows': 10000,
            'max_artifact_bytes': 16*1024**2, 'max_result_bytes': 64*1024**2,
            'input_ttl_seconds': 3600, 'output_ttl_seconds': 3600, 'ticket_seconds': 300,
            'lease_seconds': 60, 'min_free_bytes': 32*1024**3}


class Error(ValueError):
    def __init__(self, status, code, message, details=None):
        super().__init__(message)
        self.status, self.code = status, code
        self.details = details


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, code='invalid_request', status=400):
    if not condition:
        raise Error(status, code, code.replace('_', ' '))


def write_file(path, data):
    _plain_path(path)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)


def tenant_storage_limit(value):
    """Validate an explicit reservation budget without changing the global cap."""
    require(type(value) is int and 0 < value <= CONTRACT['max_global_storage_reservation_bytes'],
            'invalid_tenant_storage_reservation')
    return value


def initialize(root, tenants, *, qualification=None, qualification_verifier=None,
               tenant_storage_reservation_bytes=None):
    """Create explicit local credentials once; return plaintext only to caller."""
    root = _plain_path(root)
    require(not root.exists() and root.parent.is_dir(), 'fresh_state_required')
    require(1 <= len(tenants) <= 16 and len(set(tenants)) == len(tenants)
            and all(isinstance(n, str) and NAME.fullmatch(n) for n in tenants), 'invalid_tenants')
    limit = tenant_storage_limit(CONTRACT['max_tenant_storage_reservation_bytes']
                                 if tenant_storage_reservation_bytes is None else tenant_storage_reservation_bytes)
    root.mkdir(mode=0o700)
    for name in ('inputs', 'results'):
        (root/name).mkdir(mode=0o700)
    contract = dict(CONTRACT)
    contract['max_tenant_storage_reservation_bytes'] = limit
    if qualification is not None:
        from .profiles import verify_qualification
        (qualification_verifier or verify_qualification)(qualification)
        qualification_bytes = encoded(qualification)
        contract.update(execution='real',profile=qualification['profile_id'],qualification_sha256=digest(qualification_bytes))
        write_file(root/'qualification.json',qualification_bytes)
    write_file(root/'contract.json', encoded(contract))
    write_file(root/'secret.key', secrets.token_bytes(32))
    os.chmod(root/'secret.key', 0o600)
    db = sqlite3.connect(root/'control.sqlite')
    db.executescript('''
      PRAGMA journal_mode=WAL;
      PRAGMA synchronous=FULL;
      CREATE TABLE credentials(hash TEXT PRIMARY KEY, tenant TEXT NOT NULL);
      CREATE TABLE inputs(id TEXT PRIMARY KEY, tenant TEXT NOT NULL, state TEXT NOT NULL,
        declaration TEXT NOT NULL, bytes INTEGER NOT NULL, created REAL NOT NULL, expires REAL NOT NULL,
        token TEXT, lease REAL, manifest TEXT, error TEXT);
      CREATE TABLE files(input TEXT NOT NULL, name TEXT NOT NULL, bytes INTEGER NOT NULL,
        sha TEXT NOT NULL, state TEXT NOT NULL, token TEXT, lease REAL, PRIMARY KEY(input,name));
      CREATE TABLE jobs(id TEXT PRIMARY KEY, tenant TEXT NOT NULL, input TEXT NOT NULL,
        key TEXT NOT NULL, request TEXT NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
        token TEXT, lease REAL, created REAL NOT NULL, expires REAL, result TEXT, error TEXT,
        UNIQUE(tenant,key));
      CREATE TABLE outbox(job TEXT PRIMARY KEY, state TEXT NOT NULL);
      CREATE TABLE attempts(job TEXT NOT NULL, number INTEGER NOT NULL, state TEXT NOT NULL,
        started REAL NOT NULL, finished REAL, PRIMARY KEY(job,number));
      CREATE TABLE usage(job TEXT PRIMARY KEY, tenant TEXT NOT NULL, units INTEGER NOT NULL,
        created REAL NOT NULL, billable INTEGER NOT NULL CHECK(billable=0));
      CREATE INDEX tenant_inputs ON inputs(tenant,state);
      CREATE INDEX queued_jobs ON jobs(state,created);
    ''')
    credentials = {tenant: secrets.token_urlsafe(32) for tenant in tenants}
    db.executemany('INSERT INTO credentials VALUES (?,?)', [(digest(k.encode()), t) for t,k in credentials.items()])
    db.commit(); db.close()
    _sync_dir(root)
    return credentials


class Control:
    def __init__(self, root, *, clock=time.time, qualification_verifier=None):
        self.root = _plain_path(root).resolve(strict=True)
        self.clock, self.lock = clock, threading.RLock()
        self.contract = _decode((self.root/'contract.json').read_bytes())
        self.qualification = None
        expected = self._expected_contract(qualification_verifier)
        require(self.contract == expected, 'state_contract_mismatch', 409)
        self.execution, self.profile = self.contract['execution'], self.contract['profile']
        self.secret = (self.root/'secret.key').read_bytes()
        require(len(self.secret) == 32, 'invalid_state_secret', 500)
        self.db = sqlite3.connect(_plain_path(self.root/'control.sqlite'), timeout=5, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA synchronous=FULL')

    def _expected_contract(self, qualification_verifier):
        """Extension boundary for a separately validated, persisted service route."""
        expected = dict(CONTRACT)
        expected['max_tenant_storage_reservation_bytes'] = tenant_storage_limit(
            self.contract.get('max_tenant_storage_reservation_bytes'))
        if self.contract.get('execution')=='real':
            from .profiles import verify_qualification
            raw = (self.root/'qualification.json').read_bytes()
            self.qualification = {'record':(qualification_verifier or verify_qualification)(_decode(raw)), 'sha256':digest(raw)}
            expected.update(execution='real',profile=self.qualification['record']['profile_id'],qualification_sha256=digest(raw))
        return expected

    def close(self):
        self.db.close()

    @contextmanager
    def tx(self):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                yield self.db
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def authenticate(self, key):
        require(isinstance(key, str) and 32 <= len(key) <= 128, 'unauthorized', 401)
        with self.lock:
            row = self.db.execute('SELECT tenant FROM credentials WHERE hash=?', (digest(key.encode()),)).fetchone()
        require(row is not None, 'unauthorized', 401)
        return row['tenant']

    def capabilities(self):
        return {'model_id':MODEL,'profile_id':self.profile,'execution':self.execution,
                'qualification_sha256':self.contract.get('qualification_sha256'),
                'cancellation':{'allowed_states':['queued'],'replay_state':'cancelled',
                                'running_or_terminal_http_status':409,'interrupts_gpu':False},
                'max_running':CONTRACT['max_running'],
                'max_attempts':CONTRACT['max_attempts'],'lease_seconds':CONTRACT['lease_seconds'],
                'per_job_numerical_status':'not_evaluated'}

    def health(self):
        """Control-plane checks only; no model import, probe job or worker claim."""
        checks={}
        try:
            with self.tx() as db:
                db.execute('SELECT id FROM jobs LIMIT 1').fetchone()
                db.execute('SELECT id FROM inputs LIMIT 1').fetchone()
            checks['database_transaction']='passed'
        except (sqlite3.Error, OSError):
            checks['database_transaction']='unavailable'
        try:
            checks['state_contract']='passed' if _decode((self.root/'contract.json').read_bytes())==self.contract else 'changed'
        except (OSError,ValueError,ClientError): checks['state_contract']='unavailable'
        try:
            checks['storage_directories']='passed' if all(_plain_path(self.root/name).is_dir() for name in ('inputs','results')) else 'unavailable'
            paths=[self.root]+([Path('/mnt/c')] if Path('/mnt/c').is_dir() else [])
            checks['disk_floor']='passed' if min(shutil.disk_usage(p).free for p in paths)>=CONTRACT['min_free_bytes'] else 'below_floor'
        except (OSError,ValueError):
            checks['storage_directories']='unavailable'; checks['disk_floor']='unavailable'
        ready=all(value=='passed' for value in checks.values())
        return {'schema':'aureon.cloud.health.v1','scope':'local_control_plane_only',
                'control_plane':{'ready':ready,'state':'ready' if ready else 'degraded','checks':checks},
                'execution_readiness':{'worker':'unknown','model':'unknown','reason':'no_live_worker_attestation'},
                'limitations':['No GPU or dependency qualification from this CPU probe',
                               'Transaction availability and directory checks do not attest every future write'],
                'action':None if ready else 'Operator must inspect control storage/contract/database; retry only after recovery'}

    def status_snapshot(self, tenant):
        """Bounded aggregates for the authenticated tenant; no global activity."""
        now=self.clock()
        with self.tx() as db:
            counts={}
            for table in ('inputs','jobs'):
                rows=db.execute(f'SELECT CASE WHEN expires IS NOT NULL AND expires<=? THEN "expired" ELSE state END AS visible_state, COUNT(*) AS n FROM {table} WHERE tenant=? GROUP BY visible_state', (now,tenant)).fetchall()
                counts[table]={row['visible_state']:row['n'] for row in rows}
            used,pending=db.execute('SELECT COALESCE(SUM(bytes),0),COALESCE(SUM(CASE WHEN expires<=? THEN bytes ELSE 0 END),0) FROM inputs WHERE tenant=? AND state!="expired"', (now,tenant)).fetchone()
            lost=db.execute('SELECT COUNT(*) FROM jobs WHERE tenant=? AND state="running" AND lease<=?', (tenant,now)).fetchone()[0]
        limit=self.contract['max_tenant_storage_reservation_bytes']
        reasons=[]
        if limit-used<=CONTRACT['prepared_reserve_bytes']:
            reasons.append({'code':'tenant_input_reservation_low',
                'action':'Each retained input reserves source bytes plus 1 GiB; reuse an existing input or ask the operator to expire payloads/review the configured budget'})
        if pending:
            reasons.append({'code':'expired_inputs_awaiting_retention',
                'action':'Expired input reservations remain counted until the operator runs explicit retention'})
        if lost:
            reasons.append({'code':'own_job_lease_expired',
                'action':'A worker claim must reconcile expired leases; this snapshot does not restart or cancel work'})
        return {'schema':'aureon.cloud.tenant_status.v1','scope':'authenticated_tenant_only',
            'observed_at_epoch':now,'counts':counts,'queue':{'queued':counts['jobs'].get('queued',0),
                'running':counts['jobs'].get('running',0),'expired_running_leases':lost},
            'input_reservations':{'used_bytes':used,'limit_bytes':limit,'remaining_bytes':max(0,limit-used),
                'expired_awaiting_retention_bytes':pending,'prepared_reserve_per_input_bytes':CONTRACT['prepared_reserve_bytes'],
                'maximum_source_bytes_per_input':CONTRACT['max_source_bytes']},
            'admission':{'reasons':reasons,'scope':'Own reservations only; shared admission constraints are rechecked atomically on submission'},
            'capabilities':self.capabilities(),'health':self.health()}

    def _owned(self, db, table, tenant, identifier):
        require(isinstance(identifier, str) and IDENTIFIER.fullmatch(identifier), 'not_found', 404)
        assert table in ('inputs', 'jobs')
        row = db.execute(f'SELECT * FROM {table} WHERE id=? AND tenant=?', (identifier, tenant)).fetchone()
        require(row is not None, 'not_found', 404)
        return row

    def _ticket(self, tenant, purpose, identifier, name, expires):
        body = encoded([tenant, purpose, identifier, name, expires])
        raw = base64.urlsafe_b64encode(body).decode().rstrip('=')
        return raw+'.'+hmac.new(self.secret, raw.encode(), hashlib.sha256).hexdigest()

    def _check_ticket(self, ticket, tenant, purpose, identifier, name):
        try:
            raw, signature = ticket.split('.')
            expected = hmac.new(self.secret, raw.encode(), hashlib.sha256).hexdigest()
            value = json.loads(base64.urlsafe_b64decode(raw+'='*((-len(raw))%4)))
            valid = hmac.compare_digest(signature, expected) and value[:4] == [tenant,purpose,identifier,name]
            valid = valid and len(value) == 5 and type(value[4]) is int and self.clock() < value[4]
        except (ValueError, TypeError, IndexError, UnicodeError):
            valid = False
        require(valid, 'invalid_or_expired_ticket', 403)

    def _folder(self, kind, identifier):
        require(IDENTIFIER.fullmatch(identifier) is not None, 'invalid_identity')
        return _plain_path(self.root/kind/identifier)

    def create_input(self, tenant, body):
        files = self._validate_input(body)
        return self._reserve_input(tenant, body, files)

    def _validate_input(self, body):
        require(isinstance(body, dict) and set(body) == {'files','spec'})
        files, spec = body['files'], body['spec']
        require(isinstance(files, list) and 1 <= len(files) <= 14 and isinstance(spec, dict))
        names = []
        for item in files:
            require(isinstance(item, dict) and set(item) == {'name','bytes','sha256'})
            require(isinstance(item['name'], str) and NAME.fullmatch(item['name'])
                    and item['name'].endswith(('.tif','.xml')), 'invalid_filename')
            require(type(item['bytes']) is int and 0 < item['bytes'] <= CONTRACT['max_file_bytes'], 'invalid_size')
            require(not item['name'].endswith('.xml') or item['bytes'] <= 1024**2, 'metadata_too_large')
            require(isinstance(item['sha256'], str) and SHA.fullmatch(item['sha256']), 'invalid_hash')
            names.append(item['name'])
        require(len(set(names)) == len(names), 'duplicate_filename')
        common = {'schema','kind','units','bands','expected_crs','provenance','window'}
        if spec.get('kind') == 'aligned':
            require(set(spec) <= common | {'raster','valid_mask','validity_policy'})
            references = [spec.get('raster'), spec.get('valid_mask')]
        else:
            require(spec.get('kind') == 'l2a' and set(spec) <= common | {'scl','radiometry'})
            require(isinstance(spec.get('bands'), dict), 'invalid_bands')
            radiometry = spec.get('radiometry')
            require(isinstance(radiometry, dict) and set(radiometry) <= {'quantification_value','offsets','product_metadata'}, 'invalid_radiometry')
            references = list(spec['bands'].values()) + [spec.get('scl')]
            if 'product_metadata' in radiometry:
                references.append(radiometry['product_metadata'])
        require(all(isinstance(n, str) and n in names for n in references)
                and set(references) == set(names), 'file_references_must_be_uploaded_names')
        spectral = references[:-1] if spec.get('kind')=='l2a' and 'product_metadata' in spec['radiometry'] else references
        require(all(n.endswith('.tif') for n in spectral), 'rasters_must_be_tiff')
        if len(spectral) != len(references):
            require(references[-1].endswith('.xml'), 'metadata_must_be_xml')
        return files

    def _reserve_input(self, tenant, body, files):
        total, now, identifier = sum(f['bytes'] for f in files), self.clock(), uuid.uuid4().hex
        require(total <= CONTRACT['max_source_bytes'], 'source_byte_quota', 429)
        reservation = total+CONTRACT['prepared_reserve_bytes']
        with self.tx() as db:
            self._admit(db)
            for scope, args, limit in [('tenant=? AND', (tenant,), self.contract['max_tenant_storage_reservation_bytes']), ('', (), CONTRACT['max_global_storage_reservation_bytes'])]:
                used = db.execute(f'SELECT COALESCE(SUM(bytes),0) FROM inputs WHERE {scope} state != "expired"', args).fetchone()[0]
                if used+reservation > limit:
                    details={'scope':'tenant','used_bytes':used,'limit_bytes':limit,'requested_reservation_bytes':reservation} if scope else {'scope':'service'}
                    raise Error(429,'input_byte_quota',
                        'Input reservation unavailable: each input reserves source bytes plus 1 GiB. Reuse an existing input or ask the operator to expire retained payloads/review the budget.',details)
            # Reserve unexpanded imagery as well as upload bytes. On WSL the
            # backing Windows disk is also checked; VHD free space is not enough.
            paths = [self.root] + ([Path('/mnt/c')] if Path('/mnt/c').is_dir() else [])
            require(min(shutil.disk_usage(p).free for p in paths) >= CONTRACT['min_free_bytes']+used+reservation,
                    'storage_capacity', 429)
            require(db.execute('SELECT COUNT(*) FROM inputs').fetchone()[0] < CONTRACT['max_rows'], 'input_count_quota', 429)
            folder = self._folder('inputs', identifier)
            folder.mkdir(mode=0o700)
            (folder/'source').mkdir()
            db.execute('INSERT INTO inputs(id,tenant,state,declaration,bytes,created,expires) VALUES (?,?,?,?,?,?,?)',
                       (identifier,tenant,'uploading',encoded(body).decode(),reservation,now,now+CONTRACT['input_ttl_seconds']))
            db.executemany('INSERT INTO files(input,name,bytes,sha,state) VALUES (?,?,?,?,?)',
                [(identifier,f['name'],f['bytes'],f['sha256'],'pending') for f in files])
        return self.input(tenant, identifier)

    def input(self, tenant, identifier):
        with self.tx() as db:
            row = self._owned(db, 'inputs', tenant, identifier)
            result = {k:row[k] for k in ('id','state','expires','error')}
            result['schema'] = 'aureon.cloud.input.v1'
            if row['state']=='ready' and row['manifest'] and row['expires'] > self.clock():
                result['prepared_manifest_sha256'] = file_hash(_plain_path(self.root/row['manifest']))
            if row['expires'] <= self.clock():
                result['state'] = 'expired'
            if row['state'] == 'uploading' and row['expires'] > self.clock():
                expiry = int(min(row['expires'], self.clock()+CONTRACT['ticket_seconds']))
                result['uploads'] = [{'name':f['name'],'bytes':f['bytes'],'sha256':f['sha'],
                    'expires':expiry, 'url':f"/v1/inputs/{identifier}/files/{f['name']}?ticket="+
                    self._ticket(tenant,'upload',identifier,f['name'],expiry)}
                    for f in db.execute('SELECT * FROM files WHERE input=?', (identifier,))]
            return result

    def upload(self, tenant, identifier, name, ticket, stream, length):
        self._check_ticket(ticket, tenant, 'upload', identifier, name)
        token = uuid.uuid4().hex
        with self.tx() as db:
            row = self._owned(db,'inputs',tenant,identifier)
            require(row['state'] == 'uploading' and row['expires'] > self.clock(), 'input_closed', 409)
            file = db.execute('SELECT * FROM files WHERE input=? AND name=?', (identifier,name)).fetchone()
            require(file is not None, 'not_found', 404)
            require(length == file['bytes'], 'length_mismatch', 413)
            require(file['state'] != 'uploading' or file['lease'] < self.clock(), 'upload_busy', 409)
            db.execute('UPDATE files SET state="uploading",token=?,lease=? WHERE input=? AND name=?',
                       (token,self.clock()+30,identifier,name))
        folder = self._folder('inputs',identifier)/'source'
        temporary = _plain_path(folder/(name+'.'+token+'.part'))
        try:
            began, remaining, hash_value = time.monotonic(), length, hashlib.sha256()
            with temporary.open('xb') as output:
                while remaining:
                    require(time.monotonic()-began < 30, 'upload_deadline', 408)
                    chunk = stream.read(min(65536,remaining))
                    require(bool(chunk), 'incomplete_upload')
                    hash_value.update(chunk); output.write(chunk); remaining -= len(chunk)
                output.flush(); os.fsync(output.fileno())
            require(hash_value.hexdigest() == file['sha'], 'hash_mismatch')
            with self.tx() as db:
                current = db.execute('SELECT * FROM files WHERE input=? AND name=?', (identifier,name)).fetchone()
                row = self._owned(db,'inputs',tenant,identifier)
                require(current['token'] == token and current['lease'] > self.clock()
                        and row['state'] == 'uploading', 'stale_upload', 409)
                os.replace(temporary, _plain_path(folder/name)); _sync_dir(folder)
                db.execute('UPDATE files SET state="uploaded",token=NULL,lease=NULL WHERE input=? AND name=?', (identifier,name))
        except BaseException:
            with self.tx() as db:
                db.execute('UPDATE files SET state="pending",token=NULL,lease=NULL WHERE input=? AND name=? AND token=?', (identifier,name,token))
            raise
        finally:
            if temporary.exists():
                _plain_path(temporary).unlink()
        return {'uploaded': True, 'sha256':file['sha'], 'bytes':length}

    def complete_input(self, tenant, identifier):
        with self.tx() as db:
            row = self._owned(db,'inputs',tenant,identifier)
            require(row['expires'] > self.clock(), 'input_expired', 410)
            if row['state'] in ('verifying','validating','ready'):
                return {'id':identifier,'state':row['state']}
            require(row['state'] == 'uploading', 'input_closed', 409)
            require(db.execute('SELECT COUNT(*) FROM files WHERE input=? AND state!="uploaded"', (identifier,)).fetchone()[0] == 0,
                    'uploads_incomplete', 409)
            db.execute('UPDATE inputs SET state="verifying" WHERE id=?', (identifier,))
        return {'id':identifier, 'state':'verifying'}

    def prepare_one(self):
        """CPU executor entry: validate only operator-owned staged files, no URLs."""
        with self.tx() as db:
            db.execute('UPDATE inputs SET state="rejected",error="validation_interrupted" WHERE state="validating" AND lease<=?', (self.clock(),))
            row = db.execute('SELECT * FROM inputs WHERE state="verifying" AND expires>? ORDER BY created LIMIT 1', (self.clock(),)).fetchone()
            if row is None:
                return False
            identifier, token = row['id'], uuid.uuid4().hex
            db.execute('UPDATE inputs SET state="validating",token=?,lease=? WHERE id=?', (token,self.clock()+300,identifier))
        folder = self._folder('inputs',identifier)
        try:
            body = json.loads(row['declaration'])
            for file in body['files']:
                path = _plain_path(folder/'source'/file['name'])
                require(path.stat().st_size == file['bytes'] and file_hash(path) == file['sha256'], 'staged_file_changed')
                if file['name'].endswith('.tif'):
                    # Reject disguised VRT/remote datasets before GDAL can open
                    # references. Force the decoder as an additional check.
                    with path.open('rb') as stream:
                        require(stream.read(4) in (b'II*\x00',b'MM\x00*',b'II+\x00',b'MM\x00+'), 'tiff_header_required')
                    self._inspect_uploaded_tiff(path)
            prepared = self._prepare_input(body, folder, token, identifier)
            manifest = str(Path(prepared).relative_to(self.root))
            with self.tx() as db:
                changed = db.execute('UPDATE inputs SET state="ready",manifest=?,token=NULL,lease=NULL WHERE id=? AND token=? AND lease>?',
                                     (manifest,identifier,token,self.clock())).rowcount
                require(changed == 1, 'stale_validation', 409)
        except Exception:
            with self.tx() as db:
                db.execute('UPDATE inputs SET state="rejected",error="input_validation_failed",token=NULL,lease=NULL WHERE id=? AND token=?', (identifier,token))
        return True

    def _prepare_input(self, body, folder, token, identifier):
        from ..prepare import prepare_scene, check_scene
        spec = dict(body['spec'], scene_id=identifier)
        prepared = prepare_scene(spec, folder/('prepared-'+token), base_dir=folder/'source')
        check_scene(prepared['manifest'])
        return prepared['manifest']

    def _inspect_uploaded_tiff(self,path):
        import rasterio
        with rasterio.open(path,driver='GTiff'):pass

    def _terminal(self,db,identifier,state):
        """Optional atomic usage/entitlement hook; never external payment I/O."""
        pass

    def _validate_submission(self, body):
        require(isinstance(body, dict) and set(body) == {'input_id','model_id','profile_id'})
        require(body['model_id'] == MODEL and body['profile_id'] == self.profile, 'profile_not_available')

    def _submitted(self, db, identifier, body):
        pass

    def _admit(self, db):
        pass

    def _can_claim(self, db):
        return True

    def submit(self, tenant, body, key):
        self._validate_submission(body)
        require(isinstance(key,str) and 8 <= len(key) <= 128 and key.isascii() and all(32 <= ord(c) < 127 for c in key), 'invalid_idempotency_key')
        fingerprint, key_hash, now = digest(encoded(body)), digest(key.encode()), self.clock()
        with self.tx() as db:
            old = db.execute('SELECT * FROM jobs WHERE tenant=? AND key=?', (tenant,key_hash)).fetchone()
            if old:
                require(old['request'] == fingerprint, 'idempotency_conflict', 409)
                identifier, reused = old['id'], True
            else:
                self._admit(db)
                source = self._owned(db,'inputs',tenant,body['input_id'])
                require(source['state'] == 'ready' and source['expires'] > now, 'input_not_ready', 409)
                require(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] < CONTRACT['max_rows'], 'job_count_quota', 429)
                require(db.execute('SELECT COUNT(*) FROM jobs WHERE state IN ("queued","running")').fetchone()[0] < CONTRACT['max_active'], 'queue_full', 429)
                identifier, reused = uuid.uuid4().hex, False
                db.execute('INSERT INTO jobs(id,tenant,input,key,request,state,created) VALUES (?,?,?,?,?,"queued",?)',
                           (identifier,tenant,body['input_id'],key_hash,fingerprint,now))
                db.execute('INSERT INTO outbox VALUES (?,"pending")', (identifier,))
                self._submitted(db, identifier, body)
        return dict(self.job(tenant,identifier), reused=reused)

    def job(self, tenant, identifier):
        with self.tx() as db:
            row = self._owned(db,'jobs',tenant,identifier)
            result = {k:row[k] for k in ('id','state','attempts','created','expires','error')}
            result.update(schema='aureon.cloud.job.v1', model_id=MODEL, profile_id=self.profile,
                          execution=self.execution, validation={'status':'not_evaluated',
                          'reason':'fake_executor' if self.execution=='fake' else 'guarded_no_per_job_reference',
                          'qualification_sha256':self.contract.get('qualification_sha256')})
            if row['expires'] and row['expires'] <= self.clock():
                result['state'] = 'expired'
            return result

    def cancel(self, tenant, identifier):
        with self.tx() as db:
            row = self._owned(db,'jobs',tenant,identifier)
            require(row['state'] in ('queued','cancelled'), 'cannot_cancel_running_or_terminal', 409)
            if row['state']=='queued':
                db.execute('UPDATE jobs SET state="cancelled",expires=? WHERE id=?', (self.clock()+CONTRACT['output_ttl_seconds'],identifier))
                db.execute('UPDATE outbox SET state="finished" WHERE job=?', (identifier,))
                self._terminal(db,identifier,'cancelled')
        return self.job(tenant,identifier)

    def claim(self):
        with self.tx() as db:
            if not self._can_claim(db):
                return None
            now = self.clock()
            db.execute('UPDATE inputs SET state="rejected",error="validation_interrupted" WHERE state="validating" AND lease<=?', (now,))
            for row in db.execute('SELECT * FROM jobs WHERE state="running" AND lease<=?', (now,)).fetchall():
                state = 'failed' if row['attempts'] >= CONTRACT['max_attempts'] else 'queued'
                db.execute('UPDATE attempts SET state="lease_lost",finished=? WHERE job=? AND number=?', (now,row['id'],row['attempts']))
                db.execute('UPDATE jobs SET state=?,token=NULL,lease=NULL,error=?,expires=? WHERE id=?',
                           (state,'attempt_budget_exhausted' if state=='failed' else None,now+CONTRACT['output_ttl_seconds'] if state=='failed' else None,row['id']))
                db.execute('UPDATE outbox SET state=? WHERE job=?', ('finished' if state=='failed' else 'pending',row['id']))
                if state=='failed':self._terminal(db,row['id'],'failed')
            if db.execute('SELECT COUNT(*) FROM jobs WHERE state="running"').fetchone()[0]:
                return None
            row = db.execute('SELECT jobs.* FROM jobs JOIN outbox ON jobs.id=outbox.job WHERE jobs.state="queued" AND outbox.state="pending" ORDER BY created LIMIT 1').fetchone()
            if row is None:
                return None
            source = db.execute('SELECT * FROM inputs WHERE id=?', (row['input'],)).fetchone()
            require(source['state'] == 'ready', 'prepared_input_unavailable', 409)
            token, attempt = secrets.token_urlsafe(32), row['attempts']+1
            db.execute('UPDATE jobs SET state="running",attempts=?,token=?,lease=? WHERE id=?', (attempt,digest(token.encode()),now+CONTRACT['lease_seconds'],row['id']))
            db.execute('UPDATE outbox SET state="claimed" WHERE job=?', (row['id'],))
            db.execute('INSERT INTO attempts VALUES (?,?,"running",?,NULL)', (row['id'],attempt,now))
            return {'job_id':row['id'],'attempt':attempt,'token':token,'manifest':str(_plain_path(self.root/source['manifest']))}

    def _fenced(self, db, lease):
        row = db.execute('SELECT * FROM jobs WHERE id=?', (lease['job_id'],)).fetchone()
        require(row is not None and row['state']=='running' and row['attempts']==lease['attempt']
                and hmac.compare_digest(row['token'] or '',digest(lease['token'].encode()))
                and row['lease'] > self.clock(), 'stale_attempt', 409)
        return row

    def heartbeat(self, lease):
        with self.tx() as db:
            self._fenced(db,lease)
            db.execute('UPDATE jobs SET lease=? WHERE id=?', (self.clock()+CONTRACT['lease_seconds'],lease['job_id']))

    def publish_fake(self, lease):
        """Explicit fake artifact, never a COG or a numerical parity assertion."""
        require(self.execution=='fake', 'fake_executor_not_allowed', 409)
        artifact = encoded({'schema':'aureon.simulation.v1','job_id':lease['job_id'],'attempt':lease['attempt'],
            'model_executed':False,'input_manifest_sha256':file_hash(Path(lease['manifest'])),
            'validation':{'status':'not_evaluated','reason':'fake_executor'}})
        with self.tx() as db:
            row = self._fenced(db,lease)
            folder = self._folder('results',row['id'])
            folder.mkdir(exist_ok=True)
            attempt = folder/('attempt-'+str(lease['attempt']))
            attempt.mkdir(exist_ok=True)
            path = _plain_path(attempt/'simulation.json')
            if path.exists():
                require(path.read_bytes() == artifact, 'artifact_conflict', 409)
            else:
                write_file(path,artifact)
            record = {'simulation.json':{'path':str(path.relative_to(self.root)), 'bytes':len(artifact),'sha256':digest(artifact)}}
            now = self.clock()
            db.execute('UPDATE jobs SET state="completed",result=?,expires=?,lease=NULL WHERE id=?',
                       (encoded(record).decode(),now+CONTRACT['output_ttl_seconds'],row['id']))
            db.execute('UPDATE attempts SET state="completed",finished=? WHERE job=? AND number=?', (now,row['id'],lease['attempt']))
            db.execute('UPDATE outbox SET state="finished" WHERE job=?', (row['id'],))
            db.execute('INSERT INTO usage VALUES (?,?,1,?,0)', (row['id'],row['tenant'],now))
            self._terminal(db,row['id'],'completed')
        return {'job_id':row['id'], 'state':'completed', 'model_executed':False}

    def publish_real(self, lease, files):
        require(self.execution=='real', 'real_executor_not_allowed', 409)
        require(set(files)=={'class_map.tif','valid_mask.tif','provenance.json','job.json'}, 'incomplete_artifacts', 409)
        payload = {}
        for name,path in files.items():
            with _plain_path(path).open('rb') as stream:
                payload[name] = stream.read(CONTRACT['max_artifact_bytes']+1)
            require(len(payload[name]) <= CONTRACT['max_artifact_bytes'], 'artifact_too_large', 413)
        require(sum(map(len,payload.values())) <= CONTRACT['max_result_bytes'], 'result_too_large', 413)
        receipt, provenance = _decode(payload['job.json']), _decode(payload['provenance.json'])
        require(receipt.get('schema')=='aureon.guarded_scene_job.v2' and receipt.get('status')=='completed'
                and receipt.get('job_id')==lease['job_id'] and receipt.get('profile_id')==self.profile
                and receipt.get('validation',{}).get('numerical',{}).get('status')=='not_evaluated'
                and receipt['validation']['qualification']['sha256']==self.qualification['sha256']
                and receipt['code_sha256']==self.qualification['record']['source_sha256']
                and receipt['model_identity']['bundle_receipt_sha256']==self.qualification['record']['bundle_receipt_sha256'],
                'worker_receipt_mismatch', 409)
        require(provenance.get('schema')=='aureon.cloud.provenance.v2'
                and provenance.get('validation')==receipt['validation'], 'provenance_mismatch', 409)
        for name in ('class_map.tif','valid_mask.tif'):
            require(provenance['files'][name]['sha256']==digest(payload[name])
                    and provenance['files'][name]['bytes']==len(payload[name]), 'artifact_integrity', 409)
        with self.tx() as db:
            row = self._fenced(db,lease)
            source = db.execute('SELECT manifest FROM inputs WHERE id=?', (row['input'],)).fetchone()
            expected_manifest = file_hash(_plain_path(self.root/source['manifest']))
            require(receipt['source_manifest_sha256']==expected_manifest
                    and provenance['provenance']['source_manifest_sha256']==expected_manifest
                    and provenance['provenance']['job_id']==row['id']
                    and provenance['provenance']['model_identity']==receipt['model_identity'],
                    'result_input_binding_mismatch', 409)
            folder = self._folder('results',row['id'])
            folder.mkdir(exist_ok=True)
            attempt = folder/('attempt-'+str(lease['attempt']))
            attempt.mkdir(exist_ok=True)
            record = {}
            for name,data in payload.items():
                path = _plain_path(attempt/name)
                if path.exists():
                    require(path.read_bytes()==data, 'artifact_conflict', 409)
                else:
                    write_file(path,data)
                record[name] = {'path':str(path.relative_to(self.root)), 'bytes':len(data),'sha256':digest(data)}
            now = self.clock()
            db.execute('UPDATE jobs SET state="completed",result=?,expires=?,lease=NULL WHERE id=?',
                (encoded(record).decode(),now+CONTRACT['output_ttl_seconds'],row['id']))
            db.execute('UPDATE attempts SET state="completed",finished=? WHERE job=? AND number=?', (now,row['id'],lease['attempt']))
            db.execute('UPDATE outbox SET state="finished" WHERE job=?', (row['id'],))
            db.execute('INSERT INTO usage VALUES (?,?,1,?,0)', (row['id'],row['tenant'],now))
            self._terminal(db,row['id'],'completed')
        return {'job_id':row['id'],'state':'completed','model_executed':True}

    def fail(self, lease):
        with self.tx() as db:
            row = self._fenced(db,lease)
            db.execute('UPDATE jobs SET state="failed",error="execution_failed",expires=?,lease=NULL WHERE id=?',
                       (self.clock()+CONTRACT['output_ttl_seconds'],row['id']))
            db.execute('UPDATE attempts SET state="failed",finished=? WHERE job=? AND number=?',
                       (self.clock(),row['id'],lease['attempt']))
            db.execute('UPDATE outbox SET state="finished" WHERE job=?', (row['id'],))
            self._terminal(db,row['id'],'failed')

    def artifacts(self, tenant, identifier):
        with self.tx() as db:
            row = self._owned(db,'jobs',tenant,identifier)
            require(not row['expires'] or row['expires'] > self.clock(), 'output_expired', 410)
            require(row['state']=='completed', 'job_not_completed', 409)
            expiry = int(min(row['expires'], self.clock()+CONTRACT['ticket_seconds']))
            return {'job_id':identifier, 'expires':expiry, 'artifacts':{name:{'bytes':f['bytes'],'sha256':f['sha256'],
                'url':f'/v1/jobs/{identifier}/artifacts/{name}?ticket='+self._ticket(tenant,'download',identifier,name,expiry)}
                for name,f in json.loads(row['result']).items()}}

    def download(self, tenant, identifier, name, ticket):
        self._check_ticket(ticket,tenant,'download',identifier,name)
        with self.tx() as db:
            row = self._owned(db,'jobs',tenant,identifier)
            require(row['state']=='completed' and row['expires'] > self.clock(), 'output_unavailable', 410)
            declaration = json.loads(row['result']).get(name)
            require(declaration is not None, 'not_found', 404)
            path = _plain_path(self.root/declaration['path'])
            require(path.is_relative_to(self.root/'results'/identifier), 'artifact_path_invalid', 500)
            with path.open('rb') as stream:
                data = stream.read(CONTRACT['max_artifact_bytes']+1)
            require(len(data) == declaration['bytes'] and digest(data)==declaration['sha256'], 'artifact_integrity', 409)
            return data

    def expire(self):
        """Explicit maintenance; owns only this new service's payload directories."""
        removed = []
        with self.tx() as db:
            now = self.clock()
            db.execute('UPDATE inputs SET state="rejected",error="validation_interrupted" WHERE state="validating" AND lease<=?', (now,))
            candidates = [('results',row['id']) for row in db.execute('SELECT id FROM jobs WHERE expires<=? AND state IN ("completed","cancelled","failed")', (now,))]
            candidates += [('inputs',row['id']) for row in db.execute('SELECT id FROM inputs WHERE expires<=? AND state NOT IN ("expired","validating") AND NOT EXISTS (SELECT 1 FROM jobs WHERE jobs.input=inputs.id AND jobs.state IN ("queued","running"))', (now,))]
            for kind, identifier in candidates:
                folder = self._folder(kind,identifier)
                require(folder.parent == self.root/kind and folder.is_relative_to(self.root), 'unsafe_retention_target', 500)
                if folder.exists():
                    for directory, directories, files in os.walk(folder, followlinks=False):
                        for name in directories+files:
                            _plain_path(Path(directory)/name)
                    shutil.rmtree(folder)
                    _sync_dir(folder.parent)
                if kind=='inputs':
                    db.execute('UPDATE inputs SET state="expired",manifest=NULL WHERE id=?', (identifier,))
                else:
                    db.execute('UPDATE jobs SET state="expired",result=NULL WHERE id=?', (identifier,))
                removed.append({'kind':kind,'id':identifier})
        return removed
