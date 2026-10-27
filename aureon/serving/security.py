"""Bounded local key operations and request admission. No identity provider calls."""
import secrets
import time
import os
from pathlib import Path
from ..cloud.control import require,encoded,digest,NAME,SHA,CONTRACT
from ..scene_api.durable import _plain_path,_sync_dir

POLICY={'request_burst':256,'requests_per_second':32,'active_keys_per_tenant':8,'key_records':10000,
        'identity_provider':'unconfigured','public_signup':'requires_verified_provider_adapter','billing':'local_nonbillable'}

def write_secret(path,value):
    path=_plain_path(path)
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'wb') as stream:stream.write(encoded(value));stream.flush();os.fsync(stream.fileno())
    _sync_dir(path.parent)

def schema(control):
    with control.lock:
        control.db.executescript('''
          CREATE TABLE IF NOT EXISTS service_keys(hash TEXT PRIMARY KEY,tenant TEXT NOT NULL,created REAL NOT NULL,revoked REAL);
          CREATE TABLE IF NOT EXISTS service_gc(kind TEXT NOT NULL,id TEXT NOT NULL,state TEXT NOT NULL,receipt TEXT,PRIMARY KEY(kind,id));
          CREATE TABLE IF NOT EXISTS service_scratch(id TEXT PRIMARY KEY,path TEXT UNIQUE NOT NULL,owner TEXT NOT NULL,state TEXT NOT NULL,receipt TEXT);
          CREATE TABLE IF NOT EXISTS service_usage(job TEXT PRIMARY KEY,tenant TEXT NOT NULL,state TEXT NOT NULL,units INTEGER NOT NULL,billable INTEGER NOT NULL CHECK(billable=0));
          CREATE TABLE IF NOT EXISTS service_payment_events(provider TEXT NOT NULL,id TEXT NOT NULL,digest TEXT NOT NULL,body TEXT NOT NULL,PRIMARY KEY(provider,id));
          CREATE TABLE IF NOT EXISTS service_entitlements(tenant TEXT PRIMARY KEY,units INTEGER NOT NULL CHECK(units>=0),mode TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS service_identities(provider TEXT NOT NULL,subject TEXT NOT NULL,tenant TEXT UNIQUE NOT NULL,PRIMARY KEY(provider,subject));
          CREATE TABLE IF NOT EXISTS service_payment_sequences(tenant TEXT PRIMARY KEY,version INTEGER NOT NULL);
        ''')
        control.db.execute('INSERT OR IGNORE INTO service_keys SELECT hash,tenant,0,NULL FROM credentials')
        control.db.execute('INSERT OR IGNORE INTO service_meta VALUES (?,?)',('instance_id',encoded(secrets.token_hex(16)).decode()))
        control.db.commit()

def admit_request(control,tenant):
    now=time.monotonic()
    with control.lock:
        bucket=control._rate.get(tenant,(POLICY['request_burst'],now))
        available=min(POLICY['request_burst'],bucket[0]+max(0,now-bucket[1])*POLICY['requests_per_second'])
        require(available>=1,'request_rate_exceeded',429)
        control._rate[tenant]=(available-1,now)

def key_list(control,tenant):
    with control.lock:
        rows=control.db.execute('SELECT hash AS fingerprint,created FROM service_keys WHERE tenant=? AND revoked IS NULL ORDER BY created',(tenant,)).fetchall()
    return {'keys':[dict(r) for r in rows],'rotation':'Create a replacement, authenticate it, then revoke the old fingerprint. Lost issuance responses do not revoke the working key.'}

def create_key(control,tenant):
    require(isinstance(tenant,str) and NAME.fullmatch(tenant),'invalid_tenant')
    key=secrets.token_urlsafe(32);fingerprint=digest(key.encode())
    with control.tx() as db:
        require(db.execute('SELECT 1 FROM service_keys WHERE tenant=?',(tenant,)).fetchone() is not None,'unknown_tenant',404)
        require(db.execute('SELECT COUNT(*) FROM service_keys').fetchone()[0]<POLICY['key_records'],'key_record_quota',429)
        require(db.execute('SELECT COUNT(*) FROM credentials WHERE tenant=?',(tenant,)).fetchone()[0]<POLICY['active_keys_per_tenant'],'active_key_quota',429)
        db.execute('INSERT INTO credentials VALUES (?,?)',(fingerprint,tenant))
        db.execute('INSERT INTO service_keys VALUES (?,?,?,NULL)',(fingerprint,tenant,control.clock()))
    return {'tenant':tenant,'fingerprint':fingerprint,'key':key,'shown_once':True,'revoked_existing_keys':False}

def revoke_key(control,tenant,fingerprint):
    require(isinstance(fingerprint,str) and SHA.fullmatch(fingerprint),'invalid_key_fingerprint')
    with control.tx() as db:
        row=db.execute('SELECT * FROM service_keys WHERE hash=? AND tenant=?',(fingerprint,tenant)).fetchone()
        require(row is not None,'not_found',404)
        if row['revoked'] is None:
            if tenant=='operator':require(db.execute('SELECT COUNT(*) FROM credentials WHERE tenant="operator"').fetchone()[0]>1,'last_operator_key',409)
            db.execute('DELETE FROM credentials WHERE hash=?',(fingerprint,))
            db.execute('UPDATE service_keys SET revoked=? WHERE hash=?',(control.clock(),fingerprint))
    return {'fingerprint':fingerprint,'revoked':True,'in_flight_requests':'Already authenticated requests may finish; subsequent requests reject this key.'}

def reserve(control,db,job):
    row=db.execute('SELECT tenant FROM jobs WHERE id=?',(job,)).fetchone();tenant=row['tenant']
    entitlement=db.execute('SELECT * FROM service_entitlements WHERE tenant=?',(tenant,)).fetchone()
    if entitlement and entitlement['mode']=='test_required':
        pending=db.execute('SELECT COALESCE(SUM(units),0) FROM service_usage WHERE tenant=? AND state="reserved"',(tenant,)).fetchone()[0]
        require(entitlement['units']>pending,'entitlement_exhausted',402)
    db.execute('INSERT INTO service_usage VALUES (?,?,"reserved",1,0)',(job,tenant))

def settle(db,job,state):
    row=db.execute('SELECT * FROM service_usage WHERE job=?',(job,)).fetchone()
    if row is None or row['state']!='reserved':return
    if state=='completed':
        db.execute('UPDATE service_entitlements SET units=MAX(0,units-?) WHERE tenant=? AND mode="test_required"',(row['units'],row['tenant']))
    db.execute('UPDATE service_usage SET state=? WHERE job=?',('settled' if state=='completed' else 'released',job))

def shadow_statement(control,tenant):
    with control.lock:
        rows=control.db.execute('SELECT state,COUNT(*) AS logical_jobs,SUM(units) AS units FROM service_usage WHERE tenant=? GROUP BY state',(tenant,)).fetchall()
    return {'schema':'aureon.shadow_usage.v1','tenant':tenant,'rows':[dict(r) for r in rows],'billable':False,
      'amount':None,'currency':None,'pricing':'unconfigured','failure_policy':'Failed or cancelled jobs release the reservation; internal attempts do not add a logical charge.',
      'refund_policy':'No real charges/refunds implemented; a later provider adapter must reconcile separately idempotent settlement/refund events.'}

def test_payment_event(control,provider,event_id,body):
    """Local test adapter boundary only, deliberately absent from public HTTP routes."""
    require(provider=='local-test','payment_provider_unconfigured',503)
    require(isinstance(event_id,str) and NAME.fullmatch(event_id) and isinstance(body,dict),'invalid_test_event')
    require(set(body)=={'tenant','units','kind','sequence'} and body['kind']=='test_entitlement_snapshot'
      and type(body['units']) is int and 0<=body['units']<=10000 and type(body['sequence']) is int and 0<=body['sequence']<2**31,'invalid_test_event')
    tenant=body['tenant'];fingerprint=digest(encoded(body))
    with control.tx() as db:
        require(db.execute('SELECT 1 FROM credentials WHERE tenant=?',(tenant,)).fetchone() is not None and tenant!='operator','unknown_tenant',404)
        old=db.execute('SELECT digest FROM service_payment_events WHERE provider=? AND id=?',(provider,event_id)).fetchone()
        if old:require(old['digest']==fingerprint,'payment_event_conflict',409);return {'reused':True,'real_payment':False}
        previous=db.execute('SELECT version FROM service_payment_sequences WHERE tenant=?',(tenant,)).fetchone()
        require(previous is None or body['sequence']>previous[0],'stale_payment_event',409)
        require(db.execute('SELECT COUNT(*) FROM service_payment_events').fetchone()[0]<CONTRACT['max_rows'],'payment_event_quota',429)
        db.execute('INSERT INTO service_payment_events VALUES (?,?,?,?)',(provider,event_id,fingerprint,encoded(body).decode()))
        db.execute('INSERT INTO service_entitlements VALUES (?,? ,"test_required") ON CONFLICT(tenant) DO UPDATE SET units=excluded.units,mode=excluded.mode',(tenant,body['units']))
        db.execute('INSERT INTO service_payment_sequences VALUES (?,?) ON CONFLICT(tenant) DO UPDATE SET version=excluded.version',(tenant,body['sequence']))
    return {'reused':False,'real_payment':False,'credits_issued':False,'scope':'Synthetic admission-capacity units only; not money, free credits or verified payment events'}

def test_identity(control,provider,subject):
    """Exercise the future verified-identity adapter seam; no public signup handler."""
    require(provider=='local-test','identity_provider_unconfigured',503)
    require(isinstance(subject,str) and 1<=len(subject)<=256,'invalid_identity_subject')
    subject_hash=digest(subject.encode());tenant='customer-'+subject_hash[:16]
    key=secrets.token_urlsafe(32);fingerprint=digest(key.encode())
    with control.tx() as db:
        old=db.execute('SELECT tenant FROM service_identities WHERE provider=? AND subject=?',(provider,subject_hash)).fetchone()
        if old:return {'tenant':old['tenant'],'reused':True,'identity_verified':False,'key':None}
        require(db.execute('SELECT COUNT(DISTINCT tenant) FROM service_keys').fetchone()[0]<16,'tenant_capacity',429)
        db.execute('INSERT INTO service_identities VALUES (?,?,?)',(provider,subject_hash,tenant))
        db.execute('INSERT INTO credentials VALUES (?,?)',(fingerprint,tenant))
        db.execute('INSERT INTO service_keys VALUES (?,?,?,NULL)',(fingerprint,tenant,control.clock()))
        # No implicit GPU entitlement, credit grant or payment from registration.
        db.execute('INSERT INTO service_entitlements VALUES (?,0,"test_required")',(tenant,))
    return {'tenant':tenant,'reused':False,'identity_verified':False,'key':key,'fingerprint':fingerprint,'real_payment':False}
