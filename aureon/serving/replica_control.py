"""Explicit local replica v2/v3; one existing jobs/idempotency/usage authority.

Host RAM and filesystem credits are shared. VRAM credits belong to a physical
device. Expired heartbeats fence work but never free a possibly live GPU owner.
SQLite is confined to one Linux host/UID/namespace; this is not a cluster store.
"""
from __future__ import annotations
from collections.abc import Callable, Mapping
from ..cloud.contracts import ControlContract, Qualification
from .contracts import (BaseLease, Readiness, ReplicaSettings, ResourceUsage,
                        ReplicaWorkerSnapshot, TenantStatus)
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import uuid
from typing import cast

from .control import Control as ServiceControl, initialize as service_initialize
from .ownership import exclusive, process_identity, alive
from . import replica_capacity as capacity
from .contracts import AttemptLease, HostIdentity, PathInput, Placement, ReplicaLimits, ReplicaTopology, WorkerBudget
from ..cloud.control import CONTRACT, require, encoded, digest, file_hash, write_file
from ..scene_api.durable import _plain_path, _sync_dir

SCHEMA = 'aureon.async_replicas.v2'
ID = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}\Z')
FIELDS = {'id','device','vram_bytes','resident_vram_bytes','resident_host_bytes',
          'transient_vram_bytes','transient_host_bytes','scratch_bytes'}
LIVE = ('starting','available','running','quarantined')


def host_identity() -> HostIdentity:
    require(os.name != 'nt', 'replicas_require_linux')
    return {'machine': Path('/etc/machine-id').read_text().strip(), 'uid': os.getuid(),
            'pid_namespace': os.readlink('/proc/self/ns/pid')}


def amount(value: object) -> int:
    require(type(value) is int and 0 <= value <= 2**50, 'explicit_finite_capacity_required')
    return cast(int, value)


def topology(value: object, fake: bool, limits: ReplicaLimits | None = None) -> ReplicaTopology:
    require(isinstance(value,dict) and set(value)=={'host_bytes','output_bytes','controller_host_bytes','parser_host_bytes','workers'}, 'invalid_replica_topology')
    value=cast(ReplicaTopology,value)  # Exact outer keys checked; all members/ranges checked below.
    for k in ('host_bytes','output_bytes','controller_host_bytes','parser_host_bytes'): amount(cast(Mapping[str, object],value)[k])
    require(value['host_bytes'] > value['controller_host_bytes'] + value['parser_host_bytes'], 'host_pool_too_small')
    workers=value['workers']
    require(isinstance(workers,list) and 1<=len(workers)<=(8 if limits is None else limits['max_workers']), 'replica_count_out_of_bounds')
    ids=set(); devices=set()
    for w in workers:
        require(isinstance(w,dict) and (set(w)==FIELDS or limits is not None and set(w)==FIELDS|{'lanes'}), 'invalid_worker_budget')
        if limits is not None:
            require(type(w.get('lanes',1)) is int and w.get('lanes',1)==1,'HTTP_parallel_not_qualified')
        require(isinstance(w['id'],str) and ID.fullmatch(w['id']) and w['id'] not in ids, 'invalid_worker_id')
        require(isinstance(w['device'],str) and w['device'] not in devices, 'unique_device_required')
        if fake: require(w['device'].startswith('fake:'), 'fake_device_required')
        else:
            from .devices import device_uuid
            require(device_uuid(w['device'])==w['device'],'canonical_gpu_uuid_required')
        for k in FIELDS-{'id','device'}: amount(cast(Mapping[str, object],w)[k])
        require(w['resident_vram_bytes']+w['transient_vram_bytes']<=w['vram_bytes'], 'device_budget_too_small')
        require(w['resident_host_bytes']+w['transient_host_bytes']>0 and w['scratch_bytes']>0, 'positive_worker_budget_required')
        ids.add(w['id']); devices.add(w['device'])
    if not fake:
        require(value['parser_host_bytes']>=4*2**30 and value['controller_host_bytes']>=128*2**20, 'parser_controller_reservation_too_small')
        require(all(w['resident_host_bytes']>0 and w['resident_vram_bytes']>0
            and w['transient_host_bytes']>=CONTRACT['max_result_bytes'] and w['transient_vram_bytes']>0
            and w['scratch_bytes']>=2*CONTRACT['max_result_bytes'] for w in workers),'real_worker_reservation_required')
    result=json.loads(encoded(value))
    if limits is not None:
        for w in result['workers']:w['lanes']=1
    return cast(ReplicaTopology,result)


def execution_binding(config: PathInput | None, fake: bool) -> dict[str, object]:
    if fake:
        require(config is None, 'fake_has_no_model_config')
        return {'mode':'fake','model_executed':False}
    from ..terramind_native.offline import inspect_config
    from .backend import installed_binding
    installed=installed_binding()
    package=Path(installed['root'])
    payload={p.relative_to(package).as_posix():file_hash(p) for p in sorted(package.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    require(0<len(payload)<=2000,'runtime_inventory_bound',409)
    config=_plain_path(cast(PathInput,config)).resolve(strict=True)
    inspected=inspect_config(config)
    value=json.loads(config.read_bytes())
    return {'config_sha256':file_hash(config), 'model':value['model'], 'tokenizer':value['tokenizer'],
            **({'model_id':inspected['model_id']} if inspected['model_id']!='terramind-v1-base' else {}),
            'dependencies':value['dependencies'], 'source_manifest':value['source_manifest'],
            'mode':'engine', 'precision':'highest-fp32',
            'runtime':{'version':installed['version'],'payload_sha256':digest(encoded(payload))}}


def initialize(root: PathInput, tenants: list[str], *, resources: ReplicaTopology,
               config: PathInput | None = None, fake: bool = False,
               limits: dict[str, int] | None = None) -> dict[str, str]:
    """Fresh state only. Never migrate an existing v1 database implicitly."""
    require(isinstance(resources,dict) and isinstance(resources.get('workers'),list),'invalid_replica_topology')
    budgets=None if limits is None else capacity.limits(limits,len(resources['workers']))
    selected=topology(resources,fake,budgets); host=host_identity(); identity=execution_binding(config,fake)
    root=_plain_path(root)
    if not fake:
        from .storage import require_local
        require_local(root.parent)
    keys=service_initialize(root,tenants,model=cast(str,identity.get('model_id','terramind-v1-base')),environment='generation',
        profile='fp32-generation-v1',fake=fake,tenant_capacity=15 if budgets is None else budgets['max_tenants'])
    contract=json.loads((root/'contract.json').read_bytes())
    contract.update(schema=SCHEMA,max_running=min(CONTRACT['max_active'],len(selected['workers'])),
        replicas={'resources':selected,'host':host,'execution':identity})
    if budgets is not None:
        contract.update(schema=capacity.SCHEMA,**{k:cast(Mapping[str,int],budgets)[k] for k in ('max_running','max_active','max_rows',
            'max_tenant_storage_reservation_bytes','max_global_storage_reservation_bytes')})
        contract['replicas'].update(limits=budgets,authority=capacity.authority(host,uuid.uuid4().hex))
    db=sqlite3.connect(root/'control.sqlite')
    db.executescript('''
      CREATE TABLE replica_workers(worker TEXT NOT NULL,generation TEXT NOT NULL,device TEXT NOT NULL,
        state TEXT NOT NULL,heartbeat REAL NOT NULL,process TEXT NOT NULL,work TEXT NOT NULL UNIQUE,
        scratch_bytes INTEGER NOT NULL,closure TEXT, PRIMARY KEY(worker,generation));
      CREATE TABLE replica_assignments(job TEXT NOT NULL,attempt INTEGER NOT NULL,worker TEXT NOT NULL,
        generation TEXT NOT NULL,device TEXT NOT NULL,state TEXT NOT NULL,
        host_bytes INTEGER NOT NULL,vram_bytes INTEGER NOT NULL,output_bytes INTEGER NOT NULL,
        PRIMARY KEY(job,attempt));
      CREATE INDEX replica_assignment_owner ON replica_assignments(worker,generation,state);
      CREATE INDEX IF NOT EXISTS replica_tenant_jobs ON jobs(tenant,state,created);
      CREATE INDEX replica_live_workers ON replica_workers(state,worker,device);
      CREATE INDEX replica_scratch_credits ON replica_workers(scratch_bytes) WHERE scratch_bytes>0;
      CREATE INDEX replica_live_credits ON replica_assignments(device,host_bytes,vram_bytes,output_bytes)
        WHERE host_bytes>0 OR vram_bytes>0 OR output_bytes>0;
    '''); db.close()
    write_file(root/'contract.replicas',encoded(contract))
    os.replace(root/'contract.replicas',root/'contract.json'); _sync_dir(root)
    return keys


class ReplicaControl(ServiceControl):
    def _require_active_authority(self) -> None:
        require(not self.meta('replica_restore_retired'), 'replica_state_retired', 409)
        require(not self.meta('replica_restore_pending'), 'replica_restore_pending', 409)

    def _admit(self, db: sqlite3.Connection) -> None:
        self._require_active_authority()
        super()._admit(db)

    def _expected_contract(self, verifier: Callable[[Qualification], Qualification] | None) -> ControlContract:
        schema=self.contract.get('schema');require(schema in (SCHEMA,capacity.SCHEMA), 'explicit_replica_state_required',409)
        extra=self.contract.get('replicas',cast(ReplicaSettings,{}))
        require(set(extra)==({'resources','host','execution'} if schema==SCHEMA else {'resources','host','execution','limits','authority'}), 'invalid_replica_contract',409)
        require(extra['host']==host_identity(), 'replica_host_namespace_mismatch',409)
        budgets=None
        if schema==capacity.SCHEMA:
            budgets=capacity.limits(extra['limits'],len(extra['resources']['workers']))
            require(budgets==extra['limits'],'canonical_replica_limits_required',409)
            identifier=extra['authority'].get('id') if isinstance(extra['authority'],dict) else None
            require(isinstance(identifier,str) and re.fullmatch('[0-9a-f]{32}',identifier),'invalid_authority_identity',409)
            require(extra['authority']==capacity.authority(extra['host'],cast(str,identifier)),'invalid_placement_authority',409)
        require(topology(extra['resources'],self.contract.get('execution')=='fake',budgets)==extra['resources'],'canonical_topology_required',409)
        expected=super()._expected_contract(verifier)
        expected.update({'schema':cast(str,schema),'replicas':extra,'max_running':min(CONTRACT['max_active'],len(extra['resources']['workers']))})
        if budgets is not None:
            cast(dict[str,object],expected).update({k:cast(Mapping[str,int],budgets)[k] for k in ('max_running','max_active','max_rows',
                'max_tenant_storage_reservation_bytes','max_global_storage_reservation_bytes')})
        return expected

    @property
    def limits(self) -> ReplicaLimits | None:return self.contract['replicas'].get('limits')

    def _tenant_storage_limit(self, value: object) -> int:
        if self.limits is None:return super()._tenant_storage_limit(value)
        require(type(value) is int and value==self.limits['max_tenant_storage_reservation_bytes'],'tenant_storage_contract_mismatch',409)
        return cast(int,value)

    def tenant_capacity(self) -> int:return 15 if self.limits is None else self.limits['max_tenants']

    def _input_capacity(self) -> dict[str, int]:
        if self.limits is None:return super()._input_capacity()
        return {'tenant_bytes':self.limits['max_tenant_storage_reservation_bytes'],
            'global_bytes':self.limits['max_global_storage_reservation_bytes'],'max_rows':self.limits['max_rows']}

    def _submission_capacity(self, db: sqlite3.Connection, tenant: str) -> None:
        if self.limits is None:return super()._submission_capacity(db,tenant)
        limits=self.limits
        require(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]<limits['max_rows'],'job_count_quota',429)
        for scope,args,prefix in [('',(),''),('tenant=? AND ',(tenant,),'tenant_')]:
            counts={r['state']:r['n'] for r in db.execute('SELECT state,COUNT(*) n FROM jobs WHERE '+scope+'state IN ("queued","running") GROUP BY state',args)}
            require(sum(counts.values())<cast(Mapping[str,int],limits)['max_'+prefix+'active'],prefix+'active_capacity',429)
            require(counts.get('queued',0)<cast(Mapping[str,int],limits)['max_'+prefix+'queued'],prefix+'queued_capacity',429)

    def _next_job(self, db: sqlite3.Connection) -> sqlite3.Row | None:
        if self.limits is None:
            return cast(sqlite3.Row | None, db.execute('SELECT jobs.* FROM jobs JOIN outbox ON jobs.id=outbox.job WHERE jobs.state="queued" AND outbox.state="pending" ORDER BY created,jobs.id LIMIT 1').fetchone())
        if db.execute('SELECT COUNT(*) FROM jobs WHERE state="running"').fetchone()[0]>=self.limits['max_running']:return None
        cursor=self.meta('replica_fair_cursor','')
        tenant=db.execute('''SELECT j.tenant FROM jobs j JOIN outbox o ON j.id=o.job
            WHERE j.state="queued" AND o.state="pending"
            AND (SELECT COUNT(*) FROM jobs r WHERE r.tenant=j.tenant AND r.state="running")<?
            GROUP BY j.tenant ORDER BY (j.tenant>?) DESC,j.tenant LIMIT 1''',(self.limits['max_tenant_running'],cursor)).fetchone()
        if tenant is None:return None
        return cast(sqlite3.Row | None, db.execute('SELECT jobs.* FROM jobs JOIN outbox ON jobs.id=outbox.job WHERE jobs.tenant=? AND jobs.state="queued" AND outbox.state="pending" ORDER BY created,jobs.id LIMIT 1',(tenant[0],)).fetchone())

    def placement(self, worker: str, generation: str) -> Placement:
        spec=self.spec(worker)
        # A lease carries a snapshot. Caller edits must not alter the expected
        # host/authority identity used later to fence publication.
        return deepcopy(cast(Placement,{'schema':'aureon.replica_placement.v1','authority':self.contract['replicas'].get('authority'),
            'host':self.contract['replicas']['host'],'state_instance':self.meta('instance_id'),
            'replica':worker,'generation':generation,'device':spec['device'],'model':self.selected['model_id'],
            'profile':self.selected['profile_id'],'execution_binding_sha256':digest(encoded(self.contract['replicas']['execution'])),
            'lanes':1,'max_running_requests':1,'local_queue_slots':0}))

    @property
    def resources(self) -> ReplicaTopology: return self.contract['replicas']['resources']

    def spec(self, worker: str) -> WorkerBudget:
        require(isinstance(worker,str) and ID.fullmatch(worker), 'invalid_worker_id')
        if not hasattr(self,'_worker_specs'):self._worker_specs={w['id']:w for w in self.resources['workers']}
        require(worker in self._worker_specs,'worker_not_configured',409)
        return self._worker_specs[worker]

    def _worker(self, db: sqlite3.Connection, worker: str, generation: str, *, caller: bool = False) -> sqlite3.Row:
        row=db.execute('SELECT * FROM replica_workers WHERE worker=? AND generation=?',(worker,generation)).fetchone()
        require(row is not None,'unknown_worker_generation',409)
        if caller:
            current=process_identity(); owner=json.loads(row['process'])
            require(current is not None and all(current.get(k)==owner.get(k) for k in ('pid','start_ticks','boot_id')), 'wrong_worker_process',409)
        return cast(sqlite3.Row,row)

    def _usage(self, db: sqlite3.Connection) -> ResourceUsage:
        host=self.resources['controller_host_bytes']+self.resources['parser_host_bytes']; devices: dict[str,int]={}
        # Retained scratch remains charged even for dead/closed generations.
        # Aggregate in SQLite instead of materializing every historical row.
        output=db.execute('SELECT COALESCE(SUM(scratch_bytes),0) FROM replica_workers WHERE scratch_bytes>0').fetchone()[0]
        for row in db.execute('SELECT worker,device,COUNT(*) n FROM replica_workers WHERE state IN ("starting","available","running","quarantined") GROUP BY worker,device'):
            spec=self.spec(row['worker']);host+=row['n']*spec['resident_host_bytes']
            devices[row['device']]=devices.get(row['device'],0)+row['n']*spec['resident_vram_bytes']
        for row in db.execute('''SELECT device,SUM(host_bytes) h,SUM(vram_bytes) v,SUM(output_bytes) o FROM replica_assignments
            WHERE host_bytes>0 OR vram_bytes>0 OR output_bytes>0 GROUP BY device'''):
            host+=row['h'];output+=row['o'];devices[row['device']]=devices.get(row['device'],0)+row['v']
        return {'host_bytes':host,'output_bytes':output,'device_vram_bytes':devices}

    def _fits(self, db: sqlite3.Connection, spec: WorkerBudget, host: int, vram: int, output: int) -> bool:
        used=self._usage(db)
        return (used['host_bytes']+host<=self.resources['host_bytes']
            and used['output_bytes']+output<=self.resources['output_bytes']
            and used['device_vram_bytes'].get(spec['device'],0)+vram<=spec['vram_bytes'])

    def register(self, worker: str, work: PathInput, *, config: PathInput | None = None) -> str:
        """Caller holds its worker/device kernel leases before reserving residency."""
        self._require_active_authority()
        spec=self.spec(worker); process=process_identity(); work=_plain_path(work).resolve(strict=True)
        require(process is not None and work.is_dir() and work!=self.root
            and not work.is_relative_to(self.root) and not self.root.is_relative_to(work),'invalid_worker_work')
        require(execution_binding(config,self.execution=='fake')==self.contract['replicas']['execution'], 'replica_execution_identity_mismatch',409)
        with self.tx() as db:
            self._require_active_authority()
            if self.limits is not None:
                require(db.execute('SELECT COUNT(*) FROM replica_workers').fetchone()[0]<self.limits['max_worker_generations'],'worker_generation_record_capacity',429)
            rows=db.execute('SELECT * FROM replica_workers WHERE state IN ("starting","available","running","quarantined")').fetchall()
            require(not any(r['worker']==worker or r['device']==spec['device'] for r in rows),'worker_or_device_already_owned',409)
            require(self._fits(db,spec,spec['resident_host_bytes'],spec['resident_vram_bytes'],spec['scratch_bytes']), 'shared_residency_capacity',429)
            generation=uuid.uuid4().hex
            db.execute('INSERT INTO replica_workers VALUES (?,?,?,?,?,?,?,?,NULL)',
                (worker,generation,spec['device'],'available',self.clock(),encoded(process).decode(),str(work),spec['scratch_bytes']))
        return generation

    def pulse(self, worker: str, generation: str) -> None:
        with self.tx() as db:
            row=self._worker(db,worker,generation,caller=True)
            require(row['state'] in ('available','running'),'worker_quarantined_or_closed',409)
            db.execute('UPDATE replica_workers SET heartbeat=? WHERE worker=? AND generation=?',(self.clock(),worker,generation))

    def claim(self, **kwargs: object) -> AttemptLease | None:
        require(False,'use_atomic_replica_claim',409)
        return None  # require(False) always raises; satisfy the explicit optional return contract.

    def claim_replica(self, worker: str, generation: str) -> AttemptLease | None:
        spec=self.spec(worker)
        with self.tx() as db:
            self._require_active_authority()
            if self.meta('drain',False): return None
            row=self._worker(db,worker,generation,caller=True)
            require(row['state'] in ('available','running') and row['heartbeat']+20>self.clock(),'stale_worker_generation',409)
            if row['state']=='running': return None
            if not self._fits(db,spec,spec['transient_host_bytes'],spec['transient_vram_bytes'],CONTRACT['max_result_bytes']): return None
            job=self._next_job(db)
            if job is None: return None
            source=db.execute('SELECT * FROM inputs WHERE id=?',(job['input'],)).fetchone()
            require(source is not None and source['state']=='ready','prepared_input_unavailable',409)
            attempt=job['attempts']+1; token=secrets.token_urlsafe(32); now=self.clock()
            require(attempt<=CONTRACT['max_attempts'],'attempt_budget_exhausted',409)
            db.execute('INSERT INTO replica_assignments VALUES (?,?,?,?,?,"active",?,?,?)',
                (job['id'],attempt,worker,generation,spec['device'],spec['transient_host_bytes'],spec['transient_vram_bytes'],CONTRACT['max_result_bytes']))
            db.execute('UPDATE jobs SET state="running",attempts=?,token=?,lease=? WHERE id=?',
                (attempt,digest(token.encode()),now+CONTRACT['lease_seconds'],job['id']))
            db.execute('UPDATE outbox SET state="claimed" WHERE job=?',(job['id'],))
            db.execute('INSERT INTO attempts VALUES (?,?,"running",?,NULL)',(job['id'],attempt,now))
            db.execute('UPDATE replica_workers SET state="running" WHERE worker=? AND generation=?',(worker,generation))
            body=json.loads(db.execute('SELECT body FROM service_requests WHERE job=?',(job['id'],)).fetchone()[0])
            lease: AttemptLease={'job_id':job['id'],'attempt':attempt,'token':token,'manifest':str(_plain_path(self.root/source['manifest'])),
                'request':body,'replica_worker':worker,'replica_generation':generation,'replica_device':spec['device']}
            if self.limits is not None:
                db.execute('INSERT INTO service_meta VALUES ("replica_fair_cursor",?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(encoded(job['tenant']).decode(),))
                lease['placement']=self.placement(worker,generation)
            return lease

    def _fenced(self, db: sqlite3.Connection, lease: BaseLease) -> sqlite3.Row:
        self._require_active_authority()
        row=super()._fenced(db,lease)
        if self.limits is not None:
            require(lease.get('placement')==self.placement(cast(str,lease.get('replica_worker')),cast(str,lease.get('replica_generation'))),'stale_placement',409)
        assigned=db.execute('SELECT * FROM replica_assignments WHERE job=? AND attempt=?',(lease['job_id'],lease['attempt'])).fetchone()
        require(assigned is not None and assigned['state']=='active' and all(lease.get('replica_'+key)==assigned[column]
            for key,column in (('worker','worker'),('generation','generation'),('device','device'))),'stale_replica_assignment',409)
        worker=self._worker(db,assigned['worker'],assigned['generation'],caller=True)
        require(worker['state']=='running' and worker['heartbeat']+20>self.clock(),'stale_replica_generation',409)
        return row

    def _terminal(self, db: sqlite3.Connection, identifier: str, state: str) -> None:
        super()._terminal(db,identifier,state)
        rows=db.execute('SELECT * FROM replica_assignments WHERE job=? AND state="active"',(identifier,)).fetchall()
        for row in rows:
            if state=='completed':
                record=json.loads(db.execute('SELECT result FROM jobs WHERE id=?',(identifier,)).fetchone()[0])
                size=sum(item['bytes'] for item in record.values())
                require(size<=row['output_bytes'],'publication_reservation_exceeded',500)
                db.execute('UPDATE replica_assignments SET state="retained",host_bytes=0,vram_bytes=0,output_bytes=? WHERE job=? AND attempt=?',(size,identifier,row['attempt']))
                db.execute('UPDATE replica_workers SET state="available" WHERE worker=? AND generation=?',(row['worker'],row['generation']))
            else:
                # An execution error does not attest that outstanding CUDA work stopped.
                db.execute('UPDATE replica_assignments SET state="quarantined" WHERE job=? AND attempt=?',(identifier,row['attempt']))
                db.execute('UPDATE replica_workers SET state="quarantined" WHERE worker=? AND generation=?',(row['worker'],row['generation']))

    def _retire(self, db: sqlite3.Connection, worker: str, generation: str, closure: Mapping[str, object]) -> None:
        now=self.clock()
        assignments=db.execute('SELECT * FROM replica_assignments WHERE worker=? AND generation=? AND state IN ("active","quarantined")',(worker,generation)).fetchall()
        for a in assignments:
            job=db.execute('SELECT * FROM jobs WHERE id=?',(a['job'],)).fetchone()
            if job['state']=='running' and job['attempts']==a['attempt']:
                terminal=job['attempts']>=CONTRACT['max_attempts']; state='failed' if terminal else 'queued'
                db.execute('UPDATE attempts SET state="lease_lost",finished=? WHERE job=? AND number=?',(now,a['job'],a['attempt']))
                db.execute('UPDATE jobs SET state=?,token=NULL,lease=NULL,error=?,expires=? WHERE id=?',
                    (state,'attempt_budget_exhausted' if terminal else None,now+CONTRACT['output_ttl_seconds'] if terminal else None,a['job']))
                db.execute('UPDATE outbox SET state=? WHERE job=?',('finished' if terminal else 'pending',a['job']))
                if terminal: super()._terminal(db,a['job'],'failed')
            # Partial durable output remains charged; scratch is separately retained.
            partial=self.root/'results'/a['job']/('attempt-'+str(a['attempt']))
            output=a['output_bytes'] if partial.exists() else 0
            db.execute('UPDATE replica_assignments SET state=?,host_bytes=0,vram_bytes=0,output_bytes=? WHERE job=? AND attempt=?',
                ('retained' if output else 'released',output,a['job'],a['attempt']))
        db.execute('UPDATE replica_workers SET state="closed",closure=? WHERE worker=? AND generation=?',(encoded(closure).decode(),worker,generation))

    def close_worker(self, worker: str, generation: str, cleanup: Mapping[str, object]) -> None:
        require(isinstance(cleanup,dict) and cleanup.get('passed') is True and
            (self.execution=='fake' or cleanup.get('cuda_allocated_bytes')==cleanup.get('cuda_reserved_bytes')==0), 'verified_cleanup_required',409)
        with self.tx() as db:
            row=self._worker(db,worker,generation,caller=True)
            require(row['state'] in LIVE,'worker_already_closed',409)
            self._retire(db,worker,generation,{'reason':'verified_owned_close','cleanup':cleanup})

    def recover(self, worker: str, generation: str) -> bool:
        """Only a dead attested process plus reacquired kernel ownership can release."""
        spec=self.spec(worker)
        with self.lock: row=self._worker(self.db,worker,generation)
        require(not alive(json.loads(row['process'])),'old_worker_still_alive',409)
        from .devices import device_lease
        with exclusive(self.root,'replica-'+worker+'.lock'), (device_lease(spec['device']) if self.execution!='fake' else nullcontext()):
            with self.tx() as db:
                current=self._worker(db,worker,generation)
                require(not alive(json.loads(current['process'])),'old_worker_still_alive',409)
                if current['state'] not in LIVE: return False
                self._retire(db,worker,generation,{'reason':'attested_process_dead_and_kernel_leases_reacquired','process':json.loads(current['process'])})
        return True

    def expire(self) -> list[dict[str, str]]:
        removed=super().expire()
        with self.tx() as db:
            for row in db.execute('SELECT * FROM replica_assignments WHERE state="retained"').fetchall():
                job=db.execute('SELECT state,expires FROM jobs WHERE id=?',(row['job'],)).fetchone()
                if job and job['expires'] is not None and job['expires']<=self.clock() and not (self.root/'results'/row['job']).exists():
                    db.execute('UPDATE replica_assignments SET output_bytes=0,state="released" WHERE job=? AND attempt=?',(row['job'],row['attempt']))
        return removed

    def worker_snapshot(self) -> dict[str, object]:
        with self.lock:
            workers=[dict(r) for r in self.db.execute('SELECT worker,generation,device,state,heartbeat,scratch_bytes FROM replica_workers')]
            used=self._usage(self.db)
        value: dict[str,object]={'schema':'aureon.replica_snapshot.v2','workers':workers,'reservations':used,
            'capacities':self.resources,'shared_host_and_output_pool':True,'distributed_dispatch':False,
            'physical_multi_gpu_measured':False,'execution_binding':self.contract['replicas']['execution']}
        if self.limits is not None:
            with self.lock:
                counts={r['state']:r['n'] for r in self.db.execute('SELECT state,COUNT(*) n FROM jobs WHERE state IN ("queued","running") GROUP BY state')}
            slots=0 if self.meta('drain',False) else max(0,min(self.limits['max_active']-sum(counts.values()),self.limits['max_queued']-counts.get('queued',0)))
            value.update(schema='aureon.replica_snapshot.v3',capacity=capacity.policy(self.limits),counts=counts,
                queued_hard_bound=self.limits['max_active'],
                new_submission_slots=slots,admission_slots=slots,accepts_queued_without_live_worker=True,
                authority=self.contract['replicas']['authority'],
                placements=[self.placement(w['worker'],w['generation']) for w in workers])
        return deepcopy(value)

    def capabilities(self) -> dict[str, object]:
        value=super().capabilities()
        value.update(max_running=self.contract['max_running'],max_active=self.contract['max_active'],replica_state_schema=self.contract['schema'],
            replica_execution='one serial B1 engine per physical GPU process',batch2=False)
        cast(dict[str,object],value['stages']).update(gpu_workers=len(self.resources['workers']))
        if self.limits is not None:
            authority=self.contract['replicas']['authority']
            value.update(capacity=capacity.policy(self.limits),placement_authority={k:cast(Mapping[str,object],authority)[k] for k in
                ('schema','kind','transaction','multi_node_enabled','external_authority_required','external_adapter_installed')})
        return deepcopy(value)

    def readiness(self) -> Readiness:
        health=self.health(); snapshot=self.worker_snapshot(); drain=self.meta('drain',False)
        live=[w for w in cast(list[ReplicaWorkerSnapshot],snapshot['workers']) if w['state'] in ('available','running') and w['heartbeat']+20>self.clock()]
        ready=bool(health['control_plane']['ready'] and live and not drain
                   and not self.meta('replica_restore_retired') and not self.meta('replica_restore_pending'))
        return {'schema':'aureon.replica_readiness.v2' if self.limits is None else 'aureon.replica_readiness.v3','ready':ready,'draining':drain,
            'worker':{'state':'available' if ready else 'unavailable','model_loaded':False,'replicas_available':len(live)},
            'lazy_model_start':True,'capabilities':self.capabilities()}

    def status_snapshot(self, tenant: str) -> dict[str, object]:
        value=super().status_snapshot(tenant)
        if self.limits is not None:
            limits=self.limits;status=cast(TenantStatus,value);counts=status['counts']['jobs'];active=counts.get('queued',0)+counts.get('running',0)
            status['queue'].update(max_active=limits['max_tenant_active'],max_running=limits['max_tenant_running'],
                max_queued=limits['max_tenant_queued'],queued_hard_bound=limits['max_tenant_active'],
                tenant_admission_slots=max(0,min(limits['max_tenant_active']-active,limits['max_tenant_queued']-counts.get('queued',0))))
            for reason in status['admission']['reasons']:
                if reason['code']=='own_job_lease_expired':reason['action']='Operator must attest dead ownership and explicitly recover; expiry alone never releases a replica reservation.'
        return value
