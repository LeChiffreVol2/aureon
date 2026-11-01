"""Real SQLite/process races, fake devices/execution. No model/GPU imports."""
import aureon
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import unittest
import urllib.request
import uuid

from aureon.cloud.control import CONTRACT, Error, encoded, write_file
from aureon.serving.control import Control, initialize as initialize_one
from aureon.serving.replica_control import ReplicaControl, initialize, topology
from aureon.serving.ownership import exclusive

CTX=multiprocessing.get_context('spawn')
MIB=2**20


def capacity(host=2000*MIB, output=1000*MIB):
    return {'host_bytes':host,'output_bytes':output,'controller_host_bytes':10*MIB,'parser_host_bytes':100*MIB,
        'workers':[{'id':'w'+str(i),'device':'fake:'+str(i),'vram_bytes':1000*MIB,
            'resident_vram_bytes':100*MIB,'resident_host_bytes':100*MIB,
            'transient_vram_bytes':100*MIB,'transient_host_bytes':200*MIB,'scratch_bytes':128*MIB} for i in range(2)]}


def actor(state,worker,work,connection,barrier):
    """A real separately attested owner process, driven by a bounded test pipe."""
    control=ReplicaControl(state); work=Path(work); work.mkdir()
    with exclusive(state,'replica-'+worker+'.lock'):
        generation=control.register(worker,work); connection.send({'generation':generation})
        try:
            while connection.poll(15):
                command=connection.recv(); operation=command[0]
                try:
                    if operation=='claim-race':
                        barrier.wait(5); value=control.claim_replica(worker,generation)
                    elif operation=='claim': value=control.claim_replica(worker,generation)
                    elif operation=='publish': value=control.publish_fake(command[1])
                    elif operation=='publish-rollback':
                        terminal=control._terminal
                        def fault(db,job,state):
                            terminal(db,job,state)
                            raise Error(500,'injected_commit_fault','test fault')
                        control._terminal=fault
                        try: value=control.publish_fake(command[1])
                        finally: control._terminal=terminal
                    elif operation=='fail': value=control.fail(command[1])
                    elif operation=='pulse': value=control.pulse(worker,generation)
                    elif operation=='cancel': value=control.cancel('alpha',command[1])
                    elif operation=='close':
                        control.close_worker(worker,generation,{'passed':True,'fake':True}); connection.send({'ok':True}); break
                    else: raise ValueError(operation)
                    connection.send({'ok':True,'value':value})
                except Error as exc: connection.send({'ok':False,'code':exc.code})
        finally: control.close(); connection.close()


@unittest.skipUnless(os.name!='nt','Linux process identity/kernel locks required')
class ReplicaTests(unittest.TestCase):
    def setUp(self):
        root=Path(os.environ.get('AUREON_REPLICA_TEST_ROOT','/tmp/aureon-replica-tests'))
        root.mkdir(parents=True,exist_ok=True)
        self.root=root/(self._testMethodName+'-'+uuid.uuid4().hex); self.root.mkdir()
        self.state=self.root/'state'; self.control=None; self.actors=[]
        self.barrier=CTX.Barrier(2)

    def tearDown(self):
        for p,pipe,_ in self.actors:
            if p.is_alive():
                try:
                    pipe.send(('close',))
                    if pipe.poll(2): pipe.recv()
                except (EOFError,BrokenPipeError): pass
                p.join(2)
            if p.is_alive(): p.kill(); p.join(2)
            pipe.close()
        if self.control is not None: self.control.close()
        # Preserve all fixture scratch; no cleanup of existing work/evidence.

    def setup_state(self,resources=None):
        self.keys=initialize(self.state,['alpha'],resources=resources or capacity(),fake=True)
        self.control=ReplicaControl(self.state)
        source=uuid.uuid4().hex; folder=self.state/'inputs'/source; folder.mkdir()
        write_file(folder/'manifest.json',encoded({'schema':'explicit_fake_test_fixture','model_executed':False}))
        with self.control.tx() as db:
            db.execute('INSERT INTO inputs(id,tenant,state,declaration,bytes,created,expires,manifest) VALUES (?,?,"ready",?,0,?,?,?)',
                (source,'alpha',encoded({'fixture':True}).decode(),time.time(),time.time()+3600,str((folder/'manifest.json').relative_to(self.state))))
        self.body={'input_id':source,'model_id':'terramind-v1-base','profile_id':'fp32-generation-v1','task':'generation','parameters':{'seed':17}}
        return self.control

    def start(self,worker):
        parent,child=CTX.Pipe(); work=self.root/(worker+'-'+uuid.uuid4().hex)
        p=CTX.Process(target=actor,args=(str(self.state),worker,str(work),child,self.barrier)); p.start(); child.close()
        self.assertTrue(parent.poll(10),'worker registration deadline'); generation=parent.recv()['generation']
        item=(p,parent,generation); self.actors.append(item); return item

    def call(self,item,*command):
        item[1].send(command); self.assertTrue(item[1].poll(10),'bounded worker reply'); return item[1].recv()

    def submit(self,key): return self.control.submit('alpha',self.body,key)

    def race(self,a,b):
        a[1].send(('claim-race',)); b[1].send(('claim-race',))
        self.assertTrue(a[1].poll(10)); self.assertTrue(b[1].poll(10))
        return a[1].recv(),b[1].recv()

    def test_null_or_nonfinite_capacity_rejected_before_state_creation(self):
        for bad in (None,True,float('inf'),-1):
            x=capacity(); x['host_bytes']=bad
            with self.assertRaises(Error): topology(x,True)
        self.assertFalse(self.state.exists())

    def test_explicit_new_schema_never_modifies_old_database(self):
        initialize_one(self.state,['alpha'],model='terramind-v1-base',environment='generation',profile='fp32-generation-v1',fake=True)
        before=hashlib.sha256((self.state/'control.sqlite').read_bytes()).hexdigest()
        with self.assertRaises(Error): ReplicaControl(self.state)
        self.assertEqual(before,hashlib.sha256((self.state/'control.sqlite').read_bytes()).hexdigest())
        with self.assertRaises(Error): initialize(self.state,['alpha'],resources=capacity(),fake=True)

    def test_two_processes_claim_one_job_exactly_once(self):
        self.setup_state(); self.submit('single-job'); a=self.start('w0'); b=self.start('w1')
        replies=self.race(a,b); self.assertTrue(all(r['ok'] for r in replies))
        self.assertEqual(sum(r['value'] is not None for r in replies),1)
        self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],1)

    def test_distinct_devices_really_execute_two_concurrent_assignments(self):
        self.setup_state(); self.submit('parallel-one'); self.submit('parallel-two')
        self.assertEqual(self.control.contract['max_running'],2)
        self.assertEqual(self.control.capabilities()['max_running'],2)
        a=self.start('w0'); b=self.start('w1'); replies=self.race(a,b)
        leases=[r['value'] for r in replies]; self.assertTrue(all(leases))
        self.assertEqual(len({x['job_id'] for x in leases}),2)
        self.assertEqual(len({x['replica_device'] for x in leases}),2)
        self.assertEqual(self.control.worker_snapshot()['reservations']['host_bytes'],710*MIB)
        for owner,lease in zip((a,b),leases): self.assertTrue(self.call(owner,'publish',lease)['ok'])
        self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0],2)

    def test_shared_host_race_only_one_transient_reservation_fits(self):
        self.setup_state(capacity(host=600*MIB)); self.submit('host-one'); self.submit('host-two')
        a=self.start('w0'); b=self.start('w1'); replies=self.race(a,b)
        self.assertEqual(sum(r['value'] is not None for r in replies),1)
        self.assertEqual(self.control.worker_snapshot()['reservations']['host_bytes'],510*MIB)

    def test_shared_output_race_only_one_assignment_fits(self):
        self.setup_state(capacity(output=350*MIB)); self.submit('output-one'); self.submit('output-two')
        a=self.start('w0'); b=self.start('w1'); replies=self.race(a,b)
        self.assertEqual(sum(r['value'] is not None for r in replies),1)
        self.assertEqual(self.control.worker_snapshot()['reservations']['output_bytes'],320*MIB)

    def test_different_worker_cannot_publish_foreign_assignment(self):
        self.setup_state(); self.submit('fence-publish'); a=self.start('w0'); b=self.start('w1')
        lease=self.call(a,'claim')['value']
        self.assertFalse(self.call(b,'publish',lease)['ok'])
        wrong=dict(lease,replica_generation=b[2]); self.assertFalse(self.call(a,'publish',wrong)['ok'])
        wrong=dict(lease,replica_device='fake:1'); self.assertFalse(self.call(a,'publish',wrong)['ok'])
        self.assertTrue(self.call(a,'publish',lease)['ok'])
        self.assertFalse(self.call(a,'publish',lease)['ok'])
        self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0],1)

    def test_lease_expiry_keeps_live_owner_ram_vram_and_output(self):
        self.setup_state(); self.submit('expired-lease'); a=self.start('w0')
        lease=self.call(a,'claim')['value']; before=self.control.worker_snapshot()['reservations']
        with self.control.tx() as db: db.execute('UPDATE jobs SET lease=0 WHERE id=?',(lease['job_id'],))
        self.assertFalse(self.call(a,'publish',lease)['ok'])
        with self.assertRaises(Error) as caught: self.control.recover('w0',a[2])
        self.assertEqual(caught.exception.code,'old_worker_still_alive')
        self.assertEqual(self.control.worker_snapshot()['reservations'],before)

    def test_sigkill_recovery_fences_old_attempt_and_settles_once(self):
        self.setup_state(); original=self.submit('crash-and-retry'); a=self.start('w0')
        lease=self.call(a,'claim')['value']; a[0].kill(); a[0].join(5)
        self.assertTrue(self.control.recover('w0',a[2]))
        self.assertEqual(self.control.job('alpha',original['id'])['state'],'queued')
        replacement=self.start('w0'); new=self.call(replacement,'claim')['value']
        self.assertEqual(new['attempt'],2); self.assertEqual(new['job_id'],lease['job_id'])
        self.assertFalse(self.call(replacement,'publish',lease)['ok'])
        self.assertTrue(self.call(replacement,'publish',new)['ok'])
        replay=self.submit('crash-and-retry'); self.assertTrue(replay['reused']); self.assertEqual(replay['attempts'],2)
        self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0],1)
        self.assertEqual(self.control.db.execute('SELECT state FROM service_usage').fetchone()[0],'settled')

    def test_cancel_claim_race_has_no_cancelled_execution(self):
        self.setup_state(); job=self.submit('cancel-race'); a=self.start('w0')
        a[1].send(('claim',))
        try: self.control.cancel('alpha',job['id']); cancelled=True
        except Error: cancelled=False
        self.assertTrue(a[1].poll(10)); lease=a[1].recv()['value']
        self.assertEqual(cancelled,lease is None)
        if lease: self.assertTrue(self.call(a,'publish',lease)['ok'])
        else: self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],0)

    def test_failure_quarantines_until_verified_close(self):
        self.setup_state(); self.submit('failure-retention'); a=self.start('w0')
        lease=self.call(a,'claim')['value']; before=self.control.worker_snapshot()['reservations']
        self.assertTrue(self.call(a,'fail',lease)['ok'])
        self.assertEqual(self.control.worker_snapshot()['reservations'],before)
        self.assertFalse(self.call(a,'claim')['ok'])
        self.assertTrue(self.call(a,'close')['ok']); a[0].join(5)
        used=self.control.worker_snapshot()['reservations']
        self.assertEqual(used['host_bytes'],110*MIB); self.assertEqual(used['output_bytes'],128*MIB)

    def test_completed_output_credits_survive_worker_close(self):
        self.setup_state(); self.submit('retained-output'); a=self.start('w0')
        lease=self.call(a,'claim')['value']; self.assertTrue(self.call(a,'publish',lease)['ok'])
        used=self.control.worker_snapshot()['reservations']; self.assertGreater(used['output_bytes'],128*MIB)
        self.assertTrue(self.call(a,'close')['ok']); a[0].join(5)
        closed=self.control.worker_snapshot()['reservations']
        self.assertEqual(closed['output_bytes'],used['output_bytes'])
        self.assertEqual(closed['host_bytes'],110*MIB)

    def test_publish_rollback_keeps_reservation_and_replay_settles_once(self):
        self.setup_state(); self.submit('publish-rollback'); a=self.start('w0')
        lease=self.call(a,'claim')['value']; before=self.control.worker_snapshot()['reservations']
        self.assertFalse(self.call(a,'publish-rollback',lease)['ok'])
        self.assertEqual(self.control.worker_snapshot()['reservations'],before)
        self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0],0)
        self.assertTrue((self.state/'results'/lease['job_id']/'attempt-1'/'simulation.json').is_file())
        self.assertTrue(self.call(a,'publish',lease)['ok'])
        self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0],1)

    def test_real_worker_cli_fake_execution_and_shutdown(self):
        self.setup_state(); job=self.submit('actual-worker-cli')
        env={**os.environ,'PYTHONPATH':str(Path(aureon.__file__).resolve().parents[1]),'PYTHONDONTWRITEBYTECODE':'1'}
        result=subprocess.run([sys.executable,'-B','-m','aureon.serving','replicas','worker',
            '--state',str(self.state),'--worker','w0','--work',str(self.root/'cli-work'),
            '--fake','--max-jobs','1','--max-seconds','5'],env=env,cwd=str(self.root),capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        receipt=json.loads(result.stdout); self.assertTrue(receipt['passed']); self.assertEqual(receipt['completed'],1)
        self.assertEqual(self.control.job('alpha',job['id'])['state'],'completed')
        self.assertEqual(self.control.worker_snapshot()['workers'][0]['state'],'closed')
        self.assertEqual(self.control.worker_snapshot()['reservations']['host_bytes'],110*MIB)

    def test_http_uses_same_authority_idempotency_and_operator_registry(self):
        self.setup_state(); from aureon.cloud.http import Server
        server=Server(self.control,0); thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05}); thread.start()
        origin='http://127.0.0.1:'+str(server.server_port)
        def request(path,body=None,key=None,tenant='alpha'):
            headers={'Authorization':'Bearer '+self.keys[tenant]}
            if key: headers['Idempotency-Key']=key
            if body is not None: headers['Content-Type']='application/json'
            with urllib.request.urlopen(urllib.request.Request(origin+path,data=None if body is None else encoded(body),headers=headers),timeout=5) as response:
                return json.load(response)
        try:
            a=request('/v1/jobs',self.body,'http-replay'); b=request('/v1/jobs',self.body,'http-replay')
            self.assertEqual(a['id'],b['id']); self.assertTrue(b['reused'])
            owner=self.start('w0'); lease=self.call(owner,'claim')['value']
            self.assertTrue(self.call(owner,'publish',lease)['ok'])
            artifacts=request('/v1/jobs/'+a['id']+'/artifacts')
            simulation=request(artifacts['artifacts']['simulation.json']['url'])
            self.assertFalse(simulation['model_executed'])
            self.assertTrue(request('/v1/jobs',self.body,'http-replay')['reused'])
            self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0],1)
            registry=request('/v1/worker',tenant='operator'); self.assertTrue(registry['shared_host_and_output_pool'])
            with self.assertRaises(urllib.error.HTTPError) as caught: request('/v1/worker')
            self.assertEqual(caught.exception.code,403)
        finally: server.shutdown(); thread.join(); server.server_close()

    def test_legacy_single_worker_control_still_accepts_its_own_schema(self):
        initialize_one(self.state,['alpha'],model='terramind-v1-base',environment='generation',profile='fp32-generation-v1',fake=True)
        c=Control(self.state)
        try: self.assertEqual(c.capabilities()['max_running'],1)
        finally: c.close()


if __name__=='__main__': unittest.main()
