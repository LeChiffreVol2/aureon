"""Operator capacities and real-process claims; fake payloads/devices only."""
import hashlib,json,os,time,unittest,uuid,subprocess,sys
from types import SimpleNamespace
from unittest.mock import patch
import aureon
from pathlib import Path
import test_replica_service as legacy
from aureon.serving.replica_control import ReplicaControl,initialize,topology
from aureon.serving.replica_capacity import limits
from aureon.cloud.control import Error,encoded,write_file

def resources(n=12,**kw):
    value=legacy.capacity(host=kw.get('host',5000*legacy.MIB),output=kw.get('output',5000*legacy.MIB))
    template=value['workers'][0]
    value['workers']=[dict(template,id='w'+str(i),device='fake:'+str(i),lanes=1) for i in range(n)]
    return value

@unittest.skipUnless(os.name!='nt','Linux process identities and kernel leases required')
class CapacityTests(unittest.TestCase):
    setUp=legacy.ReplicaTests.setUp
    tearDown=legacy.ReplicaTests.tearDown
    start=legacy.ReplicaTests.start
    call=legacy.ReplicaTests.call

    def configure(self,n=12,tenants=('alpha',),budgets=None,resource=None):
        self.budgets={'max_workers':max(12,n),'max_active':12,'max_running':min(4,n),'max_queued':12,**(budgets or {})}
        self.keys=initialize(self.state,list(tenants),resources=resource or resources(n),limits=self.budgets,fake=True)
        self.control=ReplicaControl(self.state);self.bodies={}
        for tenant in tenants:
            source=uuid.uuid4().hex;folder=self.state/'inputs'/source;folder.mkdir()
            write_file(folder/'manifest.json',encoded({'fixture':True,'model_executed':False}))
            with self.control.tx() as db:
                db.execute('INSERT INTO inputs(id,tenant,state,declaration,bytes,created,expires,manifest) VALUES (?,? ,"ready",?,0,?,?,?)',
                    (source,tenant,encoded({'fixture':True}).decode(),time.time(),time.time()+3600,str((folder/'manifest.json').relative_to(self.state))))
            self.bodies[tenant]={'input_id':source,'model_id':'terramind-v1-base','profile_id':'fp32-generation-v1','task':'generation','parameters':{'seed':17}}
        return self.control

    def submit(self,key,tenant='alpha'):
        return self.control.submit(tenant,self.bodies[tenant],key)

    def claims(self,actors):
        for a in actors:a[1].send(('claim-race',))
        values=[]
        for a in actors:
            self.assertTrue(a[1].poll(10));value=a[1].recv();self.assertTrue(value['ok']);values.append(value['value'])
        return values

    def test_twelve_worker_topology_four_simultaneous_assignments_and_global_bound(self):
        self.configure();self.barrier=legacy.CTX.Barrier(4)
        for i in range(8):self.submit('capacity-job-'+str(i))
        actors=[self.start('w'+str(i)) for i in range(4)];leases=self.claims(actors)
        self.assertTrue(all(leases));self.assertEqual(len({l['job_id'] for l in leases}),4)
        other=self.start('w4');self.assertIsNone(self.call(other,'claim')['value'])
        snapshot=self.control.worker_snapshot();self.assertEqual(snapshot['counts'],{'queued':4,'running':4})
        self.assertEqual(len(snapshot['capacities']['workers']),12)
        self.assertEqual(snapshot['capacity']['limits']['max_running'],4)
        self.assertFalse(snapshot['authority']['multi_node_enabled'])
        for actor,lease in zip(actors,leases):self.assertTrue(self.call(actor,'publish',lease)['ok'])
        self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0],4)

    def test_legacy_defaults_and_existing_v2_contract_are_unchanged(self):
        old=legacy.capacity();initialize(self.state,['alpha'],resources=old,fake=True)
        before=(self.state/'contract.json').read_bytes();self.control=ReplicaControl(self.state)
        self.assertEqual(self.control.contract['schema'],'aureon.async_replicas.v2');self.assertIsNone(self.control.limits)
        self.assertEqual(self.control.contract['max_active'],3);self.assertEqual(self.control.contract['max_running'],2)
        self.assertEqual(before,(self.state/'contract.json').read_bytes())
        with self.assertRaises(Error):initialize(self.state,['alpha'],resources=old,limits={},fake=True)
        with self.assertRaises(Error):topology(resources(12),True)

    def test_invalid_capacity_or_HTTP_parallel_rejected_before_state(self):
        for invalid in ({'max_active':None},{'max_workers':True},{'max_queued':float('inf')},{'max_running':0},
                        {'max_workers':11},{'max_running':13},{'max_tenant_active':13},{'unknown':1}):
            with self.assertRaises(Error):initialize(self.state,['alpha'],resources=resources(),limits=invalid,fake=True)
            self.assertFalse(self.state.exists())
        value=resources();value['workers'][0]['lanes']=3
        with self.assertRaises(Error) as exc:initialize(self.state,['alpha'],resources=value,limits={'max_workers':12},fake=True)
        self.assertEqual(exc.exception.code,'HTTP_parallel_not_qualified');self.assertFalse(self.state.exists())

    def test_new_queue_admission_is_independent_of_running(self):
        self.configure(n=2,budgets={'max_active':4,'max_running':2,'max_queued':1})
        self.submit('queue-first');a=self.start('w0');first=self.call(a,'claim')['value']
        self.submit('queue-second')
        with self.assertRaises(Error) as exc:self.submit('queue-third')
        self.assertEqual(exc.exception.code,'queued_capacity')
        b=self.start('w1');second=self.call(b,'claim')['value'];self.assertIsNotNone(second)
        self.submit('queue-third');self.assertEqual(self.control.worker_snapshot()['counts'],{'queued':1,'running':2})
        self.assertTrue(self.call(a,'publish',first)['ok']);self.assertTrue(self.call(b,'publish',second)['ok'])

    def test_tenant_admission_cancel_replay_and_running_quota(self):
        self.configure(n=3,tenants=('alpha','beta'),budgets={'max_active':6,'max_running':3,'max_queued':6,
            'max_tenant_active':2,'max_tenant_queued':2,'max_tenant_running':1})
        first=self.submit('tenant-alpha-1');second=self.submit('tenant-alpha-2')
        with self.assertRaises(Error) as exc:self.submit('tenant-alpha-3')
        self.assertEqual(exc.exception.code,'tenant_active_capacity')
        self.assertTrue(self.submit('tenant-alpha-1')['reused'])
        beta=self.submit('tenant-beta-1','beta')
        actors=[self.start('w'+str(i)) for i in range(3)]
        leases=[self.call(a,'claim')['value'] for a in actors]
        self.assertEqual([l['job_id'] if l else None for l in leases],[first['id'],beta['id'],None])
        self.control.cancel('alpha',second['id']);self.submit('tenant-alpha-3')
        self.assertIsNone(self.call(actors[2],'claim')['value'])
        self.assertTrue(self.call(actors[0],'publish',leases[0])['ok'])
        self.assertIsNotNone(self.call(actors[2],'claim')['value'])
        self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM service_usage WHERE state="released"').fetchone()[0],1)

    def test_round_robin_is_durable_and_FIFO_within_tenant(self):
        self.configure(n=4,tenants=('alpha','beta'))
        expected=[]
        alpha=[self.submit('fair-alpha-'+str(i)) for i in range(3)]
        beta=[self.submit('fair-beta-'+str(i),'beta') for i in range(2)]
        expected=[alpha[0]['id'],beta[0]['id'],alpha[1]['id'],beta[1]['id']]
        actors=[self.start('w'+str(i)) for i in range(4)]
        actual=[]
        for a in actors:
            actual.append(self.call(a,'claim')['value']['job_id'])
            self.control.close();self.control=ReplicaControl(self.state)
        self.assertEqual(actual,expected)

    def test_host_pool_can_reject_fourth_assignment_despite_running_limit(self):
        self.configure(n=4,resource=resources(4,host=1210*legacy.MIB));self.barrier=legacy.CTX.Barrier(4)
        for i in range(4):self.submit('host-cap-'+str(i))
        actors=[self.start('w'+str(i)) for i in range(4)];leases=self.claims(actors)
        self.assertEqual(sum(l is not None for l in leases),3)
        self.assertEqual(self.control.worker_snapshot()['reservations']['host_bytes'],1110*legacy.MIB)

    def test_output_pool_can_reject_fourth_assignment_despite_running_limit(self):
        self.configure(n=4,resource=resources(4,output=704*legacy.MIB));self.barrier=legacy.CTX.Barrier(4)
        for i in range(4):self.submit('output-cap-'+str(i))
        actors=[self.start('w'+str(i)) for i in range(4)];leases=self.claims(actors)
        self.assertEqual(sum(l is not None for l in leases),3)
        self.assertEqual(self.control.worker_snapshot()['reservations']['output_bytes'],704*legacy.MIB)

    def test_dead_owner_recovery_preserves_active_credits_when_new_queue_full(self):
        self.configure(n=2,budgets={'max_active':3,'max_running':1,'max_queued':1})
        first=self.submit('recovery-first');owner=self.start('w0');lease=self.call(owner,'claim')['value']
        second=self.submit('recovery-second');owner[0].kill();owner[0].join(5)
        self.assertTrue(self.control.recover('w0',owner[2]))
        snapshot=self.control.worker_snapshot();self.assertEqual(snapshot['counts'],{'queued':2})
        self.assertEqual(snapshot['queued_hard_bound'],3);self.assertEqual(snapshot['new_submission_slots'],0)
        with self.assertRaises(Error):self.submit('recovery-third')
        self.assertTrue(self.submit('recovery-first')['reused'])
        replacement=self.start('w0');new=self.call(replacement,'claim')['value']
        self.assertEqual(new['job_id'],first['id']);self.assertEqual(new['attempt'],2)
        self.assertFalse(self.call(replacement,'publish',lease)['ok'])
        self.assertTrue(self.call(replacement,'publish',new)['ok'])
        self.control.cancel('alpha',second['id']);self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0],1)

    def test_placement_identity_is_fenced_before_publication(self):
        self.configure(n=2);self.submit('placement-one');owner=self.start('w0');lease=self.call(owner,'claim')['value']
        for field,bad in [('model','foreign-model'),('profile','foreign-profile'),('lanes',3),('state_instance','foreign-state')]:
            modified={**lease,'placement':{**lease['placement'],field:bad}}
            reply=self.call(owner,'publish',modified);self.assertFalse(reply['ok']);self.assertEqual(reply['code'],'stale_placement')
        self.assertTrue(self.call(owner,'publish',lease)['ok'])

    @patch('aureon.cloud.control.shutil.disk_usage',return_value=SimpleNamespace(free=1024*2**30))
    def test_operator_can_configure_more_than_fifteen_tenants_and_storage(self,_disk_usage):
        # Exercise logical quotas independently of the contributor's free disk.
        # No GiB payload is allocated; physical admission is checked separately.
        tenants=tuple('t'+str(i) for i in range(17))
        self.configure(n=2,tenants=tenants,budgets={'max_tenants':17,'max_tenant_storage_reservation_bytes':10*2**30,'max_global_storage_reservation_bytes':24*2**30})
        self.assertEqual(self.control.tenant_capacity(),17)
        self.assertEqual(self.control._input_capacity(),{'tenant_bytes':10*2**30,'global_bytes':24*2**30,'max_rows':10000})
        self.control.close();self.control=ReplicaControl(self.state)
        files=[{'name':'image.npy','bytes':2**30,'sha256':'a'*64}]
        for tenant,count in [('t0',5),('t1',5),('t2',2)]:
            for _ in range(count):self.control._reserve_input(tenant,{'fixture':'reservation-only-no-payload'},files)
            if tenant=='t0':
                with self.assertRaises(Error) as exc:self.control._reserve_input(tenant,{'fixture':True},files)
                self.assertEqual(exc.exception.details['scope'],'tenant')
        with self.assertRaises(Error) as exc:self.control._reserve_input('t2',{'fixture':True},files)
        self.assertEqual(exc.exception.details['scope'],'service')
        self.assertTrue(all(self.control.authenticate(self.keys[t])==t for t in tenants))
        from aureon.serving.security import test_identity
        with self.assertRaises(Error) as exc:test_identity(self.control,'local-test','eighteenth')
        self.assertEqual(exc.exception.code,'tenant_capacity')

    def test_physical_free_disk_still_blocks_input_reservation(self):
        self.configure(n=2)
        count=self.control.db.execute('SELECT COUNT(*) FROM inputs').fetchone()[0]
        files=[{'name':'image.npy','bytes':1,'sha256':'a'*64}]
        with patch('aureon.cloud.control.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaises(Error) as exc:
                self.control._reserve_input('alpha',{'fixture':'reservation-only-no-payload'},files)
        self.assertEqual(exc.exception.code,'storage_capacity')
        self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM inputs').fetchone()[0],count)

    def test_cli_contract_and_API_metadata_report_actual_capacity(self):
        import aureon
        root=Path(aureon.__file__).resolve().parent.parent
        write_file(self.root/'resources.json',encoded(resources()))
        write_file(self.root/'limits.json',encoded({'max_workers':12,'max_active':9,'max_running':4,'max_queued':6}))
        result=subprocess.run([sys.executable,'-B','-m','aureon.serving','replicas','init','--state',str(self.state),
            '--resources',str(self.root/'resources.json'),'--limits',str(self.root/'limits.json'),
            '--tenant','alpha','--credentials',str(self.root/'credentials.json'),'--fake'],
            env={**os.environ,'PYTHONPATH':str(root)},cwd='/tmp',capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout)['schema'],'aureon.async_replicas.v3')
        self.control=ReplicaControl(self.state)
        self.assertEqual(self.control.capabilities()['max_active'],9);self.assertEqual(self.control.capabilities()['max_running'],4)
        self.assertEqual(self.control.readiness()['capabilities']['max_active'],9)
        self.assertEqual(self.control.worker_snapshot()['admission_slots'],6)
        self.assertEqual(self.control.status_snapshot('alpha')['queue']['max_active'],9)
        self.control.set_meta('drain',True);self.assertEqual(self.control.worker_snapshot()['admission_slots'],0)

    def test_finite_retained_row_capacity_rejects_new_job_after_cancel(self):
        self.configure(n=2,budgets={'max_active':1,'max_running':1,'max_queued':1,'max_rows':1,'max_worker_generations':2})
        job=self.submit('row-capacity-one');self.control.cancel('alpha',job['id'])
        self.assertTrue(self.submit('row-capacity-one')['reused'])
        with self.assertRaises(Error) as exc:self.submit('row-capacity-two')
        self.assertEqual(exc.exception.code,'job_count_quota')

    def test_sql_aggregate_preserves_retained_scratch_and_releases_only_dead_owner(self):
        self.configure(n=2,budgets={'max_worker_generations':2})
        a=self.start('w0');self.assertTrue(self.call(a,'close')['ok']);a[0].join(5)
        b=self.start('w1');self.assertTrue(self.call(b,'close')['ok']);b[0].join(5)
        usage=self.control.worker_snapshot()['reservations']
        self.assertEqual(usage['output_bytes'],256*legacy.MIB)
        self.assertEqual(usage['host_bytes'],110*legacy.MIB)
        folder=self.root/'no-more-generations';folder.mkdir()
        with self.assertRaises(Error) as exc:self.control.register('w0',folder)
        self.assertEqual(exc.exception.code,'worker_generation_record_capacity')
