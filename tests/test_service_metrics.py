"""Durable aggregate and real localhost HTTP contracts; no model execution."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
import uuid

from aureon.client import APIError
from aureon.cloud.control import Error, encoded, write_file
from aureon.cloud.http import Server
from aureon.serving.client import Client
from aureon.serving.control import Control, initialize


class ServiceMetricsTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ.get('AUREON_SERVICE_TEST_ROOT', tempfile.gettempdir()))
        self.root = parent / ('metrics-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.now = 2000000000.0
        self.keys = initialize(self.root/'state', ['alpha', 'beta'], model='terramind-v1-base',
                              environment='generation', profile='fp32-generation-v1', fake=True)
        self.control = Control(self.root/'state', clock=lambda: self.now)
        self.source = uuid.uuid4().hex
        folder = self.control.root/'inputs'/self.source
        folder.mkdir()
        write_file(folder/'manifest.json', encoded({'fixture': True, 'model_executed': False}))
        with self.control.tx() as db:
            db.execute('INSERT INTO inputs(id,tenant,state,declaration,bytes,created,expires,manifest) VALUES (?,"alpha","ready","{}",123,?,?,?)',
                       (self.source, self.now, self.now+3600, str((folder/'manifest.json').relative_to(self.control.root))))
        self.body = {'input_id': self.source, 'model_id': 'terramind-v1-base',
                     'profile_id': 'fp32-generation-v1', 'task': 'generation', 'parameters': {'seed': 17}}

    def tearDown(self):
        self.control.close()
        # Keep owned test state for inspection; no workspace cleanup.

    @contextmanager
    def http(self):
        server = Server(self.control)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
        thread.start()
        url = 'http://127.0.0.1:' + str(server.server_port)
        try:
            yield Client(url, token=self.keys['alpha']), Client(url, token=self.keys['operator'])
        finally:
            server.shutdown(); thread.join(5); server.server_close()
            self.assertFalse(thread.is_alive())

    def test_empty_snapshot_is_read_only_and_has_no_private_labels(self):
        before = self.control.db.total_changes
        result = self.control.metrics()
        self.assertEqual(self.control.db.total_changes, before)
        self.assertEqual(result['schema'], 'aureon.service_metrics.v1')
        self.assertEqual(result['workers']['live'], 0)
        self.assertIsNone(result['oldest_queued_age_seconds'])
        self.assertEqual(result['input_reserved_bytes'], 123)
        self.assertEqual(result['record_capacity'], {'jobs': 0, 'inputs': 1, 'max_rows_per_table': 10000})
        self.assertEqual(result['input_reservation_limit_bytes'], 8*1024**3)
        self.assertFalse(result['billable'])
        text = json.dumps(result)
        for private in ('alpha', 'beta', self.source, str(self.control.root), *self.keys.values()):
            self.assertNotIn(private, text)

    def test_recovery_replay_cancellation_and_usage_gauges(self):
        job = self.control.submit('alpha', self.body, 'first-idempotency')
        cancel = self.control.submit('alpha', self.body, 'cancel-idempotency')
        self.control.cancel('alpha', cancel['id'])
        self.now += 7
        snapshot = self.control.metrics()
        self.assertEqual(snapshot['jobs']['queued'], 1)
        self.assertEqual(snapshot['oldest_queued_age_seconds'], 7)
        self.assertEqual(snapshot['usage']['released']['jobs'], 1)
        old = self.control.claim()
        self.now += 61
        self.assertEqual(self.control.metrics()['queue']['expired_running_leases'], 1)
        fresh = self.control.claim()
        self.assertEqual(fresh['attempt'], 2)
        with self.assertRaises(Error): self.control.publish_fake(old)
        self.control.publish_fake(fresh)
        replay = self.control.submit('alpha', self.body, 'first-idempotency')
        self.assertEqual(replay['id'], job['id']); self.assertTrue(replay['reused'])
        snapshot = self.control.metrics()
        self.assertEqual(snapshot['attempts']['lease_lost'], 1)
        self.assertEqual(snapshot['attempts']['completed'], 1)
        self.assertEqual(snapshot['jobs']['completed'], 1)
        self.assertEqual(snapshot['queue']['jobs_retried'], 1)
        self.assertEqual(snapshot['committed_usage_rows'], 1)
        self.assertEqual(snapshot['usage']['settled'], {'jobs': 1, 'units': 1})
        self.assertEqual(snapshot['usage']['reserved'], {'jobs': 0, 'units': 0})

    def test_worker_staleness_and_wall_clock_reversal_are_visible(self):
        self.control.start_worker('a'*32, model_loaded=False)
        self.control.worker_status('a'*32, state='available')
        self.control.submit('alpha', self.body, 'clock-idempotency')
        self.assertEqual(self.control.metrics()['workers']['live'], 1)
        self.now += 21
        snapshot = self.control.metrics()
        self.assertEqual(snapshot['workers']['stale_active'], 1)
        self.assertEqual(snapshot['workers']['live'], 0)
        self.now -= 100
        snapshot = self.control.metrics()
        self.assertEqual(snapshot['oldest_queued_age_seconds'], 0)
        self.assertEqual(snapshot['queue']['future_created_records'], 1)

    def test_metrics_does_not_run_retention_or_release_reservations(self):
        self.control.submit('alpha', self.body, 'retention-idempotency')
        self.control.publish_fake(self.control.claim())
        self.now += 3601
        before = self.control.db.total_changes
        snapshot = self.control.metrics()
        self.assertEqual(snapshot['retention']['expired_inputs_awaiting_gc'], 1)
        self.assertEqual(snapshot['retention']['expired_jobs_awaiting_gc'], 1)
        self.assertEqual(snapshot['input_reserved_bytes'], 123)
        self.assertEqual(self.control.db.total_changes, before)
        self.assertTrue((self.control.root/'inputs'/self.source).exists())

    def test_operator_only_HTTP_and_invalid_query(self):
        with self.http() as (customer, operator):
            with self.assertRaises(APIError) as error: customer.json('GET', '/v1/metrics')
            self.assertEqual(error.exception.status, 403)
            with self.assertRaises(APIError) as error: operator.json('GET', '/v1/metrics?ticket=x')
            self.assertEqual(error.exception.status, 400)
            observed = operator.json('GET', '/v1/metrics')
            self.assertEqual(observed, self.control.metrics())
            self.assertEqual(customer.json('GET', '/v1/status')['scope'], 'authenticated_tenant_only')

    def test_model_discovery_does_not_expose_private_contract_extensions(self):
        # Same HTTP handler serves ReplicaControl. Simulate its private durable
        # extension without provisioning a GPU or weakening control validation.
        self.control.contract['replicas'] = {'host': {'machine': 'private-host-id'},
                                             'execution': {'config_path': '/private/model.json'}}
        self.control.contract['future_private_extension'] = 'private-extension-value'
        with self.http() as (customer, operator):
            public = customer.json('GET', '/v1/models')
            private = operator.json('GET', '/v1/models')
            self.assertEqual(public['limits']['max_active'], self.control.contract['max_active'])
            self.assertEqual(public['models'][0]['model_id'], 'terramind-v1-base')
            for key in ('replicas', 'future_private_extension'):
                self.assertNotIn(key, public['limits'])
                self.assertIn(key, private['limits'])
            self.assertNotIn('private-', json.dumps(public))
            with self.assertRaises(APIError) as error: customer.json('GET', '/v1/models?ticket=x')
            self.assertEqual(error.exception.status, 400)

    def test_pending_preparation_and_restored_scratch_have_named_counts(self):
        with self.control.tx() as db:
            db.execute('UPDATE inputs SET state="verifying" WHERE id=?', (self.source,))
            db.execute('INSERT INTO service_scratch VALUES ("fixture","private-path","private-owner","foreign_restore",NULL)')
        value = self.control.metrics()
        self.assertEqual(value['inputs']['verifying'], 1)
        self.assertNotIn('unknown', value['inputs'])
        self.assertEqual(value['retention']['scratch_retained_records'], 1)

    def test_replica_aggregation_never_releases_stale_device_reservations(self):
        # The aggregate reads the same fixed replica schema; real ownership and
        # recovery are exercised by the separate process integration acceptance.
        with self.control.tx() as db:
            db.execute('CREATE TABLE replica_workers(worker TEXT,generation TEXT,device TEXT,state TEXT,heartbeat REAL)')
            db.executemany('INSERT INTO replica_workers VALUES (?,?,?,?,?)', [
                ('worker-secret-a', 'generation-a', 'GPU-secret-a', 'available', self.now),
                ('worker-secret-b', 'generation-b', 'GPU-secret-b', 'running', self.now-21),
                ('worker-secret-c', 'generation-c', 'GPU-secret-c', 'closed', self.now),
            ])
        before = self.control.db.total_changes
        value = self.control.metrics()
        self.assertEqual(value['workers'], {'retained_generations': 3, 'live': 1, 'stale_active': 1})
        self.assertEqual(self.control.db.total_changes, before)
        self.assertNotIn('secret', json.dumps(value))


if __name__ == '__main__':
    unittest.main()
