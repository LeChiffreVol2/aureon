"""Metadata must not share mutable policy/fencing state with its caller."""
from copy import deepcopy
import os
import tempfile
from pathlib import Path
import time
import unittest
import uuid

from aureon.cloud.control import Error, encoded, write_file
from aureon.serving.replica_capacity import limits, policy
from aureon.serving.replica_control import ReplicaControl, initialize
from aureon.serving.ownership import exclusive


class PolicySnapshotTests(unittest.TestCase):
    def test_policy_limits_are_a_snapshot(self):
        selected = limits({}, 1)
        expected = deepcopy(selected)
        public = policy(selected)
        public['limits']['max_active'] = 100
        self.assertEqual(selected, expected)


@unittest.skipUnless(os.name != 'nt', 'Linux replica ownership contract')
class ReplicaMetadataTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ.get('AUREON_REPLICA_TEST_ROOT', tempfile.gettempdir()))
        self.root = parent / ('metadata-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.state = self.root / 'state'
        mib = 2**20
        resources = {'host_bytes': 2000*mib, 'output_bytes': 1000*mib,
            'controller_host_bytes': 10*mib, 'parser_host_bytes': 100*mib,
            'workers': [{'id': 'w0', 'device': 'fake:0', 'lanes': 1,
                'vram_bytes': 1000*mib, 'resident_vram_bytes': 100*mib,
                'resident_host_bytes': 100*mib, 'transient_vram_bytes': 100*mib,
                'transient_host_bytes': 200*mib, 'scratch_bytes': 128*mib}]}
        initialize(self.state, ['alpha'], resources=resources, fake=True,
                   limits={'max_active': 2, 'max_queued': 2, 'max_running': 1})
        self.control = ReplicaControl(self.state)
        self.before = deepcopy(self.control.contract)
        source = uuid.uuid4().hex
        folder = self.state / 'inputs' / source
        folder.mkdir()
        write_file(folder / 'manifest.json', encoded({'fixture': True, 'model_executed': False}))
        with self.control.tx() as db:
            db.execute('INSERT INTO inputs(id,tenant,state,declaration,bytes,created,expires,manifest) '
                       'VALUES (?, ?, "ready", ?, 0, ?, ?, ?)',
                       (source, 'alpha', '{}', time.time(), time.time()+3600,
                        str((folder / 'manifest.json').relative_to(self.state))))
        self.body = {'input_id': source, 'model_id': 'terramind-v1-base',
            'profile_id': 'fp32-generation-v1', 'task': 'generation', 'parameters': {'seed': 17}}

    def tearDown(self):
        self.control.close()
        # Preserve bounded fixture evidence; never remove unrelated scratch.

    def test_capabilities_edits_do_not_change_admission(self):
        public = self.control.capabilities()
        public['capacity']['limits']['max_active'] = 99
        public['capacity']['limits']['max_queued'] = 99
        self.control.submit('alpha', self.body, 'metadata-first')
        self.control.submit('alpha', self.body, 'metadata-second')
        with self.assertRaises(Error) as caught:
            self.control.submit('alpha', self.body, 'metadata-third')
        self.assertEqual(caught.exception.code, 'active_capacity')
        self.assertEqual(self.control.contract, self.before)

    def test_nested_capabilities_lists_do_not_change_authority(self):
        public = self.control.capabilities()
        public['placement_authority']['external_authority_required'].clear()
        self.assertEqual(self.control.contract, self.before)

    def test_operator_snapshot_edits_do_not_change_resource_budget(self):
        public = self.control.worker_snapshot()
        public['capacities']['workers'][0]['vram_bytes'] = 2**40
        public['execution_binding']['mode'] = 'foreign'
        public['authority']['host']['uid'] = -1
        self.assertEqual(self.control.contract, self.before)
        self.assertEqual(self.control.spec('w0')['vram_bytes'], 1000*2**20)

    def test_nested_lease_placement_tamper_is_fenced(self):
        self.control.submit('alpha', self.body, 'metadata-lease')
        folder = self.root / 'worker'
        folder.mkdir()
        with exclusive(self.state, 'replica-w0.lock'):
            generation = self.control.register('w0', folder)
            lease = self.control.claim_replica('w0', generation)
            original = deepcopy(lease)
            lease['placement']['authority']['host']['uid'] = -1
            with self.assertRaises(Error) as caught:
                self.control.publish_fake(lease)
            self.assertEqual(caught.exception.code, 'stale_placement')
            self.assertEqual(self.control.contract, self.before)
            self.control.publish_fake(original)
            self.control.close_worker('w0', generation, {'passed': True, 'fake': True})
            self.assertEqual(self.control.db.execute('SELECT COUNT(*) FROM usage').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
