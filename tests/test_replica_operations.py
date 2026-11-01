"""Replica maintenance: real processes/HTTP, simulated input parser, stdlib only."""
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import threading
import time
import unittest
import subprocess
import sys
import struct
import shutil
from unittest.mock import patch
import uuid

from aureon.cloud.control import Error, encoded, write_file, file_hash
from aureon.cloud.http import Server
from aureon.serving.client import Client
from aureon.serving.ownership import exclusive
from aureon.serving.replica_control import ReplicaControl, initialize
from aureon.serving import replica_operations as operations
from aureon.serving.security import write_secret

CTX = multiprocessing.get_context('spawn')
MIB = 1024**2


def fixture(folder):
    """Small NPY v1 payloads encoded without NumPy; no numerical validation claim."""
    folder.mkdir()
    def npy(name, descriptor, shape, element):
        header = repr({'descr': descriptor, 'fortran_order': False, 'shape': shape}).encode('ascii')
        header += b' '*(-(10+len(header)+1) % 64)+b'\n'
        count = 1
        for dimension in shape: count *= dimension
        (folder/name).write_bytes(b'\x93NUMPY\x01\x00'+struct.pack('<H', len(header))+header+element*count)
    npy('image.npy', '<f4', (12, 224, 224), struct.pack('<f', 1.0))
    npy('mask.npy', '|b1', (224, 224), b'\x01')
    manifest = {'schema': 'aureon.terramind_native_input.v1',
        'bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
        'units': 'surface_reflectance_x10000', 'synthetic_test': True,
        'grid': {'shape': [224, 224], 'crs': 'EPSG:32647', 'transform': [10, 0, 600000, 0, -10, 1500000]}}
    for name in ('image', 'mask'):
        path = folder/(name+'.npy')
        manifest[name] = {'path': path.name, 'bytes': path.stat().st_size, 'sha256': file_hash(path)}
    write_file(folder/'manifest.json', encoded(manifest))
    return folder/'manifest.json'


def simulated_prepare(body, folder, token, identifier):
    """Only the parser is simulated; upload declarations/hashes remain real.

    These maintenance tests exercise durable files and recovery, not raster or
    model input qualification. Production parser tests have separate coverage.
    """
    mapping = body['spec']['members']
    manifest = json.loads((folder/'source'/mapping['manifest.json']).read_bytes())
    if manifest.get('synthetic_test') is not True or set(mapping) != {'manifest.json', 'image.npy', 'mask.npy'}:
        raise AssertionError('Expected the explicit synthetic maintenance fixture')
    target = folder/('prepared-'+token); target.mkdir()
    for name, uploaded in mapping.items(): shutil.copyfile(folder/'source'/uploaded, target/name)
    return target/'manifest.json'


def actor(state, work, pipe):
    control = ReplicaControl(state)
    folder = Path(work); folder.mkdir()
    try:
        with exclusive(state, 'replica-w0.lock'):
            generation = control.register('w0', folder)
            write_file(folder/'ownership.json', encoded({'worker': 'w0', 'generation': generation,
                'state': str(control.root), 'pid': os.getpid(), 'device': 'fake:0'}))
            write_file(folder/'payload.json', encoded({'synthetic': True, 'model_executed': False}))
            pipe.send(generation)
            while pipe.poll(15):
                command, value = pipe.recv()
                if command == 'claim': pipe.send(control.claim_replica('w0', generation))
                elif command == 'publish': pipe.send(control.publish_fake(value))
                elif command == 'close':
                    control.close_worker('w0', generation, {'passed': True, 'fake': True})
                    pipe.send(True); break
                elif command == 'crash': os._exit(86)
                else: raise AssertionError(command)
    finally:
        control.close(); pipe.close()


@unittest.skipUnless(os.name != 'nt', 'Linux process and kernel ownership required')
class ReplicaOperationsTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ.get('AUREON_SERVICE_TEST_ROOT', '/tmp/aureon-replica-operations'))
        self.root = parent/('operations-'+uuid.uuid4().hex); self.root.mkdir(parents=True)
        resources = {'host_bytes': 2000*MIB, 'output_bytes': 1000*MIB,
            'controller_host_bytes': 10*MIB, 'parser_host_bytes': 100*MIB,
            'workers': [{'id': 'w0', 'device': 'fake:0', 'vram_bytes': 1000*MIB,
                'resident_vram_bytes': 100*MIB, 'resident_host_bytes': 100*MIB,
                'transient_vram_bytes': 100*MIB, 'transient_host_bytes': 200*MIB, 'scratch_bytes': 128*MIB}]}
        self.state = self.root/'state'
        self.keys = initialize(self.state, ['alpha', 'beta'], resources=resources,
                               limits={'max_workers': 1}, fake=True)
        self.credentials = self.root/'keys.json'; write_secret(self.credentials, self.keys)
        self.control = ReplicaControl(self.state)
        self.actors = []; self.extra = []
        manifest = fixture(self.root/'prepared')
        with self.http(self.control) as (client, _):
            upload = client.upload_prepared(manifest, task='generation')
            with patch.object(self.control, '_prepare_input', simulated_prepare):
                self.assertTrue(self.control.prepare_one())
            self.assertEqual(client.wait_input(upload['input_id'], timeout=10)['state'], 'ready')
            self.input_id = upload['input_id']
            self.manifest_sha256 = upload['manifest_sha256']
        self.body = {'input_id': self.input_id, 'model_id': 'terramind-v1-base',
                     'profile_id': 'fp32-generation-v1', 'task': 'generation', 'parameters': {'seed': 17}}

    def tearDown(self):
        for process, pipe, _, _ in self.actors:
            if process.is_alive():
                try:
                    pipe.send(('close', None))
                    if pipe.poll(2): pipe.recv()
                except (EOFError, BrokenPipeError): pass
                process.join(3)
            if process.is_alive(): process.kill(); process.join(3)
            pipe.close()
        for control in self.extra: control.close()
        self.control.close()
        # Keep fixture evidence; only reclaim tests remove their registered,
        # newly-created disposable scratch through the explicit API under test.

    @contextmanager
    def http(self, control):
        with exclusive(control.root, 'service.lock'):
            server = Server(control)
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}); thread.start()
            url = 'http://127.0.0.1:'+str(server.server_port)
            try:
                yield Client(url, token=self.keys['alpha']), Client(url, token=self.keys['operator'])
            finally:
                server.shutdown(); thread.join(5); server.server_close()
                self.assertFalse(thread.is_alive())

    def start(self, control=None):
        control = control or self.control
        pipe, child = CTX.Pipe(); work = self.root/('scratch-'+uuid.uuid4().hex)
        process = CTX.Process(target=actor, args=(str(control.root), str(work), child))
        process.start(); child.close(); self.assertTrue(pipe.poll(10))
        item = (process, pipe, pipe.recv(), work); self.actors.append(item); return item

    def call(self, item, command, value=None):
        item[1].send((command, value))
        self.assertTrue(item[1].poll(10)); return item[1].recv()

    def close_actor(self, item):
        self.assertTrue(self.call(item, 'close')); item[0].join(5)
        self.assertEqual(item[0].exitcode, 0)

    def closed(self):
        item = self.start(); self.close_actor(item); return item

    def reclaim(self, item):
        return operations.reclaim(self.state, 'w0', item[2], work_root=self.root)

    def credits(self, item):
        return self.control.db.execute('SELECT scratch_bytes FROM replica_workers WHERE generation=?', (item[2],)).fetchone()[0]

    def snapshot(self):
        target = self.root/('backup-'+uuid.uuid4().hex)
        result = operations.backup(self.state, target, self.credentials)
        return target, result['manifest_sha256']

    def test_reclaim_closed_generation_releases_only_scratch_once(self):
        job = self.control.submit('alpha', self.body, 'operations-complete')
        item = self.start(); lease = self.call(item, 'claim')
        self.call(item, 'publish', lease); self.close_actor(item)
        before = self.control.worker_snapshot()['reservations']['output_bytes']
        with self.assertRaises(Error): operations.reclaim(self.state, 'w0', item[2], work_root=self.root/'prepared')
        self.assertTrue(item[3].exists()); self.assertEqual(self.credits(item), 128*MIB)
        result = self.reclaim(item)
        self.assertEqual(result['released_reserved_bytes'], 128*MIB)
        self.assertFalse(item[3].exists()); self.assertEqual(self.credits(item), 0)
        self.assertEqual(self.control.worker_snapshot()['reservations']['output_bytes'], before-128*MIB)
        self.assertEqual(self.control.job('alpha', job['id'])['state'], 'completed')
        self.assertTrue(self.reclaim(item)['reused'])

    def test_live_owner_active_assignment_and_HTTP_service_block_maintenance(self):
        self.control.submit('alpha', self.body, 'operations-live-owner')
        item = self.start(); self.call(item, 'claim')
        with self.assertRaises(Exception): self.reclaim(item)
        with self.assertRaises(Exception): self.snapshot()
        self.assertTrue(item[3].exists()); self.assertEqual(self.credits(item), 128*MIB)
        item[1].send(('crash', None)); item[0].join(5); self.assertEqual(item[0].exitcode, 86)
        with self.assertRaises(Error): self.reclaim(item)
        self.assertTrue(self.control.recover('w0', item[2]))
        with self.http(self.control):
            with self.assertRaises(Exception): self.snapshot()
        self.assertTrue(self.reclaim(item)['retired'])

    def test_interrupted_reclaim_preserves_credit_and_replays_frozen_inventory(self):
        item = self.closed()
        def interrupted(path):
            (Path(path)/'payload.json').unlink()
            raise OSError('injected disposable fixture removal interruption')
        with patch.object(operations.shutil, 'rmtree', interrupted):
            interrupted.avoids_symlink_attacks = True
            with self.assertRaises(OSError): self.reclaim(item)
        self.assertEqual(self.credits(item), 128*MIB)
        self.assertTrue(self.reclaim(item)['retired']); self.assertEqual(self.credits(item), 0)

    def test_changed_scratch_and_symlink_are_not_removed(self):
        item = self.closed(); outside = self.root/'outside.txt'; outside.write_text('preserve')
        (item[3]/'external').symlink_to(outside)
        with self.assertRaises(Exception): self.reclaim(item)
        self.assertEqual(outside.read_text(), 'preserve'); self.assertEqual(self.credits(item), 128*MIB)

    def test_removed_scratch_without_credit_commit_is_reconciled_once(self):
        item = self.closed(); sync = operations._sync_dir
        def interrupted(path):
            if Path(path) == self.root and not item[3].exists():
                raise OSError('injected after removal before credit release')
            return sync(path)
        with patch.object(operations, '_sync_dir', interrupted):
            with self.assertRaises(OSError): self.reclaim(item)
        self.assertFalse(item[3].exists()); self.assertEqual(self.credits(item), 128*MIB)
        self.assertEqual(self.reclaim(item)['released_reserved_bytes'], 128*MIB)
        self.assertEqual(self.reclaim(item)['released_reserved_bytes'], 0)

    def test_replaced_root_is_not_removed_after_intent(self):
        item = self.closed()
        def fail(_): raise OSError('injected before removal')
        fail.avoids_symlink_attacks = True
        with patch.object(operations.shutil, 'rmtree', fail):
            with self.assertRaises(OSError): self.reclaim(item)
        moved = item[3].with_name(item[3].name+'-preserved'); item[3].rename(moved); item[3].mkdir()
        (item[3]/'unrelated.txt').write_text('preserve')
        with self.assertRaises(Error): self.reclaim(item)
        self.assertTrue((item[3]/'unrelated.txt').exists()); self.assertTrue(moved.exists())
        self.assertEqual(self.credits(item), 128*MIB)

    def test_snapshot_handover_recovers_orphan_and_HTTP_download_usage_once(self):
        first = self.control.submit('alpha', self.body, 'operations-first-job')
        owner = self.start(); self.call(owner, 'publish', self.call(owner, 'claim')); self.close_actor(owner)
        second = self.control.submit('alpha', self.body, 'operations-orphan-job')
        orphan = self.start(); stale = self.call(orphan, 'claim')
        orphan[1].send(('crash', None)); orphan[0].join(5); self.assertEqual(orphan[0].exitcode, 86)
        old_ticket = self.control.artifacts('alpha', first['id'])['artifacts']['simulation.json']['url']
        backup, identity = self.snapshot()
        self.assertEqual(self.control.job('alpha', second['id'])['state'], 'queued')
        destination = self.root/'restored'
        result = operations.restore(backup, destination, identity)
        self.assertTrue(result['source_retired']); self.assertTrue(result['tickets_invalidated'])
        self.assertTrue(owner[3].exists()); self.assertTrue(orphan[3].exists())
        restored = ReplicaControl(destination); self.extra.append(restored)
        self.assertNotEqual(restored.meta('instance_id'), self.control.meta('instance_id'))
        self.assertNotEqual(restored.contract['replicas']['authority'], self.control.contract['replicas']['authority'])
        self.assertEqual(restored.contract['replicas']['execution'], self.control.contract['replicas']['execution'])
        self.assertEqual(restored.db.execute('SELECT SUM(scratch_bytes) FROM replica_workers').fetchone()[0], 0)
        self.control.set_meta('drain', False)
        with self.assertRaises(Error): self.control.submit('alpha', self.body, 'retired-new-job')
        with self.assertRaises(Error): self.control.claim_replica('w0', orphan[2])
        with self.assertRaises(Error): self.control.publish_fake(stale)
        restored.set_meta('drain', False)
        worker = self.start(restored); fresh = self.call(worker, 'claim')
        self.assertEqual(fresh['attempt'], 2); self.call(worker, 'publish', fresh); self.close_actor(worker)
        with self.http(restored) as (client, operator):
            job = client.json('GET', '/v1/jobs/'+second['id']); self.assertEqual(job['state'], 'completed')
            replay = client.submit_model(self.input_id, key='operations-orphan-job', model='terramind-v1-base',
                                         profile='fp32-generation-v1', task='generation')
            self.assertEqual(replay['id'], second['id']); self.assertTrue(replay['reused'])
            download = client.download_result(second['id'], self.root/'download', model='terramind-v1-base', task='generation',
                                              expected_manifest_sha256=self.manifest_sha256)
            self.assertTrue(download['fake']); self.assertTrue(download['hashes_verified'])
            with self.assertRaises(Exception): client.json('GET', old_ticket)
            self.assertEqual(operator.json('GET', '/v1/metrics')['committed_usage_rows'], 2)
            self.assertNotIn('replicas', client.json('GET', '/v1/models')['limits'])
        self.assertTrue(operations.restore(backup, destination, identity)['reused'])

    def test_backup_tampering_and_changed_source_reject_before_handover(self):
        backup, identity = self.snapshot()
        self.control.set_meta('new_activity', True)
        with self.assertRaises(Error): operations.restore(backup, self.root/'changed', identity)
        self.assertFalse((self.root/'changed').exists()); self.assertFalse(self.control.meta(operations.RETIRED))
        (backup/'credentials.json').write_text('{}')
        with self.assertRaises(Error): operations.restore(backup, self.root/'corrupt', identity)
        self.assertFalse((self.root/'corrupt').exists()); self.assertFalse(self.control.meta(operations.RETIRED))

    def test_backup_state_is_not_an_executable_queue_clone(self):
        self.control.submit('alpha', self.body, 'backup-fenced-job')
        backup, _ = self.snapshot()
        clone = ReplicaControl(backup/'state'); self.extra.append(clone)
        self.assertTrue(clone.meta(operations.PENDING)); clone.set_meta('drain', False)
        for operation, args in [(clone.submit, ('alpha', self.body, 'backup-clone-new-job')),
                                (clone.register, ('w0', self.root/'prepared')),
                                (clone.claim_replica, ('w0', '0'*32))]:
            with self.assertRaises(Error) as caught: operation(*args)
            self.assertEqual(caught.exception.code, 'replica_restore_pending')
        self.assertFalse(clone.readiness()['ready'])
        self.assertFalse(self.control.meta(operations.PENDING))
        self.control.set_meta('drain', False)
        owner = self.start(); self.assertIsNotNone(self.call(owner, 'claim')); self.close_actor(owner)

    def test_handover_publish_interruption_keeps_source_fenced_and_is_replayable(self):
        backup, identity = self.snapshot(); target = self.root/'restored'
        with patch.object(operations.os, 'rename', side_effect=OSError('injected handover publication failure')):
            with self.assertRaises(OSError): operations.restore(backup, target, identity)
        self.assertFalse(target.exists()); self.assertTrue(self.control.meta(operations.RETIRED))
        self.control.set_meta('drain', False)
        with self.assertRaises(Error): self.control.submit('alpha', self.body, 'interrupted-new-job')
        self.assertTrue(operations.restore(backup, target, identity)['source_retired'])
        self.assertTrue(operations.restore(backup, target, identity)['reused'])
        with self.assertRaises(Error): operations.restore(backup, self.root/'second-clone', identity)

    def test_pre_retirement_stage_cannot_be_used_as_a_second_authority(self):
        self.control.submit('alpha', self.body, 'pending-stage-job')
        backup, identity = self.snapshot(); target = self.root/'restored'
        verify = operations.verify_database_files
        def interrupted(root, db):
            if '.restoring-' in Path(root).name:
                raise OSError('injected before source retirement')
            return verify(root, db)
        with patch.object(operations, 'verify_database_files', interrupted):
            with self.assertRaises(OSError): operations.restore(backup, target, identity)
        self.assertFalse(self.control.meta(operations.RETIRED)); self.assertFalse(target.exists())
        stage = next(self.root.glob('restored.restoring-*'))
        staged = ReplicaControl(stage); self.extra.append(staged)
        self.assertTrue(staged.meta(operations.PENDING))
        staged.set_meta('drain', False)
        denied = [(staged.submit, ('alpha', self.body, 'pending-stage-new-job')),
                  (staged.register, ('w0', self.root/'prepared')),
                  (staged.claim_replica, ('w0', '0'*32)),
                  (staged.publish_fake, ({'job_id': '0'*32, 'attempt': 1, 'token': 'not-a-token',
                                         'manifest': str(self.root/'prepared/manifest.json')},))]
        for operation, args in denied:
            with self.assertRaises(Error) as caught: operation(*args)
            self.assertEqual(caught.exception.code, 'replica_restore_pending')
        self.assertFalse(staged.readiness()['ready'])
        with self.assertRaises(Error): operations.backup(stage, self.root/'stage-copy', self.credentials)
        self.control.set_meta('drain', False)
        owner = self.start(); self.assertIsNotNone(self.call(owner, 'claim')); self.close_actor(owner)

    def test_published_pending_state_stays_fenced_until_replay_finishes_handover(self):
        backup, identity = self.snapshot(); target = self.root/'restored'
        set_meta = ReplicaControl.set_meta
        def interrupted(control, key, value):
            if key == operations.PENDING and value is False:
                raise OSError('injected after directory publication before activation')
            return set_meta(control, key, value)
        with patch.object(ReplicaControl, 'set_meta', interrupted):
            with self.assertRaises(OSError): operations.restore(backup, target, identity)
        self.assertTrue(target.exists()); self.assertTrue(self.control.meta(operations.RETIRED))
        pending = ReplicaControl(target); self.extra.append(pending)
        with self.assertRaises(Error): pending._require_active_authority()
        self.assertTrue(operations.restore(backup, target, identity)['reused'])
        pending._require_active_authority()
        self.assertFalse(pending.meta(operations.PENDING))

    def test_explicit_CLI_backup_restore_and_reclaim_receipts(self):
        import aureon
        package = str(Path(aureon.__file__).resolve().parents[1])
        bootstrap = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('aureon.serving.replicas',run_name='__main__')"
        def command(*args):
            result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', bootstrap, package, *map(str, args)],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            for key in self.keys.values(): self.assertNotIn(key, result.stdout+result.stderr)
            return json.loads(result.stdout)
        item = self.closed()
        self.assertTrue(command('reclaim-scratch', '--state', self.state, '--worker', 'w0',
                                '--generation', item[2], '--work-root', self.root)['retired'])
        snapshot = command('backup', '--state', self.state, '--destination', self.root/'cli-backup', '--credentials', self.credentials)
        restored = command('restore', '--source', self.root/'cli-backup', '--destination', self.root/'cli-restored',
                           '--manifest-sha256', snapshot['manifest_sha256'])
        self.assertTrue(restored['source_retired']); self.assertFalse(restored['model_executed'])


if __name__ == '__main__':
    unittest.main()
