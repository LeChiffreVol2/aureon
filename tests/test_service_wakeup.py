"""Durable queue wakeups, lost-notification safety and idle write elimination."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid

from aureon.cloud.control import Error, encoded, write_file
from aureon.serving.control import Control, initialize
from aureon.serving.worker import work


class ServiceWakeupTests(unittest.TestCase):
    def setUp(self):
        parent=Path(os.environ.get('AUREON_SERVICE_TEST_ROOT',tempfile.gettempdir()))
        self.root=parent/('wake-'+uuid.uuid4().hex);self.root.mkdir(parents=True)
        self.now=2000000000.
        initialize(self.root/'state',['alpha'],model='terramind-v1-base',environment='generation',
                   profile='fp32-generation-v1',fake=True)
        self.control=Control(self.root/'state',clock=lambda:self.now)
        self.other=Control(self.root/'state',clock=lambda:self.now)
        self.source=uuid.uuid4().hex;folder=self.control.root/'inputs'/self.source;folder.mkdir()
        write_file(folder/'manifest.json',encoded({'fixture':True,'model_executed':False}))
        with self.control.tx() as db:
            db.execute('INSERT INTO inputs(id,tenant,state,declaration,bytes,created,expires,manifest) '
                       'VALUES (?,"alpha","ready","{}",123,?,?,?)',
                       (self.source,self.now,self.now+3600,str((folder/'manifest.json').relative_to(self.control.root))))
        self.body={'input_id':self.source,'model_id':'terramind-v1-base','profile_id':'fp32-generation-v1',
                   'task':'generation','parameters':{'seed':17}}

    def tearDown(self):
        self.other.close();self.control.close()

    def test_idle_probe_is_read_only(self):
        statements=[];self.control.db.set_trace_callback(statements.append)
        before=self.control.db.total_changes
        for _ in range(20):self.assertFalse(self.control._has_preparation_work())
        self.assertEqual(self.control.db.total_changes,before)
        self.assertTrue(all(s.lstrip().upper().startswith('SELECT ') for s in statements))

    def test_expired_validator_still_reaches_transactional_recovery(self):
        with self.control.tx() as db:
            db.execute('UPDATE inputs SET state="validating",lease=? WHERE id=?',(self.now-1,self.source))
        self.assertTrue(self.control._has_preparation_work())
        self.assertFalse(self.control.prepare_one())
        self.assertEqual(self.control.input('alpha',self.source)['state'],'rejected')
        self.assertFalse(self.control._has_preparation_work())

    def test_active_validator_and_expired_upload_do_not_wake_parser(self):
        with self.control.tx() as db:db.execute('UPDATE inputs SET state="validating",lease=?',(self.now+1,))
        self.assertFalse(self.control._has_preparation_work())
        with self.control.tx() as db:db.execute('UPDATE inputs SET state="verifying",expires=?',(self.now-1,))
        self.assertFalse(self.control._has_preparation_work())

    def test_input_signal_is_after_commit_and_survives_probe_wait_race(self):
        with self.control.tx() as db:db.execute('UPDATE inputs SET state="uploading"')
        self.control._preparation_ready.clear()
        self.assertFalse(self.control._has_preparation_work())
        observed=[];original=self.control._preparation_ready.set
        def notify():
            observed.append(self.other.input('alpha',self.source)['state']);original()
        with patch.object(self.control._preparation_ready,'set',side_effect=notify):
            self.control.complete_input('alpha',self.source)
        self.assertEqual(observed,['verifying'])
        self.assertTrue(self.control._preparation_ready.wait(0))
        self.assertTrue(self.control._has_preparation_work())

    def test_submission_signal_does_not_bypass_auth_or_commit(self):
        with self.assertRaises(Error):self.control.submit('unknown',self.body,'invalid-tenant')
        self.assertFalse(self.control._job_ready.is_set())
        observed=[];original=self.control._job_ready.set
        def notify():
            observed.append(self.other.db.execute('SELECT COUNT(*) FROM jobs WHERE state="queued"').fetchone()[0]);original()
        with patch.object(self.control._job_ready,'set',side_effect=notify):self.control.submit('alpha',self.body,'valid-job')
        self.assertEqual(observed,[1]);self.assertTrue(self.control._job_ready.wait(0))

    def test_worker_idle_does_not_publish_false_preparation_activity(self):
        stop=threading.Event();results=[];errors=[];observed=[];original=self.control.worker_status
        def status(*args,**kwargs):
            if 'preparation_state' in kwargs:observed.append(kwargs['preparation_state'])
            return original(*args,**kwargs)
        def run():
            try:results.append(work(self.control,self.root/'worker',fake=True,stop=stop,max_seconds=3))
            except BaseException as exc:errors.append(exc)
        with patch.object(self.control,'worker_status',side_effect=status):
            thread=threading.Thread(target=run);thread.start()
            try:
                deadline=time.monotonic()+2
                while self.control.meta('worker',{}).get('state')!='available' and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(thread.is_alive());time.sleep(.25)
            finally:stop.set();thread.join(3)
        self.assertFalse(thread.is_alive());self.assertFalse(errors)
        self.assertTrue(results[0]['passed']);self.assertEqual(observed,['closed'])

    def test_independent_control_uses_durable_queue_without_shared_event(self):
        # A separate Control object (as in a separate process) has no local
        # notification. Queue admission/claim still works via bounded polling.
        self.other.submit('alpha',self.body,'other-control')
        self.assertFalse(self.control._job_ready.is_set())
        result=work(self.control,self.root/'worker',fake=True,max_seconds=2,max_jobs=1)
        self.assertTrue(result['passed']);self.assertEqual(result['completed'],1)
        self.assertEqual(self.control.metrics()['committed_usage_rows'],1)


if __name__=='__main__':unittest.main()
