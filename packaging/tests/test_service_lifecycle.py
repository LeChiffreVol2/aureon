"""Backend lifecycle regressions using an inert session; no neural/GPU execution."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch
import uuid

from aureon.cloud.contracts import Error
from aureon.serving.backend import Backend


class StubSession:
    def __init__(self, failure=False):
        self.calls = 0
        self.failure = failure

    def close(self):
        self.calls += 1
        if self.failure:
            raise RuntimeError('private bearer credential must not appear')
        return {'passed': True, 'errors': [], 'cuda_allocated_bytes': 0, 'cuda_reserved_bytes': 0}


class BackendLifecycleTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ.get('AUREON_SERVICE_TEST_ROOT', '/tmp/aureon-service-lifecycle'))
        self.root = parent / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.config = self.root / 'assets.json'
        self.config.write_text('{}')
        self.work = self.root / 'worker'
        self.work.mkdir()
        route = {'task': 'generation', 'environment': 'generation',
                 'model_id': 'terramind-v1-base', 'profile_id': 'fp32-generation-v1',
                 'denoiser': 'graph'}
        self.control = SimpleNamespace(execution='engine', selected=route)
        self.binding = {'version': '0.1.0', 'root': '/fixture/installed/aureon', 'modules': {}}
        self.bound = self.enterContext(patch('aureon.serving.backend.installed_binding',
                                            side_effect=lambda: deepcopy(self.binding)))
        self.enterContext(patch('aureon.api.profiles.require_profile'))
        self.enterContext(patch('aureon.terramind_native.offline.inspect_config', return_value={'model_id':'terramind-v1-base'}))
        self.backend = Backend(self.control, self.work, config=self.config, execute_model=True)
        self.session = StubSession()
        self.backend.session = self.session

    def test_successful_close_is_idempotent_and_returns_detached_receipt(self):
        first = self.backend.close()
        saved = (self.work / 'cleanup.json').read_bytes()
        first['engine_cleanup']['passed'] = False
        first['installed']['modules']['forged'] = {'path': 'foreign', 'sha256': 'bad'}
        second = self.backend.close()
        self.assertTrue(second['passed'])
        self.assertTrue(second['engine_cleanup']['passed'])
        self.assertNotIn('forged', second['installed']['modules'])
        self.assertEqual(self.session.calls, 1)
        self.assertEqual((self.work / 'cleanup.json').read_bytes(), saved)

    def test_failed_cleanup_is_terminal_and_does_not_silently_retry(self):
        self.session.failure = True
        for _ in range(2):
            with self.assertRaises(Error) as caught:
                self.backend.close()
            self.assertEqual(caught.exception.code, 'cleanup_failed')
        self.assertEqual(self.session.calls, 1)
        result = json.loads((self.work / 'cleanup.json').read_text())
        self.assertFalse(result['passed'])
        self.assertNotIn('private bearer', json.dumps(result))

    def test_closed_backend_rejects_new_execution_before_input_access(self):
        self.backend.close()
        with self.assertRaises(Error) as caught:
            self.backend.run({})
        self.assertEqual(caught.exception.code, 'backend_closed')
        self.assertEqual(self.session.calls, 1)

    def test_binding_inspection_failure_still_records_owned_cleanup(self):
        self.bound.side_effect = RuntimeError('private installation path')
        with self.assertRaises(Error) as caught:
            self.backend.close()
        self.assertEqual(caught.exception.code, 'cleanup_failed')
        result = json.loads((self.work / 'cleanup.json').read_text())
        self.assertFalse(result['passed'])
        self.assertEqual(result['close_error']['code'], 'installed_binding_failed')
        self.assertEqual(self.session.calls, 1)
        self.assertNotIn('private installation', json.dumps(result))

    def test_wrong_thread_cannot_close_owner_session(self):
        errors = []
        def other():
            try:
                self.backend.close()
            except Error as exc:
                errors.append(exc.code)
        thread = threading.Thread(target=other)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ['wrong_worker_owner'])
        self.assertEqual(self.session.calls, 0)
        self.assertTrue(self.backend.close()['passed'])


if __name__ == '__main__':
    unittest.main()
