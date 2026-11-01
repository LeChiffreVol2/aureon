"""The service capability view must not mutate process-wide request policy."""
from copy import deepcopy
import os
import tempfile
from pathlib import Path
import unittest
import uuid

from aureon.serving.control import Control, initialize
from aureon.serving import security


class ServiceMetadataTests(unittest.TestCase):
    def test_security_capabilities_are_detached_from_global_policy(self):
        parent = Path(os.environ.get('AUREON_REPLICA_TEST_ROOT', tempfile.gettempdir()))
        root = parent / ('service-metadata-' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        initialize(root / 'state', ['alpha'], model='terramind-v1-base',
                   environment='generation', profile='fp32-generation-v1', fake=True)
        control = Control(root / 'state')
        expected = deepcopy(security.POLICY)
        try:
            first = control.capabilities()
            first['security']['request_burst'] = 999999
            self.assertEqual(security.POLICY, expected)
            self.assertEqual(control.capabilities()['security'], expected)
        finally:
            # A failing regression must not contaminate other tests' global state.
            security.POLICY.clear()
            security.POLICY.update(expected)
            control.close()


if __name__ == '__main__':
    unittest.main()
