"""No process lookup or signals without an attested Linux owner identity."""
import unittest
from unittest.mock import patch

from aureon.serving.supervisor import members, terminate_owned


class SupervisorIdentityTests(unittest.TestCase):
    def test_unattested_owner_is_rejected_before_scanning_or_signalling(self):
        for identity in (None, {'pid': 123, 'platform': 'windows', 'start_ticks': None}):
            for operation in (lambda: members(identity), lambda: terminate_owned(identity, 0)):
                with self.subTest(identity=identity), patch('pathlib.Path.iterdir') as scan, \
                     patch('aureon.serving.supervisor.os.kill') as kill:
                    with self.assertRaises(TypeError):
                        operation()
                    scan.assert_not_called()
                    kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
