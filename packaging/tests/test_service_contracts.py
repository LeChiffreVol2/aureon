"""Pure policy, compatibility, redaction and typed boundary contracts on CPU."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import typing
import unittest

from aureon.cloud import contracts as values
from aureon.serving.contracts import AttemptLease, BackendCleanup, ReplicaLimits
from aureon.serving.errors import safe_error
from aureon.serving.replica_capacity import authority, limits, policy


class ServiceContractTests(unittest.TestCase):
    def test_policy_import_and_invalid_validation_do_not_import_control_or_io(self):
        import aureon
        package = str(Path(aureon.__file__).resolve().parent.parent)
        code = '''
import importlib.abc,sys
sys.path.insert(0,sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in ('sqlite3','aureon.cloud.control','aureon.client') or fullname.split('.')[0] in ('torch','rasterio','terratorch','vllm') or fullname.startswith('aureon.scene_api'):
   raise AssertionError('Unexpected import: '+fullname)
sys.meta_path.insert(0,Block())
from aureon.serving.replica_capacity import limits
from aureon.serving.errors import safe_error
from aureon.cloud.contracts import Error
assert limits({},1)['max_running']==1
try:limits({'max_active':False},1)
except Error as exc:assert exc.code=='finite_positive_replica_limit'
else:raise AssertionError('invalid limit accepted')
assert safe_error(Error(507,'resident_vram_budget_exceeded','private'))['code']=='resident_vram_budget_exceeded'
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code, package],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_old_import_paths_reexport_the_same_contract_objects(self):
        from aureon.cloud import control
        from aureon.serving import ownership
        for name in ('MODEL', 'PROFILE', 'CONTRACT', 'Error', 'require', 'encoded'):
            self.assertIs(getattr(control, name), getattr(values, name))
        self.assertIs(ownership.safe_error, safe_error)

    def test_default_limits_match_the_previous_wire_values(self):
        self.assertEqual(limits({}, 2), {
            'max_workers': 8, 'max_running': 2, 'max_active': 3, 'max_queued': 3,
            'max_rows': 10000, 'max_tenants': 15, 'max_worker_generations': 10000,
            'max_tenant_active': 3, 'max_tenant_running': 2, 'max_tenant_queued': 3,
            'max_tenant_storage_reservation_bytes': 2*1024**3,
            'max_global_storage_reservation_bytes': 8*1024**3,
        })
        self.assertEqual(list(limits({}, 2)), sorted(limits({}, 2)))

    def test_capacity_rejects_invalid_worker_counts_before_coercion(self):
        for count in (False, 0, -1, 1.5, None, '2', 2**31):
            with self.subTest(count=count), self.assertRaises(values.Error) as caught:
                limits({}, count)
            self.assertEqual(caught.exception.code, 'worker_running_capacity')
            self.assertEqual(caught.exception.status, 400)

    def test_validation_does_not_mutate_input_and_policy_is_detached(self):
        supplied = {'max_workers': 12, 'max_running': 4, 'max_active': 8}
        original = deepcopy(supplied)
        selected = limits(supplied, 4)
        self.assertEqual(supplied, original)
        selected['max_running'] = 3
        self.assertEqual(supplied, original)
        public = policy(selected)
        public['limits']['max_workers'] = 1
        self.assertEqual(selected['max_workers'], 12)

    def test_authority_does_not_alias_host_identity(self):
        host = {'machine': 'fixture', 'uid': 42, 'pid_namespace': 'fixture-namespace'}
        result = authority(host, 'fixture-authority')
        result['host']['uid'] = -1
        self.assertEqual(host['uid'], 42)
        self.assertFalse(result['multi_node_enabled'])

    def test_json_encoding_and_structured_error_compatibility(self):
        self.assertEqual(values.encoded({'b': 1, 'a': '\u0e01'}), b'{"a":"\\u0e01","b":1}')
        with self.assertRaises(ValueError):
            values.encoded({'a': float('nan')})
        with self.assertRaises(values.Error) as caught:
            values.require(False, 'fixed_code', 409)
        self.assertEqual((caught.exception.status, caught.exception.code, str(caught.exception)),
                         (409, 'fixed_code', 'fixed code'))
        self.assertIsNone(caught.exception.details)

    def test_known_error_reason_survives_without_messages_or_details(self):
        result = safe_error(values.Error(507, 'resident_vram_budget_exceeded',
                                        'Bearer private', {'token': 'private'}))
        self.assertEqual(result, {'code': 'resident_vram_budget_exceeded', 'type': 'Error'})
        self.assertNotIn('private', json.dumps(result))

    def test_arbitrary_exception_codes_and_unknown_error_codes_are_redacted(self):
        error = RuntimeError('private')
        error.code = 'resident_vram_budget_exceeded'
        self.assertEqual(safe_error(error)['code'], 'worker_failed')
        self.assertEqual(safe_error(values.Error(500, 'private_customer_key', 'private'))['code'], 'worker_failed')
        self.assertEqual(safe_error(values.Error(500, 'input_changed', 'private'), 'cleanup_failed')['code'], 'cleanup_failed')

    def test_exception_type_redaction_keeps_fixed_schema(self):
        error = type('private type with spaces', (Exception,), {})('private')
        self.assertEqual(safe_error(error), {'code': 'worker_failed', 'type': 'Exception'})

    def test_supported_boundary_annotations_resolve_without_forward_reference_errors(self):
        from aureon.serving.backend import Backend
        from aureon.serving.control import Control
        from aureon.serving.replica_control import ReplicaControl
        self.assertIs(typing.get_type_hints(limits)['return'], ReplicaLimits)
        self.assertIs(typing.get_type_hints(Backend.close)['return'], BackendCleanup)
        from aureon.serving.backend import BackendControl
        self.assertIs(typing.get_type_hints(Backend.__init__)['control'], BackendControl)
        self.assertTrue(BackendControl._is_protocol)
        self.assertEqual(typing.get_type_hints(ReplicaControl.claim_replica)['return'], AttemptLease | None)
        self.assertIn('replica_generation', AttemptLease.__optional_keys__)
        self.assertIn('request', AttemptLease.__required_keys__)


if __name__ == '__main__':
    unittest.main()
