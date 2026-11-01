"""Requested-model binding at the installed service adapter; no HTTP/model calls."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aureon.serving.backend import Backend
from aureon.cloud.contracts import Error


class ServiceModelBinding(unittest.TestCase):
    def test_wrong_checkpoint_is_rejected_before_startup_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            config = work/'config.json'
            config.write_text('{}')
            control = SimpleNamespace(execution='engine', selected={
                'task': 'generation', 'environment': 'generation',
                'model_id': 'terramind-v1-large', 'denoiser': 'native', 'reuse': 'off'})
            with patch('aureon.serving.backend.installed_binding', return_value={}), \
                 patch('aureon.api.profiles.require_profile'), \
                 patch('aureon.terramind_native.offline.inspect_config', return_value={'model_id': 'terramind-v1-base'}):
                with self.assertRaises(Error) as rejected:
                    Backend(control, work, config=config, execute_model=True)
                self.assertEqual('model_config_mismatch', rejected.exception.code)
            self.assertFalse((work/'installed-startup.json').exists())

    def test_locked_base_service_rejects_other_variant_without_fallback(self):
        class StopBeforeInference(Exception):
            pass
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            config, manifest = work/'config.json', work/'input.json'
            config.write_text('{}'); manifest.write_text('{}')
            control = SimpleNamespace(execution='engine', selected={
                'task': 'generation', 'environment': 'generation',
                'model_id': 'terramind-v1-small', 'denoiser': 'native', 'reuse': 'off'})
            with patch('aureon.serving.backend.installed_binding', return_value={}), \
                 patch('aureon.api.profiles.require_profile'), \
                 patch('aureon.terramind_native.offline.inspect_config', return_value={'model_id': 'terramind-v1-small'}), \
                 patch('aureon.api.open_model_session', side_effect=StopBeforeInference) as opened:
                backend = Backend(control, work, config=config, execute_model=True)
                with self.assertRaises(Error) as rejected:
                    backend.run({'job_id': 'j', 'attempt': 1, 'manifest': str(manifest)})
                self.assertEqual(rejected.exception.code, 'selected_original_base_only')
                opened.assert_not_called()


if __name__ == '__main__':
    unittest.main()
