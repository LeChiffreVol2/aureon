"""Prerequisite diagnostics are offline and cannot accidentally load models."""
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch

from aureon.api.diagnostics import diagnose_runtime


class DiagnosticsTests(unittest.TestCase):
    def test_missing_assets_and_environment_are_actionable(self):
        with patch('aureon.api.diagnostics.inspect_profile', return_value={'status': 'failed', 'remedy': 'Use the locked environment.'}):
            report = diagnose_runtime('terramind-v1-tiny')
        self.assertEqual(report['status'], 'blocked')
        self.assertFalse(report['model_imported'])
        self.assertEqual({i['code'] for i in report['issues']}, {'environment_mismatch', 'asset_missing'})

    def test_exact_hash_passes_and_same_size_wrong_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'; path.write_bytes(b'good')
            pin = {'role': 'checkpoint', 'filename': path.name, 'bytes': 4,
                   'sha256': hashlib.sha256(b'good').hexdigest(), 'source_url': 'fixture://no-download'}
            with patch('aureon.api.diagnostics.list_assets', return_value=[pin]), \
                    patch('aureon.api.diagnostics.inspect_profile', return_value={'status': 'passed'}):
                unchecked = diagnose_runtime('terramind-v1-tiny-backbone', assets_dir=directory)
                self.assertEqual(unchecked['status'], 'incomplete')
                checked = diagnose_runtime('terramind-v1-tiny-backbone', assets_dir=directory, verify_assets=True)
                self.assertEqual(checked['status'], 'passed')
                self.assertTrue(checked['assets'][0]['identity_verified'])
                path.write_bytes(b'evil')
                bad = diagnose_runtime('terramind-v1-tiny-backbone', assets_dir=directory, verify_assets=True)
                self.assertEqual(bad['status'], 'blocked')
                self.assertEqual(bad['issues'][0]['code'], 'asset_hash_mismatch')

    def test_required_tokenizers_follow_explicit_workflow(self):
        with patch('aureon.api.diagnostics.inspect_profile', return_value={'status': 'passed'}):
            report = diagnose_runtime('terramind-v1-small-tim', tim_modalities=['S1GRD'])
            self.assertEqual([r['modality'] for r in report['assets'] if r['role'] == 'tokenizer'], ['S1GRD'])
            backbone = diagnose_runtime('terramind-v1-small-backbone')
            self.assertEqual(len(backbone['assets']), 1)

    def test_invalid_selection_fails_before_asset_io(self):
        with patch('aureon.api.diagnostics._identity', side_effect=AssertionError('No file reads')):
            for kwargs in ({'environment': 'scene'}, {'modalities': 'S2L2A'}, {'min_free_bytes': True},
                           {'tim_modalities': ['S1GRD']}, {'modalities': ['made-up']}):
                with self.assertRaises(ValueError):
                    diagnose_runtime('terramind-v1-tiny', **kwargs)

    def test_tim_overlap_matches_encoder_eligibility(self):
        with patch('aureon.api.diagnostics.inspect_profile', return_value={'status': 'passed'}):
            with self.assertRaisesRegex(ValueError, 'input/target overlap'):
                diagnose_runtime('terramind-v1-small-tim', modalities=['S2L2A'], tim_modalities=['S2L2A'])
            for tokenized in ('LULC', 'NDVI'):
                report = diagnose_runtime('terramind-v1-small-tim', modalities=[tokenized], tim_modalities=[tokenized])
                self.assertEqual(report['tim_modalities'], [tokenized])

    def test_work_inspection_creates_nothing_and_checks_disk_floor(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('aureon.api.diagnostics.inspect_profile', return_value={'status': 'passed'}), \
                patch('aureon.api.diagnostics.list_assets', return_value=[]):
            path = Path(directory) / 'not-created' / 'job'
            result = diagnose_runtime('terramind-v1-tiny', work=path, min_free_bytes=2**63)
            self.assertEqual(result['disk']['status'], 'failed')
            self.assertFalse(path.parent.exists())


if __name__ == '__main__':
    unittest.main()
