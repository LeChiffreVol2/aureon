"""Checkpoint compatibility after removing the external inference client."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from aureon.catalog import get_catalog, plan_inference
from aureon.checkpoint_bundle import build_checkpoint_config, pinned_legacy_model_args
from aureon.runtime import verify_bundle


class CheckpointBundleTests(unittest.TestCase):
    def fixture(self, folder, schema):
        model = get_catalog()['models'][1]
        args = pinned_legacy_model_args(model['id'])
        args['decoder_scale_modules'] = False
        config = build_checkpoint_config(args, as_dict=True)
        (folder / 'config.json').write_text(json.dumps(config), encoding='utf-8')
        # Verification hashes bytes only and must never deserialize this fixture.
        (folder / 'pytorch_model.bin').write_bytes(b'not a serialized model')
        hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                  for name in ('config.json', 'pytorch_model.bin')}
        receipt = dict(schema=schema, model_id=model['id'], source_identity=model['identity'],
                       verified_source_sha256=model['identity']['published_checkpoint_sha256'],
                       state_tensors_dropped=[], state_key_rewrites=[], dtype_conversion=None,
                       source_as_dict=True, files_sha256=hashes,
                       model_executed=False, downloads_started=False, server_started=False)
        (folder / 'bundle_receipt.json').write_text(json.dumps(receipt), encoding='utf-8')

    def test_old_and_new_bundle_schemas_keep_file_verification(self):
        for schema in ('aureon.vllm_bundle.v1', 'aureon.checkpoint_bundle.v1'):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                self.fixture(folder, schema)
                result = verify_bundle(folder)
                self.assertEqual(result['model_id'], 'kao-terramind-b')
                self.assertEqual(result['receipt']['schema'], schema)
                self.assertEqual(result['server_checkpoint_attestation'], 'not_available')
                (folder / 'pytorch_model.bin').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    verify_bundle(folder)

    def test_unknown_schema_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder, 'unrecognized')
            with self.assertRaisesRegex(ValueError, 'Unsupported bundle receipt schema'):
                verify_bundle(folder)

    def test_only_native_backend_is_planned_without_claiming_validation(self):
        catalog = get_catalog()
        self.assertEqual([item['id'] for item in catalog['backends']], ['pytorch'])
        for model in catalog['models']:
            plan = plan_inference(model['id'])
            self.assertEqual(plan['backend'], 'pytorch')
            self.assertFalse(plan['execution_enabled'])
            self.assertFalse(plan['ready'])
        with self.assertRaises(ValueError):
            plan_inference('kao-terramind-b', backend='vllm')

    def test_config_keeps_legacy_pyramid_and_disables_pretrained_downloads(self):
        for variant in ('a', 'b', 'c'):
            args = pinned_legacy_model_args('kao-terramind-' + variant)
            before = json.dumps(args, sort_keys=True)
            config = build_checkpoint_config(args, as_dict=True)
            model_args = config['pretrained_cfg']['model']['init_args']['model_args']
            self.assertFalse(model_args['backbone_pretrained'])
            self.assertEqual(model_args['necks'][-1], {'name': 'LearnedInterpolateToPyramidal'})
            self.assertEqual(json.dumps(args, sort_keys=True), before)

    def test_changed_sources_cannot_reuse_prior_gpu_qualification(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('source_identity', root / 'packaging/source_identity.py')
        identity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(identity)
        ledger = json.loads((root / 'packaging/review/ADAPTER-REMOVAL.json').read_text())
        for name, row in ledger['files'].items():
            data = (root / name).read_bytes()
            checked = identity.verify_source(root, name, data, row['original_sha256'])
            self.assertEqual(checked['identity'], 'behavior_changed_unqualified_candidate')
            self.assertFalse(checked['new_gpu_qualification'])
            with self.assertRaises(ValueError):
                identity.verify_source(root, name, data + b'\nMODIFIED = True\n', row['original_sha256'])


if __name__ == '__main__':
    unittest.main()
