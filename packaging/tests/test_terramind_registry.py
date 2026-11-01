"""Family dispatch, discovery and artifact identities without model execution."""
import contextlib
from dataclasses import FrozenInstanceError
import io
import json
import unittest
from unittest.mock import patch

from aureon.api import ExecutionRequired, GenerationRequest, inspect_model, list_models, open_model_session
from aureon.api.assets import list_assets
from aureon.api.cli import main
from aureon.terramind_native.common import MODEL_SHA, TOKENIZER_SHA
from aureon.terramind_native.families import (
    BACKBONE_MODELS, GENERATION_MODELS, TIM_MODELS, CHECKPOINTS, PINNED_S1GRD,
    get_terramind_spec, list_terramind_specs,
)


class TerraMindRegistry(unittest.TestCase):
    def test_twelve_explicit_modes_and_unchanged_kao(self):
        specs = list_terramind_specs()
        self.assertEqual(len(specs), 12)
        self.assertEqual(len({s.model_id for s in specs}), 12)
        for spec in specs:
            item = inspect_model(spec.model_id)
            self.assertEqual(item['upstream_name'], spec.upstream_name)
            self.assertEqual(item['mode'], spec.mode)
            self.assertFalse(item['new_variant_gpu_qualified'])
        self.assertEqual(len(list_models()), 15)
        self.assertFalse(inspect_model('kao-terramind-a')['resident'])
        self.assertTrue(inspect_model('kao-terramind-b')['resident'])

    def test_immutable_pins_and_original_base(self):
        self.assertEqual(CHECKPOINTS['base'].sha256, MODEL_SHA)
        self.assertEqual(PINNED_S1GRD.sha256, TOKENIZER_SHA)
        with self.assertRaises(TypeError):
            CHECKPOINTS['base'] = CHECKPOINTS['tiny']
        with self.assertRaises(FrozenInstanceError):
            CHECKPOINTS['base'].sha256 = 'modified'
        for spec in list_terramind_specs():
            assets = list_assets(spec.model_id)
            self.assertEqual(assets[0]['sha256'], spec.checkpoint.sha256)
            for asset in assets:
                self.assertRegex(asset['sha256'], r'^[0-9a-f]{64}$')
                self.assertRegex(asset['revision'], r'^[0-9a-f]{40}$')
                self.assertIn('/resolve/' + asset['revision'] + '/', asset['source_url'])
        self.assertEqual({a['modality'] for a in list_assets(BACKBONE_MODELS[0]) if a['role'] == 'tokenizer'}, {'LULC', 'NDVI'})
        self.assertEqual(len(list_assets(GENERATION_MODELS[0])), 2)

    def test_request_task_cannot_alias_encoder_as_generation(self):
        for model in (*BACKBONE_MODELS, *TIM_MODELS):
            with self.assertRaises(ValueError):
                GenerationRequest('missing', 'missing', 'job', '1', model_id=model)
        for model in GENERATION_MODELS:
            request = GenerationRequest('missing', 'missing', 'job', '1', model_id=model)
            self.assertEqual(request.model_id, model)
            with self.assertRaises(ExecutionRequired):
                open_model_session(model, config='missing', work='missing')

    def test_wrong_checkpoint_identity_rejected_before_session(self):
        with patch('aureon.api.models.require_profile'), \
             patch('aureon.terramind_native.offline.inspect_config', return_value={'model_id': 'terramind-v1-base'}), \
             patch('aureon.terramind_native.TerraMindSession') as session:
            for model in GENERATION_MODELS:
                if model == 'terramind-v1-base':
                    continue
                with self.assertRaisesRegex(ValueError, 'model_id'):
                    open_model_session(model, config='fixture', work='unused', execute_model=True)
            session.assert_not_called()

    def test_cli_model_and_assets_discovery(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(['models']), 0)
        self.assertEqual(len(json.loads(out.getvalue())['models']), 15)
        for model in (GENERATION_MODELS[0], BACKBONE_MODELS[-1], TIM_MODELS[1]):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(['assets', '--model', model]), 0)
            response = json.loads(out.getvalue())
            self.assertFalse(response['downloaded'])
            self.assertTrue(all(a['model_id'] == model for a in response['assets']))

    def test_names_do_not_infer_a_different_mode(self):
        for value in ('terramind-large', 'terramind_v1_base', 'terramind-v1-base-generate', 'kao-terramind-b'):
            with self.assertRaises(ValueError):
                get_terramind_spec(value)

    def test_encoder_dispatch_keeps_mode_and_explicit_options(self):
        with patch('aureon.api.terramind_encoder.TerraMindEncoder') as encoder:
            for model in (*BACKBONE_MODELS, *TIM_MODELS):
                result = open_model_session(model, model_file='weights', modalities=['S2L2A'],
                                            execute_model=True)
                self.assertIs(result, encoder.return_value)
                self.assertEqual(encoder.call_args.args, (model,))
                self.assertTrue(encoder.call_args.kwargs['execute_model'])
            with self.assertRaises(ValueError):
                open_model_session(BACKBONE_MODELS[0], model_file='weights', modalities=['S2L2A'],
                                   denoiser='graph', execute_model=True)

    def test_encoder_file_command_needs_execution_opt_in(self):
        from aureon.api.encoder_io import encode_file
        with self.assertRaises(ExecutionRequired):
            encode_file(model_id=TIM_MODELS[0], model_file='missing', input_file='missing', output='missing')


if __name__ == '__main__':
    unittest.main()
