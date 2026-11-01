"""EO admission and real SDK facade dispatch with a fake model session."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from aureon.api.eo import (EOGrid, ConditioningSpec, GeneratedTarget, SamplingRecipe,
                           EORequest, UnsupportedRoute, eo_capabilities, plan_eo, generate_eo)
from aureon.api import GenerationInput, GenerationEngine, GenerationRecipe
from aureon.api import generation_engine as facade
from aureon.terramind_native.common import INPUT_BANDS
from aureon.terramind_native.session import SessionClosed, SessionBusy


def request():
    grid = EOGrid('EPSG:32647', (10., 0., 500000., 0., -10., 1500000.), (224, 224))
    observed = ConditioningSpec('S2L2A', tuple(INPUT_BANDS), 'surface_reflectance_x10000',
                                'none', 'explicit_all_valid', grid, 'artificial-fixture', '2026-10-30T00:00:00Z')
    return EORequest('terramind-v1-base', (observed,),
                     (GeneratedTarget('S1GRD', ('VV', 'VH'), 'dB', grid),),
                     SamplingRecipe.terramind(GenerationRecipe(seed=31)))


class EOTests(unittest.TestCase):
    def test_cold_import_plan_has_no_neural_or_geospatial_dependencies(self):
        code = '''import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'numpy','torch','rasterio','terratorch','diffusers'}: raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from aureon.api import eo_capabilities
assert len(eo_capabilities()) == 4
assert all('copgen' not in c.model_id for c in eo_capabilities())
'''
        child = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_plans_are_immutable_and_keep_explicit_grids_and_sampling(self):
        value = request()
        first = plan_eo(value)
        self.assertEqual(first, plan_eo(value))
        with self.assertRaises(FrozenInstanceError):
            value.conditioning[0].grid.crs = 'EPSG:4326'
        self.assertNotEqual(first.request_sha256,
                            plan_eo(replace(value, sampling=SamplingRecipe.terramind(GenerationRecipe(seed=32)))).request_sha256)
        for capability in eo_capabilities():
            self.assertEqual(plan_eo(replace(value, model_id=capability.model_id)).capability, capability)
        self.assertEqual(value.conditioning[0].acquired_at, '2026-10-30T00:00:00Z')

    def test_unsupported_routes_fail_without_guessing_or_downloading(self):
        value = request()
        observed, target = value.conditioning[0], value.targets[0]
        bad = [replace(value, model_id='copgen-base'), replace(value, model_id='copgen-beta'),
               replace(value, targets=(replace(target, modality='S1RTC'),)),
               replace(value, targets=(replace(target, units='linear'),)),
               replace(value, targets=(replace(target, grid=replace(target.grid, shape=(112, 112))),)),
               replace(value, conditioning=(replace(observed, normalization='standardized'),)),
               replace(value, conditioning=(replace(observed, bands=('B02',)),)),
               replace(value, conditioning=(replace(observed, mask_policy='guess_cloud_mask'),)),
               replace(value, sampling=SamplingRecipe('other-diffusion-v1', 31, (('steps', 4),)))]
        for candidate in bad:
            with self.subTest(candidate=candidate), self.assertRaises(UnsupportedRoute):
                plan_eo(candidate)

    def test_invalid_values_and_hidden_mutable_recipe_rejected(self):
        for params in ([('steps', 4)], (('steps', True),), (('steps', float('nan')),),
                       (('steps', 1), ('steps', 2)), (('steps', {'class_path': 'external'}),)):
            with self.assertRaises((TypeError, ValueError)):
                SamplingRecipe('other', 1, params)
        for seed in (True, -1, 2**63):
            with self.assertRaises(ValueError): SamplingRecipe('other', seed, ())
        for seed in (2**32, 2**63 - 1):
            recipe = SamplingRecipe.terramind(GenerationRecipe(seed=seed))
            self.assertEqual(plan_eo(replace(request(), sampling=recipe)).request.sampling.seed, seed)
        with self.assertRaises(ValueError):
            EOGrid('EPSG:32647', (0., 0., 0., 0., 0., 0.), (224, 224))
        with self.assertRaises(ValueError):
            replace(request().conditioning[0], acquired_at='2026-10-30')

    def input(self, value):
        spec = value.conditioning[0]
        image = np.ones((12, 224, 224), np.float32)
        data = GenerationInput(image, bands=spec.bands, units=spec.units, valid_mask=np.ones((224, 224), bool),
                               grid=spec.grid.native(), source_id=spec.source_id)
        image[:] = 9
        self.assertTrue(np.all(data.snapshot()[0] == 1))
        return data

    def test_existing_engine_sdk_is_called_with_unchanged_recipe_and_owned_lifecycle(self):
        value = request()
        data = self.input(value)
        fake = MagicMock()
        result = {'array': np.zeros((2, 224, 224), np.float32), 'valid_mask': np.ones((224, 224), bool),
                  'provenance': {'native': {'retained': True}}}
        fake.generate.return_value = result
        fake.close.return_value = {'passed': True, 'model_load_count': 1}
        def config(**kwargs):
            Path(kwargs['output']).write_text(json.dumps({'schema': 'cpu-fixture'}))
        with tempfile.TemporaryDirectory() as tmp, patch('aureon.terramind_native.offline.create_config', side_effect=config), \
                patch.object(facade, 'open_model_session', return_value=fake) as opened:
            engine = GenerationEngine(model_file='missing', tokenizer_file='missing', work=Path(tmp) / 'work', execute_model=True)
            try:
                output = generate_eo(engine, value, data, job_id='eo-test')
                self.assertEqual(opened.call_count, 1)
                self.assertEqual(fake.generate.call_args.kwargs['recipe'], GenerationRecipe(seed=31).native())
                self.assertEqual(fake.generate.call_args.kwargs['execution_profile'], value.execution_profile)
                self.assertIs(output['array'], result['array'])
                self.assertIs(output['valid_mask'], result['valid_mask'])
                self.assertNotIn('eo_route', result['provenance'])
                self.assertFalse(output['provenance']['eo_route']['sensor_observation'])
                output['provenance']['native']['retained'] = False
                self.assertTrue(result['provenance']['native']['retained'])
                fake.close.assert_not_called()
            finally:
                engine.close()
            fake.close.assert_called_once()
            with self.assertRaises(SessionClosed): generate_eo(engine, value, data)

    def test_mismatch_and_busy_rejected_without_model_binding(self):
        value = request()
        data = self.input(value)
        with tempfile.TemporaryDirectory() as tmp, patch.object(facade, 'open_model_session', side_effect=AssertionError('model opened')):
            with GenerationEngine(model_file='missing', tokenizer_file='missing', work=Path(tmp) / 'work', execute_model=True) as engine:
                wrong = replace(value, conditioning=(replace(value.conditioning[0], source_id='different'),))
                with self.assertRaises(ValueError): generate_eo(engine, wrong, data)
                engine._direct_active = True
                try:
                    with self.assertRaises(SessionBusy): generate_eo(engine, value, data)
                finally:
                    engine._direct_active = False
                self.assertFalse(engine.work.exists())


if __name__ == '__main__': unittest.main()
