"""CPU DataLoader -> owned requests -> real queue, using a fake neural session."""
from contextlib import closing
from pathlib import Path
import gc
import random
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from aureon.api.generation_bridges import GenerationRecipe, TorchGeoBatchAdapter
from aureon.api.generation_engine import GenerationEngine
from aureon.terramind_native.common import INPUT_BANDS


def records(count=5):
    for index in range(count):
        yield {'image': torch.full((12, 224, 224), float(index + 1)),
               'valid': torch.ones((224, 224), dtype=torch.bool),
               'mask': torch.full((224, 224), 8, dtype=torch.int64),
               'crs': 'EPSG:32647', 'transform': [10., 0., index * 2240., 0., -10., 1000.],
               'id': f'scene-{index}'}


def adapter(**kwargs):
    options = dict(bands=INPUT_BANDS, units='surface_reflectance_x10000',
                   valid_mask=lambda batch, index: batch['valid'][index])
    options.update(kwargs)
    return TorchGeoBatchAdapter(**options)


def batch(count=2):
    return next(iter(DataLoader(list(records(count)), batch_size=count)))


class FakeSession:
    def __init__(self): self.calls = []; self.closes = 0
    def generate(self, value, **options):
        image, mask, metadata, _ = value.snapshot()
        self.calls.append(options)
        return {'array': image[:2].copy(), 'valid_mask': mask,
                'provenance': {'seed': options['recipe']['seed'], 'source_id': metadata['source_id']}}
    def close(self): self.closes += 1; return {'passed': True}
    def metadata(self): return {'calls': len(self.calls)}


class DataLoaderBridge(unittest.TestCase):
    def test_real_collated_loader_preserves_order_grid_and_explicit_validity(self):
        bridge = adapter(grid=lambda b, i: {'crs': b['crs'][i], 'transform': [v[i].item() for v in b['transform']]},
                         source_id=lambda b, i: b['id'][i])
        values = list(bridge.iter_inputs(DataLoader(list(records()), batch_size=2)))
        self.assertEqual(len(values), 5)
        for index, value in enumerate(values):
            image, mask, meta, _ = value.snapshot()
            self.assertTrue((image == index + 1).all())
            self.assertTrue(mask.all())  # The collated training mask contains 8, not validity.
            self.assertEqual(meta['source_id'], f'scene-{index}')
            self.assertEqual(meta['grid']['transform'][2], index * 2240.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_nested_terratorch_modality_is_explicitly_selected(self):
        source = batch(1)
        source['image'] = {'S2L2A': source['image']}
        with self.assertRaises(ValueError): next(adapter().iter_inputs([source]))
        bridge = adapter(image=lambda b: b['image']['S2L2A'])
        self.assertTrue((next(bridge.iter_inputs([source])).snapshot()[0] == 1).all())

    def test_batch_limits_reject_before_metadata_callback(self):
        source = batch(2)
        selector = unittest.mock.Mock(side_effect=AssertionError('must not read mask'))
        for limits in ({'max_batch_size': 1}, {'max_batch_image_bytes': source['image'].numel() * 4 - 1}):
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError): next(adapter(valid_mask=selector, **limits).iter_inputs([source]))
        selector.assert_not_called()

    def test_invalid_precision_shape_layout_and_masks_fail_closed(self):
        source = batch(1)
        for image in (source['image'].half(), source['image'][0], source['image'].to_sparse(),
                      np.ma.array(source['image'].numpy(), mask=False)):
            with self.subTest(type=type(image), shape=image.shape):
                with self.assertRaises((ValueError, TypeError)):
                    next(adapter().iter_inputs([{**source, 'image': image}]))
        with self.assertRaises(ValueError):
            next(adapter(valid_mask=lambda b, i: b['mask'][i]).iter_inputs([source]))
        source['image'][0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'): next(adapter().iter_inputs([source]))

    def test_units_normalization_and_callback_configuration_are_explicit(self):
        for options in ({'units': 'DN'}, {'normalization': 'standardized'}, {'bands': list(reversed(INPUT_BANDS))},
                        {'max_batch_size': True}, {'max_batch_image_bytes': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError): adapter(**options)
        with self.assertRaises(TypeError): adapter(valid_mask=None)
        with self.assertRaises(TypeError): adapter(grid={})

    def test_input_snapshot_owns_storage_and_autograd_is_unchanged(self):
        source = batch(1)
        source['image'].requires_grad_(True)
        value = next(adapter().iter_inputs([source]))
        with torch.no_grad(): source['image'].fill_(42)
        self.assertTrue(source['image'].requires_grad)
        self.assertIsNone(source['image'].grad)
        owned = value.snapshot()[0]
        self.assertTrue((owned == 1).all())
        owned.fill(91)
        self.assertTrue((value.snapshot()[0] == 1).all())

    def test_identity_seeds_survive_batch_grouping_and_do_not_touch_rng(self):
        bridge = adapter(source_id=lambda b, i: b['id'][i])
        samples = list(records(4))
        # Supply explicit non-global loader generators so loader setup itself
        # does not change the caller's global Torch RNG.
        loaders = [DataLoader(samples, batch_size=size, generator=torch.Generator().manual_seed(9)) for size in (1, 3)]
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        groups = [list(bridge.iter_jobs(loader, recipe=GenerationRecipe(seed=21), seed_policy='input_identity')) for loader in loaders]
        self.assertEqual([j.recipe.seed for j in groups[0]], [j.recipe.seed for j in groups[1]])
        self.assertEqual(len({j.recipe.seed for j in groups[0]}), 4)
        self.assertTrue(all(0 <= j.recipe.seed < 2**63 for j in groups[0]))
        self.assertEqual(random.getstate(), before[0])
        after = np.random.get_state(); self.assertEqual(after[0], before[1][0]); np.testing.assert_array_equal(after[1], before[1][1])
        self.assertEqual(after[2:], before[1][2:]); self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_same_seed_and_recipe_bridge_preserve_upstream_sampling_options(self):
        recipe = GenerationRecipe.from_terratorch(dict(modalities=['S2L2A'], output_modalities=['S1GRD'], standardize=True,
            img_size=224, decoding_steps=1, temps=1., top_p=.8, top_k=0, timesteps=12),
            seed=59, model_id='terramind-v1-base', terratorch_version='1.2.13')
        jobs = list(adapter().iter_jobs([batch(2)], recipe=recipe, execution_profile='reference', job_id=lambda value, i: f'item-{i}'))
        self.assertEqual([j.recipe for j in jobs], [recipe, recipe])
        self.assertEqual([j.job_id for j in jobs], ['item-0', 'item-1'])
        self.assertTrue(all(j.execution_profile == 'reference' for j in jobs))

    def test_max_samples_and_stop_do_not_overfetch_next_batch(self):
        reads = []; closes = []
        def source():
            try:
                for i in range(3): reads.append(i); yield batch(2)
            finally: closes.append(True)
        values = list(adapter().iter_inputs(source(), max_samples=2))
        self.assertEqual(len(values), 2); self.assertEqual(reads, [0]); self.assertEqual(closes, [True])
        reads.clear(); closes.clear()
        self.assertEqual(list(adapter().iter_inputs(source(), stop=lambda: True)), [])
        self.assertEqual(reads, [])
        class PrefetchingLoader:
            def __iter__(self): raise AssertionError('cancelled bridge must not start a loader iterator')
        self.assertEqual(list(adapter().iter_inputs(PrefetchingLoader(), stop=lambda: True)), [])
        stopped = [False]
        with closing(adapter().iter_inputs(source(), stop=lambda: stopped[0])) as stream:
            next(stream); stopped[0] = True
            self.assertEqual(list(stream), [])
        self.assertEqual(reads, [0]); self.assertEqual(closes, [True])

    def test_iterator_close_releases_current_batch_and_closes_upstream(self):
        refs = []; closed = []
        def source():
            value = batch(3); refs.append(weakref.ref(value['image']))
            try: yield value
            finally: closed.append(True)
        stream = adapter().iter_jobs(source())
        result = next(stream)
        self.assertIsNotNone(refs[0]())
        stream.close(); gc.collect()
        self.assertIsNone(refs[0]()); self.assertEqual(closed, [True])
        self.assertTrue((result.input_data.snapshot()[0] == 1).all())

    def test_primary_metadata_error_is_not_replaced_by_cleanup_error(self):
        def source():
            try: yield batch(1)
            finally: raise RuntimeError('cleanup failure')
        def broken(batch, index): raise ValueError('metadata failure')
        with self.assertRaisesRegex(ValueError, 'metadata failure') as caught:
            next(adapter(valid_mask=broken).iter_inputs(source()))
        self.assertTrue(any('cleanup failure' in note for note in getattr(caught.exception, '__notes__', [])))
        self.assertTrue(any('batch 0, item 0' in note for note in caught.exception.__notes__))

    def test_invalid_job_options_do_not_consume_batches(self):
        def source(): raise AssertionError('must not consume'); yield batch(1)
        for options in ({'seed_policy': 'index'}, {'recipe': 9}, {'job_id': 'not callable'}, {'execution_profile': 'fast'}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                next(adapter().iter_jobs(source(), **options))

    def test_real_queue_consumes_batched_loader_with_one_resident_session(self):
        fake = FakeSession()
        bridge = adapter(source_id=lambda b, i: b['id'][i])
        jobs = bridge.iter_jobs(DataLoader(list(records()), batch_size=2), seed_policy='input_identity')
        with tempfile.TemporaryDirectory() as temporary, patch('aureon.terramind_native.offline.create_config'), \
                patch('aureon.api.generation_engine.open_model_session', return_value=fake) as opening:
            with GenerationEngine(model_file='unused', tokenizer_file='unused', work=Path(temporary)/'work',
                                  max_pending=2, max_results=2, execute_model=True) as engine:
                with engine.generate_many(jobs, max_in_flight=2) as results:
                    output = list(results)
                self.assertEqual(opening.call_count, 1)
                self.assertEqual([r['provenance']['source_id'] for r in output], [f'scene-{i}' for i in range(5)])
                self.assertEqual(len(fake.calls), 5)
                self.assertLessEqual(engine.metadata()['queue']['high_water_pending'], 2)
                self.assertEqual(engine.metadata()['queue']['reserved_results'], 0)
                output[0]['array'].fill(55)
                self.assertTrue((output[1]['array'] == 2).all())
            self.assertEqual(fake.closes, 1)
        self.assertFalse(torch.cuda.is_initialized())

    def test_early_result_exit_closes_loader_and_retires_reserved_work(self):
        fake = FakeSession(); reads = []; closed = []
        def batches():
            try:
                for i in range(20): reads.append(i); yield batch(2)
            finally: closed.append(True)
        with tempfile.TemporaryDirectory() as temporary, patch('aureon.terramind_native.offline.create_config'), \
                patch('aureon.api.generation_engine.open_model_session', return_value=fake):
            with GenerationEngine(model_file='unused', tokenizer_file='unused', work=Path(temporary)/'work',
                                  max_pending=2, max_results=2, execute_model=True) as engine:
                with engine.generate_many(adapter().iter_jobs(batches()), max_in_flight=2) as results:
                    next(results)
                self.assertEqual(reads, [0]); self.assertEqual(closed, [True])
            self.assertLessEqual(len(fake.calls), 2)
            self.assertEqual(fake.closes, 1)


if __name__ == '__main__': unittest.main()
