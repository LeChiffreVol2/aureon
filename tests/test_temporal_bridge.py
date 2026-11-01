"""Actual TerraTorch TemporalWrapper CPU pooling, with local random/simple encoders."""
from dataclasses import replace
import unittest
import torch
from torch import nn

from aureon.api.temporal import TemporalBridge, TemporalInput, TemporalSpec, GeographicConditioning


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(2, 3, 1)
        self.out_channels = [3]
    def forward(self, values): return [self.projection(values)]
    def __deepcopy__(self, memo): raise AssertionError('Borrowed encoder must not be copied')


def spec(frames=4):
    return TemporalSpec(('red', 'nir'), 'caller_normalized', tuple(f'2026-10-{i+1:02d}T00:00:00Z' for i in range(frames)))


class TemporalTests(unittest.TestCase):
    def test_actual_upstream_pooling_and_subset_aggregation_matches_direct_wrapper(self):
        from terratorch.models.utils import TemporalWrapper
        encoder = Encoder().eval()
        values = torch.arange(2 * 2 * 4 * 3 * 5, dtype=torch.float32).reshape(2, 2, 4, 3, 5) / 100
        owned = TemporalInput(values, torch.ones(2, 4, 3, 5, dtype=torch.bool), spec())
        for pooling, subsets in [('mean', None), ('max', None), ('keep', None), ('concat', None),
                                 ('diff', (2, 2)), ('mean', (1, 3))]:
            with self.subTest(pooling=pooling, subsets=subsets):
                direct = TemporalWrapper(encoder, pooling=pooling, n_timestamps=4,
                                         subset_lengths=list(subsets) if subsets else None)
                direct.training = False
                with torch.no_grad(): expected = direct(values)
                rng_before = torch.get_rng_state().clone()
                with TemporalBridge(encoder, spec=spec(), pooling=pooling, subset_lengths=subsets) as bridge:
                    result = bridge.predict(owned, seed=2**40)
                    self.assertTrue(torch.equal(expected[0], result.values[0]))
                    self.assertFalse(result.values[0].is_inference())
                    self.assertEqual(result.timestamps, spec().timestamps)
                    self.assertFalse(result.coordinate_conditioning_used)
                    result.values[0].zero_()
                    self.assertTrue(torch.equal(bridge.predict(owned).values[0], expected[0]))
                self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
                self.assertFalse(encoder.training)
                self.assertTrue(encoder.projection.weight.requires_grad)

    def test_owned_temporal_data_and_exact_invalid_contracts(self):
        values = torch.ones(1, 2, 4, 3, 5)
        mask = torch.ones(1, 4, 3, 5, dtype=torch.bool)
        data = TemporalInput(values, mask, spec())
        values.zero_(); mask.zero_()
        self.assertTrue(bool(data.snapshot().all()))
        data.snapshot().zero_()
        self.assertTrue(bool(data.snapshot().all()))
        with self.assertRaises(ValueError): TemporalInput(values, mask, spec())
        with self.assertRaises(ValueError): TemporalInput(values.double(), torch.ones_like(mask), spec())
        with self.assertRaises(ValueError): replace(spec(), timestamps=tuple(reversed(spec().timestamps)))
        with self.assertRaises(ValueError): replace(spec(), timestamps=(spec().timestamps[0],) * 4)
        encoder = Encoder().eval()
        with TemporalBridge(encoder, spec=spec(), pooling='mean') as bridge:
            wrong = TemporalInput(values, torch.ones_like(mask), replace(spec(), units='different'))
            with self.assertRaises(ValueError): bridge.predict(wrong)
        with self.assertRaises(RuntimeError): bridge.predict(data)
        with self.assertRaises(ValueError): TemporalBridge(encoder, spec=spec(), pooling='diff')
        with self.assertRaises(ValueError): TemporalBridge(encoder, spec=spec(), pooling='mean', subset_lengths=(1, 1))

    def test_coordinates_are_explicit_and_never_inferred_from_crs(self):
        values = GeographicConditioning(((0., 0.), (0., 90.), (90., 0.)), 'unit_sphere_xyz').encoded()
        torch.testing.assert_close(torch.tensor(values), torch.eye(3), atol=1e-6, rtol=0)
        degrees = GeographicConditioning(((12., 100.),), 'degrees_lat_lon')
        self.assertEqual(degrees.encoded(), ((12., 100.),))
        for pair in (((91., 0.),), ((0., 181.),), ((float('nan'), 0.),)):
            with self.assertRaises(ValueError): GeographicConditioning(pair, 'degrees_lat_lon')
        with self.assertRaises(ValueError): GeographicConditioning(((1., 2.),), 'EPSG:32647')


if __name__ == '__main__': unittest.main()
