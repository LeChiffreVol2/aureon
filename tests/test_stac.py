"""Offline STAC selection rejects ambiguity and preserves declared semantics."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import unittest

from aureon.api.stac import AssetBand, STACSelection, resolve_stac_item
from test_eo import request


def fixture():
    observation = request().conditioning[0]
    selection = STACSelection(observation, tuple(AssetBand('image', i + 1, 'a' * 64)
                              for i in range(12)), AssetBand('valid', 1, 'b' * 64), 'one_is_valid_zero_is_invalid')
    item = {'type': 'Feature', 'stac_version': '1.1.0', 'id': 'explicit-item', 'geometry': None,
            'links': [], 'properties': {'datetime': observation.acquired_at},
            'assets': {'image': {'href': 'image.tif', 'type': 'image/tiff',
                                 'eo:bands': [{'name': name} for name in observation.bands]},
                       'valid': {'href': 'valid.tif', 'type': 'image/tiff'}}}
    return item, selection


class STACTests(unittest.TestCase):
    def resolve(self, item, selection):
        return resolve_stac_item(item, selections=(selection,), base_href='https://example.org/item.json')

    def test_explicit_order_owned_metadata_and_unverified_identity(self):
        item, selection = fixture()
        before = deepcopy(item)
        result, = self.resolve(item, selection)
        self.assertEqual(item, before)
        self.assertEqual(result.observation, selection.observation)
        self.assertEqual([v.band_index for v in result.bands], list(range(1, 13)))
        self.assertEqual(result.bands[0].href, 'https://example.org/image.tif')
        self.assertFalse(result.bands[0].identity_verified)
        self.assertEqual(result.asset_bytes_read, 0)
        item['assets']['image']['href'] = 'changed.tif'
        self.assertEqual(result.bands[0].href, 'https://example.org/image.tif')
        with self.assertRaises(FrozenInstanceError): result.bands[0].asset_key = 'changed'

    def test_local_resolution_does_not_require_files_or_infer_grids(self):
        item, selection = fixture()
        result, = resolve_stac_item(item, selections=(selection,), local_root=Path('nonexistent-artificial-root'))
        self.assertTrue(result.bands[0].href.endswith('image.tif'))
        self.assertEqual(result.observation.grid, selection.observation.grid)
        for href in ('../escape.tif', '/absolute.tif', '..%2fescape.tif', 'folder\\image.tif', '//remote/image.tif'):
            item['assets']['image']['href'] = href
            with self.assertRaises(ValueError):
                resolve_stac_item(item, selections=(selection,), local_root=Path('unused'))

    def test_credentials_and_unsupported_protocols_rejected_without_echo(self):
        item, selection = fixture()
        for href in ('https://user:secret@example.org/image.tif', 'https://example.org/image.tif?token=secret',
                     'https://example.org/image.tif#secret', 'http://example.org/image.tif',
                     's3://bucket/image.tif', 'file:///secret.tif', 'https://example.org/image%3ftoken=secret'):
            item['assets']['image']['href'] = href
            with self.assertRaises(ValueError) as caught: self.resolve(item, selection)
            self.assertNotIn('secret', str(caught.exception))

    def test_missing_malformed_mask_band_time_and_projection_fail(self):
        base, selection = fixture()
        changes = [lambda x: x['assets'].pop('valid'), lambda x: x['assets']['image'].pop('type'),
                   lambda x: x['properties'].update(datetime=None),
                   lambda x: x['assets']['image'].update(**{'proj:shape': [112, 112]}),
                   lambda x: x['assets']['image'].update(**{'proj:epsg': 4326}),
                   lambda x: x['assets']['image'].update(**{'proj:code': None}),
                   lambda x: x['assets']['image'].update(**{'proj:transform': [1, 0, 0, 0, -1, 0]}),
                   lambda x: x['assets']['image'].update(**{'eo:bands': [{'name': 'wrong'}]}),
                   lambda x: x['assets']['image'].update(**{'raster:bands': [{'scale': 0.0001}] * 12})]
        for change in changes:
            item = deepcopy(base)
            change(item)
            with self.assertRaises(ValueError): self.resolve(item, selection)
        with self.assertRaises(ValueError): replace(selection, mask_meaning='guess_from_cloud_bits')
        with self.assertRaises(ValueError): replace(selection, valid_mask=AssetBand('image', 1, 'a' * 64))

    def test_expected_identity_conflicts_and_missing_selection_rejected(self):
        item, selection = fixture()
        conflict = replace(selection, bands=(replace(selection.bands[0], sha256='c' * 64), *selection.bands[1:]))
        with self.assertRaises(ValueError): self.resolve(item, conflict)
        with self.assertRaises(ValueError): resolve_stac_item(item, selections=(selection,))
        with self.assertRaises(ValueError): resolve_stac_item(item, selections=(), local_root=Path('unused'))
        with self.assertRaises(ValueError): AssetBand('image', True, 'a' * 64)

    def test_stac11_raster_fields_and_unhandled_projection_are_not_ignored(self):
        base, selection = fixture()
        for field, bad_value in (('raster:scale', 0.0001), ('raster:offset', 100), ('unit', 'reflectance')):
            item = deepcopy(base)
            item['assets']['image']['bands'] = [{field: bad_value}] * 12
            with self.assertRaises(ValueError): self.resolve(item, selection)
        item = deepcopy(base)
        item['assets']['image']['raster:bands'] = [{'scale': 1, 'offset': 0}] * 12
        item['assets']['image']['bands'] = [{'raster:scale': 0.0001}] * 12
        with self.assertRaises(ValueError): self.resolve(item, selection)
        for container in ('properties', 'asset'):
            item = deepcopy(base)
            holder = item['properties'] if container == 'properties' else item['assets']['image']
            holder['raster:scale'] = 0.0001
            with self.assertRaises(ValueError): self.resolve(item, selection)
        for key in ('proj:wkt2', 'proj:projjson'):
            for where in ('properties', 'asset'):
                item = deepcopy(base)
                holder = item['properties'] if where == 'properties' else item['assets']['image']
                holder[key] = 'unhandled conflicting CRS'
                with self.assertRaises(ValueError): self.resolve(item, selection)

    def test_range_only_identity_requires_reader_validator_and_metadata_is_bounded(self):
        item, selection = fixture()
        selection = replace(selection, bands=tuple(replace(v, sha256=None) for v in selection.bands),
                            valid_mask=replace(selection.valid_mask, sha256=None))
        result, = self.resolve(item, selection)
        self.assertIsNone(result.bands[0].expected_sha256)
        self.assertTrue(result.bands[0].reader_validator_required)
        self.assertFalse(result.bands[0].identity_verified)
        item['properties']['huge'] = 'x' * 65537
        with self.assertRaises(ValueError): self.resolve(item, selection)
        item.pop('properties')
        item['properties'] = {'datetime': selection.observation.acquired_at, 'cycle': item}
        with self.assertRaises(ValueError): self.resolve(item, selection)


if __name__ == '__main__': unittest.main()
