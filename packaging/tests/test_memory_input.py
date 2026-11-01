"""Real array validation/ownership contracts; these tests do not run a model."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from aureon.api.memory_input import GenerationInput
from aureon.terramind_native.common import INPUT_BANDS


class MemoryInputTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((12, 224, 224), dtype=np.float32)
        self.image[0, 0, :3] = [-250.0, 0.0, 15000.0]
        self.mask = np.ones((224, 224), dtype=np.bool_)

    def build(self, image=None, **kwargs):
        options = dict(bands=INPUT_BANDS, units="surface_reflectance_x10000", valid_mask=self.mask)
        options.update(kwargs)
        return GenerationInput(self.image if image is None else image, **options)

    def test_no_runtime_imports_on_api_discovery(self):
        source = """import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('numpy','torch','rasterio','terratorch','diffusers'):
   raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from aureon.api.memory_input import GenerationInput
"""
        result = subprocess.run([sys.executable, "-B", "-c", source], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_snapshot_owns_values_and_does_not_clip_or_standardize(self):
        expected = self.image.copy()
        value = self.build()
        identity = value.identity
        self.image.fill(999)
        self.mask.fill(False)
        image, mask, metadata, _ = value.snapshot()
        np.testing.assert_array_equal(image, expected)
        self.assertTrue(mask.all())
        self.assertIsNone(metadata["grid"])
        image.fill(-1); mask.fill(False); metadata["bands"].clear()
        identity["sha256"] = "tampered"
        fresh = value.snapshot()
        np.testing.assert_array_equal(fresh[0], expected)
        self.assertTrue(fresh[1].all())
        self.assertEqual(fresh[2]["bands"], INPUT_BANDS)
        self.assertNotEqual(value.identity["sha256"], "tampered")

    def test_noncontiguous_view_and_canonical_uint8_mask(self):
        contiguous = self.build()
        flipped = self.image[:, :, ::-1]
        value = self.build(flipped, valid_mask=self.mask.astype(np.uint8))
        np.testing.assert_array_equal(value.snapshot()[0], flipped)
        self.assertTrue(value.snapshot()[0].flags.c_contiguous)
        self.assertEqual(contiguous.identity, self.build(valid_mask=self.mask.astype(np.uint8)).identity)
        self.assertNotEqual(value.identity, contiguous.identity)

    def test_rejects_ambiguous_units_normalization_band_order_and_precision(self):
        cases = [dict(units="stored_dn"), dict(units="reflectance"), dict(normalization="standardized"),
                 dict(normalization=True), dict(bands=INPUT_BANDS[::-1]), dict(bands=INPUT_BANDS[:-1])]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError): self.build(**options)
        for dtype in (np.float64, np.float16, np.int16):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError): self.build(self.image.astype(dtype))

    def test_bad_pixels_shapes_and_masks_fail_without_guessing(self):
        for invalid in (np.nan, np.inf):
            bad = self.image.copy(); bad[1, 10, 10] = invalid
            with self.assertRaises(ValueError): self.build(bad)
        with self.assertRaises(ValueError): self.build(self.image[None])
        with self.assertRaises(ValueError): self.build(self.image[:, :-1])
        with self.assertRaises(TypeError): self.build(self.image.tolist())
        for bad in (np.zeros_like(self.mask), np.full(self.mask.shape, 2, dtype=np.uint8), self.mask.astype(np.float32)):
            with self.assertRaises(ValueError): self.build(valid_mask=bad)

    def test_masked_arrays_do_not_lose_hidden_invalid_pixels(self):
        invalid = np.zeros(self.image.shape, bool)
        invalid[0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "masked array"):
            self.build(np.ma.array(self.image, mask=invalid))
        with self.assertRaisesRegex(ValueError, "masked array"):
            self.build(valid_mask=np.ma.array(self.mask, mask=~self.mask))

    def test_grid_optional_but_explicit_and_identity_bound_when_present(self):
        grid = {"crs": "EPSG:32647", "transform": [10, 0, 500000, 0, -10, 1500000]}
        value = self.build(grid=grid, source_id="user-scene")
        self.assertNotEqual(value.identity, self.build().identity)
        grid["transform"][2] = 1
        self.assertEqual(value.metadata["grid"]["transform"][2], 500000)
        self.assertEqual(value.metadata["grid"]["shape"], [224, 224])
        for bad in ({"crs": "EPSG:32647"}, {"crs": "not-a-crs", "transform": [1,0,0,0,-1,0]},
                    {"crs": "EPSG:32647", "transform": [0,0,0,0,0,0]},
                    {"crs": "EPSG:32647", "transform": [1,0,0,0,-1,float("nan")]},
                    {"crs": "EPSG:32647", "transform": [1e308,0,0,0,1e308,0]},
                    {**grid, "shape": None}, {**grid, "shape": [512,512]}, {**grid, "bounds": [0,0,1,1]}):
            with self.subTest(grid=bad), self.assertRaises(ValueError): self.build(grid=bad)

    def test_torchgeo_labels_and_bounds_are_not_silently_reinterpreted(self):
        sample = {"image": self.image, "mask": np.full(self.mask.shape, 9, dtype=np.uint8),
                  "crs": "EPSG:32647", "bounds": (0, 2240, 0, 2240)}
        value = GenerationInput.from_torchgeo(sample, bands=INPUT_BANDS, units="reflectance_x10000", valid_mask=self.mask)
        self.assertIsNone(value.metadata["grid"])
        self.assertTrue(value.snapshot()[1].all())
        with self.assertRaises(TypeError):
            GenerationInput.from_torchgeo(sample, bands=INPUT_BANDS, units="reflectance_x10000")
        with self.assertRaises(ValueError):
            GenerationInput.from_torchgeo({}, bands=INPUT_BANDS, units="reflectance_x10000", valid_mask=self.mask)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional CPU Torch is absent")
    def test_cpu_tensor_ownership_and_autograd_are_preserved(self):
        import torch
        tensor = torch.tensor(self.image, requires_grad=True)
        validity = torch.ones((224, 224), dtype=torch.bool)
        value = self.build(tensor, valid_mask=validity)
        self.assertTrue(tensor.requires_grad)
        self.assertIsNone(tensor.grad)
        with torch.no_grad(): tensor.fill_(33)
        validity.fill_(False)
        np.testing.assert_array_equal(value.snapshot()[0], self.image)
        self.assertTrue(value.snapshot()[1].all())
        with self.assertRaises(ValueError): self.build(tensor.to(dtype=torch.float16))


if __name__ == "__main__": unittest.main()
