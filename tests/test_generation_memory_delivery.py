"""Real CPU array/export/schema boundaries. No checkpoint or model execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio

from aureon.api import GenerationInput, GenerationRequest, export_generation, run_model
from aureon.api import profiles
from aureon.terramind_native import loader, offline
from aureon.terramind_native.common import INPUT_BANDS, MODEL_SHA, TOKENIZER_SHA
from aureon.terramind_native.memory_output import array_identity


class MemoryDeliveryTests(unittest.TestCase):
    def test_assets_only_creation_and_strict_loader_schema(self):
        with tempfile.TemporaryDirectory(prefix="aureon-assets-test-") as tmp:
            root = Path(tmp)
            for name in ("model", "tokenizer"):
                (root/name).write_bytes(b"CPU fixture, not weights")
            with patch.object(offline, "verify_file"), patch.object(profiles, "require_profile"), \
                    patch.object(offline.metadata, "version", side_effect=lambda n: profiles.PROFILES["generation"][n]):
                offline.create_config(model=root/"model", tokenizer=root/"tokenizer", output=root/"assets.json")
            config = json.loads((root/"assets.json").read_text())
            self.assertEqual(config["schema"], "aureon.terramind_assets.v1")
            self.assertNotIn("input_manifest", config)
            self.assertEqual(config["model"]["sha256"], MODEL_SHA)
            self.assertEqual(config["tokenizer"]["sha256"], TOKENIZER_SHA)
            # Exercise the shared verifier's field/schema and identity checks.
            # Artifact/dependency reads are doubles, never a qualification claim.
            verified = []
            def verify(item, base):
                verified.append(item)
                return Path(base) / item.get("path", "fixture")
            with patch.object(loader, "verify_file", side_effect=verify), \
                    patch.object(loader.metadata, "version", side_effect=lambda n: config["dependencies"][n]), \
                    patch.object(loader.metadata, "distribution"):
                checked, paths, recipe = loader.verify_config(root/"assets.json")
                self.assertNotIn("input_manifest", paths)
                self.assertEqual(len(verified), 40)  # three assets/manifests + pinned 37 source files
                bad = dict(config, input_manifest={"path":"guessed"})
                (root/"bad.json").write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "schema/fields"):
                    loader.verify_config(root/"bad.json")
                bad = dict(config, schema="aureon.terramind_native.v1")
                (root/"old-missing.json").write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "schema/fields"):
                    loader.verify_config(root/"old-missing.json")

    def test_explicit_export_roundtrip_and_reject_mutation_or_missing_grid(self):
        values = np.linspace(-35, 5, 2*224*224, dtype=np.float32).reshape(2,224,224)
        mask = np.ones((224,224), bool)
        grid = {"shape":[224,224], "crs":"EPSG:32647", "transform":[10,0,500000,0,-10,1500000]}
        result = {"array":values, "valid_mask":mask, "provenance":{
            "synthetic":True, "sensor_observation":False, "input":{"grid":grid},
            "array":array_identity(values), "valid_mask":array_identity(mask)}}
        with tempfile.TemporaryDirectory(prefix="aureon-export-test-") as tmp:
            root = Path(tmp)
            receipt = export_generation(result, root/"synthetic.tif")
            self.assertTrue(receipt["decoded_values_exact"])
            with rasterio.open(root/"synthetic.tif") as dataset:
                np.testing.assert_array_equal(dataset.read(), values)
                self.assertEqual(list(dataset.transform)[:6], grid["transform"])
            with self.assertRaises(FileExistsError):
                export_generation(result, root/"synthetic.tif")
            values[0,0,0] += 1
            with self.assertRaisesRegex(ValueError, "changed"):
                export_generation(result, root/"changed.tif")
            self.assertFalse((root/"changed.tif").exists())
            result["provenance"]["input"]["grid"] = None
            with self.assertRaisesRegex(ValueError, "requires grid"):
                export_generation(result, root/"no-grid.tif")
            self.assertFalse((root/"no-grid.tif").exists())

    def test_file_example_rejects_normalization_before_output_or_source_reads(self):
        spec = importlib.util.spec_from_file_location("prepare_export_example", Path(__file__).resolve().parents[1] / "examples/generation/prepare_export.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="aureon-export-declaration-test-") as tmp:
            output = Path(tmp)/"must-not-exist"
            kwargs = dict(raster="missing", valid_mask="missing", output=output, crs="EPSG:32647",
                          source_product_id="product", validity_policy="explicit", bands=INPUT_BANDS,
                          units="surface_reflectance_x10000", normalization="zscore")
            with self.assertRaisesRegex(ValueError, "normalization must be none"):
                module.prepare_export(**kwargs)
            kwargs.update(normalization="none", bands=list(reversed(INPUT_BANDS)))
            with self.assertRaisesRegex(ValueError, "ordered"):
                module.prepare_export(**kwargs)
            self.assertFalse(output.exists())

    def test_memory_facade_dispatch_and_mutually_exclusive_file_input(self):
        from unittest.mock import MagicMock
        data = GenerationInput(np.ones((12,224,224), np.float32), bands=INPUT_BANDS,
                               units="surface_reflectance_x10000", valid_mask=np.ones((224,224), bool))
        with self.assertRaisesRegex(ValueError, "not both"):
            GenerationRequest("c", "w", "j", "a", input_data=data, input_manifest="m")
        with self.assertRaises(TypeError):
            GenerationRequest("c", "w", "j", "a", input_data={"image":"not validated"})
        fake = MagicMock(); fake.__enter__.return_value = fake
        fake.generate.return_value = {"array":np.zeros((2,224,224),np.float32), "provenance":{}}
        with patch("aureon.api.models.open_model_session", return_value=fake):
            result = run_model(GenerationRequest("c", "w", "j", "a", input_data=data,
                                                 execution_profile="reference"), execute_model=True)
        fake.generate.assert_called_once_with(data, job_id="j", attempt_id="a", recipe=None)
        fake.run.assert_not_called()
        self.assertEqual(result["job"]["array"].shape, (2,224,224))

    def test_file_example_prepares_pixels_once_and_preserves_grid(self):
        from aureon.prepare import check_scene
        spec = importlib.util.spec_from_file_location("prepare_export_example", Path(__file__).resolve().parents[1] / "examples/generation/prepare_export.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        values = np.full((12,32,32), 1000, dtype=np.float32)
        values[:, 0, :3] = [-500, 0, 15000]
        transform = rasterio.Affine(10,0,500000,0,-10,1500000)
        with tempfile.TemporaryDirectory(prefix="aureon-file-example-test-") as tmp:
            root = Path(tmp)
            for name, pixels in (("image",values), ("mask",np.ones((1,32,32),np.uint8))):
                with rasterio.open(root/(name+".tif"), "w", driver="GTiff", width=32,height=32,
                                   count=pixels.shape[0],dtype=pixels.dtype,crs="EPSG:32647",transform=transform,
                                   tiled=True,blockxsize=16,blockysize=16) as ds:
                    ds.write(pixels)
                    if name == "image": ds.descriptions = tuple(INPUT_BANDS)
            record = module.prepare_export(raster=root/"image.tif", valid_mask=root/"mask.tif", output=root/"prepared",
                       crs="EPSG:32647", source_product_id="CPU-test", validity_policy="explicit all-valid fixture",
                       bands=INPUT_BANDS, units="surface_reflectance_x10000", normalization="none")
            self.assertEqual(check_scene(record["manifest"])["valid_pixels"], 32*32)
            with rasterio.open(root/"prepared/scene.tif") as ds:
                np.testing.assert_array_equal(ds.read(), values)
                self.assertEqual(ds.transform, transform)


if __name__ == "__main__":
    unittest.main()
