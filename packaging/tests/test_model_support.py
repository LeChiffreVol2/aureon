"""Model dispatch/admission and offline package contracts; no model execution."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from aureon.api import GenerationRequest, SceneRequest, list_models, inspect_model, run_model, open_model_session
from aureon.api import models, profiles, ExecutionRequired
from aureon.terramind_native import offline


class ModelSupport(unittest.TestCase):
    def test_discovery_and_help_without_runtime_imports(self):
        code = '''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torchvision','terratorch','numpy','rasterio','vllm','diffusers'):
   raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from aureon.api.cli import main
from aureon.terramind_native import offline
main(['models']); main(['profiles']); main(['check','--environment','generation'])
for args in (['--help'],['generate','--help'],['generation-config','--help']):
 try: main(args)
 except SystemExit as exc: assert exc.code == 0
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_registry_detached_and_honest(self):
        a = list_models(); a[0]["outputs"].clear()
        self.assertTrue(list_models()[0]["outputs"])
        self.assertEqual(len(a), 4)
        self.assertFalse(inspect_model("kao-terramind-a")["resident"])
        self.assertTrue(inspect_model("terramind-v1-base")["synthetic"])
        with self.assertRaises(ValueError): inspect_model("terramind-large")

    def test_generation_exact_profile_without_vllm(self):
        self.assertNotIn("vllm", profiles.PROFILES["generation"])
        self.assertEqual(profiles.PROFILES["generation"]["torch"], "2.10.0")
        self.assertEqual(profiles.PROFILES["generation"]["huggingface-hub"], "0.36.2")

    def test_unsupported_combinations_before_loading(self):
        with patch.object(models, "require_profile", side_effect=AssertionError("accessed runtime")):
            for model in ("kao-terramind-a", "kao-terramind-c", "unknown"):
                with self.assertRaises(ValueError): open_model_session(model, execute_model=True)
            with self.assertRaises(ValueError):
                GenerationRequest("missing", "out", "j", "a", model_id="kao-terramind-b")
            with self.assertRaises(ValueError):
                open_model_session("terramind-v1-base", config="missing", work="out", profile_id="fp16", execute_model=True)
            with self.assertRaises(ValueError):
                run_model(SceneRequest("m", "b", "o", "kao-terramind-a", "fp32-full-reference-v1"), execute_model=True)

    def test_execution_opt_in_before_io(self):
        with patch.object(models, "require_profile", side_effect=AssertionError("accessed runtime")):
            with self.assertRaises(ExecutionRequired): run_model(GenerationRequest("missing", "out", "j", "a"))
            with self.assertRaises(ExecutionRequired):
                open_model_session("terramind-v1-base", config="missing", work="out")

    def test_invalid_recipe_and_attempt_before_io(self):
        with self.assertRaises(ValueError): GenerationRequest("missing", "out", "../j", "a")
        with self.assertRaises(ValueError): GenerationRequest("missing", "out", "j", "a", recipe={"token_steps": 999})

    def test_scene_dispatch_keeps_original_request(self):
        request = SceneRequest("m", "b", "o", "kao-terramind-c", "fp32-full-reference-v1", "scene-legacy")
        with patch.object(models, "run_scene", return_value="original-result") as run:
            self.assertEqual(run_model(request, execute_model=True), "original-result")
            run.assert_called_once_with(request, execute_model=True)

    def test_generation_returns_distinct_schema_and_cleanup(self):
        session = MagicMock(); session.__enter__.return_value = session
        session.run.return_value = {"cog": "synthetic", "model_load_count": 1}
        session.close.return_value = {"passed": True}
        with patch.object(models, "open_model_session", return_value=session) as opening:
            result = run_model(GenerationRequest("config", "work", "j", "first", denoiser="graph"), execute_model=True)
            self.assertEqual(result["schema"], "aureon.generation_result.v1")
            self.assertTrue(result["synthetic"])
            self.assertTrue(result["cleanup"]["passed"])
            self.assertEqual(opening.call_args.kwargs["denoiser"], "graph")
            session.run.assert_called_once_with(None, job_id="j", attempt_id="first", recipe=None)

    def test_generation_delegates_exact_session(self):
        with patch.object(models, "require_profile"), patch.object(offline, "inspect_config") as check, \
                patch("aureon.terramind_native.TerraMindSession") as native:
            result = open_model_session("terramind-v1-base", config="c", work="w", denoiser="graph", execute_model=True)
            check.assert_called_once_with("c")
            native.assert_called_once_with(config="c", work="w", denoiser="graph", execute_model=True)
            self.assertIs(result, native.return_value)

    def test_offline_config_relative_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ("model", "tokenizer", "input"):
                (root/name).write_text("local fixture")
            def version(name):
                return "2.10.0+cu128" if name == "torch" else profiles.PROFILES["generation"][name]
            with patch.object(offline, "verify_file"), patch.object(offline, "_input_metadata"), \
                    patch.object(profiles, "require_profile"), patch.object(offline.metadata, "version", side_effect=version):
                offline.create_config(model=root/"model", tokenizer=root/"tokenizer", input_manifest=root/"input", output=root/"config.json")
                config=json.loads((root/"config.json").read_text())
                self.assertEqual(config["model"]["path"], "model")
                self.assertEqual(config["model"]["sha256"], offline.MODEL_SHA)
                self.assertNotIn("vllm", config["dependencies"])
                self.assertEqual(config["dependencies"]["torch"], "2.10.0+cu128")
                with self.assertRaises(ValueError):
                    offline.create_config(model="missing", tokenizer="missing", input_manifest="missing", output=root/"config.json")

    def test_offline_rejects_unpinned_assets_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/"wrong").write_bytes(b"wrong weights")
            with patch.object(profiles, "require_profile"), patch.object(offline.metadata, "version", return_value="fixture"):
                with self.assertRaises(ValueError):
                    offline.create_config(model=root/"wrong", tokenizer=root/"wrong", input_manifest=root/"wrong", output=root/"config.json")
            self.assertFalse((root/"config.json").exists())

    def test_packaged_source_inventory_is_portable(self):
        p=Path(offline.__file__).with_name("upstream-source.json")
        data=json.loads(p.read_text())
        self.assertEqual(len(data["files"]),37)
        for row in data["files"]:
            self.assertTrue(row["path"].startswith("terratorch/"))
            self.assertNotIn("..",Path(row["path"]).parts)
            self.assertEqual(len(row["sha256"]),64)


if __name__ == "__main__": unittest.main()
