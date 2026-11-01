"""Canonical SDK/CLI dispatch and compatibility, with no model execution."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from aureon.api import GenerationRequest, GenerationSession, open_model_session, run_model
from aureon.api import cli, models
from aureon.api.best import BestTerraMindSession
from aureon.api.best_workflows import BestSceneRequest, BestWorkQueue, run_best_scene


class SingleEngineDefaultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aureon-default-route-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.value = {"schema": "aureon.terramind_native.v1", "model": {}, "tokenizer": {},
                      "source_manifest": {}, "dependencies": {}, "input_manifest": {}, "recipe": {}}
        self.config.write_text(json.dumps(self.value))

    def test_base_default_selects_exact_core_without_extra_asset_hashing(self):
        with patch("aureon.api.best.BestTerraMindSession") as selected, \
             patch("aureon.terramind_native.offline.inspect_config", side_effect=AssertionError("extra asset hashing")), \
             patch("aureon.terramind_native.TerraMindSession", side_effect=AssertionError("wrong core")):
            actual = open_model_session("terramind-v1-base", config=self.config, work="work", execute_model=True)
            selected.assert_called_once_with(self.config, "work", execute_model=True, profile="best_fp32")
            self.assertIs(actual, selected.return_value)

    def test_reference_is_explicit_same_frozen_implementation(self):
        with patch("aureon.api.best.BestTerraMindSession") as selected:
            open_model_session("terramind-v1-base", config=self.config, work="work", execute_model=True,
                               execution_profile="reference")
            self.assertEqual(selected.call_args.kwargs["profile"], "reference_fp32")

    def test_assets_only_and_wrong_model_fail_before_constructing_resident(self):
        for value in ({**self.value, "schema": "aureon.terramind_assets.v1"},
                      {**self.value, "model_id": "terramind-v1-large"}):
            self.config.write_text(json.dumps(value))
            with patch("aureon.api.best.BestTerraMindSession") as selected:
                with self.assertRaises(ValueError):
                    open_model_session("terramind-v1-base", config=self.config, work="work", execute_model=True)
                selected.assert_not_called()

    def test_unsupported_expert_pair_is_not_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, "auto/graph"):
            GenerationRequest("c", "w", "j", "1", denoiser="native")
        with self.assertRaisesRegex(ValueError, "input_kind"):
            open_model_session("terramind-v1-base", config="c", work="w", input_kind="automatic")

    def test_nonbase_and_explicit_array_keep_their_existing_interfaces(self):
        for model_id, extra in (("terramind-v1-small", {}),
                                ("terramind-v1-base", {"input_kind": "array"})):
            with patch.object(models, "require_profile"), \
                 patch("aureon.terramind_native.offline.inspect_config", return_value={"model_id": model_id}), \
                 patch("aureon.terramind_native.TerraMindSession") as family, \
                 patch("aureon.api.best.BestTerraMindSession", side_effect=AssertionError("base-only alias")):
                actual = open_model_session(model_id, config="c", work="w", execute_model=True, **extra)
                self.assertIs(actual, family.return_value)
                self.assertEqual(family.call_args.kwargs["model_id"], model_id)
                self.assertEqual(family.call_args.kwargs["denoiser"], "native")

    def test_run_model_uses_file_default_and_preserves_result_schema(self):
        session = MagicMock(); session.__enter__.return_value = session
        session.run.return_value = {"raw": "preserved"}; session.close.return_value = {"passed": True}
        with patch.object(models, "open_model_session", return_value=session) as opened:
            result = run_model(GenerationRequest("c", "w", "j", "1"), execute_model=True)
        self.assertEqual(opened.call_args.kwargs["input_kind"], "file")
        session.run.assert_called_once_with(None, job_id="j", attempt_id="1", recipe=None)
        self.assertEqual(result["schema"], "aureon.generation_result.v1")
        self.assertEqual(result["job"], {"raw": "preserved"})

    def test_canonical_names_are_identity_aliases_not_new_wrappers(self):
        from aureon.api import PreparedSceneRequest, GenerationWorkQueue, run_prepared_scene
        self.assertIs(GenerationSession, BestTerraMindSession)
        self.assertIs(PreparedSceneRequest, BestSceneRequest)
        self.assertIs(GenerationWorkQueue, BestWorkQueue)
        self.assertIs(run_prepared_scene, run_best_scene)

    def test_plain_generate_defaults_base_and_accepts_work_alias(self):
        with patch.object(cli, "run_model", return_value={"status": "completed"}) as run, redirect_stdout(StringIO()):
            self.assertEqual(cli.main(["generate", "--config", "c", "--work", "w", "--execute-model"]), 0)
        request = run.call_args.args[0]
        self.assertEqual(request.model_id, "terramind-v1-base")
        self.assertEqual(request.output, "w")
        self.assertIsNone(request.denoiser)
        self.assertIsNone(request.reuse)

    def test_plain_generate_scene_uses_selected_resident(self):
        session = MagicMock(); session.__enter__.return_value = session
        with patch.object(cli, "open_model_session", return_value=session) as opened, \
             patch("aureon.api.best_workflows.inspect_best_scene", return_value={"status":"planned"}) as preflight, \
             patch("aureon.api.best_workflows.run_best_scene", return_value={"status": "completed"}) as run, \
             patch("aureon.api.scene_generation.run_generation_scene", side_effect=AssertionError("wrong engine")), \
             redirect_stdout(StringIO()):
            code = cli.main(["generate-scene", "--config", "c", "--work", "w", "--image", "i",
                             "--valid-mask", "m", "--output", "o", "--execute-model"])
        self.assertEqual(code, 0)
        self.assertEqual(opened.call_args.args, ("terramind-v1-base",))
        self.assertIs(run.call_args.args[0], session)
        self.assertIsInstance(run.call_args.args[1], BestSceneRequest)
        self.assertEqual(run.call_args.args[1].max_output_bytes, 512*1024**2)
        preflight.assert_called_once_with(run.call_args.args[1])

    def test_invalid_scene_preflight_does_not_construct_resident(self):
        with patch("aureon.api.best_workflows.inspect_best_scene", side_effect=ValueError("invalid scene budget")), \
             patch.object(cli, "open_model_session") as opened, redirect_stderr(StringIO()), \
             self.assertRaises(SystemExit) as raised:
            cli.main(["generate-scene", "--config", "c", "--work", "w", "--image", "i",
                      "--valid-mask", "m", "--output", "o", "--execute-model"])
        self.assertEqual(raised.exception.code, 2)
        opened.assert_not_called()

    def test_scene_plan_requires_no_weights_or_config(self):
        with patch("aureon.api.best_workflows.inspect_best_scene", return_value={"status": "planned"}) as inspect, \
             patch.object(cli, "open_model_session", side_effect=AssertionError("model opened")), redirect_stdout(StringIO()):
            self.assertEqual(cli.main(["scene-plan", "--image", "i", "--valid-mask", "m", "--output", "o"]), 0)
        inspect.assert_called_once()

    def test_scene_execution_permission_and_unsupported_legacy_base_args(self):
        cases = (["--config", "missing", "--work", "missing"],
                 ["--model-file", "m", "--tokenizer-file", "t", "--work", "w", "--execute-model"])
        for extra in cases:
            with patch.object(cli, "open_model_session", side_effect=AssertionError("model opened")), \
                 redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                cli.main(["generate-scene", "--image", "i", "--valid-mask", "m", "--output", "o", *extra])
            self.assertEqual(raised.exception.code, 2)

    def test_other_family_scene_preserves_explicit_compatibility_route(self):
        with patch("aureon.api.scene_generation.run_generation_scene", return_value={"status": "completed"}) as run, \
             patch.object(cli, "open_model_session", side_effect=AssertionError("base opened")), redirect_stdout(StringIO()):
            code = cli.main(["generate-scene", "--model", "terramind-v1-small", "--model-file", "model",
                "--tokenizer-file", "tokenizer", "--work", "w", "--image", "i", "--valid-mask", "m",
                "--output", "o", "--execute-model"])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args[0].model, "terramind-v1-small")
        self.assertEqual(run.call_args.args[0].max_output_bytes, 2*1024**3)

    def test_registry_discloses_base_default_and_other_family_scope(self):
        base, small = models.inspect_model("terramind-v1-base"), models.inspect_model("terramind-v1-small")
        self.assertEqual((base["default_file_runtime"], base["denoiser_default"]), ("selected_core", "graph"))
        self.assertEqual(small["default_file_runtime"], "family_runtime")
        self.assertIn("no base performance inheritance", small["default_file_scope"])


if __name__ == "__main__":
    unittest.main()
