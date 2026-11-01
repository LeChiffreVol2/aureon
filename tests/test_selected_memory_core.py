"""CPU delivery/delegation checks; fake arithmetic is not model qualification."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from aureon._reference.integrity import verify_core
from aureon.api import GenerationInput, open_model_session
from aureon.api import models
from aureon.terramind_native.session import _NativeRuntime, TerraMindSession
from aureon.terramind_native.selected_core import load_primitives
from aureon.terramind_native.denoiser_lanes import LaneGraphs
from test_generation_batch import FakeModel, RECIPE


class CPUAdapter(_NativeRuntime):
    device = "cpu"

    def __init__(self, core):
        super().__init__()
        self.numerical_core = core
        self.torch = torch
        self.model = FakeModel()
        self.versions = self._versions()
        self.config = {"model": {}, "tokenizer": {}, "source_manifest": {}, "dependencies": {}}
        self.loaded_identity = {"cpu_double": True}

    def _policy(self):
        pass

    def _assert_policy(self):
        assert not torch.is_grad_enabled()


class SelectedMemoryCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity = verify_core()
        cls.core = load_primitives(cls.identity)

    def test_provider_objects_are_frozen_implementations(self):
        for field, module, name in (
            ("schedule", "compat", "schedule_compatibility"), ("rng", "rng", "rng_capsule"),
            ("reuse", "reuse", "exact_conditioning_reuse"), ("decision", "session", "_decision"),
            ("graph_factory", "denoiser_graph", "DenoiserGraph"),
            ("binding_factory", "denoiser_binding", "S1GRDDecoderBinding"),
            ("tensor_graph_factory", "graph_runtime", "FixedTensorGraph")):
            expected = getattr(importlib.import_module("aureon._reference.terramind." + module), name)
            self.assertIs(getattr(self.core, field), expected)
        self.assertIs(self.core.patch_lock, importlib.import_module("aureon._reference.terramind.denoiser_graph")._PATCH_LOCK)

    def run_memory(self, runtime, mode="off", denoiser="native"):
        image = np.ones((12, 224, 224), dtype=np.float32)
        mask = np.ones((224, 224), dtype=bool)
        # No geospatial grid is fabricated. Any accidental persistence fails.
        with patch("numpy.save", side_effect=AssertionError("array spill")), \
             patch("aureon.terramind_native.geo.write_cog", side_effect=AssertionError("implicit COG")), \
             patch("aureon.terramind_native.session.write_json", side_effect=AssertionError("delivery JSON spill")), \
             patch("aureon.terramind_native.session.verify_core", side_effect=AssertionError("per-job core hashing")):
            result = runtime.run(image, mask, {"source_id": "CPU-no-grid"}, {"kind": "array"}, RECIPE,
                                 Path("unused-output"), mode=mode, decoder_context=None, capture_logits=False,
                                 session_identity={}, job_id="j", attempt_id="1", denoiser=denoiser,
                                 delivery="memory")
        self.assertNotIn("grid", result["provenance"]["input"])
        self.assertFalse(np.shares_memory(result["array"], image))
        self.assertFalse(np.shares_memory(result["valid_mask"], mask))
        self.assertTrue(result["rng"]["restored"])
        self.assertTrue(result["scheduler_restored"])
        self.assertTrue(result["decoding_steps_restored"])
        self.assertTrue(result["trace"]["methods_restored"])
        self.assertEqual(result["numerical_core"]["manifest_sha256"], self.identity.manifest_sha256)
        self.assertEqual(result["numerical_core"]["execution_adapter"], "memory")
        return result

    def test_memory_reference_preserves_rng_and_returns_owned_arrays_without_grid_or_files(self):
        runtime = CPUAdapter(self.core)
        before = torch.get_rng_state().clone()
        first = self.run_memory(runtime)
        second = self.run_memory(runtime)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(np.array_equal(first["array"].view("uint32"), second["array"].view("uint32")))
        self.assertEqual(first["trace"]["tokens"], second["trace"]["tokens"])
        first["array"].fill(0)
        self.assertFalse(np.array_equal(first["array"], second["array"]))

    def test_recommended_graph_and_reuse_are_delegated_without_changing_delivery(self):
        calls = []
        @contextmanager
        def reuse(model, *, enabled):
            calls.append(("reuse", enabled))
            yield {"requested": enabled, "cache_hits": 0, "cache_released": True}
        class Graph:
            @contextmanager
            def request(self, model):
                calls.append(("graph", True))
                yield {"profile_id": "cpu-delegation-double", "enabled": True, "restored": True}
        core = replace(self.core, reuse=reuse, graph_eligibility=lambda *a: (True, "cpu-double"),
                       graph_factory=lambda model: Graph())
        self.run_memory(CPUAdapter(core), "auto", "graph")
        self.assertEqual(calls, [("reuse", True), ("graph", True)])

    def test_lane_graphs_use_selected_binding_graph_and_shared_patch_lock(self):
        lanes = LaneGraphs(FakeModel(), max_lanes=3, numerical_core=self.core)
        self.assertIs(lanes.binding_factory, self.core.binding_factory)
        self.assertIs(lanes.graph_factory, self.core.tensor_graph_factory)
        self.assertIs(lanes.numerical_core.patch_lock, self.core.patch_lock)
        self.assertEqual(lanes.graphs, [])  # No allocations until actual admitted lanes.
        self.assertEqual(lanes.max_lanes, 3)
        self.assertTrue(lanes.close()["released"])

    def test_default_array_factory_selects_recommended_and_other_families_stay_separate(self):
        for model_id, expected in (("terramind-v1-base", "recommended"), ("terramind-v1-small", None)):
            with patch.object(models, "require_profile"), \
                 patch("aureon.terramind_native.offline.inspect_config", return_value={"model_id": model_id}), \
                 patch("aureon.terramind_native.TerraMindSession") as session:
                open_model_session(model_id, config="assets", work="w", input_kind="array", execute_model=True)
                self.assertEqual(session.call_args.kwargs.get("execution_profile"), expected)

    def test_explicit_runtime_load_binds_base_only_and_constructs_one_model(self):
        for model_id in ("terramind-v1-base", "terramind-v1-small"):
            with self.subTest(model_id=model_id), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = {"model_id": model_id, "recipe": RECIPE}
                path = root / "assets.json"
                path.write_text(json.dumps(config))
                runtime = _NativeRuntime()
                runtime.device = "cpu"
                runtime.core_identity = self.identity if model_id.endswith("base") else None
                model = FakeModel()
                # The load boundary holds weak references; these stand-ins are weakrefable.
                model.sampler.model = MagicMock()
                model.tokenizer["tok_sen1grd@224"] = MagicMock()
                with patch("aureon.terramind_native.loader.load_native", return_value=(model, {}, config, {})) as loader, \
                     patch("aureon.terramind_native.selected_core.load_primitives", wraps=load_primitives) as select, \
                     patch("aureon.terramind_native.session.verify_core", side_effect=AssertionError("already verified")), \
                     patch.object(runtime, "_policy"), patch.object(torch, "set_num_threads"):
                    runtime.load(path, root)
                loader.assert_called_once_with(path)
                self.assertIs(runtime.model, model)
                if model_id.endswith("base"):
                    select.assert_called_once_with(self.identity)
                    self.assertIs(runtime.numerical_core.rng, self.core.rng)
                else:
                    select.assert_not_called()
                    self.assertIsNone(runtime.numerical_core)
                self.assertFalse(torch.cuda.is_initialized())

    def test_constructor_verifies_core_once_without_loading_primitives_or_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "assets.json"
            config.write_text(json.dumps({"recipe": RECIPE}))
            with patch("aureon.terramind_native.session.verify_core", wraps=verify_core) as verify, \
                 patch("aureon.terramind_native.selected_core.load_primitives", side_effect=AssertionError("eager primitives")), \
                 patch.object(_NativeRuntime, "load", side_effect=AssertionError("eager model")):
                with TerraMindSession(config, root / "work", execute_model=True) as session:
                    self.assertEqual(verify.call_count, 1)
                    self.assertEqual(session.metadata()["numerical_core"]["manifest_sha256"], self.identity.manifest_sha256)
                    session.metadata()
                    self.assertEqual(verify.call_count, 1)


if __name__ == "__main__":
    unittest.main()
