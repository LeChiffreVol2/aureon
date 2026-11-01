"""Resident lifecycle and native-boundary CPU doubles; no native model forwards."""
from contextlib import contextmanager
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch
from aureon.terramind_native import TerraMindSession, SessionBusy, SessionClosed, SessionFailed
from aureon.terramind_native import session as native
from aureon.terramind_native.common import INPUT_BANDS, binding, read_json, write_json
from aureon.terramind_native.rng import snapshot, state_digest

TARGET = "tok_sen1grd@224"
RECIPE = {"token_steps": 8, "diffusion_steps": 10, "temperature": 1., "top_p": .8, "top_k": 0, "seed": 17}


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_embeddings = torch.nn.ModuleDict({"untok_sen2l2a@224": torch.nn.Identity()})
        self.weight = torch.nn.Parameter(torch.tensor(2.))
        self.calls = 0

    def forward_encoder(self, x, mask):
        self.calls += 1
        return x*self.weight + mask.to(x.dtype)

    def forward_logits(self, encoded):
        return {TARGET: encoded.mean().reshape(1, 1).repeat(196, 3)}


class Sampler(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model = Core()

    def generate(self, data, schedule, *, num_tokens=None, **kwargs):
        position = 0
        for step in schedule:
            count = num_tokens or step["num_tokens"]
            encoded = self.model.forward_encoder(data["conditioning"], data["mask"])
            self.model.forward_logits(encoded)
            data[TARGET]["tensor"][:, position:position+count] = torch.randint(0, 50, (1, count))
            position += count
        return data


class Tokenizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.noise_scheduler = types.SimpleNamespace(timesteps=[1, 2, 3])

    def decode_tokens(self, tokens, **kwargs):
        self.noise_scheduler.timesteps.append(99)
        return tokens.float().mean().expand(1, 2, 224, 224).clone() + torch.rand((1, 2, 224, 224))/100


class Model(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.sampler = Sampler()
        self.tokenizer = torch.nn.ModuleDict({TARGET: Tokenizer()})
        self.modalities = ["S2L2A"]; self.output_modalities = [TARGET]
        self.decoding_steps = 8
        self.events = events
        self.eval().requires_grad_(False)

    def forward(self, data, **kwargs):
        self.events.append(("forward", threading.get_ident()))
        counts = [196] if self.decoding_steps == 1 else [25, 25, 25, 25, 24, 24, 24, 24]
        schedule = [{"num_tokens": n, "target_domain": TARGET, "scheme": "roar", "cfg_scale": 1., "temperature": 1.} for n in counts]
        x = data["S2L2A"][:, :2, :2, :2]
        state = {TARGET: {"tensor": torch.zeros((1, 196), dtype=torch.long)}, "conditioning": x,
                 "mask": torch.zeros_like(x, dtype=torch.bool)}
        result = self.sampler.generate(state, schedule, num_tokens=196)
        return {"S1GRD": self.tokenizer[TARGET].decode_tokens(result[TARGET]["tensor"])}


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.events = []; self.sessions = []; self.refs = []
        np.save(self.root / "image.npy", np.ones((12, 224, 224), dtype=np.float32))
        np.save(self.root / "mask.npy", np.ones((224, 224), dtype=bool))
        manifest = {"schema": "aureon.terramind_native_input.v1", "bands": INPUT_BANDS,
                    "units": "surface_reflectance_x10000", "image": binding(self.root / "image.npy"),
                    "mask": binding(self.root / "mask.npy"),
                    "grid": {"shape": [224, 224], "crs": "EPSG:32647", "transform": [10, 0, 632260, 0, -10, 1583830]}}
        self.manifest = self.root / "input.json"; write_json(self.manifest, manifest)
        self.config = {"recipe": RECIPE, "input_manifest": binding(self.manifest),
                       "model": {"sha256": "a"*64}, "tokenizer": {"sha256": "b"*64},
                       "source_manifest": {"sha256": "c"*64}, "dependencies": {"test_runtime": "CPU-double"}}
        self.config_path = self.root / "config.json"; write_json(self.config_path, self.config)

        def load(path):
            self.events.append(("load", threading.get_ident()))
            model = Model(self.events); self.refs.append(weakref.ref(model))
            return model, {"test_double": True}, copy.deepcopy(self.config), {"input_manifest": self.manifest}

        self.patches = [patch("aureon.terramind_native.loader.load_native", new=load),
                        patch.object(native._NativeRuntime, "device", "cpu"),
                        patch.object(torch.cuda, "synchronize", new=lambda: None)]
        for p in self.patches: p.start()

    def tearDown(self):
        for session in self.sessions:
            try: session.close()
            except SessionFailed: pass
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def session(self, **kwargs):
        value = TerraMindSession(self.config_path, self.root / ("session-%d" % len(self.sessions)), execute_model=True, **kwargs)
        self.sessions.append(value); return value

    def test_light_import_and_execution_is_explicit(self):
        code = "import sys; from aureon.terramind_native import TerraMindSession; assert not ({'torch','numpy','rasterio','terratorch'} & sys.modules.keys())"
        checked = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        with self.assertRaisesRegex(ValueError, "execute_model"):
            TerraMindSession(self.config_path, self.root / "not-created")
        self.assertFalse((self.root / "not-created").exists()); self.assertEqual(self.events, [])

    def test_single_request_uses_subclass_runtime_factory_on_owner_thread(self):
        from aureon.api.lane_generation import ExperimentalLaneSession
        from aureon.terramind_native.denoiser_lanes import LaneRuntime
        value=ExperimentalLaneSession(self.config_path,self.root/'lane-single',execute_model=True,
                                      allow_unqualified_parallel=True)
        self.sessions.append(value)
        value.run(job_id='single',attempt_id='1')
        self.assertIsInstance(value._runtime,LaneRuntime)
        self.assertEqual(value.metadata()['model_load_count'],1)
        self.assertIsNone(value._runtime.parallel_lanes)

    def test_lazy_load_two_jobs_same_owner_exact_outputs_and_idempotent_close(self):
        session = self.session()
        self.assertEqual(self.events, [])
        before = state_digest(snapshot(torch, np, ()))
        direct = session.run(job_id="scene", attempt_id="direct", reuse="off", capture_logits=True)
        reused = session.run(job_id="scene", attempt_id="reuse", capture_logits=True)
        self.assertEqual(direct["raw"]["sha256"], reused["raw"]["sha256"])
        self.assertEqual(direct["trace"]["tokens"]["sha256"], reused["trace"]["tokens"]["sha256"])
        self.assertEqual(len(reused["trace"]["logits"]), 8)
        self.assertEqual(direct["reuse"]["encoder_calls"], 8)
        self.assertEqual(reused["reuse"]["encoder_calls"], 1)
        self.assertEqual(reused["reuse"]["cache_hits"], 7)
        self.assertEqual(self.refs[0]().tokenizer[TARGET].noise_scheduler.timesteps, [1, 2, 3])
        self.assertEqual(before, state_digest(snapshot(torch, np, ())))
        cleanup = session.close()
        self.assertTrue(cleanup["passed"]); self.assertEqual(cleanup, session.close())
        self.assertEqual(cleanup["model_load_count"], 1); self.assertEqual(cleanup["jobs_completed"], 2)
        self.assertEqual({i for _, i in self.events}, {cleanup["owner_thread_id"]})
        self.assertNotEqual(cleanup["owner_thread_id"], threading.get_ident())
        self.assertIsNone(self.refs[0]())
        self.assertEqual(read_json(session.work / "session-cleanup.json"), cleanup)
        with self.assertRaises(SessionClosed): session.run(job_id="later", attempt_id="one")

    def test_auto_one_step_has_no_hook_and_attempt_replay_cannot_overwrite(self):
        session = self.session()
        record = session.run(job_id="scene", attempt_id="one", recipe={**RECIPE, "token_steps": 1})
        self.assertFalse(record["reuse_decision"]["enabled"])
        self.assertEqual(record["reuse_decision"]["reason"], "no_repeated_encoder")
        self.assertEqual(record["reuse"]["cache_hits"], 0)
        raw = Path(record["raw"]["path"]).read_bytes()
        with self.assertRaises(FileExistsError): session.run(job_id="scene", attempt_id="one")
        self.assertEqual(raw, Path(record["raw"]["path"]).read_bytes())
        next_record = session.run(job_id="scene", attempt_id="two", recipe={**RECIPE, "token_steps": 1}, reuse="off")
        self.assertEqual(record["raw"]["sha256"], next_record["raw"]["sha256"])
        self.assertEqual(record["trace"]["tokens"]["sha256"], next_record["trace"]["tokens"]["sha256"])
        self.assertEqual(record["rng"]["request_post_rng_sha256"], next_record["rng"]["request_post_rng_sha256"])
        self.assertNotIn("encoder_calls", next_record["reuse"])
        self.assertEqual(next_record["model_load_count"], 1)

    def test_bad_input_recipe_and_path_fail_before_load_without_consuming_attempt(self):
        session = self.session()
        with self.assertRaises(ValueError): session.run(job_id="../escape", attempt_id="x")
        with self.assertRaises(ValueError): session.run(job_id="scene", attempt_id="x", recipe={**RECIPE, "token_steps": 3})
        invalid = read_json(self.manifest); invalid["grid"]["transform"] = [0, 0, 1, 0, 0, 2]
        bad = self.root / "bad.json"; write_json(bad, invalid)
        with self.assertRaises(ValueError): session.run(bad, job_id="scene", attempt_id="x")
        self.assertEqual(self.events, []); self.assertEqual(session.metadata()["state"], "new")
        session.run(job_id="scene", attempt_id="x")

    def test_per_request_decoder_override_runs_on_owner_inside_private_state(self):
        observed = []
        @contextmanager
        def hook(model, recipe):
            observed.append((threading.get_ident(), torch.is_grad_enabled(), model.tokenizer[TARGET].noise_scheduler))
            receipt = {"profile_id": "test-decoder", "restored": False}
            try: yield receipt
            finally: receipt["restored"] = True
        session = self.session(decoder_context=hook)
        selected = session.run(job_id="scene", attempt_id="selected")
        baseline = session.run(job_id="scene", attempt_id="native", decoder_context=None)
        self.assertEqual(selected["decoder"], {"profile_id": "test-decoder", "restored": True})
        self.assertEqual(baseline["decoder"]["profile_id"], "native")
        self.assertEqual(selected["raw"]["sha256"], baseline["raw"]["sha256"])
        self.assertEqual(observed[0][:2], (selected["owner_thread_id"], False))
        self.assertIsNot(observed[0][2], self.refs[0]().tokenizer[TARGET].noise_scheduler)

    def test_overlap_and_second_session_reject_without_consuming_identity(self):
        entered = threading.Event(); release = threading.Event(); errors = []
        @contextmanager
        def blocked(model, recipe):
            entered.set()
            if not release.wait(10): raise RuntimeError("Test did not release hook")
            yield {"profile_id": "blocked-test", "restored": True}
        first = self.session(); second = self.session()
        def call():
            try: first.run(job_id="scene", attempt_id="held", decoder_context=blocked)
            except BaseException as exc: errors.append(str(exc))
        worker = threading.Thread(target=call); worker.start()
        try:
            self.assertTrue(entered.wait(10))
            with self.assertRaises(SessionBusy): first.run(job_id="scene", attempt_id="retryable")
            with self.assertRaises(SessionBusy): first.close()
            with self.assertRaises(SessionBusy): second.run(job_id="scene", attempt_id="retryable")
        finally:
            release.set(); worker.join(10)
        self.assertFalse(worker.is_alive()); self.assertEqual(errors, [])
        first.run(job_id="scene", attempt_id="retryable")
        first.close()
        second.run(job_id="scene", attempt_id="retryable")

    def test_decoder_failure_restores_rng_and_closes_model_without_retry(self):
        before = state_digest(snapshot(torch, np, ()))
        @contextmanager
        def broken(model, recipe):
            original = model.tokenizer[TARGET].decode_tokens
            def fail(*a, **k):
                torch.rand(3)
                raise RuntimeError("injected decode failure")
            model.tokenizer[TARGET].decode_tokens = fail
            receipt = {"profile_id": "broken-test", "restored": False}
            try: yield receipt
            finally:
                del model.tokenizer[TARGET].decode_tokens
                receipt["restored"] = True
        session = self.session()
        with self.assertRaisesRegex(SessionFailed, "injected decode failure"):
            session.run(job_id="scene", attempt_id="bad", decoder_context=broken)
        self.assertEqual(before, state_digest(snapshot(torch, np, ())))
        self.assertEqual(session.metadata()["state"], "failed")
        gc.collect(); self.assertIsNone(self.refs[0]())
        self.assertTrue(session.close()["passed"])
        self.assertTrue((session.work / "jobs/scene/bad/failure.json").is_file())
        with self.assertRaises(SessionClosed): session.run(job_id="scene", attempt_id="new")

    def test_false_decoder_restoration_fails_closed(self):
        @contextmanager
        def unrestored(model, recipe):
            yield {"profile_id": "unrestored-test", "restored": False}
        session = self.session()
        with self.assertRaisesRegex(SessionFailed, "restoration"):
            session.run(job_id="scene", attempt_id="bad", decoder_context=unrestored)
        self.assertTrue(session.close()["passed"])

    def test_directory_admission_failure_releases_new_process_owner(self):
        first = self.session(); second = self.session()
        original = Path.mkdir
        def fail(path, *args, **kwargs):
            if path == first.work / "jobs/scene/attempt": raise PermissionError("injected output admission failure")
            return original(path, *args, **kwargs)
        with patch.object(Path, "mkdir", new=fail), self.assertRaises(PermissionError):
            first.run(job_id="scene", attempt_id="attempt")
        self.assertEqual(self.events, [])
        second.run(job_id="scene", attempt_id="attempt")
        second.close()
        first.run(job_id="scene", attempt_id="attempt")

    def test_unused_context_manager_closes_without_loading_or_cuda(self):
        with self.session() as session:
            self.assertEqual(session.metadata()["model_load_count"], 0)
        self.assertEqual(self.events, [])
        self.assertEqual(session.close()["model_load_count"], 0)
        self.assertFalse(torch.cuda.is_initialized()); self.assertFalse(torch.cuda.is_available())

    def test_memory_and_file_share_runtime_without_array_files(self):
        from aureon.api import GenerationInput
        from aureon.terramind_native.memory_output import array_identity
        image = np.ones((12, 224, 224), dtype=np.float32)
        data = GenerationInput(image, bands=INPUT_BANDS, units="surface_reflectance_x10000",
                               valid_mask=np.ones((224, 224), dtype=bool))
        session = self.session()
        saved = session.run(job_id="file", attempt_id="reference", reuse="off")
        array = session.generate(data, job_id="memory", attempt_id="reference", execution_profile="reference")
        # This compares the delivery branches with a CPU double, not model parity.
        np.testing.assert_array_equal(array["array"], np.load(saved["raw"]["path"]))
        self.assertEqual(array["trace"]["tokens"], array_identity(np.load(saved["trace"]["tokens"]["path"])))
        self.assertIsNone(array["provenance"]["input"]["grid"])
        self.assertEqual(array["provenance"]["input_identity"], data.identity)
        self.assertEqual(array["execution_selection"]["effective"], {"reuse":"off", "denoiser":"native"})
        memory_dir = session.work / "jobs/memory/reference"
        self.assertEqual({p.name for p in memory_dir.iterdir()}, {"receipt.json"})
        receipt = read_json(memory_dir / "receipt.json")
        self.assertNotIn("array", receipt)
        self.assertEqual(receipt["provenance"]["array"], array_identity(array["array"]))
        array["array"].fill(999)
        next_result = session.generate(data, job_id="memory", attempt_id="recommended", execution_profile="recommended")
        self.assertEqual(next_result["execution_selection"]["effective"], {"reuse":"auto", "denoiser":"native"})
        self.assertTrue(next_result["execution_selection"]["fallback"])
        self.assertIn("CUDA profile required", next_result["execution_selection"]["reasons"][-1])
        np.testing.assert_array_equal(next_result["array"], np.load(saved["raw"]["path"]))
        self.assertEqual(session.metadata()["model_load_count"], 1)
        with self.assertRaises(FileExistsError):
            session.generate(data, job_id="memory", attempt_id="reference")
        with self.assertRaises(ValueError):
            session.run(data, job_id="memory", attempt_id="logits", capture_logits=True)
        self.assertFalse((session.work / "jobs/memory/logits").exists())

    def test_assets_only_config_needs_no_input_file_for_memory(self):
        from aureon.api import GenerationInput
        config = read_json(self.config_path)
        del config["input_manifest"]
        config["schema"] = "aureon.terramind_assets.v1"
        assets = self.root / "assets.json"
        write_json(assets, config)
        session = TerraMindSession(assets, self.root/"assets-session", execute_model=True, execution_profile="reference")
        self.sessions.append(session)
        with self.assertRaisesRegex(ValueError, "Assets-only config"):
            session.run(job_id="missing", attempt_id="1")
        self.assertEqual(self.events, [])
        data = GenerationInput(np.ones((12,224,224), np.float32), bands=INPUT_BANDS,
                               units="surface_reflectance_x10000", valid_mask=np.ones((224,224), bool))
        with patch.object(native, "_input", side_effect=AssertionError("file input used")):
            result = session.generate(data, job_id="array", attempt_id="1")
        self.assertEqual(result["array"].shape, (2,224,224))
        self.assertEqual(result["provenance"]["delivery"], "memory")

    def test_memory_failure_restores_state_and_closes_without_success_arrays(self):
        from aureon.api import GenerationInput
        data = GenerationInput(np.ones((12,224,224), np.float32), bands=INPUT_BANDS,
                               units="surface_reflectance_x10000", valid_mask=np.ones((224,224), bool))
        before = state_digest(snapshot(torch, np, ()))
        session = self.session(execution_profile="reference")
        with patch.object(Model, "forward", side_effect=RuntimeError("memory forward failed")):
            with self.assertRaisesRegex(SessionFailed, "memory forward failed"):
                session.generate(data, job_id="array", attempt_id="failed")
        self.assertEqual(before, state_digest(snapshot(torch, np, ())))
        self.assertTrue(session.close()["passed"])
        self.assertEqual({p.name for p in (session.work/"jobs/array/failed").iterdir()}, {"failure.json"})
        self.assertIsNone(self.refs[0]())

    def test_generation_engine_two_calls_use_same_core_and_lazy_owned_config(self):
        from aureon.api import GenerationEngine, GenerationInput
        data = GenerationInput(np.ones((12,224,224), np.float32), bands=INPUT_BANDS,
                               units="surface_reflectance_x10000", valid_mask=np.ones((224,224), bool))
        def config(**kwargs):
            value = copy.deepcopy(self.config)
            value.pop("input_manifest")
            value["schema"] = "aureon.terramind_assets.v1"
            write_json(kwargs["output"], value)
        work = self.root / "facade"
        with patch("aureon.terramind_native.offline.create_config", side_effect=config), \
                patch("aureon.api.models.require_profile"), patch("aureon.terramind_native.offline.inspect_config", return_value={"model_id":"terramind-v1-base"}):
            with GenerationEngine(model_file="fixture", tokenizer_file="fixture", work=work, execute_model=True) as engine:
                self.assertFalse(work.exists())
                reference = engine.generate(data, seed=17, execution_profile="reference")
                recommended = engine.generate(data, seed=17)
                self.assertEqual(engine.metadata()["session"]["model_load_count"], 1)
                self.assertEqual(recommended["execution_selection"]["requested"], "recommended")
                np.testing.assert_array_equal(reference["array"], recommended["array"])
                self.assertEqual(reference["rng"]["request_post_rng_sha256"], recommended["rng"]["request_post_rng_sha256"])
            self.assertTrue(engine.close()["passed"])
        self.assertFalse(list(work.rglob("*.npy")))
        self.assertFalse(list(work.rglob("*.tif")))
        self.assertNotIn("input_manifest", read_json(work/"assets.json"))


if __name__ == "__main__": unittest.main()
