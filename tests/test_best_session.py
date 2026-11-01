"""CPU lifecycle checks through the original resident run path; no model calls."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from aureon.api.best import BestSessionBusy, BestTerraMindSession
from aureon._reference.integrity import verify_core

CORE = importlib.import_module("aureon._reference.terramind.session")
BROAD = importlib.import_module("aureon.terramind_native.session")
RECIPE = {"token_steps": 8, "diffusion_steps": 10, "temperature": 1.0,
          "top_p": 0.8, "top_k": 0, "seed": 17}


class CpuRuntime:
    """Only exercises lifecycle/admission/delivery; makes no numerical claim."""
    calls = []
    fail = False
    cleanup_passes = True
    entered = None
    resume = None

    def load(self, config, work):
        self.calls.append(("load", threading.get_ident()))

    def run(self, image, mask, metadata, identity, recipe, output, **kwargs):
        self.calls.append(("run", threading.get_ident()))
        if self.entered is not None:
            self.entered.set()
            if not self.resume.wait(5):
                raise TimeoutError("CPU test coordination failed")
        if self.fail:
            raise ValueError("CPU seam execution failure")
        return {"recipe": recipe, "reuse": kwargs["mode"], "denoiser": kwargs["denoiser"],
                "job_id": kwargs["job_id"], "attempt_id": kwargs["attempt_id"],
                "decoder_hook": kwargs["decoder_context"], "capture_logits": kwargs["capture_logits"]}

    def close(self):
        self.calls.append(("close", threading.get_ident()))
        return {"passed": self.cleanup_passes}


class BestSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aureon-best-cpu-", dir=os.environ.get("AUREON_TEST_SCRATCH"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"recipe": RECIPE, "input_manifest": {"path": "input.json"}}))
        (self.root / "input.json").write_text("{}")
        CpuRuntime.calls = []
        CpuRuntime.fail = False
        CpuRuntime.cleanup_passes = True
        CpuRuntime.entered = CpuRuntime.resume = None
        self.patchers = [patch.object(CORE, "_NativeRuntime", CpuRuntime),
                         patch.object(CORE, "_input", return_value=(None, None, {}, {}))]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)

    def session(self, name="work", **kwargs):
        result = BestTerraMindSession(self.config, self.root / name, execute_model=True, **kwargs)
        self.addCleanup(result.close)
        return result

    def test_execution_permission_and_profile_rejected_before_io(self):
        with self.assertRaises(ValueError):
            BestTerraMindSession("missing", self.root / "no-work")
        with self.assertRaises(ValueError):
            BestTerraMindSession("missing", self.root / "no-work", execute_model=True, profile="fp16")
        self.assertFalse((self.root / "no-work").exists())

    def test_original_run_loads_once_preserves_recipe_and_does_not_rehash_core(self):
        with patch("aureon.api.best.verify_core", wraps=verify_core) as verify:
            with self.session() as session:
                identity = session.core_fingerprint
                first = session.run(job_id="j", attempt_id="a")
                second = session.run(job_id="j", attempt_id="b", recipe={**RECIPE, "seed": 101})
                self.assertEqual(verify.call_count, 1)
                self.assertIs(session.core_fingerprint, identity)
                self.assertEqual(first["reuse"], "auto")
                self.assertEqual(first["denoiser"], "graph")
                self.assertIsNone(first["decoder_hook"])
                self.assertFalse(first["capture_logits"])
                self.assertEqual(first["recipe"]["seed"], 17)
                self.assertEqual(second["recipe"]["seed"], 101)
                self.assertEqual(second["model_load_count"], 1)
                self.assertEqual(session.metadata()["jobs_completed"], 2)
                saved = json.loads((session.work / "jobs/j/b/receipt.json").read_text())
                self.assertEqual(saved, second)
                with self.assertRaises(FileExistsError):
                    session.run(job_id="j", attempt_id="b")
                coremeta = session.metadata()["core"]
                coremeta["wheel_sha256"] = "caller mutation"
                self.assertNotEqual(session.metadata()["core"]["wheel_sha256"], "caller mutation")
            self.assertEqual([c[0] for c in CpuRuntime.calls], ["load", "run", "run", "close"])
            self.assertEqual(len({c[1] for c in CpuRuntime.calls}), 1)
            self.assertNotEqual(CpuRuntime.calls[0][1], threading.get_ident())
            self.assertFalse(BROAD._PROCESS_OWNER.locked())

    def test_reference_profile_selects_off_native_without_numeric_hook(self):
        with self.session(profile="reference_fp32") as session:
            result = session.run(job_id="reference", attempt_id="one")
            self.assertEqual((result["reuse"], result["denoiser"]), ("off", "native"))

    def test_broad_and_selected_ownership_is_mutually_exclusive(self):
        BROAD._PROCESS_OWNER.acquire()
        try:
            with self.session("blocked") as waiting:
                with self.assertRaises(BestSessionBusy):
                    waiting.run(job_id="blocked", attempt_id="1")
            self.assertEqual(CpuRuntime.calls, [])
        finally:
            BROAD._PROCESS_OWNER.release()
        with self.session() as first:
            first.run(job_id="first", attempt_id="1")
            self.assertFalse(BROAD._PROCESS_OWNER.acquire(blocking=False))
            with self.session("waiting") as second:
                with self.assertRaises(BestSessionBusy):
                    second.run(job_id="second", attempt_id="1")
        with self.session("second"):
            pass

    def test_run_and_lifecycle_are_original_not_forwarding_wrappers(self):
        for name in ('run', '_run_owned', 'close', '_close_owned', '__enter__', '__exit__'):
            self.assertIs(getattr(BestTerraMindSession,name),getattr(CORE.TerraMindSession,name))
        with patch('subprocess.Popen',side_effect=AssertionError('No model process adapter')):
            with self.session() as session:
                self.assertFalse(hasattr(session,'_session'))
                session.run(job_id='direct',attempt_id='1')

    def test_constructor_failure_releases_process_owner(self):
        with self.assertRaises(FileNotFoundError):
            BestTerraMindSession(self.root / "missing", self.root / "bad", execute_model=True)
        self.assertFalse(BROAD._PROCESS_OWNER.locked())
        with self.session():
            pass

    def test_busy_close_does_not_release_owner(self):
        CpuRuntime.entered, CpuRuntime.resume = threading.Event(), threading.Event()
        session = self.session()
        errors = []
        def run():
            try:
                session.run(job_id="busy", attempt_id="one")
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(CpuRuntime.entered.wait(3))
            with self.assertRaises(BestSessionBusy):
                session.close()
            with self.assertRaises(BestSessionBusy):
                session.run(job_id="other", attempt_id="one")
            self.assertTrue(BROAD._PROCESS_OWNER.locked())
        finally:
            CpuRuntime.resume.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        session.close()
        self.assertFalse(BROAD._PROCESS_OWNER.locked())

    def test_execution_failure_is_terminal_and_close_releases_owner(self):
        CpuRuntime.fail = True
        session = self.session()
        with self.assertRaises(CORE.SessionFailed):
            session.run(job_id="bad", attempt_id="one")
        with self.assertRaises(CORE.SessionClosed):
            session.run(job_id="bad", attempt_id="two")
        self.assertTrue((session.work / "jobs/bad/one/failure.json").is_file())
        session.close()
        self.assertFalse(BROAD._PROCESS_OWNER.locked())

    def test_config_change_is_rejected_before_model_load(self):
        session = self.session()
        self.config.write_text("{}")
        with self.assertRaisesRegex(ValueError, "config changed"):
            session.run(job_id="changed", attempt_id="one")
        self.assertEqual(CpuRuntime.calls, [])

    def test_cleanup_failure_preserves_primary_error_and_releases_owner(self):
        CpuRuntime.cleanup_passes = False
        session = BestTerraMindSession(self.config, self.root / "work", execute_model=True)
        with self.assertRaisesRegex(ValueError, "primary") as raised:
            with session:
                session.run(job_id="cleanup", attempt_id="one")
                raise ValueError("primary")
        self.assertTrue(any("cleanup failed" in note for note in raised.exception.__notes__))
        self.assertFalse(BROAD._PROCESS_OWNER.locked())

    def test_cold_public_import_does_not_load_neural_dependencies(self):
        code = '''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','terratorch','numpy','rasterio','diffusers'}:
   raise AssertionError('Unexpected cold dependency: '+fullname)
sys.meta_path.insert(0,Block())
from aureon.api import BestTerraMindSession
assert 'aureon.terramind_native.session' not in sys.modules
assert 'aureon.api.model_pool' not in sys.modules
'''
        code = "import sys; sys.path.insert(0, " + repr(str(Path(__file__).resolve().parents[1])) + ");\n" + code
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
