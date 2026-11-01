"""Selection/admission contracts only; doubles do not qualify model execution."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from aureon.api import GenerationRequest, ExecutionRequired, run_model, plan_generation_execution
from aureon.terramind_native import session as native
from aureon.terramind_native.execution import completed_selection

RECIPE = dict(token_steps=8, diffusion_steps=10, temperature=1., top_p=.8, top_k=0, seed=17)


class SelectionTests(unittest.TestCase):
    def test_cpu_plan_does_not_import_or_probe_runtime(self):
        code = '''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','numpy','rasterio','terratorch','vllm'):
   raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from aureon.api.cli import main
main(['generation-plan'])
'''
        child = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        plan = json.loads(child.stdout)
        self.assertEqual(plan["planned"], dict(reuse="auto", denoiser="graph"))
        self.assertIsNone(plan["effective"])
        self.assertEqual(plan["cuda"], "not_probed")

    def test_reference_and_out_of_scope_recipes_remain_native(self):
        for steps in (1, 8):
            for diffusion in (1, 10, 50):
                recipe = dict(RECIPE, token_steps=steps, diffusion_steps=diffusion)
                ref = plan_generation_execution(execution_profile="reference", recipe=recipe)
                self.assertEqual(ref["planned"], dict(reuse="off", denoiser="native"))
                auto = plan_generation_execution(recipe=recipe)
                self.assertEqual(auto["fallback"], (steps, diffusion) != (8, 10))
                if auto["fallback"]:
                    self.assertEqual(auto["planned"], ref["planned"])

    def test_old_defaults_explicit_execution_and_conflicts(self):
        request = GenerationRequest("missing", "out", "job", "attempt")
        self.assertEqual((request.execution_profile, request.reuse, request.denoiser), (None, None, None))
        for profile in (None, "recommended", "reference"):
            with self.assertRaises(ExecutionRequired):
                run_model(GenerationRequest("missing", "out", "job", "attempt", execution_profile=profile))
        for kwargs in ({"reuse":"off"}, {"denoiser":"native"}):
            with self.assertRaisesRegex(ValueError, "owns reuse/denoiser"):
                GenerationRequest("missing", "out", "job", "attempt", execution_profile="recommended", **kwargs)
        with self.assertRaises(ValueError):
            plan_generation_execution(execution_profile="fast", recipe=RECIPE)

    def test_runtime_fallback_receipt_is_detached_and_honest(self):
        plan = plan_generation_execution(recipe=RECIPE)
        result = completed_selection(plan, {"enabled": False, "reason": "unsupported_topology_or_mode"},
                                     {"profile_id":"native", "enabled":False, "fallback":True, "reason":"CUDA profile required"})
        self.assertEqual(result["requested"], "recommended")
        self.assertEqual(result["effective"], dict(reuse="off", denoiser="native"))
        self.assertTrue(result["fallback"])
        self.assertEqual(len(result["reasons"]), 2)
        self.assertEqual(plan["reasons"], [])
        self.assertEqual(result["qualification"], "not_established_by_selection")
        # A rejected first signature must report native even after graph admission.
        signature = completed_selection(plan, {"enabled":True, "reason":"eligible"},
                                        {"profile_id":"graph", "enabled":False, "fallback":True, "reason":"signature"})
        self.assertEqual(signature["effective"]["denoiser"], "native")

    def test_resident_selection_recomputed_for_each_recipe_and_closed(self):
        calls = []
        class Runtime:
            def load(self, *args): pass
            def run(self, *args, **kw):
                calls.append(kw)
                return {"status":"completed"}
            def close(self): return {"passed":True}
        with tempfile.TemporaryDirectory(prefix="aureon-selection-test-") as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps({"recipe":RECIPE, "input_manifest":{"path":"unused"}}))
            with patch.object(native, "_NativeRuntime", Runtime), patch.object(native, "_input", return_value=(None, None, {}, {})):
                with native.TerraMindSession(config, root/"work", execute_model=True, execution_profile="recommended") as session:
                    session.run(job_id="job", attempt_id="eight")
                    session.run(job_id="job", attempt_id="one", recipe=dict(RECIPE, token_steps=1))
                    session.run(job_id="job", attempt_id="ref", execution_profile="reference")
                    session.run(job_id="job", attempt_id="expert", execution_profile=None, reuse="off")
                self.assertTrue(session.close()["passed"])
                self.assertEqual(session.metadata()["model_load_count"], 1)
            self.assertEqual([(x["mode"], x["denoiser"]) for x in calls],
                             [("auto","graph"), ("off","native"), ("off","native"), ("off","native")])
            self.assertEqual(calls[0]["execution_plan"]["requested"], "recommended")
            self.assertNotIn("execution_plan", calls[3])
        self.assertFalse(root.exists())

    def test_facade_propagates_selection(self):
        from unittest.mock import MagicMock
        fake = MagicMock()
        fake.__enter__.return_value = fake
        fake.run.return_value = {"status":"completed"}
        with patch("aureon.api.models.open_model_session", return_value=fake) as opened:
            run_model(GenerationRequest("c", "w", "j", "a", execution_profile="reference"), execute_model=True)
        self.assertEqual(opened.call_args.kwargs["execution_profile"], "reference")


if __name__ == "__main__":
    unittest.main()
