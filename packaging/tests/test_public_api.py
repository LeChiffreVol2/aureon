"""CPU contracts for the new facade; fake executors never establish model parity."""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from aureon.api import SceneRequest, ExecutionRequired, run_scene, ProfileError
from aureon.api import profiles, v1


class PublicAPI(unittest.TestCase):
    def request(self, root, **kwargs):
        data = dict(prepared_input=root/"manifest.json",bundle=root/"bundle",output=root/"output",
                    model_id="kao-terramind-b",profile_id="fp32-stream-guarded-v3",qualification=root/"qualification.json")
        data.update(kwargs)
        return SceneRequest(**data)

    def test_import_help_no_optional_runtime(self):
        code = """import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','terratorch','numpy','rasterio','vllm'):
   raise AssertionError('optional runtime imported: '+fullname)
sys.meta_path.insert(0,Block())
import aureon.api
from aureon.api.cli import main
main(['profiles'])
assert not any(n in sys.modules for n in ('torch','numpy','rasterio'))
"""
        child = subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True,timeout=30)
        self.assertEqual(child.returncode,0,child.stderr)

    def test_explicit_execution_before_io(self):
        with patch.object(v1,"require_profile",side_effect=AssertionError("read")):
            with self.assertRaises(ExecutionRequired):
                run_scene(self.request(Path("missing")))

    def test_model_profile_and_immutable_request(self):
        with self.assertRaisesRegex(ValueError,"KAO-TerraMind B"):
            self.request(Path("missing"),model_id="kao-terramind-c")
        for model in v1.MODELS:
            r = self.request(Path("missing"),model_id=model,profile_id="fp32-full-reference-v1",qualification=None,environment="scene-legacy")
            with self.assertRaises(FrozenInstanceError):
                r.model_id = "changed"

    def test_actionable_environment_mismatch(self):
        with patch.object(profiles.metadata,"version",return_value="0.0"):
            with self.assertRaisesRegex(ProfileError,"separate Python 3.12 Linux environment"):
                profiles.require_profile("scene")
        with self.assertRaisesRegex(ProfileError,"exactly one"):
            profiles.inspect_profile("scene,vllm")

    def fake(self, root, *, fail=False, bad_cleanup=False):
        (root/"bundle").mkdir()
        (root/"manifest.json").write_text('{"fixture":true}')
        (root/"qualification.json").write_text(json.dumps({"model_id":"kao-terramind-b","profile_id":"fp32-stream-guarded-v3"}))
        owner = self
        self.closed = 0
        class Executor:
            def __init__(self,bundle,work,qualification,*,execute_model):
                owner.assertTrue(execute_model)
                owner.assertNotIn("qualification_mode",qualification)
                self.work = work
                work.mkdir()
            def execute(self,manifest,job_id):
                if fail:
                    raise RuntimeError("synthetic execution failure")
                job = self.work/job_id
                job.mkdir()
                data = {"status":"completed","source_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "validation":{"numerical":{"status":"not_evaluated"}}}
                paths = {}
                for name in ("class_map.tif","valid_mask.tif","provenance.json","job.json"):
                    paths[name] = job/name
                    paths[name].write_text(json.dumps(data))
                return paths
            def close(self):
                owner.closed += 1
                return {"cleanup_passed":not bad_cleanup,"full_identity_check_passed":True,"errors":[]}
        module = types.SimpleNamespace(engine_for=lambda _:types.SimpleNamespace(
            verify_qualification=lambda r:r,ResidentExecutor=Executor))
        return module

    def test_guarded_completion_and_owned_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = self.fake(root)
            with patch.dict(sys.modules,{"aureon.cloud.profiles":fake}), patch.object(v1,"require_profile",return_value={"status":"passed"}), patch.object(v1.metadata,"version",return_value="test"):
                result = run_scene(self.request(root),execute_model=True)
            self.assertEqual(self.closed,1)
            self.assertEqual(result.numerical_status,"not_evaluated")
            record = json.loads(result.receipt.read_text())
            self.assertEqual(record["schema"],"aureon.engine_result.v1")
            self.assertEqual(record["task_accuracy"],"not_evaluated")
            self.assertEqual(len(record["artifacts"]),4)

    def test_failure_closes_without_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = self.fake(root,fail=True)
            with patch.dict(sys.modules,{"aureon.cloud.profiles":fake}), patch.object(v1,"require_profile",return_value={}):
                with self.assertRaisesRegex(RuntimeError,"synthetic"):
                    run_scene(self.request(root),execute_model=True)
            self.assertEqual(self.closed,1)
            self.assertFalse((root/"output/engine-result.json").exists())
            self.assertTrue((root/"output/engine-failure.json").exists())

    def test_cleanup_failure_prevents_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = self.fake(root,bad_cleanup=True)
            with patch.dict(sys.modules,{"aureon.cloud.profiles":fake}), patch.object(v1,"require_profile",return_value={}):
                with self.assertRaisesRegex(RuntimeError,"cleanup"):
                    run_scene(self.request(root),execute_model=True)
            self.assertFalse((root/"output/engine-result.json").exists())

    def test_qualification_profile_mismatch_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = self.fake(root)
            (root/"qualification.json").write_text('{"profile_id":"fp32-guarded-v2"}')
            with patch.dict(sys.modules,{"aureon.cloud.profiles":fake}), patch.object(v1,"require_profile",return_value={}):
                with self.assertRaisesRegex(ValueError,"Qualification"):
                    run_scene(self.request(root),execute_model=True)
            self.assertFalse((root/"output").exists())


if __name__ == "__main__":
    unittest.main()
