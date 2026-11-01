"""Lazy facade ownership, forwarding and lifecycle; never load model weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from aureon.api import GenerationEngine, GenerationInput, ExecutionRequired
from aureon.api import generation_engine as facade
from aureon.terramind_native.common import INPUT_BANDS
from aureon.terramind_native.session import SessionBusy, SessionClosed, SessionFailed


class GenerationEngineTests(unittest.TestCase):
    def data(self):
        return GenerationInput(np.ones((12,224,224), np.float32), bands=INPUT_BANDS,
                               units="surface_reflectance_x10000", valid_mask=np.ones((224,224), bool))

    def test_import_and_unused_constructor_are_lazy(self):
        code = '''import sys,importlib.abc,tempfile
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','numpy','rasterio','terratorch','diffusers'):
   raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from aureon.api import GenerationEngine
with tempfile.TemporaryDirectory() as tmp:
 work=Path(tmp)/'never-created'
 with GenerationEngine(model_file='missing',tokenizer_file='missing',work=work,execute_model=True) as engine:
  assert engine.metadata()['execution_profile']=='recommended'
  assert engine.metadata()['session'] is None
 assert not work.exists()
 assert engine.close()['model_load_count']==0
'''
        result = subprocess.run([sys.executable,"-B","-c",code], capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_retired_tensor_batch_option_rejects_without_side_effects(self):
        with tempfile.TemporaryDirectory(prefix="aureon-retired-option-") as tmp:
            work = Path(tmp) / "work"
            with patch.object(facade, "open_model_session", side_effect=AssertionError("must stay lazy")):
                with self.assertRaisesRegex(TypeError, "neural_batch"):
                    GenerationEngine(model_file="missing", tokenizer_file="missing", work=work,
                                     execute_model=True, neural_batch=object())
            self.assertFalse(work.exists())

    def test_unsupported_selection_and_execution_optin_before_paths(self):
        args = dict(model_file="missing",tokenizer_file="missing",work="missing-parent/new")
        with self.assertRaises(ExecutionRequired): GenerationEngine(**args)
        for model in ("kao-terramind-b", "terramind-v1-large-backbone"):
            with self.assertRaisesRegex(ValueError,"generation model_id"):
                GenerationEngine(model,execute_model=True,**args)
        for profile in (None,"fast"):
            with self.assertRaises(ValueError):
                GenerationEngine(execute_model=True,execution_profile=profile,**args)

    def test_one_owned_config_resident_default_and_seed_reference_forwarding(self):
        data = self.data()
        with tempfile.TemporaryDirectory(prefix="aureon-facade-test-") as tmp:
            work=Path(tmp)/"work"
            fake=MagicMock(); fake.close.return_value={"passed":True,"model_load_count":1}
            fake.generate.return_value={"array":np.zeros((2,224,224),np.float32),"provenance":{}}
            def create(**kw):
                self.assertNotIn("input_manifest",kw)
                Path(kw["output"]).write_text(json.dumps({"schema":"fixture"}))
            with patch("aureon.terramind_native.offline.create_config",side_effect=create) as config, \
                    patch.object(facade,"open_model_session",return_value=fake) as opening:
                with GenerationEngine(model_file="model",tokenizer_file="tokenizer",work=work,execute_model=True) as engine:
                    self.assertFalse(work.exists())
                    first=engine.generate(data,seed=123)
                    second=engine.generate(data,seed=456,execution_profile="reference")
                    self.assertEqual(config.call_count,1)
                    self.assertEqual(opening.call_count,1)
                    self.assertEqual(opening.call_args.kwargs["execution_profile"],"recommended")
                    self.assertEqual(opening.call_args.kwargs["input_kind"],"array")
                    self.assertEqual(opening.call_args.kwargs["config"],work/"assets.json")
                    self.assertEqual(opening.call_args.kwargs["work"],work/"session")
                    calls=fake.generate.call_args_list
                    self.assertEqual(calls[0].kwargs["recipe"]["seed"],123)
                    self.assertEqual(calls[1].kwargs["recipe"]["seed"],456)
                    self.assertEqual([c.kwargs["execution_profile"] for c in calls],["recommended","reference"])
                    self.assertIsNone(calls[0].kwargs["job_id"])
                self.assertTrue(engine.close()["passed"])
                fake.close.assert_called_once()
                with self.assertRaises(SessionClosed): engine.generate(data)
                self.assertTrue(work.exists())  # Engine retains owned execution evidence after close.

    def test_array_engine_uses_real_factory_without_file_schema_validation(self):
        """Mock model construction only: exercise Engine -> real factory dispatch."""
        with tempfile.TemporaryDirectory(prefix="aureon-array-factory-") as tmp:
            fake = MagicMock()
            fake.close.return_value = {"passed": True}
            fake.generate.return_value = {"array": np.zeros((2,224,224), np.float32), "provenance": {}}
            def create(**options):
                Path(options["output"]).write_text(json.dumps({"schema": "assets-only-test"}))
            with patch("aureon.terramind_native.offline.create_config", side_effect=create), \
                 patch("aureon.api.models.require_profile"), \
                 patch("aureon.terramind_native.offline.inspect_config", return_value={"model_id": "terramind-v1-base"}), \
                 patch("aureon.terramind_native.TerraMindSession", return_value=fake) as constructor:
                with GenerationEngine(model_file="model", tokenizer_file="tokenizer",
                                      work=Path(tmp)/"work", execute_model=True) as engine:
                    result = engine.generate(self.data())
                self.assertEqual(result["array"].shape, (2,224,224))
                constructor.assert_called_once()
                self.assertNotIn("input_kind", constructor.call_args.kwargs)
                fake.close.assert_called_once()

    def test_invalid_input_seed_and_job_do_not_create_work(self):
        data=self.data()
        with tempfile.TemporaryDirectory(prefix="aureon-facade-test-") as tmp:
            work=Path(tmp)/"work"
            with GenerationEngine(model_file="missing",tokenizer_file="missing",work=work,execute_model=True) as engine:
                with self.assertRaises(TypeError): engine.generate({"image":"unvalidated"})
                for seed in (-1,True,2**63):
                    with self.assertRaises(ValueError): engine.generate(data,seed=seed)
                with self.assertRaises(ValueError): engine.generate(data,job_id="")
                with self.assertRaises(ValueError): engine.generate(data,attempt_id="../escape")
                self.assertFalse(work.exists())

    def test_directory_race_never_adopts_foreign_work(self):
        data=self.data()
        with tempfile.TemporaryDirectory(prefix="aureon-facade-test-") as tmp:
            work=Path(tmp)/"work"
            engine=GenerationEngine(model_file="missing",tokenizer_file="missing",work=work,execute_model=True)
            work.mkdir(); (work/"foreign.txt").write_text("preserve")
            with patch("aureon.terramind_native.offline.create_config",side_effect=AssertionError("foreign write")):
                with self.assertRaises(FileExistsError): engine.generate(data)
            self.assertEqual({p.name for p in work.iterdir()},{"foreign.txt"})
            self.assertTrue(engine.close()["execution_failed"])

    def test_failure_closes_and_overlap_is_rejected(self):
        data=self.data(); entered=threading.Event(); release=threading.Event(); errors=[]
        with tempfile.TemporaryDirectory(prefix="aureon-facade-test-") as tmp:
            fake=MagicMock(); fake.close.return_value={"passed":True}
            def fail(*a,**kw):
                entered.set()
                if not release.wait(10): raise RuntimeError("test timeout")
                raise SessionFailed("injected runtime failure")
            fake.generate.side_effect=fail
            with patch("aureon.terramind_native.offline.create_config"), patch.object(facade,"open_model_session",return_value=fake):
                engine=GenerationEngine(model_file="m",tokenizer_file="t",work=Path(tmp)/"w",execute_model=True)
                def run():
                    try: engine.generate(data)
                    except SessionFailed as exc: errors.append(str(exc))
                thread=threading.Thread(target=run); thread.start()
                try:
                    self.assertTrue(entered.wait(10))
                    with self.assertRaises(SessionBusy): engine.generate(data)
                    with self.assertRaises(SessionBusy): engine.close()
                finally:
                    release.set(); thread.join(10)
                self.assertEqual(errors,["injected runtime failure"])
                self.assertFalse(thread.is_alive())
                self.assertTrue(engine.close()["passed"])
                with self.assertRaises(SessionClosed): engine.generate(data)
                fake.close.assert_called_once()


if __name__ == "__main__": unittest.main()
