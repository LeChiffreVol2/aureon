"""CPU lifecycle and attempt isolation tests; fakes do not establish parity."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from aureon.api import EngineSession, ExecutionRequired, SessionBusy, SessionClosed, SessionFailed
from aureon.api import session as api

REAL_BACKEND = api._Backend


class Sessions(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.bundle = self.root/"bundle"
        self.bundle.mkdir()
        self.manifest = self.root/"manifest.json"
        self.manifest.write_text('{"scene_id":"fixture"}')
        self.calls = []
        self.fail = self.bad_cleanup = self.mutate = False
        self.wait = None
        self.entered = threading.Event()
        self.verify_calls = 0
        owner = self

        class Backend:
            def __init__(self, bundle, work, profile, qualification):
                self.work = work
                work.mkdir()
                self.jobs = 0
                self.worker_id = work.parent.name
                owner.calls.append(("construct", threading.get_ident()))

            def run(self, manifest, job_id, attempt_id):
                owner.calls.append(("run", threading.get_ident()))
                owner.entered.set()
                if owner.wait is not None:
                    if not owner.wait.wait(5):
                        raise TimeoutError("Fake deadline")
                if owner.fail:
                    raise RuntimeError("synthetic failure")
                output = self.work/attempt_id/job_id
                output.mkdir(parents=True)
                record = {"status":"completed", "job_id":job_id,
                          "source_manifest_sha256":api._hash(manifest), "parity":{"passed":True},
                          "validation":{"numerical":{"status":"not_evaluated"}}}
                paths = {}
                for name in ("job.json", "class_map.tif", "valid_mask.tif", "provenance.json"):
                    paths[name] = output/name
                    paths[name].write_text(json.dumps(record))
                self.jobs += 1
                if owner.mutate:
                    manifest.write_text('{"scene_id":"changed"}')
                return paths

            def metadata(self):
                owner.calls.append(("metadata", threading.get_ident()))
                return {"worker_id":self.worker_id,"model_load_count":1,"jobs_completed":self.jobs,"state":"ready"}

            def close(self):
                owner.calls.append(("close", threading.get_ident()))
                return {"cleanup_passed":not owner.bad_cleanup,"full_identity_check_passed":True,
                        "errors":[],"model_load_count":1,"jobs_completed":self.jobs,
                        "cuda_allocated_bytes_after_cleanup":0,"cuda_reserved_bytes_after_cleanup":0}

        runtime = types.ModuleType("aureon.runtime")
        def verify(bundle):
            self.verify_calls += 1
            return {"model_id":"kao-terramind-b","receipt_sha256":"bundle-hash"}
        runtime.verify_bundle = verify
        self.stack.enter_context(patch.dict("sys.modules", {"aureon.runtime": runtime}))
        self.stack.enter_context(patch.object(api, "require_profile", return_value={"status":"passed"}))
        self.stack.enter_context(patch.object(api.packages, "version", return_value="test"))
        self.stack.enter_context(patch.object(api, "_Backend", Backend))
        # Test isolation includes the deliberately poisoned-process case.
        self.stack.enter_context(patch.object(api, "_PROCESS_OWNER", threading.Lock()))
        self.sessions = []
        self.addCleanup(self.close_sessions)

    def close_sessions(self):
        for s in self.sessions:
            s.close()

    def create(self, name="session", **kwargs):
        options = dict(profile_id="fp32-full-reference-v1", execute_model=True)
        options.update(kwargs)
        result = EngineSession(self.bundle, self.root/name, **options)
        self.sessions.append(result)
        return result

    def test_execution_opt_in_precedes_io(self):
        for value in (False, 1, None, "yes"):
            with self.assertRaises(ExecutionRequired):
                EngineSession("missing", "missing", profile_id="fp32-full-reference-v1", execute_model=value)
        self.assertFalse(self.calls)

    def test_unsupported_models_environment_and_profiles(self):
        for change in ({"model_id":"kao-terramind-a"}, {"model_id":"kao-terramind-c"},
                       {"environment":"scene-legacy"}, {"profile_id":"amp_fp16"}):
            with self.assertRaises(ValueError):
                self.create(**change)
        self.assertFalse(self.calls)

    def test_environment_failure_precedes_directory_or_model(self):
        with patch.object(api,"require_profile",side_effect=ValueError("environment")):
            with self.assertRaisesRegex(ValueError,"environment"):
                self.create()
        self.assertFalse((self.root/"session").exists())
        self.assertFalse(self.calls)

    def test_guarded_requires_exact_qualification(self):
        with self.assertRaises(ValueError):
            self.create(profile_id="fp32-stream-guarded-v3")
        qual = self.root/"q.json"
        qual.write_text(json.dumps({"profile_id":"fp32-stream-guarded-v3", "model_id":"kao-terramind-b"}))
        module = types.ModuleType("aureon.cloud.profiles")
        module.engine_for = lambda profile: types.SimpleNamespace(verify_qualification=lambda record: (_ for _ in ()).throw(ValueError("Qualified Engine runtime source changed")))
        with patch.dict("sys.modules", {"aureon.cloud.profiles":module}):
            with self.assertRaisesRegex(ValueError,"runtime source changed"):
                self.create(profile_id="fp32-stream-guarded-v3",qualification=qual)
        self.assertFalse(self.calls)

    def test_lazy_close_is_idempotent_without_model(self):
        s = self.create()
        self.assertEqual(s.metadata()["model_load_count"],0)
        first = s.close()
        self.assertTrue(first["cleanup_passed"])
        self.assertEqual(first,s.close())
        self.assertFalse(self.calls)
        with self.assertRaises(SessionClosed):
            s.run(self.manifest)

    def test_two_jobs_share_model_and_owner(self):
        caller = threading.get_ident()
        s = self.create()
        a = s.run(self.manifest)
        b = s.run(self.manifest)
        self.assertNotEqual(a.output,b.output)
        self.assertEqual(s.metadata()["model_load_count"],1)
        self.assertEqual(s.metadata()["worker"]["jobs_completed"],2)
        self.assertEqual(json.loads(a.receipt.read_text())["session_shutdown"],"not_evaluated")
        cleanup = s.close()
        self.assertTrue(cleanup["cleanup_passed"])
        self.assertEqual(cleanup["jobs_completed"],2)
        self.assertEqual([name for name,_ in self.calls].count("construct"),1)
        owners = {tid for _,tid in self.calls}
        self.assertEqual(len(owners),1)
        self.assertNotIn(caller,owners)

    def test_logical_retry_uses_new_attempt_without_receipt_rewrite(self):
        s = self.create()
        job = "a"*32
        first = s.run(self.manifest,job_id=job,attempt_id="b"*32)
        original = first.job_receipt.read_bytes()
        second = s.run(self.manifest,job_id=job,attempt_id="c"*32)
        self.assertNotEqual(first.output,second.output)
        self.assertEqual(first.job_receipt.read_bytes(),original)
        for result in (first,second):
            self.assertEqual(json.loads(result.job_receipt.read_text())["job_id"],job)
            self.assertEqual(json.loads(result.provenance.read_text())["job_id"],job)
        self.assertEqual(s.metadata()["model_load_count"],1)

    def test_duplicate_attempt_never_executes_twice(self):
        s = self.create()
        s.run(self.manifest,job_id="a"*32,attempt_id="b"*32)
        with self.assertRaises(FileExistsError):
            s.run(self.manifest,job_id="a"*32,attempt_id="b"*32)
        self.assertEqual([n for n,_ in self.calls].count("run"),1)
        self.assertEqual(s.metadata()["state"],"ready")

    def test_stale_attempt_has_no_publication_result(self):
        s = self.create()
        outcomes = iter((True,False))
        published = []
        with self.assertRaisesRegex(SessionFailed,"stale after"):
            published.append(s.run(self.manifest,job_id="a"*32,attempt_id="b"*32,
                                   is_current_attempt=lambda:next(outcomes)))
        self.assertFalse(published)
        self.assertFalse(list((self.root/"session").rglob("engine-result.json")))
        self.assertEqual(s.metadata()["state"],"failed")
        self.assertTrue(s.close()["cleanup_passed"])

    def test_stale_before_execution_starts_no_model(self):
        s = self.create()
        with self.assertRaisesRegex(SessionFailed,"stale before"):
            s.run(self.manifest,is_current_attempt=lambda:False)
        self.assertFalse(self.calls)
        self.assertEqual(self.verify_calls,0)

    def test_busy_process_does_not_consume_retry_identity(self):
        one, two = self.create("one"), self.create("two")
        one.run(self.manifest)
        job = "a"*32
        with self.assertRaises(SessionBusy):
            two.run(self.manifest,job_id=job)
        self.assertEqual(list((self.root/"two/admissions").iterdir()),[])
        one.close()
        result = two.run(self.manifest,job_id=job)
        self.assertEqual(json.loads(result.job_receipt.read_text())["job_id"],job)

    def test_overlapping_calls_fail_without_queue(self):
        s = self.create()
        self.wait = threading.Event()
        failures = []
        def run():
            try:s.run(self.manifest)
            except BaseException as error:failures.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(self.entered.wait(5))
            with self.assertRaises(SessionBusy):s.run(self.manifest)
            with self.assertRaises(SessionBusy):s.close()
            with self.assertRaises(SessionBusy):s.metadata()
        finally:
            self.wait.set(); thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        self.assertEqual([n for n,_ in self.calls].count("run"),1)

    def test_failure_is_terminal_and_cleanup_runs_on_owner(self):
        s = self.create()
        self.fail = True
        with self.assertRaisesRegex(SessionFailed,"synthetic failure"):s.run(self.manifest)
        with self.assertRaises(SessionClosed):s.run(self.manifest)
        before = list(self.calls)
        cleanup = s.close()
        self.assertEqual(before,self.calls)
        self.assertTrue(cleanup["cleanup_passed"])
        self.assertEqual({t for n,t in self.calls if n in ("run","close")}, {self.calls[0][1]})

    def test_source_mutation_fails_closed(self):
        self.mutate = True
        s = self.create()
        with self.assertRaisesRegex(SessionFailed,"identity mismatch"):s.run(self.manifest)
        self.assertFalse(list((self.root/"session").rglob("engine-result.json")))

    def test_invalid_ids_and_path_collision_precede_model(self):
        s = self.create()
        for value in ("../escape", "a"*31, "A"*32, 1):
            with self.assertRaises(ValueError):s.run(self.manifest,job_id=value)
            with self.assertRaises(ValueError):s.run(self.manifest,attempt_id=value)
        with self.assertRaises(ValueError):self.create()
        self.assertFalse(self.calls)

    def test_failed_cleanup_blocks_another_session(self):
        first, second = self.create("first"), self.create("second")
        first.run(self.manifest)
        self.bad_cleanup = True
        self.assertFalse(first.close()["cleanup_passed"])
        with self.assertRaises(SessionBusy):second.run(self.manifest)

    def test_context_manager_closes_and_reports_failure(self):
        s = self.create()
        self.bad_cleanup = True
        with self.assertRaisesRegex(SessionFailed,"shutdown failed"):
            with s:s.run(self.manifest)
        self.assertEqual(s.metadata()["state"],"closed")

    def test_full_reference_bridge_registers_new_scene_without_reload(self):
        constructed = []
        calls = []
        module = types.ModuleType("aureon.scene_api.worker")
        prepare = types.ModuleType("aureon.prepare")
        prepare.check_scene = lambda path: json.loads(path.read_text())
        class Worker:
            def __init__(self,bundle,scenes,output,**kwargs):
                constructed.append(self); self.scenes = scenes
            def _manifest(self,path):return json.loads(path.read_text()),None,api._hash(path)
            def run_job(self,scene,job,output):
                calls.append((scene,job,output,dict(self.scenes)))
                return {"download_artifacts":{"job.json":{"path":"job.json"}}}
            def close(self):return {"cleanup_passed":True}
        module.ResidentSceneWorker = Worker
        with patch.dict("sys.modules",{"aureon.scene_api.worker":module,"aureon.prepare":prepare}):
            backend = REAL_BACKEND(self.bundle,self.root/"bridge","fp32-full-reference-v1",None)
            backend.run(self.manifest,"a"*32,"b"*32)
            other = self.root/"other.json";other.write_text('{"scene_id":"other"}')
            backend.run(other,"a"*32,"c"*32)
            backend.close()
        self.assertEqual(len(constructed),1)
        self.assertEqual(calls[1][0],"other")
        self.assertEqual(set(calls[1][3]),{"other"})
        self.assertNotEqual(calls[0][2],calls[1][2])

    def test_guarded_bridge_separates_attempt_parent_and_logical_job(self):
        seen = []
        module = types.ModuleType("aureon.cloud.profiles")
        prepare = types.ModuleType("aureon.prepare")
        prepare.check_scene = lambda path: (_ for _ in ()).throw(AssertionError("Redundant guarded input validation"))
        class Executor:
            def __init__(self,bundle,work,qualification,**kwargs):self.work=work;work.mkdir()
            def execute(self,manifest,job):
                seen.append((self.work,job));return {"job.json":self.work/job/"job.json"}
        module.engine_for = lambda profile:types.SimpleNamespace(ResidentExecutor=Executor)
        with patch.dict("sys.modules",{"aureon.cloud.profiles":module,"aureon.prepare":prepare}):
            backend = REAL_BACKEND(self.bundle,self.root/"guarded","fp32-stream-guarded-v3",{})
            first = backend.run(self.manifest,"a"*32,"b"*32)
            second = backend.run(self.manifest,"a"*32,"c"*32)
        self.assertNotEqual(first["job.json"],second["job.json"])
        self.assertEqual([job for _,job in seen],["a"*32,"a"*32])
        self.assertEqual(backend.executor.work,backend.work)


if __name__ == "__main__":
    unittest.main()
