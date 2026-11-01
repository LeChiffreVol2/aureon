"""Replica control and bounded stage tests; not a GPU performance benchmark."""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from aureon.scaling import (ArtifactStore, Broker, CoordinatorServer, Job, Pipeline,
                            RemoteClient, ResidentAdapter, ResidentWorker, Stage, WorkerSpec)


class PipelineTests(unittest.TestCase):
    def test_stages_and_result_memory_credits(self):
        with Pipeline([Stage("read", lambda x, c: x + 1), Stage("infer", lambda x, c: x * 2),
                       Stage("export", lambda x, c: str(x))], max_requests=2, max_live_bytes=20) as pipeline:
            handle = pipeline.submit("one", 3, tenant="t", seed=1, reservation_bytes=20)
            with self.assertRaises(BufferError):
                pipeline.submit("two", 2, tenant="t", seed=1, reservation_bytes=1)
            self.assertEqual(handle.result(2), "8")
            self.assertEqual(pipeline.metrics()["reserved_bytes"], 0)
            second = pipeline.submit("two", 2, tenant="t", seed=1, reservation_bytes=20)
            self.assertEqual(second.result(2), "6")

    def test_fairness_and_no_head_of_line_per_tenant(self):
        gate, entered, order = threading.Event(), threading.Event(), []
        def run(value, context):
            order.append(context.request_id)
            if value == 0:
                entered.set()
                gate.wait(2)
            return value
        with Pipeline([Stage("infer", run)], max_requests=8, max_live_bytes=8) as pipeline:
            handles = [pipeline.submit("a0", 0, tenant="a", seed=0, reservation_bytes=1)]
            self.assertTrue(entered.wait(2))
            for name, tenant in [("a1", "a"), ("a2", "a"), ("b1", "b"), ("a3", "a")]:
                handles.append(pipeline.submit(name, 1, tenant=tenant, seed=1, reservation_bytes=1))
            gate.set()
            [handle.result(2) for handle in handles]
        self.assertEqual(order, ["a0", "a1", "b1", "a2", "a3"])

    def test_cancel_exception_and_terminal_close(self):
        gate, entered = threading.Event(), threading.Event()
        def run(value, context):
            entered.set()
            gate.wait(2)
            return value
        pipeline = Pipeline([Stage("infer", run)], max_requests=1, max_live_bytes=1)
        handle = pipeline.submit("x", 1, tenant="a", seed=0, reservation_bytes=1)
        self.assertTrue(entered.wait(2))
        handle.cancel()
        gate.set()
        with self.assertRaises(CancelledError):
            handle.result(2)
        self.assertEqual(pipeline.metrics()["reserved_bytes"], 0)
        pipeline.close()
        with self.assertRaises(RuntimeError):
            pipeline.submit("z", 1, tenant="a", seed=0, reservation_bytes=1)

    def test_stage_failure_and_discard_release(self):
        def fail(value, context):
            raise ValueError("bad data")
        with Pipeline([Stage("prepare", fail)], max_requests=1, max_live_bytes=1) as pipeline:
            handle = pipeline.submit("x", 1, tenant="a", seed=1, reservation_bytes=1)
            with self.assertRaises(ValueError):
                handle.result(2)
            self.assertEqual(pipeline.metrics()["reserved_bytes"], 0)

    def test_close_drains_unconsumed_results(self):
        pipeline = Pipeline([Stage("s", lambda x, c: x)], max_requests=2, max_live_bytes=4)
        handles = [pipeline.submit(str(i), i, tenant="a", seed=i, reservation_bytes=2) for i in range(2)]
        pipeline.close()
        self.assertEqual([h.result() for h in handles], [0, 1])
        self.assertEqual(pipeline.metrics()["reserved_bytes"], 0)

    def test_explicit_thread_safe_stage_overlap(self):
        barrier = threading.Barrier(2)
        def concurrent(value, context):
            barrier.wait(2)
            return value + context.seed
        with Pipeline([Stage("prepare", concurrent, workers=2)], max_requests=2, max_live_bytes=2) as pipeline:
            handles = [pipeline.submit(str(i), 10, tenant="a", seed=i, reservation_bytes=1) for i in range(2)]
            self.assertEqual([h.result(3) for h in handles], [10, 11])

    def test_discard_releases_payload_even_when_handle_is_retained(self):
        import gc
        import weakref
        class Payload:
            pass
        with Pipeline([Stage("s", lambda x, c: x)], max_requests=1, max_live_bytes=1) as pipeline:
            payload = Payload()
            reference = weakref.ref(payload)
            handle = pipeline.submit("one", payload, tenant="a", seed=0, reservation_bytes=1)
            del payload
            handle.discard()
            for _ in range(100):
                gc.collect()
                if reference() is None:
                    break
                time.sleep(.01)
            self.assertIsNone(reference())
            self.assertEqual(pipeline.metrics()["reserved_bytes"], 0)
            with self.assertRaises(RuntimeError):
                handle.result()

    def test_resident_cleanup_runs_on_same_owner_thread(self):
        calls = []
        def infer(value, context):
            calls.append(("infer", threading.get_ident()))
            return value
        def close():
            calls.append(("close", threading.get_ident()))
        with Pipeline([Stage("infer", infer, finalize=close)], max_requests=1, max_live_bytes=1) as pipeline:
            pipeline.submit("one", 1, tenant="a", seed=0, reservation_bytes=1).result(2)
        self.assertEqual([row[0] for row in calls], ["infer", "close"])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertNotEqual(calls[0][1], threading.get_ident())

    def test_cleanup_error_preserves_primary_exception(self):
        def close():
            raise ValueError("cleanup")
        error = RuntimeError("primary")
        with self.assertRaisesRegex(RuntimeError, "primary"):
            with Pipeline([Stage("s", lambda v, c: v, finalize=close)], max_requests=1, max_live_bytes=1):
                raise error
        self.assertIn("cleanup also failed", error.__notes__[0])


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "jobs.db"
        self.clock = [100.]
        self.broker = Broker(self.path, max_jobs=20, lease_seconds=10, clock=lambda: self.clock[0])
        self.broker.register(WorkerSpec("small", ("terra",), 10, 1))
        self.broker.register(WorkerSpec("large", ("terra", "copgen"), 100, 2))

    def tearDown(self):
        self.broker.close()
        self.temp.cleanup()

    def job(self, name, *, tenant="a", route="terra", size=5, attempts=3):
        return Job.create(name, tenant=tenant, route=route, payload={"input": "identity"}, seed=17,
                          required_bytes=size, max_attempts=attempts)

    def test_capacity_route_and_tenant_fairness(self):
        for job in [self.job("big", size=50), self.job("a1"), self.job("a2"), self.job("b1", tenant="b")]:
            self.broker.submit(job)
        first = self.broker.claim("small")
        self.assertEqual(first.job.job_id, "a1")
        self.assertIsNone(self.broker.claim("small"))
        self.broker.complete(first, {"n": 1})
        self.assertEqual(self.broker.claim("small").job.job_id, "b1")
        self.assertEqual(self.broker.claim("large").job.job_id, "big")

    def test_retry_fencing_restart_and_exactly_one_commit(self):
        self.broker.submit(self.job("one"))
        old = self.broker.claim("large")
        self.broker.close()
        self.clock[0] += 11
        self.broker = Broker(self.path, lease_seconds=10, clock=lambda: self.clock[0])
        new = self.broker.claim("small")
        self.assertEqual(new.attempt, 2)
        with self.assertRaises(RuntimeError):
            self.broker.complete(old, {"old": True})
        self.broker.complete(new, {"new": True})
        self.broker.complete(new, {"new": True})
        with self.assertRaises(ValueError):
            self.broker.complete(new, {"new": False})
        self.assertEqual(self.broker.status("one")["result"], {"new": True})

    def test_heartbeat_drain_cancel_and_attempt_budget(self):
        self.broker.submit(self.job("one", attempts=1))
        lease = self.broker.claim("small")
        self.clock[0] += 8
        self.broker.heartbeat(lease)
        self.clock[0] += 8
        self.assertEqual(self.broker.status("one")["state"], "running")
        self.broker.drain("small")
        self.broker.submit(self.job("two"))
        self.assertIsNone(self.broker.claim("small"))
        self.clock[0] += 3
        self.assertEqual(self.broker.status("one")["state"], "failed")
        lease2 = self.broker.claim("large")
        self.broker.cancel("two")
        with self.assertRaises(RuntimeError):
            self.broker.complete(lease2, {})

    def test_idempotency_retention_and_conflicts(self):
        job = self.job("one")
        self.broker.submit(job)
        self.broker.submit(job)
        with self.assertRaises(ValueError):
            self.broker.submit(self.job("one", size=7))
        with self.assertRaises(RuntimeError):
            self.broker.forget("one")
        self.broker.cancel("one")
        self.broker.forget("one")
        with self.assertRaises(KeyError):
            self.broker.status("one")

    def test_errors_are_bounded_and_retries_explicit(self):
        self.broker.submit(self.job("one"))
        lease = self.broker.claim("small")
        self.broker.fail(lease, "temporary", retry=True)
        new = self.broker.claim("small")
        self.assertEqual(new.attempt, 2)
        self.broker.fail(new, "terminal")
        self.assertIsNone(self.broker.claim("small"))

    def test_different_route_cannot_consume_worker_and_limits_retained(self):
        self.broker.submit(self.job("cop", route="copgen"))
        self.assertIsNone(self.broker.claim("small"))
        lease = self.broker.claim("large")
        self.assertEqual(lease.job.route, "copgen")
        with self.assertRaises(RuntimeError):
            self.broker.register(WorkerSpec("large", ("terra",), 1))
        for index in range(19):
            self.broker.submit(self.job("j" + str(index)))
        with self.assertRaises(BufferError):
            self.broker.submit(self.job("overflow"))


class WireTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.broker = Broker(self.root / "jobs.db", lease_seconds=2)
        self.broker.register(WorkerSpec("gpu0", ("terra", "cpu"), 100, 1))
        self.broker.register(WorkerSpec("gpu1", ("cpu",), 100, 1))
        self.store = ArtifactStore(self.root / "blobs", max_bytes=2**20, max_artifact_bytes=2**19)
        self.client_key, self.worker_key, self.other_key = [secrets.token_urlsafe(32) for _ in range(3)]
        self.server = CoordinatorServer(self.broker, self.store, client_key=self.client_key,
                                       worker_keys={"gpu0": self.worker_key, "gpu1": self.other_key})
        self.server.start()
        self.client = RemoteClient(self.server.url, self.client_key, allow_loopback_http=True)
        self.worker = RemoteClient(self.server.url, self.worker_key, allow_loopback_http=True)

    def tearDown(self):
        self.server.close()
        self.broker.close()
        self.temp.cleanup()

    def submit(self, name="one"):
        job = Job.create(name, tenant="t", route="cpu", payload={}, seed=17, required_bytes=5)
        self.client.call("submit", job=asdict(job))
        return job

    def test_full_job_artifact_flow_and_lease_heartbeat(self):
        data = b"real-transfer" * 10000
        digest = hashlib.sha256(data).hexdigest()
        self.client.upload(io.BytesIO(data), size=len(data), digest=digest)
        self.submit()
        def infer(context):
            destination = self.root / "input"
            self.worker.download(digest, destination)
            context.check()
            self.assertEqual(destination.read_bytes(), data)
            return {"artifacts": {"output": digest}, "seed": context.job.seed}
        worker = ResidentWorker(self.worker, {"cpu": infer}, heartbeat_seconds=.1)
        self.assertEqual(worker.run_until_idle(max_jobs=2), 1)
        status = self.client.call("status", job_id="one")
        self.assertEqual(status["state"], "succeeded")
        self.client.download(digest, self.root / "output")
        self.assertEqual((self.root / "output").read_bytes(), data)
        handle = self.client.job("one")
        self.assertEqual(handle.result(timeout=0)["seed"], 17)
        paths = handle.download_artifacts(self.root / "delivered")
        self.assertEqual(paths["output"].read_bytes(), data)

    def test_typed_client_submission_reattach_and_timeout(self):
        job = Job.create("typed", tenant="t", route="cpu", payload={}, seed=17, required_bytes=5)
        handle = self.client.submit(job)
        self.assertEqual(handle.job_id, "typed")
        with self.assertRaises(TimeoutError):
            handle.result(timeout=0)
        self.client.job("typed").cancel()
        with self.assertRaises(RuntimeError):
            handle.result(timeout=0)

    def test_permission_boundaries_and_stale_worker(self):
        self.submit()
        with self.assertRaises(urllib.error.HTTPError):
            self.client.call("claim")
        lease = self.worker.claim()
        other = RemoteClient(self.server.url, self.other_key, allow_loopback_http=True)
        with self.assertRaises(urllib.error.HTTPError):
            other.call("complete", lease=asdict(lease), result={"artifacts": {}})
        with self.assertRaises(urllib.error.HTTPError):
            self.worker.call("status", job_id="one")
        self.client.call("cancel", job_id="one")
        with self.assertRaises(urllib.error.HTTPError):
            self.worker.call("complete", lease=asdict(lease), result={"artifacts": {}})

    def test_missing_output_not_committed_and_digest_rejected(self):
        self.submit()
        lease = self.worker.claim()
        with self.assertRaises(urllib.error.HTTPError):
            self.worker.call("complete", lease=asdict(lease), result={"artifacts": {"cog": "0" * 64}})
        self.assertEqual(self.broker.status("one")["state"], "running")
        with self.assertRaises(urllib.error.HTTPError):
            self.client.upload(io.BytesIO(b"abc"), size=3, digest="0" * 64)
        self.assertFalse((self.store.root / ("0" * 64)).exists())

    def test_no_remote_plaintext_or_path_download(self):
        with self.assertRaises(ValueError):
            RemoteClient("http://10.0.0.1:9000", self.worker_key)
        with self.assertRaises(ValueError):
            self.client.download("../secret", self.root / "bad")

    def test_separate_process_worker_real_json_transport(self):
        self.submit()
        package_root = Path(os.environ.get("AUREON_SCALING_TEST_RUNTIME", str(Path(__file__).resolve().parents[1])))
        script = self.root / "worker.py"
        script.write_text("import sys,json\nfrom pathlib import Path\n"
                          "config=json.loads(Path(sys.argv[1]).read_text())\n"
                          "sys.path.insert(0,config['root'])\n"
                          "from aureon.scaling import RemoteClient,ResidentWorker\n"
                          "client=RemoteClient(config['url'],config['key'],allow_loopback_http=True)\n"
                          "worker=ResidentWorker(client,{'cpu':lambda c:{'artifacts':{},'seed':c.job.seed}})\n"
                          "assert worker.run_until_idle(max_jobs=2)==1\n")
        config = self.root / "worker-config.json"
        config.write_text(json.dumps({"root": str(package_root), "url": self.server.url, "key": self.worker_key}))
        completed = subprocess.run([sys.executable, "-I", "-B", str(script), str(config)],
                                   capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.client.call("status", job_id="one")["state"], "succeeded")

    def test_artifact_dedup_quota_and_corruption(self):
        data = b"x" * 100
        digest = hashlib.sha256(data).hexdigest()
        store = ArtifactStore(self.root / "tiny", max_bytes=100, max_artifact_bytes=100)
        store.put(io.BytesIO(data), size=100, digest=digest)
        store.put(io.BytesIO(data), size=100, digest=digest)
        with self.assertRaises(BufferError):
            store.put(io.BytesIO(b"y"), size=1, digest=hashlib.sha256(b"y").hexdigest())
        (store.root / digest).write_bytes(b"corrupt")
        with self.assertRaises(ValueError), store.open(digest):
            pass

    def test_long_callback_heartbeat_survives_original_lease(self):
        self.submit()
        def handler(context):
            # Longer than the real 2-second lease; native work stays on owner thread.
            time.sleep(2.3)
            context.check()
            return {"artifacts": {}, "seed": context.job.seed}
        worker = ResidentWorker(self.worker, {"cpu": handler}, heartbeat_seconds=.2)
        self.assertTrue(worker.run_once())
        self.assertEqual(self.broker.status("one")["state"], "succeeded")

    def test_cancellation_observed_during_worker_callback(self):
        self.submit()
        def handler(context):
            self.client.call("cancel", job_id=context.job.job_id)
            self.assertTrue(context.cancelled.wait(2))
            context.check()
            return {"artifacts": {}}
        worker = ResidentWorker(self.worker, {"cpu": handler}, heartbeat_seconds=.05)
        with self.assertRaises(RuntimeError):
            worker.run_once()
        self.assertEqual(self.broker.status("one")["state"], "cancelled")

    def test_unknown_credentials_and_redirect_refused(self):
        wrong = RemoteClient(self.server.url, secrets.token_urlsafe(32), allow_loopback_http=True)
        with self.assertRaises(urllib.error.HTTPError):
            wrong.call("claim")
        from aureon.scaling.transport import _NoRedirect
        from http.client import HTTPMessage
        with self.assertRaises(ValueError):
            _NoRedirect().redirect_request(urllib.request.Request(self.server.url), io.BytesIO(), 302,
                                            "redirect", HTTPMessage(), "https://elsewhere.invalid")


class AdapterTests(unittest.TestCase):
    def test_common_lifecycle_preserves_explicit_seed_and_output(self):
        from aureon.scaling import WorkContext
        job = Job.create("one", tenant="a", route="terra", payload={}, seed=19, required_bytes=4)
        adapter = ResidentAdapter(prepare=lambda c: c.job.seed, infer=lambda value, c: value * 2,
                                  deliver=lambda value, c: {"artifacts": {}, "value": value})
        self.assertEqual(adapter(WorkContext(job, threading.Event()))["value"], 38)

    def test_native_bridge_forwards_attempt_and_rejects_seed_mismatch(self):
        from aureon.scaling import WorkContext, terramind_worker_adapter, copgen_worker_adapter
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        job = Job.create("one", tenant="a", route="terra", payload={}, seed=19, required_bytes=4)
        context = WorkContext(job, threading.Event(), attempt=3)
        request, data, engine = SimpleNamespace(sampling=SimpleNamespace(seed=19)), object(), Mock()
        adapter = terramind_worker_adapter(engine, prepare=lambda c: (request, data),
                                           deliver=lambda value, c: {"artifacts": {}, "value": value})
        with patch("aureon.api.eo.generate_eo", return_value={"array": "owned"}) as generate:
            self.assertEqual(adapter(context)["value"], {"array": "owned"})
            generate.assert_called_once_with(engine, request, data, job_id="one", attempt_id="3")
        request.sampling.seed = 20
        with patch("aureon.api.eo.generate_eo") as generate, self.assertRaises(ValueError):
            adapter(context)
        generate.assert_not_called()
        adapter2 = copgen_worker_adapter(engine, prepare=lambda c: (request, data),
                                         deliver=lambda value, c: {"artifacts": {}, "value": value})
        with self.assertRaises(ValueError):
            adapter2(context)
        engine.generate.assert_not_called()
        request.sampling.seed = 19
        engine.generate.return_value = {"output": "S1RTC"}
        self.assertEqual(adapter2(context)["value"], {"output": "S1RTC"})
        engine.generate.assert_called_once_with(request, data)


if __name__ == "__main__":
    unittest.main()
