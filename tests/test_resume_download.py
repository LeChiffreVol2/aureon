"""Recovery example: read-only requests, bound downloads and unchanged usage."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("resume_download_example", ROOT / "examples/client/resume_download.py")
example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = example
spec.loader.exec_module(example)


class RecoveryExampleTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ.get("AUREON_SERVICE_TEST_ROOT", tempfile.gettempdir()))
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="resume-example-", dir=parent))
        self.request = example.RecoveryRequest("a" * 32, "terramind-v1-base", "fp32-generation-v1", "generation", "d" * 64)

    def status(self, **changes):
        return {"id": self.request.job_id, "model_id": self.request.model,
                "profile_id": self.request.profile, "task": self.request.task,
                "state": "completed", **changes}

    def log(self, *, extra=(), job=True):
        rows = [{"event": "request", "model": self.request.model, "profile": self.request.profile,
                 "task": self.request.task, "manifest_sha256": self.request.manifest_sha256},
                {"event": "input_created", "input_id": "b" * 32, "manifest_sha256": self.request.manifest_sha256}]
        if job:
            rows.append({"event": "job_submitted", "job_id": self.request.job_id, "input_id": "b" * 32})
        rows.extend(extra)
        path = self.root / "request.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return path

    def test_recorded_request_can_be_resumed(self):
        self.assertEqual(example.read_recovery(self.log()), self.request)

    def test_uncertain_submission_is_not_invented(self):
        with self.assertRaisesRegex(example.RecoveryError, "uncertain submission"):
            example.read_recovery(self.log(job=False))

    def test_conflicting_input_binding_rejected(self):
        for extra in ({"event": "input_ready", "input_id": "c" * 32},
                      {"event": "input_created", "manifest_sha256": "e" * 64},
                      {"event": "job_submitted", "job_id": "f" * 32}):
            with self.subTest(extra=extra), self.assertRaises(example.RecoveryError):
                example.read_recovery(self.log(extra=[extra]))

    def test_large_or_malformed_log_is_rejected(self):
        path = self.root / "oversize.jsonl"
        path.write_bytes(b"x" * (256 * 1024 + 1))
        with self.assertRaises(example.RecoveryError):
            example.read_recovery(path)
        path.write_text("[]\n")
        with self.assertRaises(example.RecoveryError):
            example.read_recovery(path)

    def test_completed_job_only_reads_then_downloads(self):
        client = Mock(spec=["json", "wait_job", "download_result"])
        client.json.return_value = self.status()
        client.download_result.return_value = {"hashes_verified": True}
        result = example.resume_download(client, self.request, self.root / "result")
        self.assertTrue(result["hashes_verified"])
        client.json.assert_called_once_with("GET", "/v1/jobs/" + self.request.job_id)
        client.wait_job.assert_not_called()
        client.download_result.assert_called_once_with(self.request.job_id, self.root / "result",
            expected_manifest_sha256=self.request.manifest_sha256, model=self.request.model, task=self.request.task)

    def test_pending_and_terminal_failures_do_not_download(self):
        for state in ("queued", "running", "failed", "cancelled", "expired", "unknown"):
            client = Mock();client.json.return_value = self.status(state=state)
            with self.subTest(state=state), self.assertRaises(example.RecoveryError):
                example.resume_download(client, self.request, self.root / "result")
            client.download_result.assert_not_called();client.wait_job.assert_not_called()

    def test_optional_wait_rechecks_job_identity(self):
        client = Mock();client.json.return_value = self.status(state="running")
        client.wait_job.return_value = self.status(model_id="wrong-model")
        with self.assertRaisesRegex(example.RecoveryError, "identity"):
            example.resume_download(client, self.request, self.root / "result", wait_seconds=2)
        client.wait_job.assert_called_once_with(self.request.job_id, timeout=2)
        client.download_result.assert_not_called()

    def test_existing_output_and_invalid_wait_refuse_before_http(self):
        client = Mock()
        for seconds in (-1, float("nan"), float("inf"), 3601):
            with self.assertRaises(example.RecoveryError):
                example.resume_download(client, self.request, self.root / "result", wait_seconds=seconds)
        with self.assertRaises(example.RecoveryError):
            example.resume_download(client, self.request, self.root)
        client.json.assert_not_called()

    def test_real_HTTP_download_replay_does_not_add_usage(self):
        import test_service_workflows as workflows
        fixture = workflows.ServiceWorkflowTests();fixture.setUp()
        try:
            control, keys, _ = fixture.create()
            body = fixture.ready(control)
            job = control.submit("alpha", body, "recover-download-once")
            lease = control.claim()
            control.publish_engine(lease, fixture.files(control, lease))
            request = example.RecoveryRequest(job["id"], body["model_id"], body["profile_id"],
                "features", workflows.file_hash(Path(lease["manifest"])))
            before = control.metrics()["committed_usage_rows"]
            with fixture.http(control, keys) as (client, other, _):
                for n in range(2):
                    result = example.resume_download(client, request, self.root / ("download-" + str(n)))
                    self.assertTrue(result["hashes_verified"])
                    self.assertTrue((self.root / ("download-" + str(n)) / "download.json").is_file())
                with self.assertRaises(workflows.APIError):
                    example.resume_download(other, request, self.root / "other-tenant")
            self.assertEqual(before, 1)
            self.assertEqual(control.metrics()["committed_usage_rows"], before)
            self.assertIsNone(control.claim())
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
