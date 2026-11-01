"""Command validation and routing without model execution or installation."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from aureon.api.cli import main


class BestCLITests(unittest.TestCase):
    def test_cost_routes_through_public_cli(self):
        output = StringIO()
        with redirect_stdout(output):
            code = main(["best", "cost", "--resource-seconds", "3600",
                         "--completed-outputs", "10", "--hourly-rate", "1"])
        data = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(data["total"], 1.)
        self.assertEqual(data["per_completed_output"], .1)
        self.assertFalse(data["billable"])

    def test_capacity_reports_process_plan_without_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"devices.json"
            path.write_text(json.dumps([{"device":"GPU-example", "total_bytes":8000, "free_bytes":7000}]))
            output = StringIO()
            with redirect_stdout(output):
                code = main(["best", "capacity", "--devices", str(path), "--per-replica-bytes", "1000",
                             "--reserve-bytes", "1000", "--replicas-per-device", "5"])
            self.assertEqual(code, 0)
            data = json.loads(output.getvalue())
            self.assertEqual(data["replicas"], 5)
            self.assertEqual(data["processes_launched"], 0)

    def test_generation_refuses_execution_before_touching_missing_assets(self):
        with redirect_stderr(StringIO()) as errors, self.assertRaises(SystemExit) as raised:
            main(["best", "generate", "--config", "absent", "--manifest", "absent", "--work", "absent"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--execute-model", errors.getvalue())

    def test_nan_cost_is_rejected(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            main(["best", "cost", "--resource-seconds", "nan", "--completed-outputs", "1", "--hourly-rate", "1"])
        self.assertEqual(raised.exception.code, 2)

    def test_scene_refuses_execution_before_touching_assets(self):
        with redirect_stderr(StringIO()) as errors, self.assertRaises(SystemExit) as raised:
            main(["best", "scene", "--image", "absent", "--valid-mask", "absent",
                  "--output", "absent", "--job-id", "test", "--config", "absent", "--work", "absent"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--execute-model", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
