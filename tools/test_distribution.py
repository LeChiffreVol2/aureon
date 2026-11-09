"""Installer fault and publication-drift checks; no install or network access."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load("install")
guard = load("check_distribution")
index_metadata = load("check_index_metadata")


class Response(io.BytesIO):
    def geturl(self):
        return "https://release-assets.githubusercontent.com/test.whl"


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aureon-install-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = b"test artifact payload"
        self.row = {"filename": "test.whl", "url": installer.RELEASE_URL + "test.whl",
                    "sha256": hashlib.sha256(self.data).hexdigest(), "bytes": len(self.data)}
        self.target = self.root / "test.whl"

    def test_actual_release_and_locks_pass(self):
        self.assertEqual(installer.release()["sha256"], installer.BEST)

    def test_download_verified_without_network(self):
        with patch.object(installer, "urlopen", return_value=Response(self.data)):
            installer.acquire(self.row, self.target)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertEqual(list(self.root.iterdir()), [self.target])

    def test_wrong_hash_does_not_publish(self):
        with patch.object(installer, "urlopen", return_value=Response(b"x" * len(self.data))):
            with self.assertRaisesRegex(ValueError, "size/hash"):
                installer.acquire(self.row, self.target)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_oversized_download_is_bounded(self):
        with patch.object(installer, "urlopen", return_value=Response(self.data + b"extra")):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                installer.acquire(self.row, self.target)
        self.assertFalse(self.target.exists())

    def test_existing_artifact_preserved(self):
        self.target.write_bytes(b"existing")
        with patch.object(installer, "urlopen") as network:
            with self.assertRaisesRegex(ValueError, "already exists"):
                installer.acquire(self.row, self.target)
            network.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b"existing")

    def test_interrupted_download_does_not_publish(self):
        with patch.object(installer, "urlopen", side_effect=OSError("disconnected")):
            with self.assertRaises(OSError):
                installer.acquire(self.row, self.target)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_insecure_redirect_rejected(self):
        response = Response(self.data)
        response.geturl = lambda: "http://example.invalid/wheel"
        with patch.object(installer, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "Insecure"):
                installer.acquire(self.row, self.target)

    def test_existing_environment_refused(self):
        with self.assertRaisesRegex(ValueError, "new --prefix"):
            installer.destination_path(self.root)

    def test_source_checkout_prefix_refused(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            installer.destination_path(self.root / "env", self.root)

    def test_external_prefix_allowed(self):
        result = installer.destination_path(self.root / "env", self.root / "checkout")
        self.assertEqual(result, self.root / "env")

    def test_environment_excludes_override_and_cuda(self):
        with patch.dict(os.environ, {"PIP_EXTRA_INDEX_URL": "https://example.invalid", "PYTHONPATH": "bad"}):
            result = installer.environment()
        self.assertNotIn("PIP_EXTRA_INDEX_URL", result)
        self.assertNotIn("PYTHONPATH", result)
        self.assertEqual(result["CUDA_VISIBLE_DEVICES"], "")

    def test_plan_has_no_side_effects(self):
        with patch("sys.argv", ["install.py", "--plan"]), patch("sys.stdout", new_callable=io.StringIO) as output:
            with patch.object(installer, "install") as operation, patch.object(installer, "urlopen") as network:
                self.assertEqual(installer.main(), 0)
                operation.assert_not_called()
                network.assert_not_called()
        self.assertFalse(json.loads(output.getvalue())["writes"])


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aureon-distribution-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        names = set(guard.GUIDES) | {"packaging/review/SELECTED-ENGINE.json", "release/v0.1.0.json",
                                    "release/quality-v0.1.0.json", "pyproject.toml"}
        for name in names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())

    def change_quality(self, key, value):
        path = self.root / "release/quality-v0.1.0.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record[key] = value
        path.write_text(json.dumps(record), encoding="utf-8")

    def test_actual_distribution_passes(self):
        self.assertTrue(guard.inspect(self.root)["passed"])

    def test_historical_speed_cannot_be_promoted_by_metadata(self):
        self.change_quality("performance_applies_to_selected_wheel", True)
        with self.assertRaisesRegex(ValueError, "measured cohort"):
            guard.inspect(self.root)

    def test_wrong_selected_wheel_rejected(self):
        self.change_quality("selected_wheel_sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "identities"):
            guard.inspect(self.root)

    def test_stale_download_advice_rejected(self):
        path = self.root / "packaging/SUPPORT.md"
        path.write_text("The local candidate is unpublished", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Stale"):
            guard.inspect(self.root)

    def test_missing_dimension_rejected(self):
        record = json.loads((self.root / "release/quality-v0.1.0.json").read_text(encoding="utf-8"))
        self.change_quality("evidence", record["evidence"][:-1])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            guard.inspect(self.root)

    def test_unreviewed_pypi_status_rejected(self):
        self.change_quality("pypi_open", True)
        with self.assertRaisesRegex(ValueError, "Review evidence"):
            guard.inspect(self.root)

    def test_embedded_old_selection_rejected(self):
        text = "**Runtime selection:** use [selected](https://example.invalid), `0fa5ef2b...`."
        self.assertIn("embedded_readme_selects_historical_runtime", index_metadata.description_issues(text))

    def test_historical_comparison_is_allowed_in_description(self):
        text = "**Runtime selection:** use `e09e710c`. Historical performance comes from `0fa5ef2b`."
        self.assertEqual(index_metadata.description_issues(text), [])


if __name__ == "__main__":
    unittest.main()
