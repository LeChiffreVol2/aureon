"""Reject private tools and test payloads even if mistakenly allowlisted."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("library_scope", ROOT / "packaging/check_library_scope.py")
scope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scope)


class LibraryScopeTests(unittest.TestCase):
    def test_private_modules_cannot_be_promoted_by_runtime_allowlist(self):
        for name in ("aureon/runpod_worker/dispatcher.py", "aureon/cloud/bridge.py",
                     "aureon/hosted_local/ledger.py", "aureon/web_server/contact.py",
                     "aureon/cloud/billing.py", "aureon/internal/state.py",
                     "aureon/research/helper.py", "aureon/../private.py",
                     "aureon//cloud/control.py", "aureon/finite_scan.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                scope.validate_runtime_names([name])

    def test_tests_and_data_never_become_wheel_runtime(self):
        for name in ("aureon/tests/test_engine.py", "aureon/test_engine.py",
                     "aureon/fixtures/input.npy", "aureon/weights.pt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                scope.validate_runtime_names([name])

    def test_portable_service_contracts_remain_supported(self):
        scope.validate_runtime_names(["aureon/cloud/control.py", "aureon/cloud/engine.py",
                                      "aureon/serving/control.py", "aureon/api/models.py"])

    def test_current_checkout_has_complete_local_imports(self):
        self.assertTrue(scope.inspect(ROOT)["passed"])

    def test_private_imports_are_rejected_including_lazy_and_relative(self):
        for source in ("import cloud.stage", "from cloud import stage",
                       "from aureon import runpod_worker",
                       "from . import hosted_local",
                       "def run():\n    from cloud import check",
                       "import importlib\nimportlib.import_module('cloud.stage')"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "aureon").mkdir()
                (root / "packaging").mkdir()
                (root / "aureon/__init__.py").write_text(source, encoding="utf-8")
                (root / "packaging/runtime-files.json").write_text(
                    json.dumps({"files": ["aureon/__init__.py"]}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    scope.inspect(root)


if __name__ == "__main__":
    unittest.main()
