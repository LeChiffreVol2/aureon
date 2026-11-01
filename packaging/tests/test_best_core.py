"""Pinned closure, reproduction and fail-closed build/type exception checks."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from aureon._reference.integrity import FILENAMES, MANIFEST_SHA256, PREFIX, verify_core

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "packaging" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BestCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aureon-core-check-", dir=os.environ.get("AUREON_TEST_SCRATCH"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / PREFIX, self.root / PREFIX)
        self.inventory = json.loads((ROOT / "packaging/runtime-files.json").read_text())["files"]

    def test_exact_inventory_and_checker_exception(self):
        result = load("verify_frozen_core").verify_frozen_runtime(self.root, self.inventory)
        self.assertTrue(result["verified"])
        self.assertEqual(result["manifest_sha256"], MANIFEST_SHA256)
        self.assertEqual(set(result["frozen_python_paths"]), {PREFIX+n for n in FILENAMES if n.endswith(".py")})
        self.assertEqual(len(result["frozen_python_paths"]), 12)
        self.assertNotIn("aureon/api/best.py", result["frozen_python_paths"])
        self.assertNotIn("aureon/_reference/integrity.py", result["frozen_python_paths"])

    def test_tampering_fails_before_type_exemption(self):
        target = self.root / PREFIX / "session.py"
        target.write_bytes(target.read_bytes() + b"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load("verify_frozen_core").verify_frozen_runtime(self.root, self.inventory)

    def test_manifest_rewriting_cannot_expand_exception(self):
        target = self.root / PREFIX / "SOURCE.json"
        value = json.loads(target.read_text())
        value["files"].append({"path": "aureon/api/best.py", "sha256": "0"*64, "bytes": 0})
        target.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            verify_core(self.root)

    def test_missing_or_extra_code_fails_closed(self):
        target = self.root / PREFIX / "new.py"
        target.write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "Undeclared"):
            verify_core(self.root)

    def test_incomplete_allowlist_fails_closed(self):
        names = [n for n in self.inventory if n != PREFIX + "rng.py"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load("verify_frozen_core").verify_frozen_runtime(self.root, names)

    def test_only_required_helpers_are_shipped(self):
        tree = ast.parse((self.root / PREFIX / "harness.py").read_bytes())
        self.assertEqual({n.name for n in tree.body if isinstance(n, ast.FunctionDef)},
                         {"capture_failure", "trace_native"})
        self.assertFalse((self.root / PREFIX / "offline.py").exists())
        self.assertFalse((self.root / PREFIX / "decoder_profile.py").exists())

    def test_selected_pair_reproduces_every_payload(self):
        wheel, source = os.environ.get("AUREON_SELECTED_WHEEL"), os.environ.get("AUREON_SELECTED_SOURCE")
        if not wheel or not source:
            self.skipTest("Exact archived pair not supplied; installed source hashes are checked separately")
        data = load("freeze_best_core").payloads(Path(wheel), Path(source))
        for name, content in data.items():
            self.assertEqual(content, (ROOT / PREFIX / name).read_bytes(), name)

    def test_build_rejects_overlays_on_frozen_source(self):
        from unittest.mock import patch
        packaging = self.root / "packaging"
        (packaging / "review").mkdir(parents=True)
        (packaging / "runtime-files.json").write_text(json.dumps({"files": self.inventory}))
        (packaging / "review/SOURCE-NOTICE-OVERLAYS.json").write_text(json.dumps(
            {"overlays": {PREFIX + "session.py": "# Would change frozen bytes\n"}}))
        (self.root / "pyproject.toml").write_text('[project]\nlicense-files = ["legal.txt"]\n')
        builder = load("build_review")
        with patch.object(builder, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "Frozen core cannot receive"):
                builder.stage(self.root / "candidate")
        self.assertFalse((self.root / "candidate").exists())

    def test_wrong_archives_rejected_before_any_extraction(self):
        bad = self.root / "bad.whl"
        bad.write_bytes(b"not the selected artifact")
        with self.assertRaisesRegex(ValueError, "exact selected"):
            load("freeze_best_core").payloads(bad, bad)


if __name__ == "__main__":
    unittest.main()
