"""CPU-only regression checks for review staging and archive admission.

Synthetic repositories and ZIP/TAR archives exercise release boundaries without
running a build backend, installing a dependency, or importing a neural runtime.
Actual backend builds and noneditable-wheel acceptance are separate evidence.
"""
from __future__ import annotations

import ast
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile


BUILDER_PATH = Path(__file__).resolve().parents[1] / "build_review.py"
SPEC = importlib.util.spec_from_file_location("aureon_review_builder_under_test", BUILDER_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
LEGAL = {"LICENSE", "NOTICE", "THIRD-PARTY-NOTICES.md", "REVIEW-STATUS.md",
         "SOURCE-NOTICE-OVERLAYS.json"}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


class ReviewFixture(unittest.TestCase):
    def setUp(self):
        scratch_parent = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory(prefix="aureon-packaging-test-", dir=scratch_parent)
        self.base = Path(self.temporary.name)
        if not self.base.resolve().is_relative_to(scratch_parent):
            raise AssertionError("Temporary cleanup target escaped the test directory")
        self.addCleanup(self.temporary.cleanup)
        self.root = self.base / "checkout"
        self.root.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(BUILDER, "ROOT", self.root).start()
        for name in BUILDER.FIXED:
            write(self.root / name, "# Synthetic review fixture\n" if name.endswith(".py") else "Synthetic review fixture\n")
        license_patterns = ["packaging/review/" + name for name in sorted(LEGAL)]
        license_patterns.append("packaging/review/third-party-licenses/upstream-LICENSE.txt")
        write(self.root / "pyproject.toml", "[project]\nname = \"aureon-inference\"\nversion = " +
              json.dumps(BUILDER.VERSION) + "\nlicense-files = " + json.dumps(license_patterns) + "\n")
        self.runtime = ["aureon/__init__.py", "aureon/example.py", "aureon/static/scene-console.html",
                        "aureon/static/scene-console.js", "aureon/static/scene-console.css"]
        for name in self.runtime:
            write(self.root / name, "VALUE = 'ภูมิศาสตร์'\n" if name.endswith(".py") else "fixture\n")
        self.set_runtime(self.runtime)
        self.set_overlays({})
        write(self.root / "packaging/review/third-party-licenses/upstream-LICENSE.txt", "Upstream notice fixture\n")
        write(self.root / "maintenance/prepare-sentinel2-scene.py", "def main():\n    return 0\n")
        self.output = self.base / "candidate"

    def set_runtime(self, names):
        write(self.root / "packaging/runtime-files.json", json.dumps({"files": names}))

    def set_overlays(self, overlays):
        write(self.root / "packaging/review/SOURCE-NOTICE-OVERLAYS.json", json.dumps({"overlays": overlays}))

    def stage(self):
        return BUILDER.stage(self.output)

    def assert_preflight_rejects(self):
        with self.assertRaises((ValueError, UnicodeError, SyntaxError)):
            self.stage()
        self.assertFalse(self.output.exists(), "Rejected source selection must not create a candidate")


class StagingTests(ReviewFixture):
    def test_path_admission_rejects_escape_absolute_windows_and_forbidden_members(self):
        for name in ("../outside.txt", "/absolute.txt", "aureon/../../outside.txt",
                     "aureon\\example.py", "X:/absolute.txt", ".env", "aureon/.env.local",
                     "models/weights.txt", "inputs/scene.txt", "runs/report.txt", ".archive/old.txt",
                     ".git/config", "aureon/__pycache__/code.txt", "node_modules/package.txt",
                     "aureon//example.py", "aureon/./example.py", "aureon/example.py\ninclude notes/*"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                BUILDER.member(self.root, name)

    def test_missing_member_rejected(self):
        with self.assertRaises(ValueError):
            BUILDER.member(self.root, "aureon/missing.py")

    def test_symlink_member_rejected(self):
        link = self.root / "aureon/link.py"
        try:
            link.symlink_to(self.root / "aureon/example.py")
        except (OSError, NotImplementedError) as error:
            self.skipTest("Platform does not permit creating a test symlink: " + str(error))
        with self.assertRaises(ValueError):
            BUILDER.member(self.root, "aureon/link.py")

    def test_duplicate_and_out_of_package_runtime_selection_rejected(self):
        for names in (self.runtime + self.runtime[:1], self.runtime + ["examples/not-runtime.py"]):
            with self.subTest(names=names):
                self.set_runtime(names)
                self.assert_preflight_rejects()

    def test_existing_output_preserved(self):
        self.output.mkdir()
        marker = self.output / "evidence.json"
        marker.write_text("immutable", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.stage()
        self.assertEqual(marker.read_text(encoding="utf-8"), "immutable")
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_parent_must_exist(self):
        with self.assertRaises(ValueError):
            BUILDER.stage(self.base / "absent" / "candidate")
        self.assertFalse((self.base / "absent").exists())

    def test_personal_runtime_paths_and_credential_like_content_fail_before_writes(self):
        # Assemble fixtures so this test source itself passes the release scan.
        forbidden = ["C:" + "/Users/fixture/project", "/mnt/c/" + "Users/fixture/project",
                     "/root/" + "aureon-runtime/venv-fixture/bin/python",
                     "sk-" + "A" * 32, "hf_" + "B" * 32,
                     "-----BEGIN " + "PRIVATE KEY-----"]
        for text in forbidden:
            with self.subTest(kind=text[:12]):
                write(self.root / "aureon/example.py", "# " + text + "\nVALUE = 1\n")
                self.assert_preflight_rejects()

    def test_non_utf8_source_is_rejected(self):
        write(self.root / "aureon/static/scene-console.html", b"\xff")
        self.assert_preflight_rejects()

    def test_overlay_cannot_add_executable_statement_or_target_unselected_file(self):
        for overlays in ({"aureon/example.py": "raise RuntimeError('changed')\n"},
                         {"aureon/unselected.py": "# attribution\n"},
                         {"aureon/static/scene-console.js": "# attribution\n"},
                         {"aureon/example.py": "# no terminating newline"}):
            with self.subTest(overlays=overlays):
                self.set_overlays(overlays)
                self.assert_preflight_rejects()

    def test_comment_encoding_cookie_cannot_change_python_constants(self):
        self.set_overlays({"aureon/example.py": "# coding: latin-1\n"})
        self.assert_preflight_rejects()

    def test_comment_overlay_preserves_original_ast_hashes_and_is_idempotent(self):
        original = (self.root / "aureon/example.py").read_bytes()
        prefix = "# Attribution fixture\n# Modified for review\n\n"
        self.set_overlays({"aureon/example.py": prefix})
        source, inventory = self.stage()
        staged = (source / "aureon/example.py").read_bytes()
        self.assertEqual(staged, prefix.encode() + original)
        self.assertEqual((self.root / "aureon/example.py").read_bytes(), original)
        self.assertEqual(ast.dump(ast.parse(original)), ast.dump(ast.parse(staged)))
        entry = next(item for item in inventory["files"] if item["path"] == "aureon/example.py")
        self.assertEqual(entry["original_sha256"], BUILDER.sha(original))
        self.assertEqual(entry["staged_sha256"], BUILDER.sha(staged))
        self.assertEqual(entry["bytes"], len(staged))
        self.assertTrue(entry["notice_prefix_added"])
        with patch.object(BUILDER, "ROOT", source):
            again, second = BUILDER.stage(self.base / "restaged")
        self.assertEqual((again / "aureon/example.py").read_bytes(), staged)
        second_entry = next(item for item in second["files"] if item["path"] == "aureon/example.py")
        self.assertFalse(second_entry["notice_prefix_added"])

    def test_unselected_personal_files_never_enter_source_inventory(self):
        write(self.root / "notes/private.txt", "fixture not selected\n")
        write(self.root / ".env", "UNSELECTED=fixture\n")
        source, inventory = self.stage()
        selected = {entry["path"] for entry in inventory["files"]}
        self.assertNotIn("notes/private.txt", selected)
        self.assertNotIn(".env", selected)
        self.assertFalse((source / "notes").exists())
        self.assertFalse((source / ".env").exists())
        self.assertTrue(inventory["review_only"])
        self.assertFalse(inventory["published"])


class ArchiveTests(ReviewFixture):
    def setUp(self):
        super().setUp()
        self.source, self.inventory = self.stage()
        self.dist = self.output / "dist"
        self.dist.mkdir()
        self.info = "aureon_inference-" + BUILDER.VERSION + ".dist-info"
        self.tar_root = "aureon_inference-" + BUILDER.VERSION
        self.metadata = ("Metadata-Version: 2.4\nName: aureon-inference\nVersion: " + BUILDER.VERSION +
                         "\nClassifier: Development Status :: 3 - Alpha\nLicense-Expression: Apache-2.0\n\n").encode()
        self.wheel = {entry["path"]: (self.source / entry["path"]).read_bytes()
                      for entry in self.inventory["files"] if entry["path"].startswith("aureon/")}
        for entry in self.inventory["files"]:
            name = entry["path"]
            if name.startswith("packaging/review/third-party-licenses/") or (
                    name.startswith("packaging/review/") and Path(name).name in LEGAL):
                self.wheel[self.info + "/licenses/" + name] = (self.source / name).read_bytes()
        self.wheel[self.info + "/METADATA"] = self.metadata
        self.wheel[self.info + "/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        self.wheel[self.info + "/entry_points.txt"] = b"[console_scripts]\n"
        self.wheel[self.info + "/top_level.txt"] = b"aureon\n"
        self.sdist = {entry["path"]: (self.source / entry["path"]).read_bytes()
                      for entry in self.inventory["files"]}
        for name in ("SOURCE-INVENTORY.json", "MANIFEST.in"):
            self.sdist[name] = (self.source / name).read_bytes()
        self.sdist["PKG-INFO"] = self.metadata

    def archives(self, *, wheel_extra=(), tar_extra=(), tar_links=()):
        record_path = self.info + "/RECORD"
        record = io.StringIO(newline="")
        writer = csv.writer(record, lineterminator="\n")
        for name, data in self.wheel.items():
            if name != record_path:
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                writer.writerow((name, "sha256=" + digest, str(len(data))))
        writer.writerow((record_path, "", ""))
        self.wheel[record_path] = record.getvalue().encode()
        wheel_path = self.dist / ("aureon_inference-" + BUILDER.VERSION + "-py3-none-any.whl")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(wheel_path, "w") as archive:
                for name, data in list(self.wheel.items()) + list(wheel_extra):
                    # ZipInfo normalizes backslashes on Windows. Preserve raw
                    # adversarial names so this is genuinely an admission test.
                    member = zipfile.ZipInfo(name)
                    member.filename = member.orig_filename = name
                    archive.writestr(member, data)
        tar_path = self.dist / (self.tar_root + ".tar.gz")
        with tarfile.open(tar_path, "w:gz") as archive:
            pairs = [(self.tar_root + "/" + name, data) for name, data in self.sdist.items()]
            for name, data in pairs + list(tar_extra):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            for name in tar_links:
                member = tarfile.TarInfo(self.tar_root + "/" + name)
                member.type = tarfile.SYMTYPE
                member.linkname = "../outside.txt"
                archive.addfile(member)

    def audit(self):
        return BUILDER.audit_distributions(self.source, self.dist, self.inventory)

    def test_exact_synthetic_archives_pass_and_return_artifact_hashes(self):
        self.archives()
        report = self.audit()
        self.assertEqual(len(report), 2)
        for entry in report:
            data = (self.dist / entry["file"]).read_bytes()
            self.assertEqual(entry["sha256"], BUILDER.sha(data))
            self.assertEqual(entry["bytes"], len(data))

    def test_missing_runtime_and_changed_runtime_rejected(self):
        saved = dict(self.wheel)
        for replacement in (None, b"VALUE = 999\n"):
            with self.subTest(replacement=replacement):
                self.wheel = dict(saved)
                if replacement is None:
                    del self.wheel["aureon/example.py"]
                else:
                    self.wheel["aureon/example.py"] = replacement
                self.archives()
                with self.assertRaises(ValueError):
                    self.audit()

    def test_unallowlisted_runtime_rejected(self):
        self.wheel["aureon/unreviewed.py"] = b"VALUE = 3\n"
        self.archives()
        with self.assertRaises(ValueError):
            self.audit()

    def test_wheel_traversal_disguised_as_metadata_rejected(self):
        for name in ("../evil.dist-info/extra.txt", self.info + "/../../extra.txt",
                     "/" + self.info + "/extra.txt", self.info + "/bad\\extra.txt",
                     self.info + "/extra.txt\x00hidden"):
            with self.subTest(name=name):
                self.archives(wheel_extra=[(name, b"fixture\n")])
                with self.assertRaises(ValueError):
                    self.audit()

    def test_duplicate_zip_member_rejected_even_when_bytes_match(self):
        self.archives(wheel_extra=[("aureon/example.py", self.wheel["aureon/example.py"])])
        with self.assertRaises(ValueError):
            self.audit()

    def test_sdist_traversal_disguised_as_egg_info_rejected(self):
        self.archives(tar_extra=[(self.tar_root + "/aureon_inference.egg-info/../../extra.txt", b"fixture\n")])
        with self.assertRaises(ValueError):
            self.audit()

    def test_sdist_multiple_top_level_roots_rejected(self):
        self.archives(tar_extra=[("other-root/aureon/example.py", self.sdist["aureon/example.py"])])
        with self.assertRaises(ValueError):
            self.audit()

    def test_duplicate_tar_member_rejected_even_when_bytes_match(self):
        self.archives(tar_extra=[(self.tar_root + "/aureon/example.py", self.sdist["aureon/example.py"])])
        with self.assertRaises(ValueError):
            self.audit()

    def test_sdist_links_rejected(self):
        self.archives(tar_links=["linked.txt"])
        with self.assertRaises(ValueError):
            self.audit()

    def test_wheel_legal_bytes_are_bound_to_staged_source(self):
        for name in LEGAL | {"third-party-licenses/upstream-LICENSE.txt"}:
            path = self.info + "/licenses/packaging/review/" + name
            original = self.wheel[path]
            with self.subTest(name=name):
                self.wheel[path] = b"Changed legal material\n"
                self.archives()
                with self.assertRaises(ValueError):
                    self.audit()
                self.wheel[path] = original

    def test_generated_source_inventory_and_manifest_must_exist_with_exact_bytes(self):
        for name in ("SOURCE-INVENTORY.json", "MANIFEST.in"):
            original = self.sdist[name]
            for replacement in (None, b"Changed generated evidence\n"):
                with self.subTest(name=name, replacement=replacement):
                    if replacement is None:
                        self.sdist.pop(name)
                    else:
                        self.sdist[name] = replacement
                    self.archives()
                    with self.assertRaises(ValueError):
                        self.audit()
                    self.sdist[name] = original

    def test_credential_like_or_personal_content_in_metadata_rejected(self):
        for forbidden in ("hf_" + "C" * 32, "C:" + "/Users/fixture/private"):
            with self.subTest(kind=forbidden[:8]):
                self.wheel[self.info + "/METADATA"] = self.metadata + forbidden.encode()
                self.archives()
                with self.assertRaises(ValueError):
                    self.audit()


if __name__ == "__main__":
    unittest.main(verbosity=2)
