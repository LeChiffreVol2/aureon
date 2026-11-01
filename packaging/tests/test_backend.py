"""Distribution boundary checks without building or importing a model."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("review_builder",ROOT/"packaging/build_review.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BackendContract(unittest.TestCase):
    def test_root_routes_through_staging_backend(self):
        config = tomllib.loads((ROOT/"pyproject.toml").read_text())
        self.assertEqual(config["build-system"]["build-backend"],"build_backend")
        self.assertNotIn("find",config["tool"]["setuptools"].get("packages",{}))
        self.assertFalse(any("*" in n for n in config["project"]["license-files"]))

    def test_stage_exact_files_and_no_provider_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, inventory = builder.stage(Path(tmp)/"stage")
            selected = {r["path"] for r in inventory["files"]}
            actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
            self.assertEqual(actual,selected|{"SOURCE-INVENTORY.json","MANIFEST.in"})
            self.assertIn("aureon/cloud/stream_guarded.py",selected)
            # The async service reuses local durable control; provider dispatch
            # remains outside this explicitly reviewed package footprint.
            self.assertIn("aureon/cloud/control.py",selected)
            self.assertIn("aureon/serving/__main__.py",selected)
            self.assertNotIn("aureon/cloud/__main__.py",selected)
            self.assertFalse(any(n.startswith("aureon/runpod_worker/") or "/test_" in n for n in selected if n.startswith("aureon/")))
            for record in inventory["files"]:
                self.assertEqual(builder.sha((source/record["path"]).read_bytes()),record["staged_sha256"])
            for path, digest in builder.QUALIFIED_NOTICE_SOURCES.items():
                identity = builder.verify_qualified_source(path, (source/path).read_bytes())
                self.assertEqual(identity["original_sha256"], digest)
                self.assertFalse(identity["new_gpu_qualification"])

    def test_qualification_numerical_bytes_preserved(self):
        # Root qualification files are external evidence; this test also works
        # from sdist by checking the immutable numerical source AST overlays.
        with tempfile.TemporaryDirectory() as tmp:
            source, inventory = builder.stage(Path(tmp)/"stage")
            for record in inventory["files"]:
                if record["path"].endswith(".py") and record["notice_prefix_added"]:
                    self.assertEqual(ast.dump(ast.parse((ROOT/record["path"]).read_bytes())),
                                     ast.dump(ast.parse((source/record["path"]).read_bytes())))

    def test_annotation_transition_rejects_changed_computation(self):
        spec = importlib.util.spec_from_file_location("identity_guard", ROOT / "packaging/source_identity.py")
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        ledger = json.loads((ROOT / "packaging/review/ANNOTATION-TRANSITION.json").read_text())
        name = "aureon/recipe.py"
        current = (ROOT / name).read_bytes()
        altered = current.replace(b"STRIDE = 168", b"STRIDE = 169")
        self.assertNotEqual(current, altered)
        self.assertNotEqual(guard.executable_digest(current), guard.executable_digest(altered))
        # Updating the byte digest alone must not authorize changed arithmetic/configuration.
        ledger["files"][name]["candidate_sha256"] = builder.sha(altered)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "packaging/review"
            folder.mkdir(parents=True)
            (folder / "ANNOTATION-TRANSITION.json").write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, "Unreviewed qualified-source transition"):
                guard.verify_source(root, name, altered, builder.QUALIFIED_NOTICE_SOURCES[name])



if __name__ == "__main__":
    unittest.main()
