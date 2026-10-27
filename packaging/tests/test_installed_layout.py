"""Run against a NONEDITABLE installed wheel from outside its source checkout.

Example (the parent coordinator owns build/install authorization):
  /tmp/aureon-review/bin/python -I -B /checkout/packaging/tests/test_installed_layout.py \
    --checkout-root /checkout --expected-install-root /tmp/aureon-review \
    --scratch-root /tmp --report /checkout/packaging/review/installed-tests.json

No model execution, network requests, installation or dependency changes occur.
The COG test uses synthetic pixels and makes no model parity/quality/speed claim.
"""
import argparse
import base64
import hashlib
import importlib
import importlib.abc
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest


CONFIG = None
OBSERVED = {}
NEURAL = {"torch", "terratorch", "vllm"}


class BlockNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in NEURAL:
            raise AssertionError("Packaging CPU check attempted neural import: " + fullname)
        return None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InstalledLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = importlib.import_module("aureon")
        cls.package_root = Path(cls.package.__file__).resolve().parent
        if cls.package_root.is_relative_to(CONFIG.checkout_root):
            raise AssertionError("Checkout import cannot prove installed-wheel behavior")
        if not cls.package_root.is_relative_to(CONFIG.expected_install_root):
            raise AssertionError("Imported AUREON lies outside the declared installation")
        cls.distribution = metadata.distribution("aureon-inference")
        cls.before = {str(path.relative_to(cls.package_root)): sha(path)
                      for path in cls.package_root.rglob("*") if path.is_file()
                      and "__pycache__" not in path.parts}
        OBSERVED.update(package_root=str(cls.package_root), version=cls.distribution.version,
                        distribution_root=str(cls.distribution.locate_file("")),
                        package_files_before=len(cls.before))

    @classmethod
    def tearDownClass(cls):
        after = {str(path.relative_to(cls.package_root)): sha(path)
                 for path in cls.package_root.rglob("*") if path.is_file()
                 and "__pycache__" not in path.parts}
        if after != cls.before:
            raise AssertionError("Installed package bytes changed during read-only checks")
        if NEURAL.intersection(sys.modules):
            raise AssertionError("A neural runtime was imported")
        OBSERVED["installed_package_bytes_unchanged"] = True

    def test_distribution_is_noneditable_and_contains_required_sources_and_assets(self):
        direct = self.distribution.read_text("direct_url.json")
        if direct:
            record = json.loads(direct)
            self.assertIsNot(record.get("dir_info", {}).get("editable"), True)
            OBSERVED["installation_direct_url"] = record
        members = {str(path).replace("\\", "/") for path in self.distribution.files or ()}
        required = {"aureon/__init__.py", "aureon/recipe.py", "aureon/reference.py", "aureon/client.py",
            "aureon/scene_api/durable.py", "aureon/scene_api/stream_worker.py",
            "aureon/scene_pipeline/streaming.py", "aureon/scene_pipeline/stream_io.py",
            "aureon/scene_pipeline/stream_audit.py", "aureon/experimental_precision.py",
            "aureon/scene_pipeline/job.py", "aureon/scene_pipeline/audit.py",
            "aureon/scene_pipeline/georaster.py", "aureon/scene_api/worker.py",
            "aureon/scene_api/__main__.py", "aureon/scene_api/audit.py",
            "aureon/prepare/__init__.py", "aureon/prepare/__main__.py", "aureon/prepare/local.py",
            "aureon/preflight.py", "aureon/scene_cli.py", "aureon/bundle_storage.py",
            "aureon/static/scene-console.html", "aureon/static/scene-console.js",
            "aureon/static/scene-console.css"}
        self.assertTrue(required <= members, sorted(required - members))
        for name in required:
            path = Path(self.distribution.locate_file(name)).resolve()
            self.assertTrue(path.is_relative_to(CONFIG.expected_install_root), name)
            self.assertTrue(path.is_file(), name)
        self.assertFalse(any(name.startswith(("models/", "inputs/", "runs/", ".archive/", ".venv/"))
                             for name in members))

    def test_installed_legal_inventory_matches_reviewed_source_and_wheel_record(self):
        project = tomllib.loads((CONFIG.checkout_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        expected = {}
        for pattern in project["license-files"]:
            for path in CONFIG.checkout_root.glob(pattern):
                if path.is_file():
                    self.assertTrue(path.resolve().is_relative_to(CONFIG.checkout_root))
                    expected[path.relative_to(CONFIG.checkout_root).as_posix()] = sha(path)
        mandatory = {"packaging/review/" + name for name in
            ("LICENSE", "NOTICE", "THIRD-PARTY-NOTICES.md", "REVIEW-STATUS.md", "SOURCE-NOTICE-OVERLAYS.json")}
        self.assertTrue(mandatory <= set(expected))
        self.assertTrue(any(name.startswith("packaging/review/third-party-licenses/") for name in expected))
        installed = {}
        for member in self.distribution.files or ():
            name = str(member).replace("\\", "/")
            if ".dist-info/licenses/" not in name:
                continue
            relative = name.split(".dist-info/licenses/", 1)[1]
            self.assertNotIn(relative, installed)
            path = Path(self.distribution.locate_file(member)).resolve()
            self.assertTrue(path.is_relative_to(CONFIG.expected_install_root))
            digest = sha(path)
            self.assertIsNotNone(member.hash, "Installed license lacks a RECORD digest")
            self.assertEqual(member.hash.mode, "sha256")
            record_digest = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode()
            self.assertEqual(member.hash.value, record_digest, name)
            installed[relative] = digest
        self.assertEqual(installed, expected)
        OBSERVED["installed_legal_material"] = {"file_count": len(installed),
            "matches_reviewed_checkout": True, "record_sha256_verified": True,
            "sha256": installed}

    def test_imports_and_explicit_execution_gates_need_no_neural_runtime(self):
        from aureon.client import SceneClient
        SceneClient("https://example.invalid", token="x" * 40)
        from aureon.scene_pipeline.job import run_job
        from aureon.scene_api.worker import ResidentSceneWorker
        output = CONFIG.work / "must-not-be-created"
        with self.assertRaises(ValueError):
            run_job(CONFIG.work / "missing-manifest.json", CONFIG.work / "external-bundle", output)
        with self.assertRaises(ValueError):
            ResidentSceneWorker(CONFIG.work / "external-bundle", {"scene": CONFIG.work / "missing.json"}, output)
        self.assertFalse(output.exists())
        self.assertFalse(NEURAL.intersection(sys.modules))

    def test_source_snapshot_and_seals_resolve_installed_files_from_arbitrary_cwd(self):
        from aureon.scene_pipeline import job
        from aureon.scene_api.worker import source_hashes
        from aureon.scene_api.identity import FileSeal
        self.assertEqual((job.ROOT / "aureon").resolve(), self.package_root)
        hashes = source_hashes()
        self.assertIn("aureon/recipe.py", hashes)
        self.assertIn("aureon/scene_api/worker.py", hashes)
        self.assertIn("aureon/bundle_storage.py", hashes)
        self.assertIn("aureon/prepare/local.py", hashes)
        self.assertIn("aureon/scene_cli.py", hashes)
        for name, digest in hashes.items():
            source = (job.ROOT / name).resolve()
            self.assertTrue(source.is_relative_to(self.package_root), name)
            self.assertEqual(sha(source), digest)
        snapshot = CONFIG.work / "snapshot"
        snapshot.mkdir()
        job._snapshot(snapshot, hashes)
        for name, digest in hashes.items():
            self.assertEqual(sha(snapshot / "source-snapshot" / name), digest)
        self.assertEqual(FileSeal([job.ROOT / name for name in hashes]).verify(), len(hashes))
        OBSERVED["installed_source_members_snapshotted"] = len(hashes)

    def test_scene_console_resources_are_packaged_and_snapshot_without_model_start(self):
        from aureon.scene_api.console import load_console_assets
        assets = load_console_assets(self.package_root / "static")
        self.assertEqual(set(assets.manifest()["files"]),
                         {"scene-console.html", "scene-console.js", "scene-console.css"})
        self.assertEqual(assets.get("/console"), assets.get("/console/"))
        self.assertIsNone(assets.get("/console/../../token.txt"))
        OBSERVED["console_assets"] = assets.manifest()

    def test_installed_module_cli_help_is_available_without_neural_imports(self):
        code = '''import importlib.abc, runpy, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.',1)[0] in {'torch','terratorch','vllm'}:
            raise AssertionError('Neural import during help: '+fullname)
sys.meta_path.insert(0, Reject())
module=sys.argv[1]
sys.argv=[module,'--help']
try:
    runpy.run_module(module, run_name='__main__')
except SystemExit as error:
    if error.code not in (0,None): raise
assert not {'torch','terratorch','vllm'}.intersection(sys.modules)
'''
        modules = ("aureon", "aureon.client", "aureon.scene_pipeline", "aureon.scene_api",
                   "aureon.prepare", "aureon.preflight", "aureon.scene_cli",
                   "aureon.scene_pipeline.audit", "aureon.scene_api.audit")
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, "-I", "-B", "-c", code, module],
                    cwd=CONFIG.work, env=environment, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())
        OBSERVED["module_help_checked"] = list(modules)

    def test_external_prepared_manifest_to_cog_roundtrip_is_cpu_only(self):
        import numpy as np
        import rasterio
        from aureon.recipe import INPUT_UNITS, S2_BANDS
        from aureon.scene_pipeline.job import load_manifest, _read_source
        from aureon.scene_pipeline.georaster import write_segmentation_geotiffs
        folder = CONFIG.work / "external-scene"
        folder.mkdir()
        transform = rasterio.Affine(10, 0, 632260, 0, -10, 1583830)
        values = np.full((12, 11, 13), 2000, dtype=np.float32)
        valid = np.ones((11, 13), dtype=np.uint8)
        valid[2, 3] = 0
        values[:, 0, 0] = 0  # A genuine valid zero is not nodata.
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(folder / "scene.tif", "w", driver="GTiff", count=12, height=11,
                               width=13, dtype="float32", crs="EPSG:32647", transform=transform) as ds:
                ds.write(values)
                ds.write_mask(valid * 255)
                for index, band in enumerate(S2_BANDS, 1):
                    ds.set_band_description(index, band)
            with rasterio.open(folder / "valid_mask.tif", "w", driver="GTiff", count=1, height=11,
                               width=13, dtype="uint8", crs="EPSG:32647", transform=transform) as ds:
                ds.write(valid, 1)
        def record(name):
            return {"path": name, "sha256": sha(folder / name), "bytes": (folder / name).stat().st_size}
        manifest = {"schema": "aureon.sentinel2_scene.v1", "scene_id": "synthetic-packaging-only",
            "raster": {**record("scene.tif"), "bands": list(S2_BANDS), "units": INPUT_UNITS,
                       "crs": "EPSG:32647", "shape": [12, 11, 13], "dtype": "float32",
                       "transform": list(transform)[:6]},
            "valid_mask": {**record("valid_mask.tif"), "valid_pixels": int(valid.sum())}}
        manifest_path = folder / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        admitted, files, unused = load_manifest(manifest_path)
        source = _read_source(admitted, files)
        self.assertTrue(source.valid_mask[0, 0])
        self.assertTrue(np.array_equal(source.image[:, 0, 0], np.zeros(12)))
        classes = np.full((11, 13), 3, dtype=np.uint8)
        classes[valid == 0] = 255
        result = SimpleNamespace(class_map=classes, valid_mask=valid.astype(bool))
        output = CONFIG.work / "synthetic-cog"
        receipt = write_segmentation_geotiffs(output, result, source,
            provenance={"synthetic_packaging_check": True, "model_executed": False}, cog=True)
        self.assertEqual(receipt["reopen_validation"], "passed")
        for name, expected, nodata in (("class_map.tif", classes, 255), ("valid_mask.tif", valid, 0)):
            with rasterio.open(output / name) as ds:
                self.assertEqual(ds.crs.to_epsg(), 32647)
                self.assertEqual(ds.transform, transform)
                self.assertEqual(ds.nodata, nodata)
                self.assertEqual(ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT"), "COG")
                self.assertTrue(np.array_equal(ds.read(1), expected))
        self.assertEqual(sha(folder / "scene.tif"), manifest["raster"]["sha256"])
        with self.assertRaises(FileExistsError):
            write_segmentation_geotiffs(output, result, source, provenance={"synthetic": True}, cog=True)
        OBSERVED["synthetic_georaster_roundtrip"] = {"passed": True, "crs": "EPSG:32647",
            "shape": [11, 13], "valid_pixels": int(valid.sum()), "model_executed": False,
            "numerical_model_parity_tested": False, "speed_or_accuracy_claim": False}

    def test_optional_explicit_external_bundle_identity_without_model_loading(self):
        if CONFIG.bundle is None:
            self.skipTest("No explicit external bundle provided; no weights inferred or acquired")
        from aureon.runtime import verify_bundle
        self.assertFalse(CONFIG.bundle.is_relative_to(self.package_root))
        checked = verify_bundle(CONFIG.bundle)
        self.assertEqual(checked["model_id"], "kao-terramind-b")
        OBSERVED["external_bundle"] = {"model_id": checked["model_id"],
            "receipt_sha256": checked["receipt_sha256"], "files_sha256": checked["files_sha256"],
            "cpu_identity_only": True, "model_loaded": False}


def main(argv=None):
    global CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--expected-install-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(argv)
    for key in ("checkout_root", "expected_install_root", "scratch_root"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    if not sys.flags.isolated:
        parser.error("Use python -I so cwd/PYTHONPATH cannot supply checkout imports")
    if args.expected_install_root.is_relative_to(args.checkout_root) or args.scratch_root.is_relative_to(args.checkout_root):
        parser.error("Installation and scratch directory must both be outside the source checkout")
    if args.bundle:
        args.bundle = args.bundle.resolve(strict=True)
    args.report = args.report.absolute()
    if args.report.exists() or not args.report.parent.is_dir():
        parser.error("Report must be fresh with an existing parent")
    forbidden = NEURAL.intersection(sys.modules)
    if forbidden:
        parser.error("Neural runtime already imported before packaging tests")
    guard = BlockNeuralImports()
    sys.meta_path.insert(0, guard)
    previous = Path.cwd()
    temporary = tempfile.TemporaryDirectory(prefix="aureon-installed-layout-", dir=args.scratch_root)
    args.work = Path(temporary.name).resolve()
    CONFIG = args
    OBSERVED.update(interpreter=sys.executable, isolated_mode=bool(sys.flags.isolated),
                    arbitrary_cwd=str(args.work), checkout_root=str(args.checkout_root))
    try:
        os.chdir(args.work)
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InstalledLayoutTests))
        report = {"schema": "aureon.installed_layout_cpu_tests.v1", "status": "passed" if result.wasSuccessful() else "failed",
            "tests_run": result.testsRun, "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
            "failures": [{"test": str(test), "detail": detail} for test, detail in result.failures + result.errors],
            "observed": OBSERVED, "test_source_sha256": sha(__file__),
            "scope": {"model_executed": False, "installation_performed_by_test": False,
                "dependency_resolution_proven": False, "neural_import_blocker_active": True,
                "meaning": "Installed layout, import gates, source/resource availability and synthetic geospatial correctness only"}}
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "tests_run": result.testsRun,
                          "report": str(args.report)}), flush=True)
        return 0 if result.wasSuccessful() else 1
    finally:
        os.chdir(previous)
        sys.meta_path.remove(guard)
        target = Path(temporary.name).resolve()
        if target.parent != args.scratch_root or not target.name.startswith("aureon-installed-layout-"):
            raise RuntimeError("Refusing cleanup outside owned scratch directory")
        temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
