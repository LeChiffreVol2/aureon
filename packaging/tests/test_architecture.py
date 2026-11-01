"""Packaging, reverse-dependency and import-identity regressions; no models."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("engine_architecture", ROOT / "packaging/check_architecture.py")
architecture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(architecture)


class ArchitectureTests(unittest.TestCase):
    def test_checkout_has_closed_packages_and_extracted_boundaries(self):
        self.assertTrue(architecture.inspect(ROOT)["passed"])

    def test_planning_cannot_reach_frameworks_or_orchestration(self):
        for target in ("torch", "numpy", "aureon.api.models", "aureon.serving", "cloud.runtime"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                architecture.validate_edge("aureon.planning.routing", target)

    def test_bridges_cannot_depend_on_api_or_service_owners(self):
        for target in ("aureon.api", "aureon.cloud", "aureon.serving", "cloud.stage", "tests.helpers"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                architecture.validate_edge("aureon.integrations.model_bridge", target)
        architecture.validate_edge("aureon.integrations.model_bridge", "aureon.terramind_native.rng")

    def test_omitted_package_and_stub_are_rejected(self):
        names = json.loads((ROOT / "packaging/runtime-files.json").read_text())["files"]
        for missing in ("package", "stub"):
            project = tomllib.loads((ROOT / "pyproject.toml").read_text())
            if missing == "package":
                project["tool"]["setuptools"]["packages"].remove("aureon.integrations")
            else:
                project["tool"]["setuptools"]["package-data"].pop("aureon.planning")
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                architecture.validate_packages(names, project)

    def test_import_orders_preserve_modules_patch_targets_and_old_pickles(self):
        # New process catches cycles and verifies the default import remains light.
        program = '''
import importlib, importlib.abc, pickle, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','rasterio','terratorch','diffusers','transformers','cloud'}:
            raise AssertionError('Import side effect: '+fullname)
sys.meta_path.insert(0, BlockRuntime())
groups = {'planning': ('routing','resource_policy','pool_placement'),
          'integrations': ('model_bridge','borrowed_model','preprocessing','terratorch_task','upstream_task')}
for group, members in groups.items():
    for name in members:
        paths = ['aureon.api.'+name, 'aureon.'+group+'.'+name]
        if sys.argv[2] == 'new-first': paths.reverse()
        first, second = map(importlib.import_module, paths)
        assert first is second, paths
from aureon.api import ModelBridge, TensorLayout
from aureon.integrations.model_bridge import ModelBridge as CanonicalBridge
from aureon.planning.resource_policy import TensorLayout as CanonicalLayout
assert ModelBridge is CanonicalBridge and TensorLayout is CanonicalLayout
import aureon.integrations.terratorch_task as task
with patch('aureon.api.terratorch_task.build_terratorch_task') as factory:
    assert task.build_terratorch_task is factory
for module, name, args in [('resource_policy','TensorLayout',((2,3),)),
                            ('pool_placement','WorkerCapabilities',())]:
    cls = getattr(importlib.import_module('aureon.planning.'+module), name)
    value = cls(*args); canonical = cls.__module__
    try:
        cls.__module__ = 'aureon.api.'+module
        legacy = pickle.dumps(value, protocol=4)
    finally: cls.__module__ = canonical
    restored = pickle.loads(legacy)
    assert type(restored) is cls and restored == value
    assert pickle.loads(pickle.dumps(value)) == value
print('import order, old pickle, identity and patch target passed')
'''
        for order in ("old-first", "new-first"):
            result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(ROOT), order],
                                    capture_output=True, text=True, timeout=60)
            with self.subTest(order=order):
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
