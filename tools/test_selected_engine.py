"""Exercise source-lock rejection using disposable copies, without model imports."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('selected_engine_check', ROOT / 'tools/check_selected_engine.py')
assert spec is not None and spec.loader is not None
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class SelectedSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix='aureon-source-lock-')
        self.root = Path(self.temporary.name)
        assert self.root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.temporary.cleanup)
        self.manifest_path = Path('packaging/review/SELECTED-ENGINE.json')
        self.manifest = json.loads((ROOT / self.manifest_path).read_text(encoding='utf-8'))
        names = {r['path'] for r in self.manifest['runtime']} | set(self.manifest['build_inputs']) | {self.manifest_path.as_posix()}
        for name in names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def test_complete_source_passes(self) -> None:
        self.assertTrue(guard.inspect(self.root)['passed'])

    def test_modified_runtime_rejected(self) -> None:
        path = self.root / 'aureon/api/models.py'
        path.write_bytes(path.read_bytes() + b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'source drift'):
            guard.inspect(self.root)

    def test_missing_runtime_rejected(self) -> None:
        (self.root / 'aureon/api/models.py').unlink()
        with self.assertRaisesRegex(ValueError, 'inventory drift'):
            guard.inspect(self.root)

    def test_extra_runtime_rejected(self) -> None:
        (self.root / 'aureon/unreviewed.py').write_text('')
        with self.assertRaisesRegex(ValueError, 'inventory drift'):
            guard.inspect(self.root)

    def test_dependency_change_rejected(self) -> None:
        path = self.root / 'packaging/lock-generation-linux-py312-cu128.txt'
        path.write_bytes(path.read_bytes() + b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'build input drift'):
            guard.inspect(self.root)

    def test_different_selection_rejected(self) -> None:
        self.manifest['wheel_sha256'] = guard.ORIGIN
        (self.root / self.manifest_path).write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'artifact role drift'):
            guard.inspect(self.root)

    def test_wrong_wheel_rejected(self) -> None:
        wheel = self.root / 'another.whl'
        wheel.write_bytes(b'not the selected wheel')
        with self.assertRaisesRegex(ValueError, 'not the locked Best'):
            guard.inspect(self.root, wheel)


if __name__ == '__main__':
    unittest.main(verbosity=2)
