"""Verify the selected v0.1.0 Engine source and optional wheel, without imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SELECTED = 'e09e710c9e9ace17b7bf2214d8c9eff2c4ed605a7d45e8c4e649b672f6f88a64'
ORIGIN = '0fa5ef2b9440fcc310a17c76f78df52278cbe2d354015be2046a8e4196715c97'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def member(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or relative.as_posix() != name:
        raise ValueError('Invalid manifest path: ' + name)
    path = root / name
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError('Manifest path escapes the checkout: ' + name)
    return path


def inspect(root: Path, wheel: Path | None = None) -> dict[str, object]:
    root = root.resolve(strict=True)
    manifest = json.loads((root / 'packaging/review/SELECTED-ENGINE.json').read_text(encoding='utf-8'))
    if (manifest['version'] != '0.1.0' or manifest['wheel_sha256'] != SELECTED
            or manifest['performance_origin_wheel_sha256'] != ORIGIN):
        raise ValueError('Selected artifact role drift; review an intentional qualification change')
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    if project['name'] != 'aureon-inference' or project['version'] != '0.1.0':
        raise ValueError('Project name/version drift')
    rows = manifest['runtime']
    names = [row['path'] for row in rows]
    inventory = json.loads((root / 'packaging/runtime-files.json').read_text(encoding='utf-8'))['files']
    actual = {p.relative_to(root).as_posix() for p in (root / 'aureon').rglob('*') if p.is_file()}
    if len(names) != 225 or len(set(names)) != len(names) or set(names) != set(inventory) or set(names) != actual:
        raise ValueError('Selected runtime inventory drift')
    for row in rows:
        path = member(root, row['path'])
        if path.stat().st_size != row['source_bytes'] or digest(path) != row['source_sha256']:
            raise ValueError('Selected source drift: ' + row['path'])
    for name, expected in manifest['build_inputs'].items():
        if digest(member(root, name)) != expected:
            raise ValueError('Selected dependency/build input drift: ' + name)
    wheel_checked = False
    if wheel is not None:
        if digest(wheel) != SELECTED:
            raise ValueError('Wheel is not the locked Best artifact')
        with zipfile.ZipFile(wheel) as archive:
            wheel_names = {n for n in archive.namelist() if n.startswith('aureon/') and not n.endswith('/')}
            if wheel_names != set(names):
                raise ValueError('Wheel runtime inventory differs')
            for row in rows:
                payload = archive.read(row['path'])
                if len(payload) != row['wheel_bytes'] or hashlib.sha256(payload).hexdigest() != row['wheel_sha256']:
                    raise ValueError('Installed payload drift: ' + row['path'])
        wheel_checked = True
    return {'passed': True, 'version': '0.1.0', 'wheel_sha256': SELECTED,
            'performance_origin_wheel_sha256': ORIGIN, 'runtime_files': len(rows),
            'build_inputs': len(manifest['build_inputs']), 'actual_wheel_verified': wheel_checked,
            'model_calls': 0, 'network_used': False,
            'scope': 'Source/build-input identity. Existing numerical acceptance is retained; no new performance or GPU qualification.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--wheel', type=Path, help='Optional existing selected wheel; never downloaded')
    parser.add_argument('--report', type=Path, help='Fresh receipt outside the checkout')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.report and (args.report.exists() or args.report.resolve().is_relative_to(root)):
        parser.error('Report must be fresh and outside the checkout')
    result = inspect(root, args.wheel)
    if args.report:
        with args.report.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
