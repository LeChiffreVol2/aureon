"""Verify the exact local v0.1.0 wheel and optional installed profile; CPU only.

Run from any directory with Python 3.12. No install, model load or network call.
"""
import argparse
import base64
import csv
import hashlib
import importlib.abc
from importlib import metadata
import io
import json
from pathlib import Path
import sys
import zipfile

class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'terratorch', 'vllm', 'diffusers'}:
            raise RuntimeError('Installation check must not load neural runtime: ' + fullname)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument('--manifest', type=Path, help='Release kit ARTIFACTS.json, verified through its external ZIP checksum')
    identity.add_argument('--sha256', help='Expected wheel SHA256 from a separately trusted release record')
    parser.add_argument('--environment', choices=('scene', 'generation', 'scene-legacy'))
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    expected = args.sha256
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        if manifest.get('version') != '0.1.0' or manifest['wheel']['filename'] != wheel.name:
            parser.error('Release manifest version/filename does not match the wheel')
        expected = manifest['wheel']['sha256']
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        parser.error('Expected a lowercase SHA256 from the release record')
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
        parser.error('Wheel SHA256 does not match this delivery revision')
    result = {'status': 'passed', 'wheel_sha256': expected,
              'model_executed': False, 'neural_imports': False}
    with zipfile.ZipFile(wheel) as archive:
        record = next(n for n in archive.namelist() if n.endswith('.dist-info/RECORD'))
        checked = 0
        for name, digest, size in csv.reader(io.StringIO(archive.read(record).decode())):
            if not digest:
                continue
            payload = archive.read(name)
            algorithm, expected = digest.split('=', 1)
            actual = base64.urlsafe_b64encode(hashlib.new(algorithm, payload).digest()).decode().rstrip('=')
            if actual != expected or len(payload) != int(size):
                raise RuntimeError('Wheel RECORD mismatch: ' + name)
            checked += 1
        result['record_payloads_checked'] = checked
        if args.environment:
            if sys.version_info[:2] != (3, 12) or sys.platform != 'linux':
                parser.error('Supported profile requires Linux and Python 3.12')
            sys.meta_path.insert(0, NoNeuralImports())
            import aureon
            from aureon.api import API_VERSION
            from aureon.api.profiles import inspect_profile
            distribution = metadata.distribution('aureon-inference')
            if distribution.version != '0.1.0':
                raise RuntimeError('Installed version differs')
            package = Path(aureon.__file__).resolve().parent
            expected_package = Path(distribution.locate_file('aureon')).resolve()
            if package != expected_package or not package.is_relative_to(Path(sys.prefix).resolve()):
                raise RuntimeError('Expected noneditable installed package inside this environment')
            count = 0
            for name in archive.namelist():
                if name.endswith('/') or name == record:
                    continue
                target = Path(distribution.locate_file(name))
                if not target.is_file() or target.read_bytes() != archive.read(name):
                    raise RuntimeError('Installed payload differs from wheel: ' + name)
                count += 1
            checks = inspect_profile(args.environment)
            if checks['status'] != 'passed':
                raise RuntimeError('Dependency profile mismatch: ' + json.dumps(checks))
            result.update(version=distribution.version, api_version=API_VERSION,
                          environment=args.environment, installed_payloads_checked=count,
                          profile=checks, package_root=str(package))
            # Exercise installed console entry points with a neural import blocker.
            from contextlib import redirect_stdout
            commands = [('aureon-engine', ['models']), ('aureon-engine', ['profiles']),
                        ('aureon-engine', ['--help']), ('aureon-service', ['--help']),
                        ('aureon-service-client', ['--help']), ('aureon-prepare', ['--help'])]
            entries = {e.name: e for e in distribution.entry_points if e.group == 'console_scripts'}
            for command, arguments in commands:
                old_argv = sys.argv
                try:
                    sys.argv = [command, *arguments]
                    with redirect_stdout(io.StringIO()):
                        try:
                            code = entries[command].load()()
                        except SystemExit as exited:
                            code = exited.code
                    if code not in (None, 0):
                        raise RuntimeError('CLI check failed: ' + command)
                finally:
                    sys.argv = old_argv
            result['cli_checks'] = len(commands)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
