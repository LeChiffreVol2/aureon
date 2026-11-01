"""CPU EO/STAC contracts with NumPy/Rasterio; no neural imports, downloads or GPU."""
from __future__ import annotations
import argparse
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    runtime = (args.runtime or source).resolve(strict=True)
    scratch, output = args.scratch.absolute(), args.output.absolute()
    if scratch.exists() or output.exists(): parser.error('Scratch/output must be fresh')
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'rasterio')}
    scratch.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    tempfile.tempdir = str(scratch)
    os.environ.update(PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                      TEMP=str(scratch), TMP=str(scratch), TMPDIR=str(scratch))
    sys.path[:0] = [str(runtime), str(source / 'tests')]
    forbidden = []
    class NoNeural(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'torch','torchvision','terratorch','diffusers','transformers'}:
                forbidden.append(fullname)
                raise AssertionError('Neural import blocked: ' + fullname)
    sys.meta_path.insert(0, NoNeural())
    def no_network(*args, **kwargs):
        forbidden.append('network')
        raise AssertionError('Network is forbidden in EO CPU tests')
    with patch.object(socket.socket, 'connect', no_network), patch.object(socket, 'create_connection', no_network):
        suite = unittest.defaultTestLoader.loadTestsFromNames(('test_eo', 'test_stac'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'aureon' or name.startswith('aureon.'):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(runtime): raise AssertionError('Import outside declared runtime: ' + name)
            paths[name] = str(path.relative_to(runtime))
    report = dict(passed=result.wasSuccessful() and not result.skipped and not forbidden,
                  tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  forbidden_attempts=forbidden, runtime=str(runtime), source=str(source), cwd=str(scratch),
                  imports=paths, dependencies=versions, model_calls=0, GPU_calls=0,
                  scope='CPU contracts and real SDK facade with fake model session; no model parity or performance claim')
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in {'imports'}}))
    return 0 if report['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
