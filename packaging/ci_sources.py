"""Real Rasterio and loopback HTTP source contracts; no external network/models."""
from __future__ import annotations
import argparse
import importlib.abc
import importlib.metadata
import ipaddress
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
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    runtime = (args.runtime or source).resolve(strict=True)
    scratch, output = args.scratch.absolute(), args.output.absolute()
    if scratch.exists() or output.exists(): parser.error('Scratch/output must be fresh')
    scratch.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    os.chdir(scratch)
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                      TMP=str(scratch), TEMP=str(scratch), TMPDIR=str(scratch))
    sys.path[:0] = [str(runtime), str(source / 'tests')]
    forbidden = []
    class NoNeural(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'torch', 'torchvision', 'terratorch', 'diffusers', 'transformers'}:
                forbidden.append(fullname)
                raise AssertionError('Neural import forbidden: ' + fullname)
    sys.meta_path.insert(0, NoNeural())
    original_connect, original_dns = socket.socket.connect, socket.getaddrinfo
    def loopback_connect(sock, address):
        if not isinstance(address, tuple) or not ipaddress.ip_address(address[0]).is_loopback:
            forbidden.append('external_connection')
            raise AssertionError('External connection forbidden')
        return original_connect(sock, address)
    def loopback_dns(host, *args, **kwargs):
        if host not in ('127.0.0.1', 'localhost', '::1'):
            forbidden.append('external_dns')
            raise AssertionError('External DNS forbidden')
        return original_dns(host, *args, **kwargs)
    with patch.object(socket.socket, 'connect', loopback_connect), patch.object(socket, 'getaddrinfo', loopback_dns):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(
            ['test_http_range_source', 'test_stac_raster', 'test_scene_generation']))
    paths = {name: str(Path(mod.__file__).resolve()) for name, mod in sys.modules.items()
             if name == 'aureon' or name.startswith('aureon.')}
    assert all(Path(path).is_relative_to(runtime) for path in paths.values())
    receipt = dict(passed=result.wasSuccessful() and not result.skipped and not forbidden,
                   tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                   skipped=len(result.skipped), forbidden_attempts=forbidden, runtime_imports=paths,
                   cwd=str(scratch), source=str(source),
                   dependencies={n: importlib.metadata.version(n) for n in ('numpy', 'rasterio')},
                   scope='Actual Rasterio and loopback HTTP; fake engine; no model or live-provider qualification')
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'runtime_imports'}))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
