"""COP-GEN contract/lifecycle CPU fixtures; no upstream model/assets or GPU calls."""
import argparse
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
                      TEMP=str(scratch), TMP=str(scratch), TMPDIR=str(scratch))
    sys.path[:0] = [str(runtime), str(source/'tests')]
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append('network_or_cuda_init')
        raise AssertionError('Network/CUDA initialization forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(torch.cuda, '_lazy_init', forbidden):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_copgen_native'))
    imports = {n: str(Path(m.__file__).resolve()) for n, m in sys.modules.items()
               if n == 'aureon' or n.startswith('aureon.')}
    assert all(Path(p).is_relative_to(runtime) for p in imports.values())
    receipt = dict(passed=result.wasSuccessful() and not result.skipped and not attempts and not torch.cuda.is_initialized(),
                   tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                   forbidden_attempts=attempts, cuda_initialized=torch.cuda.is_initialized(),
                   torch=torch.__version__, cwd=str(scratch), runtime_imports=imports,
                   scope='CPU fixtures only; no native COP-GEN execution or installed GPU qualification')
    output.write_text(json.dumps(receipt, indent=2)+'\n')
    return 0 if receipt['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
