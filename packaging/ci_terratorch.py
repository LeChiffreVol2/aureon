"""Actual random-weight upstream CPU contracts; no asset/network/GPU access."""
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[1];runtime=(args.runtime or source).resolve(strict=True)
    scratch=args.scratch.absolute();output=args.output.absolute()
    if scratch.exists() or output.exists():parser.error('Scratch and output must be fresh')
    scratch.mkdir(parents=True);output.parent.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='',LIGHTLY_DID_VERSION_CHECK='True',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NO_ALBUMENTATIONS_UPDATE='1',TEMP=str(scratch),TMP=str(scratch),TMPDIR=str(scratch),PYTHONDONTWRITEBYTECODE='1')
    tempfile.tempdir=str(scratch)
    os.chdir(scratch)
    sys.path[:0]=[str(runtime),str(source/'tests')]
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    attempts=[]
    def forbidden(*args,**kwargs):
        attempts.append('network_or_cuda_init');raise RuntimeError('Network/CUDA initialization forbidden')
    with patch.object(socket.socket,'connect',forbidden),patch.object(torch.cuda,'_lazy_init',forbidden):
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(['test_terratorch_task','test_model_bridge','test_upstream_task','test_borrowed_model','test_temporal_bridge','test_adapter_registry']))
    imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
             if name=='aureon' or name.startswith('aureon.')}
    assert all(Path(path).is_relative_to(runtime) for path in imports.values())
    receipt={'passed':result.wasSuccessful() and not result.skipped and not attempts and not torch.cuda.is_initialized(),
        'cwd':str(Path.cwd()),'source':str(source),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
        'forbidden_attempts':attempts,'cuda_initialized':torch.cuda.is_initialized(),'torch':torch.__version__,
        'runtime_imports':imports,'scope':'Actual TerraTorch random-weight CPU forward; no pretrained/performance qualification'}
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    return 0 if receipt['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
