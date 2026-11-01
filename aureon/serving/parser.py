"""CPU-only prepared-input decoder in a bounded child process, before model admission."""
from __future__ import annotations
import threading
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .workflow_config import WorkflowBinding
from ..cloud.contracts import UploadBody
from .contracts import PathInput
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from ..cloud.control import require, encoded, write_file, file_hash
from ..scene_api.durable import _plain_path
from .timing import elapsed

LIMITS={'wall_seconds':90,'cpu_seconds':60,'address_space_bytes':4*1024**3,'file_bytes':1024**3,'open_files':64}

def run(body: UploadBody, folder: PathInput, token: str, task: str, *, timeout: float | None = None, stop: threading.Event | None = None,
        workflow: WorkflowBinding | None = None) -> Path:
    folder=_plain_path(folder).resolve(strict=True)
    request=folder/('parse-'+token+'.json');receipt=folder/('parse-'+token+'-result.json')
    write_file(request,encoded({'body':body,'folder':str(folder),'token':token,'task':task,'receipt':str(receipt),
                               **({'workflow':workflow} if workflow is not None else {})}))
    environment_names=('PATH','SystemRoot','WINDIR','TEMP','TMP','TMPDIR','LANG','LC_ALL','TZ')
    env={**{k:os.environ[k] for k in environment_names if k in os.environ},'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','CUDA_VISIBLE_DEVICES':'',
      'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','GDAL_NUM_THREADS':'1',
      'GDAL_CACHEMAX':'64','GDAL_PAM_ENABLED':'NO','GDAL_DISABLE_READDIR_ON_OPEN':'EMPTY_DIR',
      'PROJ_NETWORK':'OFF','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    env.pop('PYTHONPATH',None)
    # Explicit package root works in an installed wheel and in source CPU tests.
    code="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('aureon.serving.parser',run_name='__main__')"
    command=[sys.executable,'-I','-B','-c',code,str(Path(__file__).resolve().parents[2]),str(request)]
    child=subprocess.Popen(command,cwd=folder,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=elapsed()+(LIMITS['wall_seconds'] if timeout is None else timeout)
        while child.poll() is None:
            if stop is not None and stop.is_set():
                child.kill();child.wait();raise ValueError('input_parser_stopped')
            remaining=deadline-elapsed()
            if remaining<=0:
                child.kill();child.wait();raise ValueError('input_parser_deadline')
            try:child.wait(timeout=min(.1,remaining))
            except subprocess.TimeoutExpired:pass
        require(child.returncode==0 and receipt.is_file(),'input_parser_rejected')
        value=json.loads(receipt.read_bytes())
        require(value.get('passed') and value.get('model_imported') is False,'input_parser_rejected')
        path=_plain_path(folder/value['manifest'])
        require(path.is_relative_to(folder/('prepared-'+token)) and file_hash(path)==value['manifest_sha256'],'parser_result_binding')
        return path
    finally:
        if child.poll() is None:child.kill();child.wait()

def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('request',type=Path);args=p.parse_args()
    resource_limits=False
    if sys.platform=='linux':
        import resource
        for kind,value in [(resource.RLIMIT_AS,LIMITS['address_space_bytes']),(resource.RLIMIT_CPU,LIMITS['cpu_seconds']),
                           (resource.RLIMIT_FSIZE,LIMITS['file_bytes']),(resource.RLIMIT_NOFILE,LIMITS['open_files']),(resource.RLIMIT_CORE,0)]:
            resource.setrlimit(kind,(value,value))
        resource_limits=True
    request=json.loads(args.request.read_bytes());folder=Path(request['folder']);receipt=Path(request['receipt'])
    result={'schema':'aureon.input_parser.v1','passed':False,'limits':LIMITS,'linux_resource_limits':resource_limits,
      'network_isolation':'No URL inputs; restricted local formats. OS network namespace/seccomp is a deployment boundary, not provided by rlimits.'}
    try:
        from .inputs import prepare
        path=prepare(request['body'],folder,request['token'],request['task'],workflow=request.get('workflow'))
        require('torch' not in sys.modules,'parser_model_import')
        result.update(passed=True,manifest=path.relative_to(folder).as_posix(),manifest_sha256=file_hash(path),model_imported=False)
    except Exception:
        result.update(error='input_validation_failed',model_imported='torch' in sys.modules)
    write_file(receipt,encoded(result))
    return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
