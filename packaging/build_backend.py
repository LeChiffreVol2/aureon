"""PEP 517 backend: every supported hook builds an exact staged allowlist.

No editable hook: editable discovery cannot satisfy this distribution contract.
Experiment sources remain in the checkout, outside the built artifact.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import build_review


def _hook(name, destination, config_settings=None):
    if config_settings:
        raise ValueError("Build overrides are unsupported; use the reviewed configuration")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aureon-allowlist-") as temp:
        source, inventory = build_review.stage(Path(temp)/"candidate")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SOURCE_DATE_EPOCH=build_review.EPOCH)
        code = ("from setuptools import build_meta; import json,sys; "
                "result=getattr(build_meta,sys.argv[1])(sys.argv[2]); "
                "print('AUREON_BUILD_RESULT='+json.dumps(result))")
        process = subprocess.run([sys.executable,"-B","-c",code,name,str(destination)],
                                 cwd=source,env=env,capture_output=True,text=True,timeout=180)
        if process.returncode:
            raise RuntimeError(process.stdout+process.stderr)
        print(process.stdout,end="")
        result = json.loads(next(line.split("=",1)[1] for line in process.stdout.splitlines()
                                 if line.startswith("AUREON_BUILD_RESULT=")))
        if name in {"build_wheel","build_sdist"}:
            build_review.audit_distributions(source,destination,inventory,selected=[destination/result])
        return result


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory,config_settings=None):
    return _hook("prepare_metadata_for_build_wheel",metadata_directory,config_settings)


def build_wheel(wheel_directory,config_settings=None,metadata_directory=None):
    return _hook("build_wheel",wheel_directory,config_settings)


def build_sdist(sdist_directory,config_settings=None):
    return _hook("build_sdist",sdist_directory,config_settings)
