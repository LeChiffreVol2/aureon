"""Explicit alternative distribution metadata; default native pins remain intact."""
from __future__ import annotations
import json
import re
import tomllib


def apply_core_profile(text: str) -> str:
    data=tomllib.loads(text)
    if data.get('tool',{}).get('aureon',{}).get('distribution-profile')=='core':
        if data['project'].get('dependencies') != []:
            raise ValueError('Core marker conflicts with mandatory dependencies')
        return text
    dependencies=data['project']['dependencies']
    if not dependencies or any('==' not in value for value in dependencies):
        raise ValueError('Core profile requires an explicit exact native dependency source')
    extras={
        'generation':dependencies,
        'borrowed-torch210':['numpy==2.2.6','torch==2.10.0'],
        'borrowed-torch211':['numpy==2.2.6','torch==2.11.0'],
        'geo':['numpy==2.2.6','rasterio==1.5.2'],
        'copgen':dependencies+data['project']['optional-dependencies']['copgen'],
        'scene-legacy':dependencies+data['project']['optional-dependencies']['scene-legacy'],
    }
    result,count=re.subn(r'^dependencies = \[[^\n]*\]$', 'dependencies = []',text,flags=re.M)
    if count != 1:raise ValueError('Expected one explicit mandatory dependency declaration')
    block='[project.optional-dependencies]\n'+'\n'.join(k+' = '+json.dumps(v) for k,v in extras.items())+'\n\n'
    result,count=re.subn(r'\[project.optional-dependencies\]\n.*?(?=\n\[)',block,result,flags=re.S)
    if count != 1:raise ValueError('Expected one optional dependency table')
    result += '\n[tool.aureon]\ndistribution-profile = "core"\n'
    result=result.replace('description = "Inference for Generative Foundation Models for Earth Observation"',
        'description = "Core interfaces for Generative Foundation Models for Earth Observation (explicit backend extras)"')
    checked=tomllib.loads(result)
    assert checked['project']['dependencies']==[]
    assert checked['project']['optional-dependencies']==extras
    assert checked['project']['name']==data['project']['name']
    assert checked['project']['version']==data['project']['version']
    return result


def select_build_profile(settings: dict[str, object] | None) -> str | None:
    if not settings:return None
    if set(settings) != {'aureon-profile'} or settings['aureon-profile'] not in ('core','generation'):
        raise ValueError('Only explicit aureon-profile=core or generation is supported')
    return str(settings['aureon-profile'])
