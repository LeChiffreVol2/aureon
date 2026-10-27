"""Explicit process-local THP policy; never writes system-wide memory controls."""
import ctypes
import os
from pathlib import Path
import platform
import sys

PR_SET_THP_DISABLE=41
PR_GET_THP_DISABLE=42


def _prctl(option, value=0):
    libc=ctypes.CDLL(None,use_errno=True)
    function=libc.prctl
    function.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
    function.restype=ctypes.c_int
    result=function(option,value,0,0,0)
    if result<0:
        error=ctypes.get_errno()
        raise OSError(error,os.strerror(error))
    return result


def configure_process_memory(policy):
    if sys.platform!='linux' or policy not in ('inherit','disable'):
        raise ValueError('Linux process THP policy must be inherit or disable')
    before=_prctl(PR_GET_THP_DISABLE)
    if policy=='disable':
        _prctl(PR_SET_THP_DISABLE,1)
    after=_prctl(PR_GET_THP_DISABLE)
    if after!=(1 if policy=='disable' else before):
        raise RuntimeError('Requested process THP policy was not applied')
    root=Path('/sys/kernel/mm/transparent_hugepage')
    controls={}
    for name in ('enabled','defrag','khugepaged/max_ptes_none','khugepaged/scan_sleep_millisecs'):
        try:
            controls[name]=(root/name).read_text().strip()
        except OSError as error:
            controls[name]={'unavailable':type(error).__name__}
    return {'schema':'aureon.process_memory_policy.v1','requested':policy,'pid':os.getpid(),
            'pr_get_thp_disable_before':before,'pr_get_thp_disable_after':after,
            'kernel':platform.release(),'system_controls_read_only':controls,
            'scope':'Only this explicitly started service process and its descendants; no sysfs write, cache clearing, allocator trim or forced GC'}
