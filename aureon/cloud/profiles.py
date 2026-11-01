"""Explicit local qualification selection; no import path supplied by API users."""
from __future__ import annotations
from .contracts import GuardedModule, Qualification


def engine_for(profile: str | None) -> GuardedModule:
    if profile == 'fp32-guarded-v2':
        from . import engine
    elif profile == 'fp32-stream-guarded-v3':
        from . import stream_guarded
        return stream_guarded
    else:
        raise ValueError('Unknown guarded profile')
    return engine


def verify_qualification(record: Qualification) -> Qualification:
    return engine_for(record.get('profile_id')).verify_qualification(record)
