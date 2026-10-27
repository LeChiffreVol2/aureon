"""Explicit local qualification selection; no import path supplied by API users."""


def engine_for(profile):
    if profile == 'fp32-guarded-v2':
        from . import engine
    elif profile == 'fp32-stream-guarded-v3':
        from . import stream_guarded as engine
    else:
        raise ValueError('Unknown guarded profile')
    return engine


def verify_qualification(record):
    return engine_for(record.get('profile_id')).verify_qualification(record)
