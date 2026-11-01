"""Elapsed durations only; persistent expiry/lease timestamps stay in control.clock."""
import time

CLOCK = 'CLOCK_MONOTONIC_RAW' if hasattr(time, 'CLOCK_MONOTONIC_RAW') else 'perf_counter'


def elapsed() -> float:
    if CLOCK == 'CLOCK_MONOTONIC_RAW':
        return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    return time.perf_counter()
