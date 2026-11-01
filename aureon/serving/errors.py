"""Deterministic error redaction, separate from ownership/filesystem operations."""
from ..cloud.contracts import Error
from .contracts import SafeError

# Only fixed internal codes may replace the generic execution code. Exception
# messages, details and arbitrary .code attributes may contain customer secrets.
_DIAGNOSTIC_CODES = frozenset({
    'device_capacity_unavailable', 'shared_residency_capacity',
    'resident_vram_budget_exceeded', 'peak_vram_budget_exceeded',
    'resident_host_budget_exceeded', 'worker_scratch_capacity',
    'physical_disk_floor', 'result_storage_capacity', 'result_too_large',
    'stale_attempt', 'stale_placement', 'stale_replica_assignment',
    'stale_replica_generation', 'stale_worker_generation',
    'worker_quarantined_or_closed', 'replica_lease_lost',
    'cleanup_failed', 'cleanup_not_verified', 'backend_closed',
    'input_changed', 'config_changed',
})


def safe_error(exc: BaseException, code: str = 'worker_failed') -> SafeError:
    """Keep explicit caller context; preserve only allowlisted execution reasons."""
    if code == 'worker_failed' and isinstance(exc, Error):
        reason = exc.code
        if isinstance(reason, str) and reason in _DIAGNOSTIC_CODES:
            code = reason
    name = type(exc).__name__
    return {'code': code,
            'type': name if name.isascii() and name.isidentifier() and len(name) < 80 else 'Exception'}
