"""Deterministic finite capacity policy; no control-plane, filesystem or DB imports."""
from copy import deepcopy
from typing import Final, cast

from ..cloud.contracts import CONTRACT, require
from .contracts import CapacityPolicy, HostIdentity, PlacementAuthority, ReplicaLimits

SCHEMA: Final = 'aureon.async_replicas.v3'
FIELDS: Final = frozenset({
    'max_workers', 'max_running', 'max_active', 'max_queued', 'max_rows',
    'max_tenants', 'max_worker_generations', 'max_tenant_active',
    'max_tenant_running', 'max_tenant_queued',
    'max_tenant_storage_reservation_bytes', 'max_global_storage_reservation_bytes',
})


def limits(value: object, worker_count: int) -> ReplicaLimits:
    """Validate external configuration and return an independent canonical record."""
    require(type(worker_count) is int and 0 < worker_count <= 2**31-1,
            'worker_running_capacity')
    require(isinstance(value, dict) and not set(value)-FIELDS, 'invalid_replica_limits')
    supplied = cast(dict[str, object], value)
    for key, number in supplied.items():
        maximum = 2**50 if key.endswith('_bytes') else 2**31-1
        require(type(number) is int and 0 < number <= maximum, 'finite_positive_replica_limit')
    # All supplied members have passed runtime type/range validation. Defaults
    # use the same canonical contract as the transactional control plane.
    result: dict[str, int] = {
        'max_workers': 8, 'max_running': min(3, worker_count), 'max_active': 3,
        'max_queued': 3, 'max_rows': CONTRACT['max_rows'], 'max_tenants': 15,
        'max_tenant_storage_reservation_bytes': CONTRACT['max_tenant_storage_reservation_bytes'],
        'max_global_storage_reservation_bytes': CONTRACT['max_global_storage_reservation_bytes'],
    }
    result.update(cast(dict[str, int], supplied))
    result.setdefault('max_worker_generations', result['max_rows'])
    for key, parent in [('max_tenant_active', 'max_active'),
                        ('max_tenant_running', 'max_running'), ('max_tenant_queued', 'max_queued')]:
        result.setdefault(key, result[parent] if key == 'max_tenant_active'
                          else min(result[parent], result['max_tenant_active']))
    for key, number in result.items():
        maximum = 2**50 if key.endswith('_bytes') else 2**31-1
        require(type(number) is int and 0 < number <= maximum, 'finite_positive_replica_limit')
    require(worker_count <= result['max_workers'] and result['max_running'] <= worker_count,
            'worker_running_capacity')
    require(result['max_worker_generations'] >= worker_count, 'worker_generation_capacity')
    require(result['max_running'] <= result['max_active'] <= result['max_rows']
            and result['max_queued'] <= result['max_active'], 'inconsistent_request_capacity')
    require(result['max_tenant_running'] <= result['max_tenant_active']
            and result['max_tenant_queued'] <= result['max_tenant_active'], 'inconsistent_tenant_capacity')
    require(all(result[key] <= result[parent] for key, parent in [
        ('max_tenant_active', 'max_active'), ('max_tenant_running', 'max_running'),
        ('max_tenant_queued', 'max_queued'),
        ('max_tenant_storage_reservation_bytes', 'max_global_storage_reservation_bytes')]),
        'tenant_limit_exceeds_service')
    # Preserve the old canonical key order without a redundant JSON round-trip.
    return cast(ReplicaLimits, dict(sorted(result.items())))


def policy(selected: ReplicaLimits) -> CapacityPolicy:
    return {
        'limits': deepcopy(selected),
        'hard_outstanding_bound': 'max_active includes all queued/running jobs and recovered attempts',
        'queued_admission': 'max_queued limits new submissions; recovery retains existing active credits and may requeue up to max_active',
        'tenant_queued_admission': 'same recovery policy, bounded by max_tenant_active',
        'running': 'global and per-tenant limits checked in the same transaction as resource/attempt assignment',
        'fairness': 'durable round-robin among eligible tenants; FIFO within a tenant; no latency/SLA guarantee',
        'worker_queue': 0, 'requests_per_worker': 1, 'HTTP_lanes_per_worker': 1,
        'HTTP_parallel_enabled': False,
        'library_lanes': 'separate explicitly qualified library execution option',
    }


def authority(host: HostIdentity, identifier: str) -> PlacementAuthority:
    return {
        'schema': 'aureon.placement_authority.v1', 'id': identifier, 'kind': 'single_host_sqlite',
        'host': deepcopy(host), 'transaction': 'BEGIN IMMEDIATE', 'multi_node_enabled': False,
        'external_authority_required': [
            'atomic shared resource reservations and attempt assignment',
            'unique tenant idempotency keys and exactly-once usage settlement',
            'worker generation and physical-device fencing',
            'attested ownership release before retry or credit reuse',
            'durable artifact visibility and retention accounting',
        ],
        'external_adapter_installed': False,
    }
