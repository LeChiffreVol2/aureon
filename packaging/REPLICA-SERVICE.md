# Local replica service and finite capacity

**Best v0.1.0 interfaces.** The replica modules in this checkout
are absent from the unchanged published GitHub wheel. Prior installed GPU
acceptance belongs to separately identified private artifacts; it is not
acceptance of a new public-source build. This guide describes the source
contracts. Physical multi-GPU scaling and sustained load remain unmeasured.

One Linux host, UID and PID namespace own one local SQLite authority. All HTTP
submissions use its existing jobs, idempotency keys, attempts, outbox and usage
tables. Separate processes each own one full physical GPU UUID and one serial
resident generation engine with independent request state. Tensor-batch experiments are retired from the current facade.
Keep the existing exact FP32 profile. No cross-host/network SQLite or model sharding
is supported by this adapter.

## Configurable capacity: new v3 source contract

The integrated capacity extension has prior CPU/process contract coverage.
Its imported source is distinct from the published release; a new build needs
its own artifact binding and installed validation.

Omitting `--limits` retains v2 defaults. Supplying an explicit limits JSON when
creating a **fresh state** creates `aureon.async_replicas.v3`; opening a v2 state
does not migrate or rewrite its contract. Existing v1 backup/supervisor commands
still do not accept replica state. Capacity is fixed in the new state contract;
editing live SQLite rows or contract files is not a reconfiguration interface.

Example limits for a topology with at least4 workers (each distinct physical GPU
UUID must actually exist before real workers start):

```json
{
  "max_workers": 12,
  "max_running": 4,
  "max_active": 12,
  "max_queued": 8,
  "max_rows": 10000,
  "max_worker_generations": 10000,
  "max_tenants": 32,
  "max_tenant_active": 4,
  "max_tenant_running": 2,
  "max_tenant_queued": 4,
  "max_tenant_storage_reservation_bytes": 10737418240,
  "max_global_storage_reservation_bytes": 25769803776
}
```

```sh
aureon-service replicas init --state /data/new-capacity-state \
  --resources /data/resources.json --limits /data/limits.json \
  --tenant customer-a --tenant customer-b --config /data/generation.json \
  --credentials /data/new-capacity-credentials.json
```

Capacity counts must be integers1..2^31-1 and storage limits1..2^50 bytes;
booleans, nulls and nonfinite numbers are rejected. Omitted fields receive safe legacy defaults. Defaults
are8 configured workers,3 active/new queued jobs, at most3 running jobs (also
bounded by configured worker count),15 customer tenants,10000 retained rows and
worker generations,2GiB input reservation per tenant and8GiB globally. Tenant
concurrency defaults follow their service limits. Larger requested limits do not
allocate workers, GPUs or memory and do not establish measured capacity.

`max_active` is the **hard bound on accepted unfinished jobs**, including recovery.
`max_queued` is a new-submission threshold. A recovered existing job retains its
active credit: it can return to the queue even when new admission is full, so
recovery may raise queued depth up to `max_active`. The same rule applies to the
tenant queued threshold and hard `max_tenant_active` bound. Replays of existing
idempotency keys remain available when admission is full. Cancellation releases
the existing logical reservation; it never interrupts a running model call.

Global and tenant running quotas, host/output budgets, attempt assignment and
fencing are checked under the same `BEGIN IMMEDIATE` transaction. Claiming uses
durable round-robin among tenants that have eligible pending jobs and running
capacity, with FIFO order within each tenant. This is dispatch fairness, not a
latency/SLA guarantee. Input creation enforces the configured tenant/global byte
reservations and retained row bound, and these values survive contract reopen.
HTTP request-rate/key policies remain separate from job capacity.

The worker topology, service queue, and library lanes are different controls.
V3 worker records explicitly carry `lanes: 1`; higher HTTP lane counts are rejected
until a separately qualified integration exists. Each current worker has one
running request and no private request queue. Generic N-lane `GenerationEngine`
development does not silently activate parallel execution in this HTTP service.

Reservation queries aggregate in SQLite using live-credit indexes. Closed worker
scratch and retained outputs remain charged; quarantined resident/transient credit
cannot be reused before verified close/death. The generation-history row cap is
separate from configured worker count. No automatic retention or reclamation is
introduced by capacity configuration.

Operator snapshots expose host/state/authority, full device UUID, replica and
generation, model/profile, execution-binding digest, and serial lane/request
capacity. These placement identities are checked again when a lease publishes.
Tenant capabilities expose capacity policy without internal host identity. This
adapter remains one Linux host, UID, PID namespace and local SQLite authority.
Multi-node support would require an external transactional authority for atomic
resource/attempt claims, idempotency, fencing, settlement and artifact visibility;
no such adapter or orchestrator is installed here.

## New-state commands

`aureon-service replicas --help` exposes the commands. Existing v1 state is rejected
before schema mutation; there is no in-place migration. Create a fresh state and
credentials file with `replicas init`. Generation config must bind the existing
pinned assets, dependencies and upstream source. Real workers must use a noneditable
installed package whose full runtime payload matches the initialization identity.

An operator supplies finite byte capacities in a JSON topology. Example structure
(replace the device placeholder with the actual full lowercase physical UUID):

```json
{
  "host_bytes": 21474836480,
  "output_bytes": 8589934592,
  "controller_host_bytes": 268435456,
  "parser_host_bytes": 4294967296,
  "workers": [{
    "id": "gpu0",
    "device": "GPU-<full physical UUID>",
    "vram_bytes": 7516192768,
    "resident_vram_bytes": 3221225472,
    "resident_host_bytes": 6442450944,
    "transient_vram_bytes": 1073741824,
    "transient_host_bytes": 1073741824,
    "scratch_bytes": 2147483648
  }]
}
```

These are explicit example reservations, not measured limits for arbitrary GPUs
or models. Leave physical OS/other-application headroom. Host/output capacities
are shared once across the service. Adding a worker never adds another copy of
the shared host or disk allowance. Resident reservations count once per worker;
request transient reservations count once per active attempt. Null, boolean,
negative and nonfinite capacities reject. A parser/controller reserve is held
even while the controller is offline. The existing active-job bound is three.

```sh
aureon-service replicas init --state /data/new-replicas --tenant customer \
  --resources /data/topology.json --config /data/assets.json \
  --credentials /data/new-replica-keys.json

# Controller and its single bounded CPU parser; this process loads no model.
aureon-service replicas serve --state /data/new-replicas --port 8042 --max-seconds 3600

# In a separate process, explicitly execute one configured worker.
aureon-service replicas worker --state /data/new-replicas --worker gpu0 \
  --work /data/fresh-worker-work --config /data/assets.json \
  --execute-model --max-seconds 3600 --max-jobs 100
```

For another physical GPU, add a distinct configured worker/UUID before initializing
and launch its own worker process with a different fresh work directory. Ordinary
authenticated upload/submit/status/artifact-download endpoints remain the same.
Only the operator can read the full registry through `GET /v1/worker`.

## Assignment, fencing and recovery

A `BEGIN IMMEDIATE` transaction checks the current worker generation, shared
host/output credits and device VRAM credits, then creates the existing attempt
token, resource reservation and worker assignment together. There is no second
queue or second usage authority. Publication rechecks token, attempt, worker,
generation, device and calling process identity. Committing the artifact manifest,
usage and credit transition is one transaction. Duplicate publication cannot
settle another usage unit. Queued cancellation races atomically with claim;
running cancellation still returns 409 and does not preempt a CUDA kernel.

Expired heartbeat/attempt leases fence publication but **do not release resources**.
Execution failure quarantines credits until verified owned closure. Recovery
requires the recorded Linux PID/start-ticks/boot identity to be dead and requires
reacquiring both the worker and device kernel leases. A dead attempt requeues
within the existing attempt budget; a late old generation cannot publish.

```sh
aureon-service replicas status --state /data/new-replicas
aureon-service replicas recover --state /data/new-replicas \
  --worker gpu0 --generation RECORDED_GENERATION
```

Recovery is explicit in this slice. A replacement worker needs a new work directory.
A controller shutdown drains the shared service; a single worker closing does not
drain healthy peers. Use `replicas serve --resume` to reopen a deliberately drained
authority. Worker deadlines/signals stop admission and finish current synchronous
execution before owned cleanup; an external supervisor must bound a stuck GPU call.
The original single-worker supervisor/backup/restore commands do not accept v2.

## Resource ownership and limits

Completed output credits remain charged by actual durable artifact bytes until
explicit expiry removes the owned result directory. Partial publication is charged
conservatively after a crash. Worker scratch reserves remain charged after close
because those files still exist; this slice retains scratch and does not provide
automatic scratch reclamation. A depleted pool rejects admission rather than
claiming absent capacity. Do not manually edit reservation rows to free credits.

```sh
# Explicit retention of this service's expired artifacts only.
aureon-service replicas expire --state /data/new-replicas
```

The worker checks physical disk headroom and actual cached CUDA/RSS against its
resident budget before releasing request credits. A budget overrun fails closed.
Reservations are admission accounting, not a cgroup hard-memory limit or protection
from unrelated GPU consumers. External process/container limits and hosted security
remain operator responsibilities. No model quality, multi-GPU speedup or production
SLA is inferred from CPU fixtures or the single-GPU functional acceptance.
