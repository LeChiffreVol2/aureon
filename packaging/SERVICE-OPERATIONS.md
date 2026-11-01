# Local pilot service operations

## Local source extension: bounded preparation and device ownership

**Integrated source, unreleased · v0.1.0.** These operations describe current
source. The unchanged published release and prior private GPU acceptance retain
their own artifact identities. No new build or GPU qualification is implied.
Use [release installation](START-HERE.md) for the published wheel and the
[replica guide](REPLICA-SERVICE.md) for the separate replica-state contract.

One preparation thread may validate one uploaded input while the worker runs a
ready job. The parser remains a separate CPU process with its existing limits.
There is still one durable queue, at most three active jobs and one running job;
GPU execution and durable publication stay serial. No second `GenerationEngine`
queue is inserted behind the durable queue. Queued cancellation retains its
existing atomic semantics; it never interrupts active GPU work. Stop cancels the
owned parser child, joins preparation/heartbeat threads and closes the backend.
An interrupted input validation becomes rejected; already queued jobs retain
their prepared inputs for explicit resume. Publication checks aggregate file
sizes before reading results, keeps the 64 MiB payload budget and verifies the
fenced attempt again before committing artifacts and usage.

`serve`, `worker` and `supervise` now accept optional `--device GPU-<full UUID>`
for the resident TerraMind generation route. Ordinals, shortened UUIDs and MIG
identifiers are unsupported. An inherited CUDA allowlist must itself contain the
full selected UUID; binding cannot expand that allowlist. Device binding occurs
before CUDA initialization, and the explicitly executing backend checks the
actual visible device UUID before loading weights. A process with initialized
CUDA cannot be rebound. The ordinary one-worker mode uses separate state/work/port ownership. Shared
replica state requires the explicit [replica mode](REPLICA-SERVICE.md); do not
launch multiple ordinary workers against one state.

The default without `--device` preserves host-wide exclusion for that service
UID. Explicit devices acquire a shared legacy gate plus an exclusive UUID lock,
so different UUIDs may coexist, while the same UUID and older host-wide workers
conflict. Locks are kernel-owned until verified close or process exit; they are
not stealable time leases. If cleanup cannot attest zero CUDA allocated/reserved,
the lock is quarantined until process exit. Lock files are never unlinked by the
worker. Use the same host UID and same local `AUREON_GPU_LOCK_DIR` mount for all
cooperating workers. Different UIDs/lock namespaces and unrelated GPU programs
are not coordinated. Scoped KAO-TerraMind workers are deliberately rejected until
ownership of their GPU child processes can be carried through a parent crash.

The operator-only `GET /v1/worker` returns stable state-instance worker ID,
per-start generation, configured route, device scope, stage state and queue
counts. It contains no bearer credentials or attempt tokens. Customer access is
403. Resource estimates and exact `ExecutionKey` are explicitly absent pending
qualified bindings; this status is not directly an admitted placement. A local
integration can revalidate the selected generation through
`Control.claim(expected_owner=generation)` inside the existing durable claim
transaction. Heartbeats and publication of such claims are also owner-fenced.
Do not treat `api.routing.plan_placement` as a resource reservation or dispatch.

Worker, parser and supervisor elapsed durations use `CLOCK_MONOTONIC_RAW` on
Linux and `perf_counter` on Windows, with the clock name in worker/supervisor
receipts. Persistent expiry and lease timestamps retain the control clock.
These service diagnostics have no per-run host clock witness; comparative GPU
benchmarks still need the separately qualified RAW/QPC measurement protocol.
No global Python clock replacement, numerical kernel, FP32 recipe or checkpoint
was changed by this extension.

## Finite supervision and ownership

The installed service entrypoint accepts `supervise`, which starts `serve` in an owned process session, retains attempt logs/receipts, uses a fresh scratch child for each start, and limits total elapsed service time, completed durable jobs and restarts. Default restarts are zero; an explicit value0..2 enables bounded backoff. Each logical job still has at most three leased attempts. Model exceptions terminate a worker; failed jobs remain failed. A later successful idle worker does not hide newly failed durable jobs in the supervisor result.

```sh
python -m aureon.serving supervise --state /data/b-state --work /scratch/b-run-001 \
  --bundle /assets/kao-terramind-b --execute-model --port 8042 \
  --max-jobs 100 --max-seconds 3600 --max-restarts 1 --backoff 2 \
  --stop-grace 30 --drain-seconds 30 --ready-file /data/b-bind-001.json
```

The state must already select KAO-TerraMind B and an **exact-artifact qualified guarded profile**; full-reference remains a deliberate audit alternative. For TerraMind use a separate state/environment and replace `--bundle` with the verified generation `--config`. All commands above are operator examples, not instructions to rerun frozen acceptance. Work and ready paths must be fresh. A restarted child writes a distinct `-restart-N` ready file recorded in the supervisor receipt; choose a fixed port for a stable local endpoint.

The CLI owns `service.lock`; the worker owns `worker.lock`; the supervisor owns `supervisor.lock`. Backup requires all three to be free. These locks protect the ordinary installed CLI path; an embedding application using `Control` directly must implement the same ownership contract and stop all writers before maintenance. No file lock can protect against unrelated software deliberately editing state.

Real workers additionally hold a conservative host/service-user GPU lock (`aureon-gpu-<uid>.lock`). By default it is in `/tmp`. Containers must share a local operator-owned `AUREON_GPU_LOCK_DIR` mount and service UID to coordinate the same GPU; isolated container `/tmp` or different UIDs do not provide a cross-container lock. The lock serializes all GPUs for that service user; it is not a multi-GPU scheduler. No other application's GPU process is stopped.

SIGTERM/operator drain stops admission and allows active work to close. The supervisor's finite deadline sends SIGTERM to only its owned process session, then SIGKILL after its configured grace if needed. Forced termination is a failure and never a zero-CUDA cleanup claim. The process interruption test uses a fake executor and an injected lease expiry; it is not a real CUDA-crash qualification. Restart follows existing durable leases, fencing and idempotency. Native per-job output and service shutdown remain separate receipts.

## Backup, restore and rollback

Stop the supervisor/server first. Backup takes exclusive locks, durably drains the state, uses SQLite's backup API, copies the complete state and an explicitly supplied active credential bundle, verifies referenced prepared files/results, and checks source files again. It checks the unchanged32GiB disk floor plus the snapshot allocation on both the destination and WSL's backing drive. The backup directory is private; copies are mode0600 on Linux. It contains signing secrets and bearer keys: it is an operator backup, never a customer download or a public artifact.

```sh
python -m aureon.serving backup --state /data/b-state \
  --credentials /data/operator-credentials.json --output /backups/b-snapshot-001
python -m aureon.serving restore --backup /backups/b-snapshot-001 \
  --manifest-sha256 VERIFIED_BACKUP_MANIFEST_SHA256 --state /data/b-restored-001
```

Keep the reported manifest SHA separately with the exact wheel/config/qualification/runtime identity. Hash verification requires that trusted external hash; a hash stored beside attacker-modified contents is not an authenticity guarantee. Every supplied credential must still authenticate to its declared tenant and the bundle must include an operator key. One-time customer keys held only by customers cannot be reconstructed from database hashes. Protect additional customer/operator credential files separately or include them in the supplied active bundle.

Restore validates the complete snapshot before publishing a fresh state directory. It preserves logical job/key/usage records, invalidates old attempt tokens, requeues interrupted attempts only within their existing attempt budget, and preserves completed jobs for replay/download. It rotates the signing secret so old ticket URLs fail, preserves bearer credential hashes, and writes protected `restored-credentials.json`. The restored state is **drained**; inspect it before an explicit `--resume` with fresh scratch. Restored scratch registry entries are marked foreign and cannot reclaim directories from the old instance. Partial restore staging is retained for diagnosis and is never the published destination.

Use the prior compatible frozen wheel/environment with its own pre-change backup for an operator rollback; do not run two restored copies or old/new binaries against the same state. There is no automatic cross-version data migration guarantee. The current CPU receipt proves backup/restore/replay and stale-token rejection with this source schema. An installed real-model rollback is separate evidence. Backup does not cover model assets, external scratch or a provider's volume lifecycle; retain those immutable assets/qualifications separately.

## Retention and scratch reclamation

```sh
python -m aureon.serving expire --state /data/b-state
python -m aureon.serving reclaim --state /data/b-state --work /scratch/b-run-001/worker-0
```

`expire` now commits inaccessible input/result tombstones and a durable GC intent **before** deleting payloads. A failed/partial deletion can be retried; idempotency keys and usage are not erased. Queued/running jobs pin their inputs, and active validation/upload leases are excluded. Record quotas remain bounded, so indefinite tombstone retention eventually requires a separately designed archival/migration policy rather than silently reusing customer keys.

`reclaim` accepts only a scratch directory registered by this state instance with a successful closed-worker receipt; real worker receipts must attest CUDA0/0. It stores the complete scratch file hash inventory and worker result in durable maintenance state before deleting. It can recover from interrupted deletion and repeated calls. Unregistered paths, symlinks, nested mounts, state overlap, failed workers and foreign restored-instance scratch are rejected. Failed/abandoned scratch remains for explicit diagnosis/export; it is not swept automatically. Durable customer output remains subject to its own retention policy. Frozen research evidence is never registered as scratch.

The32GiB control disk floor,34GiB scratch pre-claim floor and4GiB scratch admission bound remain unchanged. Uploads reserve source bytes plus1GiB; the default2GiB tenant limit does not permit two independent nonempty uploads. Reuse ready inputs when appropriate. Supervisor bounds include retained scratch across its attempts and a16MiB log limit. Account for both the local filesystem and any backing-store capacity. CPU tests may inject capacity to exercise logic; real model runs must pass actual storage admission.

## Key lifecycle, HTTP and parser boundary

Tenant key rotation is two-stage: create a replacement, authenticate it, then revoke the old fingerprint. Losing the create response leaves the old key working. Key material is shown once; list returns fingerprints only. Active keys are limited to8 per tenant and historical records to10,000. Revocation is durable and tenant-scoped; the last operator key cannot be revoked. Requests already authenticated may finish, but subsequent requests using a revoked key fail. All ticketed transfers also require bearer authentication.

The authenticated endpoints are `GET/POST /v1/keys`, `POST /v1/keys/revoke` with `{fingerprint}`, and `GET /v1/usage`. Offline operator equivalents are:

```sh
python -m aureon.serving key-create --state /data/b-state --tenant customer --output /data/customer-new-key.json
python -m aureon.serving key-revoke --state /data/b-state --tenant customer --fingerprint OLD_KEY_SHA256
```

The new key file is private from its first write, and the client CLI accepts it directly with `--tenant customer`. Validate the new key before revocation. Neither CLI output nor structured failure logs contain the key. HTTP access logging remains disabled; thread/worker failures use stable codes/types rather than exception messages, request bodies or signed URLs. Customer readiness omits worker PID and global job counts; the operator retains detailed local readiness.

HTTP retains mandatory bearer auth, strict Host allowlists, no browser Origin support, no transfer-encoding/chunked request bodies, unique Content-Length,64KiB JSON request bodies, bounded uploads and four concurrent handler slots. Aggregate headers are limited to32KiB, connection lifetime to45seconds, per-read waits to10seconds and authenticated requests to a256-request burst replenished at32requests/second per tenant. This is a local process limit, not a distributed rate limiter. Public HTTPS termination, verified identity, proxy/network policy and abuse protection remain hosted boundaries. Never treat a caller-supplied forwarded header as identity proof.

Uploaded TIFF/NPY/JSON parsing runs in a separate CPU process before model admission. The child receives an allowlisted environment, no inherited application credentials and no visible CUDA device. Linux limits are90seconds wall,60seconds CPU,4GiB address space,1GiB per file,64 file descriptors and no core dumps. It admits only declared local paths, forces TIFF decoding, bounds raster dimensions to the supported4096px/12band scene envelope, rejects pickled NumPy and preserves existing input validation. It does not crop/resample the generation224px input.

**Process/rlimit isolation is not a hostile-code sandbox.** A native decoder exploit could still act with the service UID's filesystem privileges. Network namespaces/seccomp, unprivileged identities, secret separation and read-only mounts require a separately built/qualified image. Windows fake tests have subprocess/wall-time isolation but not Linux rlimits. No claim of a publicly hardened decoder or HTTPS endpoint follows from these tests.

## Public launch and payment seams

The product direction is public library first, followed by a public self-service hosted API when payments are ready. No invitation-only launch gate is introduced. Public access does not mean free or unlimited GPU work.

Current identity/payment adapters are deliberately **local-test only**, absent from public HTTP signup/webhook routes. The test identity seam maps a hashed provider subject to one tenant/key with zero test entitlement; it does not verify an external identity or issue money/free credits. A future verified provider adapter must supply that trust boundary. The test payment seam stores idempotent events, rejects conflicting duplicates and stale sequence snapshots, and changes only synthetic admission-capacity units. It is not a checkout/webhook/charge/refund integration.

New jobs atomically reserve one nonbillable logical unit. Success settles it once with artifact publication; cancellation/failure releases it. Internal attempts and lost acknowledgements do not add customer jobs or usage. The shadow statement has no price, amount or currency. Paid settlement/refund reconciliation, real provider event verification, invoice accuracy, pricing and support terms remain unconfigured. Local doubles do not qualify them. Existing0.6 jobs retain their original usage; the new shadow ledger does not retroactively charge them.

## CPU evidence and installed integration

The separately implemented [replica v2 source extension](REPLICA-SERVICE.md) adds
one shared durable authority with per-device workers and atomic resource/attempt
assignment. It requires explicit fresh state; the existing one-worker commands
and published wheel remain unchanged. Its CPU/fake-process qualification and
remaining installed/multi-GPU limits are documented in that guide.

Prior CPU/process checks exercised fake-worker interruption, watchdog
termination, lease fencing, replay, redaction, parser limits and backup/restore.
Injected failures do not qualify real CUDA recovery. Prior installed GPU results
belong to their original private artifacts. Review the exact source, dependency
profile and artifact identity before using a new build.

## Operator monitoring

`GET /v1/metrics` returns `aureon.service_metrics.v1` JSON to an authenticated
operator. Customer credentials receive HTTP403. Keep operator credentials in
your secret store; do not put them in browser code or metric labels.

```python
from aureon.serving.client import Client

operator = Client("http://127.0.0.1:8042", token=operator_key)
metrics = operator.json("GET", "/v1/metrics")
```

The snapshot provides durable job/input/attempt states, queue capacity and oldest
queued age, active expired leases, live/stale worker counts, input reservations,
record capacity, pending retention and once-only nonbillable usage. It covers
single-worker and replica controllers. It contains no tenant/job IDs, paths,
credentials, device UUIDs, or unbounded event/trace lists. All state aggregates
are read in one transaction; the probe never claims, cancels, retries, deletes,
loads a model or queries CUDA. Metrics remain available when workers are stopped.

Use `/v1/ready` for worker/admission readiness and `/v1/health` for control-plane
health; a successful metrics request alone does not mean the model is ready.
Watch `queue.expired_running_leases`, `workers.stale_active`, growing queue age,
reservation/record limits and pending retention. Recovery and cleanup remain
explicit operator actions under the existing ownership and retention rules.

These fields are gauges of retained state, not monotonic Prometheus counters.
Terminal-record expiry can change state counts. Queue age uses the persisted
wall clock; a backwards clock exposes `queue.future_created_records` and clamps
negative ages to zero. Do not use these values as GPU latency or throughput
benchmarks, actual VRAM/RSS measurements, invoices, or hosted SLA evidence.

## Explicit offline replica maintenance

Replica maintenance is Linux-only, on one host/UID/PID namespace. Stop the
service and workers first. These commands take every controller/worker lock and
the configured real-device leases; they reject live or uncertain owners. They
do not load models, start workers or schedule housekeeping.

Retire one **closed, dead, registered generation's** scratch under the explicit
parent you own:

```bash
python -m aureon.serving.replicas reclaim-scratch \
  --state /srv/aureon/state --worker gpu0 --generation GENERATION_ID \
  --work-root /srv/aureon/scratch
```

The directory must be the registered direct child of `--work-root`, carry its
matching ownership record, and have no active assignment/resident credits.
Links, replaced roots and changed files fail closed. The command records an
inventory intent before removal and releases the scratch reservation only after
successful removal. An interrupted removal can be replayed with the same
command; repeated success releases zero additional credits. Durable job
artifacts and usage records remain intact. This command is explicit deletion
of registered scratch, not a workspace cleanup or a background retention rule.

Create a fresh offline snapshot with the current credentials file:

```bash
python -m aureon.serving.replicas backup \
  --state /srv/aureon/state --destination /srv/aureon/backups/snapshot-01 \
  --credentials /srv/aureon/secrets/keys.json
```

Backup drains the source, attests dead owners under all kernel locks, reconciles
their finite attempts, and verifies SQLite plus input/output hashes. Source
scratch is preserved and remains charged. The snapshot contains secrets; keep
it access-restricted and protect its returned `manifest_sha256` independently.
The copied `state/` is fenced before its contract becomes valid; it cannot be
started as a second service by clearing drain. Use the restore handover command.

Restore is a **same-host handover**, using that exact digest and a fresh path:

```bash
python -m aureon.serving.replicas restore \
  --source /srv/aureon/backups/snapshot-01 \
  --destination /srv/aureon/restored-state --manifest-sha256 MANIFEST_SHA256
```

The original state must remain accessible, drained and unchanged since the
snapshot. Restore verifies the snapshot, then durably retires the original
execution authority before publishing the destination. The retired source
rejects new admission, worker registration, claims and publication even if
someone clears its drain flag. Its files remain intact. This prevents a copied
pending queue from running in two authorities. Replaying the same handover
finishes an interrupted directory publication; a second destination is refused.
Every staging directory is fenced before its runtime contract is made valid.
Clearing its drain flag cannot make it executable. The pending fence stays set
through publication and clears only after source retirement and validation of
the published destination, so pre-retirement staging cannot fork the queue.

The destination starts drained, preserves logical job/idempotency/usage records,
rotates instance/ticket/placement identity and keeps the same model/profile
binding. Old scratch paths are detached and cannot be reclaimed through the new
state. Resume explicitly with the existing replica serve/worker commands and
fresh scratch directories. This is not cross-host disaster recovery or a
distributed ownership system. Physical multi-GPU behavior still requires its
own hardware qualification.

Job/input/generation record caps remain finite, intentional backpressure.
Payload expiry and scratch retirement do not silently erase idempotency/usage
history or make those record limits unlimited. Operators must size those limits
for the intended deployment and review lifecycle needs before exhaustion.

## Operational handoff for the locked integrated package

Use this checklist with the current wheel identity before opening a service.
Keep operator credentials outside source and customer downloads. These checks
are local CPU/simulation evidence unless an installed GPU receipt says otherwise.

| Failure or decision | Required behavior / operator action | Reproducible CPU coverage |
| --- | --- | --- |
| Upload or submission acknowledgement lost | Retain input/job ID and original idempotency key; do not start another logical request | `test_service_workflows`: HTTP upload/preparation/submit/download |
| Download interrupted | Resume the same completed job into a fresh directory; preserve partial output and verify identity/hashes | `test_resume_download`: real HTTP replay, unchanged usage and tenant rejection |
| Queue/output reservation exhausted | Refuse before execution; wait for capacity or review quota, never bypass it | `test_service_workflows`: atomic quota/cancel/replay |
| Queued cancellation or running job | Release queued work according to the state transition; running/terminal cancellation may return conflict | Same quota/cancel test; no forced-GPU cleanup claim |
| Worker lease expires | Fence the stale attempt; allow only the valid attempt to publish and settle once | Same recovery/once-only usage test |
| Retention deletion fails | Keep durable tombstone/GC intent and reservation; retry maintenance, not inference | `test_service_workflows`: failed output credit/expiry GC |
| Download/native source is corrupt | Fail verification; do not treat a partial file as delivery | stream persistence and `test_best_workflows` recovery corruption tests |
| Backend close fails | Keep failure receipt; reject new execution and do not silently retry cleanup | `test_service_lifecycle` |

Run `packaging/ci_cpu.py` for the package contracts and download-recovery tests.
The additional named cases above live in `tests/test_service_workflows.py` and
`tests/test_best_workflows.py`; the CI shortcut does not run every listed case.
Use the current checkout for the new recovery example/tests; these were added
after the frozen source archive and do not form another runtime package.
A passed fake-worker test does not certify CUDA crash recovery or scientific
output quality. Preserve each installed-wheel receipt and never infer its scope
from version `0.1.0` alone. Public HTTPS, hostile-parser sandboxing, real provider
provisioning and sustained/multi-GPU operation still require separate qualification.
