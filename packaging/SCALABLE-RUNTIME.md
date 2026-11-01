# Scalable execution and replica contracts

These opt-in Python interfaces are in the local v0.1.0 candidate. They are not a
replacement for the published artifact or a claim of measured multi-GPU speedup.
No model, socket, worker, download or CUDA context starts on import.

## Bounded stages before transport

`aureon.scaling.Pipeline` executes explicitly declared `Stage` callbacks. Stages
can represent read, preparation, encoding, generation, decoding and export; an
adapter may keep non-interruptible native operations together. Each stage uses
tenant round-robin and explicit worker counts. The default is one worker per
stage: sharing an existing CUDA resident session across threads is not implied.

Each request reserves **peak live request bytes across its stages**, including
expanded inputs, latent/token state and retained output. Admission rejects when
request or memory credits are exhausted. Results retain credits until consumed
or discarded; resident model/graph allocations require their separate device
budget. These are conservative accounting contracts, not allocator enforcement.
Data ownership transfers into the pipeline until completion. Never mutate a
submitted tensor concurrently. `StageContext` carries request identity, seed,
cancellation and deadline; the adapter creates its own generators and sampler.

Callbacks check cancellation between native calls. Cancelling does not terminate
a CUDA kernel. `close()` drains; `close(cancel=True)` requests cooperative stop.
`Stage(finalize=...)` closes stage-owned resources on the same worker thread that
ran its callbacks. Create resident sessions lazily on that thread, keep per-worker
state in thread-local storage when using multiple workers, and close them in the
finalizer. Cleanup failures are reported without replacing a primary context error.
Stage seconds/calls, active stages and retained credits are available via
`metrics()`. They do not establish GPU utilization or scientific output quality.

## Replicas and remote execution

`Broker` provides a durable SQLite work queue. Register each worker's routes,
execution slots and available request-memory budget explicitly. Capacity is
operator-declared, not remote hardware attestation. Claims skip requests that do
not fit a worker and rotate tenants. Draining prevents new leases while current
work can finish. Heterogeneous workers therefore use the same job contract.

Each lease has an unpredictable token, attempt and expiry. Heartbeats renew it.
After failure/expiry, another replica may execute the request again; only the
current lease can commit. Execution is **at least once**, with **one fenced result
commit**. This is not exactly-once computation or billing. Model callbacks must
not perform billing or other unprotected external side effects. Completed commit
retries with identical metadata are idempotent. Cancellation rejects late results.

The coordinator uses its own wall clock for expiry; keep its clock stable. SQLite
belongs on a local durable disk behind **one coordinator**, not a network share
used by independent coordinators. Restart reopens jobs and expires old leases on
the next operation. Keep database/WAL together for backup. This release does not
implement replicated coordinator consensus or high availability.

`CoordinatorServer` is a bounded stdlib reference transport on loopback. It uses
JSON only, distinct client and worker credentials, and worker-scoped lease
operations. A worker cannot claim as another worker or submit customer jobs.
For separate machines, expose this operator endpoint through a **verified TLS
reverse proxy or authenticated tunnel**; `RemoteClient` rejects non-loopback HTTP,
disabled certificate checks and redirects. No pickle, import path, arbitrary URL
or server-selected local file path is accepted. Deployment and TLS provider
configuration remain the operator's responsibility.

This is a **trusted operator replica protocol**, not a public multi-tenant API.
The client credential is an operator credential. Worker credentials can access
blobs in this deployment; use separate deployments/stores for different trust
domains. Public identity, tenant authorization, quotas and billing belong above
this interface. It does not silently expose the private AUREON Cloud service.

`ArtifactStore` writes bounded SHA256-addressed blobs atomically and verifies
checksums on upload/download. The wire result contains `artifacts: {name: digest}`;
all listed blobs must exist and verify before commit. Export COG and provenance
as separate immutable blobs and return their digests. No model data belongs in
control messages (64 KiB metadata bound). Choose disk/per-artifact budgets and
retention explicitly. The library never automatically deletes model/evidence
files. Orphaned retry blobs consume quota until operator retention; terminal-job
`forget()` removes idempotency history and does not remove artifacts.

Client flow is `handle = client.submit(job)` → `handle.status()` →
`handle.result(timeout=...)` → `handle.download_artifacts(new_directory)`.
Reattach after restart with `client.job(job_id)`. A timeout leaves the job and
handle usable; retry an uncertain submission with exactly the same `Job` identity.
Keys are supplied by the operator and never belong in the JSON job or its logs.

## One adapter lifecycle, different model algorithms

`ResidentAdapter(prepare=..., infer=..., deliver=...)` is a typed common lifecycle.
The owner creates and closes the model once on the worker thread. A
`ResidentWorker` runs bounded explicit `run_once()`/`run_until_idle(max_jobs=...)`
calls and renews leases in a separate heartbeat thread. Native model work never
moves to the heartbeat thread.

Use `terramind_worker_adapter` with a caller-owned `GenerationEngine` and a trusted
loader returning `(EORequest, GenerationInput)`. It calls existing `generate_eo`,
including route/grid/units/seed validation and job/attempt provenance.
`copgen_worker_adapter` instead uses `(EORequest, CopgenInput)` and the resident
`CopgenEngine`; it preserves COP-GEN's native schedule and S1RTC semantics. Both
reject a job seed different from its sampling recipe. Delivery callbacks export
through the existing model-specific COG writer and upload its result/provenance.
They do not pretend S1GRD and S1RTC are interchangeable.

Use one visible GPU per native resident worker process while upstream global
state requires it. More replicas are different from more tensor batch elements.
Tensor/sequence/pipeline sharding of one model is not implemented by this queue.
Do not infer support from a node label, a device UUID or a simulated capacity.

## Verification scope

`packaging/ci_scaling.py` runs CPU stage/fairness/backpressure checks and actual
loopback HTTP/blob transfer, including a separate worker process. It checks
restart, retry fencing, heartbeats, cancellation, credential scope, metadata/disk
limits and checksum corruption. This proves control flow, not physical GPU
throughput, cross-host TLS deployment, native worker concurrency, sustained
production load or scientific usefulness. Same-profile FP32 remains the reference;
new precision/backend/hardware profiles need their own numerical and EO checks.
