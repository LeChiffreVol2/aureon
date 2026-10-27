# Installed AUREON Engine examples

The public facade is `aureon.api` (contract `aureon.engine.v1`). Importing it or running help does not import NumPy, Rasterio or a neural runtime, acquire data, or execute a model.

```python
from aureon.api import SceneRequest, run_scene

request = SceneRequest(
    prepared_input="/absolute/prepared/manifest.json",
    bundle="/absolute/verified/kao-terramind-b",
    output="/absolute/results/new-sdk-job",
    model_id="kao-terramind-b",
    profile_id="fp32-full-reference-v1",
    environment="scene",
)
result = run_scene(request, execute_model=True)
print(result.class_map, result.valid_mask, result.receipt)
assert result.numerical_status == "passed"
```

The same explicit example is installed as `python -m aureon.api.examples.sdk_job --help`. CLI execution uses `aureon-engine run` with the same arguments. Full-reference execution omits qualification. Guarded execution requires a passing qualification for the actual installed source and dependencies; earlier Engine r2 records reject the changed manifest loader. KAO-TerraMind A/C retain the existing facade's separately scoped historical environment evidence; the resident session below supports B only.

For several prepared scenes, use one explicit resident session:

```python
from aureon.api import EngineSession

with EngineSession("/absolute/verified/kao-terramind-b", "/absolute/new-session",
                   profile_id="fp32-full-reference-v1", execute_model=True) as engine:
    first = engine.run("/absolute/prepared/first/manifest.json")
    second = engine.run("/absolute/prepared/second/manifest.json")
    assert engine.metadata()["model_load_count"] == 1
cleanup = engine.close()  # Idempotent; the context already closed it.
assert cleanup["cleanup_passed"] and not cleanup["errors"]
```

`python -m aureon.api.examples.resident_session --help` describes the installed example. The session constructs, runs and closes the existing worker on one owner thread. Overlapping calls raise `SessionBusy` immediately; there is no waiting caller queue. Only one session may own the process runtime. This guard applies to `EngineSession`; do not run legacy `run_scene` or direct workers concurrently in that process. Close explicitly or use a context manager.

`run(manifest, job_id=..., attempt_id=..., is_current_attempt=...)` accepts 32-character lowercase hexadecimal IDs. The logical job ID stays in the numerical receipts. A fresh attempt ID gives that logical job a fresh output namespace; repeating the same pair is rejected without another model execution. Omitted attempt IDs default to the job ID. Busy rejection leaves the identity retryable. An optional no-argument predicate must return exactly `True` before execution and before the session success receipt; services must still fence publication atomically because a lease can expire after return.

`SceneResult` points to the existing artifacts. A resident job receipt confirms completed artifacts while the session still owns resources; it does not attest shutdown. Inspect `session-cleanup.json` separately. Internal errors terminate the session and trigger owner-thread cleanup; there is no reload, fallback, hidden retry or partial-output reuse. Unsupported generation, reduced precision and KAO-TerraMind A/C are rejected by `EngineSession`.

For asynchronous local jobs, start `aureon-scene-api --execute-model --streaming --model kao-terramind-b --bundle /absolute/bundle --manifest /absolute/prepared/manifest.json --output /absolute/new-session --port 8031 --max-runtime-seconds 900`. It is the unchanged strict scene-v1 **full-reference** API, not a guarded-v2/v3 HTTP endpoint.

```python
from aureon.api import SceneClient
client = SceneClient.from_token_file("http://127.0.0.1:8031", "/absolute/new-session/token.txt")
job = client.submit("REGISTERED_SCENE_ID", idempotency_key="my-request-001")
client.wait(job.job_id, timeout=900)
delivery = client.download(job.job_id, "/absolute/new-download")
print(delivery.receipt)
```

All output directories must be fresh. The original `run_scene` owns and closes one worker per call; its `engine-result.json` is written only after successful shutdown and artifact checks. Its behavior and `SceneClient` timeout, idempotency and error behavior are unchanged. `SceneClient` does not accept guarded receipts as v1. The explicit `EngineSession` lifecycle above is additive.

Highest FP32/TF32 off and the accepted recipe are preserved. Guarded jobs report no per-job numerical comparison; an explicit qualification binds exact code, checkpoint, dependencies and GPU. Canonical JSON identity can differ from a qualification file's byte hash. Numerical/geospatial fidelity is not labeled accuracy or generalization. No generation, hosted service, unrestricted input size, remote weight attestation or performance advantage is promised.

## Both model families

`python -m aureon.api.examples.model_support --help` documents a two-job resident TerraMind example through common dispatch. Execution requires an offline pinned config, fresh work directory, separate generation environment and explicit `--execute-model`. `aureon-engine models` lists KAO-TerraMind A/B/C and TerraMind capabilities without importing models.
