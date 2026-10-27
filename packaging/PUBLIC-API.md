# Public Python API v1

Import `aureon.api` after [installation](START-HERE.md). `API_VERSION` is
`aureon.engine.v1`; package version is 0.1.0. Importing the facade does not load
neural runtimes or weights. Model execution is opt-in. Paths in these examples
refer to your own prepared inputs/assets and fresh output directories.

## TerraMind generation

First [create a config with your pinned assets and prepared input](START-HERE.md#first-terramind-result).
This example executes a model job; choose a fresh output directory.

```python
from aureon.api import GenerationRequest, run_model
result = run_model(GenerationRequest(
    config='/absolute/config/generation.json', output='/absolute/work/tm-first',
    job_id='first', attempt_id='one', reuse='auto', denoiser='graph'), execute_model=True)
print(result['job'], result['cleanup'])
```

The generation result is JSON-compatible with schema `aureon.generation_result.v1`,
task, model, synthetic flag, native `job` record and separate `cleanup` receipt.
Do not interpret generated continuous imagery as a segmentation map.

## KAO-TerraMind segmentation

Use a separate scene environment and your [prepared B bundle](START-HERE.md#first-kao-terramind-b-result).

```python
from aureon.api import SceneRequest, run_model
request = SceneRequest(
    prepared_input='/absolute/work/prepared-first/manifest.json',
    bundle='/absolute/assets/kao-terramind-b', output='/absolute/work/b-first',
    model_id='kao-terramind-b', profile_id='fp32-full-reference-v1', environment='scene')
result = run_model(request, execute_model=True)
print(result.class_map, result.valid_mask, result.provenance, result.receipt)
```

`SceneResult` exposes paths plus `profile_id` and `numerical_status`. The facade
owns worker shutdown. `run_scene` is the scene-specific equivalent. Full-reference
A/C use their model IDs and `environment='scene-legacy'`.

## Resident sessions and errors

For multiple serial jobs, use `open_model_session('terramind-v1-base', config=...,
work=..., reuse='auto', denoiser='graph', execute_model=True)` as a context manager.
Call `session.run(job_id=..., attempt_id=..., recipe=...)` for explicit seed recipes.
Inspect `session.close()` for cleanup after leaving the context. The installed
`python -m aureon.api.examples.model_support --help` example demonstrates two seeds.
For B, `EngineSession` owns the resident worker with an explicit guarded
qualification. The installed `resident_session` example documents its parameters.
One job may be in flight per session; concurrent use raises `SessionBusy`.
Internal execution failure terminates the session; recovery uses a new session.

`ExecutionRequired`, `ProfileError`, `SessionBusy`, `SessionClosed`, and
`SessionFailed` are explicit public failures; malformed input also raises
`ValueError`/`OSError`. Do not treat an error as an empty valid prediction.
CLI errors return nonzero; successful commands emit JSON.

`aureon.api.assets.list_assets(model_id=None)` provides metadata only. Core
low-level modules and underscore-prefixed functions remain implementation details.
For authenticated HTTP use `aureon.serving.client.Client` and the CLI documented
in [ASYNC-SERVICE.md](ASYNC-SERVICE.md). HTTP transport and SDK invoke the same
Engine. API v1 changes will preserve its documented request/result semantics or
explicitly introduce a new version; 0.x does not imply a production SLA.
