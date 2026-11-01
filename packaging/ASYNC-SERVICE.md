# Authenticated local upload → job → verified result

> **Installed-package scope:** These features ship in the single v0.1.0
> package and call the resident Best directly. No source controller or model IPC
> is required. See [installation](CANDIDATE-QUICKSTART.md); original benchmark
> scope and published alpha artifacts remain unchanged.


Install the exact release in a [separate selected environment](START-HERE.md).
Keep state, scratch and credentials on an owned native Linux filesystem (not a
Windows mount/network share). Use separate state/work paths per service/model.
Default loopback binding is intentional. No external deployment is implied.

**TerraMind HTTP limit:** the locked worker executes TerraMind base only for
`generation` and `generation_scene`. Its shared configuration schema also lists
other sizes, but those generation jobs are rejected before a model session opens;
do not treat discovery/config acceptance as support. The `features` executor is
also rejected; use Backbone/TiM SDK/CLI for actual feature inference. The feature
transport schema remains testable with simulated outputs. Earlier multi-family
SDK evidence remains separate.

## KAO-TerraMind B

For the locked integrated TerraMind package, start with the generation section
below or [whole-scene HTTP](RESIDENT-WORKFLOWS.md#configure-a-scene-service).
The B instructions retain their separate dependency/artifact qualification.

Use the original GitHub v0.1.0 wheel and separate scene lock for this qualified
route. The revised generation candidate has incompatible default dependencies;
see [the profile migration table](LOCKED-INSTALL.md).

Create a bundle and prepared scene as in the first-result guide. In terminal1:

```sh
mkdir -p "$HOME/aureon-service" "$HOME/aureon-downloads"
aureon-service init --state "$HOME/aureon-service/b-state" \
  --credentials "$HOME/aureon-service/b-credentials.json" --tenant local \
  --model kao-terramind-b --environment scene --profile fp32-full-reference-v1
aureon-service serve --state "$HOME/aureon-service/b-state" \
  --work "$HOME/aureon-service/b-worker" --bundle "$HOME/aureon-assets/kao-terramind-b" \
  --execute-model --port 8042 --max-jobs 1 --max-seconds 300 --drain-seconds 60
```

In terminal2, activate the same environment:

```sh
aureon-service-client --url http://127.0.0.1:8042 \
  --credentials "$HOME/aureon-service/b-credentials.json" --tenant local run \
  --manifest "$HOME/aureon-work/prepared-first/manifest.json" \
  --model kao-terramind-b --profile fp32-full-reference-v1 --task scene \
  --idempotency-key b-first --recovery-file "$HOME/aureon-downloads/b-first-request.jsonl" \
  --output "$HOME/aureon-downloads/b-first"
```

Guarded B operators may select `fp32-stream-guarded-v3` with `--qualification`
during init, using a separately valid receipt. Use that same profile in the client.
The full-reference route above needs no operator-created qualification.

## TerraMind generation

Use the generation environment, prepared224 manifest and generated config.
In terminal1:

```sh
mkdir -p "$HOME/aureon-service" "$HOME/aureon-downloads"
aureon-service init --state "$HOME/aureon-service/tm-state" \
  --credentials "$HOME/aureon-service/tm-credentials.json" --tenant local \
  --model terramind-v1-base --environment generation --profile fp32-generation-v1 --denoiser graph
aureon-service serve --state "$HOME/aureon-service/tm-state" \
  --work "$HOME/aureon-service/tm-worker" --config "$HOME/aureon-config/generation.json" \
  --execute-model --port 8043 --max-jobs 1 --max-seconds 300 --drain-seconds 60
```

In terminal2:

```sh
aureon-service-client --url http://127.0.0.1:8043 \
  --credentials "$HOME/aureon-service/tm-credentials.json" --tenant local run \
  --manifest "$HOME/aureon-work/generation-input-first/manifest.json" \
  --model terramind-v1-base --profile fp32-generation-v1 --task generation --seed 17 \
  --idempotency-key tm-first --recovery-file "$HOME/aureon-downloads/tm-first-request.jsonl" \
  --output "$HOME/aureon-downloads/tm-first"
```

The client uploads declared evidence, submits, polls status, downloads and verifies
saved COG/provenance bytes. Numerical comparison is an Engine qualification, not
something transport verification proves. The service stops admission after its
finite job/time budget and retains a bounded read-only delivery grace. Preserve
state and request records for recovery. Use `status --job-id ID` with the same
credentials after uncertainty. The CLI's `run --input-id ID` reuses an existing
upload; keep the original logical idempotency key. Changed payloads under that key
are conflicts, not free retries. Never submit a fresh key just to download again.

Python clients use `aureon.serving.client.Client`; inspect its installed API and
CLI help for explicit resume/download methods. Credentials stay in owner-only
files and outside archives/logs. Default input quota is2GiB and storage reservation
includes source bytes plus1GiB per retained input; begin with a small crop. Host
binding, TLS/proxy, multi-host storage and provider deployment are separate work.
Only one model worker owns a service at a time. The
[original public release](https://github.com/LeChiffreVol2/aureon/releases/tag/v0.1.0)
records one guarded-v3 B512 and one TerraMind224 HTTP job on the unchanged
`a11b67e8…c2c1bf` wheel, with retained FP32 parity, verified saved artifacts,
replay/once-only usage and clean shutdown. This does not qualify every example
route on new hardware or hosted production. No rerun is required by this
documentation closeout.

The locked integrated wheel `e09e710c...` has its own installed acceptance:
26 calls across the original comparison, four-tile scene, local HTTP and private
Cloud routes, exact FP32/reference checks and seven zero-CUDA resident closures.
Those checks are separate from the original published alpha and historical
performance table. They do not qualify every family, sustained load or physical
multi-GPU serving.

See [resident scene HTTP, feature schemas and SDK dispatch](RESIDENT-WORKFLOWS.md)
for the locked installed workflows, including process ownership, finite budgets,
large artifact transport and separate source/installed/hardware qualification.

## Resume a download without submitting again

Keep the JSONL written by `run --recovery-file`. If the job was submitted but
the connection or download failed, use the current checkout companion example
with the same installed wheel, service URL and tenant:

```sh
python -I -B /absolute/aureon-checkout/examples/client/resume_download.py \
  --url http://127.0.0.1:8043 \
  --credentials "$HOME/aureon-service/tm-credentials.json" --tenant local \
  --recovery-file "$HOME/aureon-downloads/tm-first-request.jsonl" \
  --output "$HOME/aureon-downloads/tm-first-recovered" --wait-seconds 300
```

This example only reads status and downloads through the installed SDK. It
never uploads, submits, cancels or changes usage. It verifies job/model/profile/
task identity, then verifies result hashes and the original input binding.
The output must be fresh; retain partial downloads for inspection. If the log
has no `job_submitted` event, submission is uncertain: keep the original input ID
and idempotency key and resolve that request before creating another job.
Expired or failed jobs are reported without retrying computation.

The example is a source companion added after the immutable archive; it is not
claimed to be inside that earlier sdist. No Engine source overlay is needed.
