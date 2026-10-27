# Authenticated local upload → job → verified result

Install the exact release in a [separate selected environment](START-HERE.md).
Keep state, scratch and credentials on an owned native Linux filesystem (not a
Windows mount/network share). Use separate state/work paths per service/model.
Default loopback binding is intentional. No external deployment is implied.

## TerraMind generation

Use the generation environment, prepared224 manifest and generated config.
In terminal 1, activate the generation environment:

```sh
source "$HOME/aureon-generation-0.1.0/bin/activate"
mkdir -p "$HOME/aureon-service" "$HOME/aureon-downloads"
aureon-service init --state "$HOME/aureon-service/tm-state" \
  --credentials "$HOME/aureon-service/tm-credentials.json" --tenant local \
  --model terramind-v1-base --environment generation --profile fp32-generation-v1 --denoiser graph
aureon-service serve --state "$HOME/aureon-service/tm-state" \
  --work "$HOME/aureon-service/tm-worker" --config "$HOME/aureon-config/generation.json" \
  --execute-model --port 8043 --max-jobs 1 --max-seconds 300 --drain-seconds 60
```

In terminal 2, activate that same environment:

```sh
source "$HOME/aureon-generation-0.1.0/bin/activate"
aureon-service-client --url http://127.0.0.1:8043 \
  --credentials "$HOME/aureon-service/tm-credentials.json" --tenant local run \
  --manifest "$HOME/aureon-work/generation-input-first/manifest.json" \
  --model terramind-v1-base --profile fp32-generation-v1 --task generation --seed 17 \
  --idempotency-key tm-first --recovery-file "$HOME/aureon-downloads/tm-first-request.jsonl" \
  --output "$HOME/aureon-downloads/tm-first"
```

## KAO-TerraMind B

Create a bundle and prepared scene as in the first-result guide. In terminal 1:

```sh
source "$HOME/aureon-scene-0.1.0/bin/activate"
mkdir -p "$HOME/aureon-service" "$HOME/aureon-downloads"
aureon-service init --state "$HOME/aureon-service/b-state" \
  --credentials "$HOME/aureon-service/b-credentials.json" --tenant local \
  --model kao-terramind-b --environment scene --profile fp32-full-reference-v1
aureon-service serve --state "$HOME/aureon-service/b-state" \
  --work "$HOME/aureon-service/b-worker" --bundle "$HOME/aureon-assets/kao-terramind-b" \
  --execute-model --port 8042 --max-jobs 1 --max-seconds 300 --drain-seconds 60
```

In terminal 2, activate the same environment:

```sh
source "$HOME/aureon-scene-0.1.0/bin/activate"
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

## Results and retries

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
Only one model worker owns a service at a time. The final v0.1.0 wheel passed
B512 guarded-v3 and TerraMind224 HTTP functional acceptance, including retained
FP32 parity, replay/once-only usage and shutdown. See the
[support matrix](../docs/MODEL-SUPPORT.md). This does not qualify every example
route on other hardware or hosted production.
