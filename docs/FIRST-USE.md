# Try AUREON and record friction

Start with [installation](../packaging/CANDIDATE-QUICKSTART.md), then
[first input and result](../packaging/START-HERE.md). The optional installer
creates a fresh environment, verifies the released wheel and runs a CPU
preparation example. It never downloads weights or starts a model job.

## Try the onboarding improvements

The installer and updated guides are available on the review branch. Clone the
public repository into a new directory; no private repository access is needed:

```sh
git clone --branch feature/developer-onboarding --single-branch https://github.com/LeChiffreVol2/aureon.git aureon-review
cd aureon-review
```

This uses the existing released v0.1.0 Engine. The review branch supplies the
installer and guides; it does not replace the release. Use Linux x86_64 / WSL2
with Python 3.12 for the installation below. Choose a new environment path.

```sh
python3.12 tools/install.py --plan
python3.12 tools/install.py --execute --prefix "$HOME/aureon-env"
. "$HOME/aureon-env/bin/activate"
```

Run from a current checkout, or give the absolute path to `tools/install.py`.
The prefix must be new and outside the checkout. Linux x86_64 / WSL2 and Python
3.12 are required for execution. Download time is separate from user effort.
Pinned third-party dependencies can require source builds; AUREON's released
wheel is installed unchanged. Logs and `install-result.json` remain in the
prefix on failure. Choose a fresh prefix after correcting the cause. No existing
environment is upgraded or removed.

On success, `install-result.json` reports `passed: true`. The CPU walkthrough
reports `pixels_and_grid_verified: true`. Record `git rev-parse HEAD` with any
feedback so maintainers can reproduce the guide and installer version used.

For an offline install add `--wheel /absolute/aureon_inference-0.1.0-py3-none-any.whl`
and `--wheelhouse /absolute/archives`; that directory must contain all archives
named by the two generation locks, including the pinned source distributions.
Then run `aureon-engine doctor` as shown in the first-result guide, supply the
official assets and prepare your own Sentinel-2 imagery.

## Tasks

| Task | Successful result | Record |
| --- | --- | --- |
| Install | Hash-verified wheel, dependency checks and CPU walkthrough pass | OS/Python, elapsed setup/download time, error codes, assistance |
| Prepare imagery | Declared bands/units/mask/grid pass input checks | Active minutes, corrections, unsupported input encountered |
| Generate one result | Completed job and saved COG/provenance verified | Model recipe, artifact identities, elapsed time, completion or failure |
| Inspect in GIS | Expected CRS, extent, nodata and band semantics; synthetic label visible | Tool used, interpretation errors, task usefulness |
| Recover/reuse | Status/download replay for the same job without duplicate execution | Assistance, retries, duplicate work or inconsistent artifacts |

Model execution is explicit in the guide. CPU preparation is a successful
installation exercise, not a generated EO result. A whole scene can contain
multiple native tiles; check coverage and seams before downstream use.

## Feedback template

Copy this into a local note or [installation issue](https://github.com/LeChiffreVol2/aureon/issues/new/choose).
Share only what you choose. Exclude API keys, private imagery, customer names
and personal filesystem paths. Report vulnerabilities privately via
[SECURITY.md](../SECURITY.md).

```text
Role/experience: GeoAI practitioner / general developer; EO and Python experience
Environment: OS, Python, GPU/driver if used
Artifact: wheel SHA256 and source commit
Task: install / prepare / generate / inspect / recover
Outcome: completed-unassisted / completed-assisted / failed / abandoned
Elapsed seconds:
Active user seconds:
Download/wait seconds:
Errors and assistance needed:
What was unclear:
Result useful for your intended task? Why:
Ease rating: 1 (very difficult) to 5 (very easy)
Sanitized receipt/error excerpt:
```

A first small group finds friction; it does not establish a population-wide
success rate. Report the number of people, attempted tasks and failures alongside
any percentage. See [evaluation methods](EVALUATION.md).
