# Contributing

Use Python3.12 and the exact source companion. Changes to numerical recipes,
preprocessing identities, reuse/graph guards or dependency pins need focused
compatibility evidence. Do not rewrite prior acceptance records or infer model
quality from synthetic tests. Keep checkpoints, imagery, keys and environments
out of the source archive.

Run `python -B packaging/ci_cpu.py --output /absolute/fresh/cpu.json` for the
standard-library contracts. Installed layout/public-alpha tests use the selected
locked environment outside the source checkout; they block neural imports and
do not qualify GPU performance. Review `packaging/runtime-files.json` and the
explicit support-file allowlist before adding runtime/package resources.

Describe concrete behavior, the tested scope and remaining limitations. Contact
norviccompanylimited@gmail.com for contributions/support; report security issues
privately through [SECURITY.md](SECURITY.md). Proposed contributions are reviewed
under the accompanying Apache-2.0 terms; do not claim ownership of third-party
code. This public alpha does not imply a hosted SLA. See the [repository contribution guide](../CONTRIBUTING.md).
