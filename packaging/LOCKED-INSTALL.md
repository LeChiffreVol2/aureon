# Exact dependency installation

Follow [START-HERE.md](START-HERE.md) for the complete installed workflow.
Use Linux x86_64, CPython 3.12, one new environment per profile, and the exact wheel
identified in the release kit's `ARTIFACTS.json`. No editable checkout is required.

| Workflow | Bootstrap | Dependency lock | Runtime environment |
| --- | --- | --- | --- |
| KAO-TerraMind B | lock-bootstrap-scene.txt | lock-scene-linux-py312-cu130.txt | scene |
| TerraMind base | lock-bootstrap-generation.txt | lock-generation-linux-py312-cu128.txt | generation |
| KAO-TerraMind A/C | generation bootstrap | generation lock | scene-legacy |

Install bootstrap first, then the selected lock with `--require-hashes
--no-build-isolation`, then the wheel with `--no-deps --no-index`.
The locks include source distributions where no compatible upstream wheel exists.
The host venv bootstraps pip; record `python --version` and `python -m pip --version`
when reporting a problem. Keep installed dependencies out of source and release
archives. `dependency-inventory.json` records origins, hashes and declared
dependencies. Hashes establish identity, not absence of vulnerabilities.

CUDA-enabled dependencies consume several GiB. A compatible NVIDIA driver is
separate from this kit. The verified model machine is RTX3070/WSL; successful pip
installation alone does not qualify a new driver/device. No automatic environment
switching occurs. Do not merge locks, add all extras, or silently upgrade a failed
pin. Use [support](SUPPORT.md) for actionable diagnostics.

The release manifest records actual installed CPU checks. Final model/HTTP
acceptance is summarized in the [support matrix](../docs/MODEL-SUPPORT.md); it is not inferred from packaging.
