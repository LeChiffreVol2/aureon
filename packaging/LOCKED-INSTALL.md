# Dependency locks and compatible environments

Use the [single current AUREON distribution](CANDIDATE-QUICKSTART.md).
Generation dependencies are included by default; no separate feature/core wheel
is needed. `[generation]` and `[geo]` remain compatibility extras on the same
distribution. Plain pip installation uses the normal resolver and build isolation.
The current working delivery remains unpublished; PyPI is not open.

The tested generation stack uses Linux/WSL Python 3.12, Torch 2.10.0,
TorchVision 0.25.0, TerraTorch 1.2.13 and NumPy 2.2.6. Use matching source-archive
locks to reproduce the dependency snapshot, or validate your existing compatible
environment before installing the same wheel with `--no-deps`.

```sh
python -m pip install --require-hashes -r packaging/lock-bootstrap-generation.txt
python -m pip install --require-hashes --no-build-isolation -r packaging/lock-generation-linux-py312-cu128.txt
python -m pip install --no-deps /absolute/aureon_inference-0.1.0-py3-none-any.whl
python -m pip check
```

These bootstrap/no-build-isolation flags belong to locked reproduction, not normal
pip installation. Offline use needs all locked archives via `--no-index --find-links`.
The lock includes source distributions; `dependency-inventory.json` records their
origins and hashes. Installation does not acquire imagery/checkpoints or run models.

KAO-TerraMind full-reference support uses the same distribution with the optional
`scene-legacy` dependencies (including TorchGeo 0.9.0) and that explicit environment
contract. Its previous GPU receipts keep their own artifact scope. The original
published alpha's scene stack uses different Torch/NumPy versions and is not a
compatible environment into which to force this generation wheel. Keep historical
release instructions with their release; they are not additional current builds.

vLLM is not required by the public inference workflow. Old core-profile build
tooling and comparator dependencies are retained for development/reproduction,
not offered as competing current installations. A shared version string is not
an artifact identity: verify the wheel/source against the accompanying receipt.
