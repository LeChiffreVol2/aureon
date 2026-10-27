# AUREON

**Specialized inference infrastructure for Earth foundation models.**

AUREON Foundation by Norvique provides a Python library, CLI and authenticated
local API for geospatial model inference. It turns user-owned Sentinel-2 inputs
into georeferenced results with explicit bands, units, masks and provenance.

**Public alpha v0.1.0** · [Release](https://github.com/LeChiffreVol2/aureon/releases/tag/v0.1.0)
· [Documentation](docs/README.md) · [Release notes](docs/releases/v0.1.0.md)

| Workflow | Input | Output |
| --- | --- | --- |
| KAO-TerraMind B segmentation | Prepared twelve-band Sentinel-2 L2A | Class and validity-mask COGs with provenance |
| TerraMind 1.0 base generation | All-valid 224×224 S2L2A crop | Synthetic S1GRD VV/VH COG, raw output and provenance |

The Engine includes resident execution, exact eligible encoder reuse, guarded
decoder CUDA Graphs, FP32 reference checks and verified artifact delivery.
Model execution is explicit. Installation, import and inspection never download
weights or execute a model. Assets and imagery are supplied separately.

## Install

Use **Linux x86_64 / Python 3.12**, including WSL2. Scene and generation use
separate locked environments. Download the release kit and verify its SHA256:

```sh
mkdir aureon-download && cd aureon-download
curl -fL https://github.com/LeChiffreVol2/aureon/releases/download/v0.1.0/aureon-0.1.0-public-alpha.zip \
  -o aureon-0.1.0-public-alpha.zip
printf '%s  %s\n' '3e844b14ba7a4a8138a3386ab2dd928e426ffcbf018f3183acf91a854297db07' \
  'aureon-0.1.0-public-alpha.zip' | sha256sum --check
unzip aureon-0.1.0-public-alpha.zip
cd aureon-0.1.0-public-alpha
sha256sum --check SHA256SUMS
```

Then follow the [installation and first-result guide](packaging/START-HERE.md).
It covers the exact dependency locks, pinned asset sources, creating your own
KAO-TerraMind bundle, preparing user-owned Sentinel-2 input and an explicit first
job. No operator-created bundle, private scripts or development checkout is needed.
The kit contains the wheel, matching source archive, examples, locks and verifier.

The official wheel is `aureon-inference==0.1.0`, SHA256
`a11b67e859ce0e2e89277bf5e35e1b8d917bfa0aa577617bf9e66d8860c2c1bf`.
Use these release assets rather than an unverified package-index name.

## Use the Engine

After installation:

```sh
aureon-engine models
aureon-engine assets --model terramind-v1-base
aureon-engine check --environment generation
```

- [Python and resident sessions](packaging/PUBLIC-API.md)
- [Input preparation and geospatial contract](packaging/INPUTS.md)
- [Local upload → job → verified download](packaging/ASYNC-SERVICE.md)
- [Pinned external model assets](packaging/ASSETS.md)
- [Supported models and limitations](docs/MODEL-SUPPORT.md)
- [Errors and support](packaging/SUPPORT.md)

The exact release wheel passed local KAO-TerraMind B512 and TerraMind224 HTTP
functional acceptance on RTX 3070 8 GiB / WSL, retaining FP32 reference parity,
replay without additional inference or usage, and clean shutdown. This alpha
does not imply hosted availability, a production SLA, arbitrary GPU support or
scientific validation on new scenes. Generated S1GRD is synthetic, not observed SAR.
KAO-TerraMind A/C retain their explicitly earlier-artifact support scope.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Standard-library CPU contracts and static
checks run without neural runtimes, weights, GPU jobs or benchmarks. The
repository's `aureon/` files match all 139 runtime/resource payloads in the
official wheel byte for byte. Public documentation was updated after the frozen
archive was built; a source rebuild is not the official wheel's artifact identity.

## License and contact

AUREON-owned contributions are licensed under [Apache-2.0](LICENSE) by
**Norvique Company**. AUREON Foundation is a project under the company, not a
separate incorporated foundation. Retain the [NOTICE](NOTICE) and
[third-party attribution](packaging/review/THIRD-PARTY-NOTICES.md).

The [approved scoped rights statement](docs/RIGHTS.md) records the treatment of
upstream constants/class names and the preserved README discrepancy. Frozen
archive review text predates that approval; the statement supersedes its pending
owner-disposition status only. Weights, imagery and private training data are
not included or licensed by this release.

Contact: **norviccompanylimited@gmail.com**. For vulnerabilities, use the private
reporting procedure in [SECURITY.md](SECURITY.md); do not open a public issue with
sensitive details. No response-time or remediation SLA is promised.
