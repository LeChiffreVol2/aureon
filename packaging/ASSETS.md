# Official pinned assets

AUREON never downloads these assets automatically. Obtain them separately
from the official revision-specific sources below under their model/data terms.
Do not download the upstream samples for these examples; use your own imagery.
Do not execute a repository helper or enable remote Python code.

Start with the **TerraMind base checkpoint and S1GRD tokenizer** for generation.
Together they are about 2.67 GB. The [first-result guide](START-HERE.md#supply-assets)
provides download and checksum commands. KAO-TerraMind B is only needed for the
separate segmentation route.

`aureon-engine assets --model MODEL_ID` returns these identities as JSON.
Check the complete local file with `sha256sum FILE` before use.

## terramind-v1-base — checkpoint

- [Official file](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base/resolve/fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a/TerraMind_v1_base.pt) · [model card](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base/blob/fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a/README.md)
- Filename: `TerraMind_v1_base.pt`
- Revision: `fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a`
- Bytes: 1518713958
- SHA256: `83c3a0938067c83867a46e564443c2fa38383bf4f966d931b11cb025b847d7ec`

## terramind-v1-base — tokenizer

- [Official file](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1GRD/resolve/38937fd35012c2dcf81987b400cec725b5eb0466/TerraMind_Tokenizer_S1GRD.pt) · [model card](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1GRD/blob/38937fd35012c2dcf81987b400cec725b5eb0466/README.md)
- Filename: `TerraMind_Tokenizer_S1GRD.pt`
- Revision: `38937fd35012c2dcf81987b400cec725b5eb0466`
- Bytes: 1146593994
- SHA256: `d83e43ccbdbfb787724d42a4e09dce6abb122e8f286889a379af59ffbe979ed1`

## kao-terramind-b — checkpoint

- [Official file](https://huggingface.co/kaopanboonyuen/KAO-TerraMind-B/resolve/9698b737e6fdbdbc3f404a4a64cb70799472c2c8/terramind_v1_kaoB_best.pt) · [model card](https://huggingface.co/kaopanboonyuen/KAO-TerraMind-B/blob/9698b737e6fdbdbc3f404a4a64cb70799472c2c8/README.md)
- Filename: `terramind_v1_kaoB_best.pt`
- Revision: `9698b737e6fdbdbc3f404a4a64cb70799472c2c8`
- Bytes: 398251303
- SHA256: `d5cd2c97ce853971c847eb1a423c6d5ba996eff8ebc0e1e64d116bed64e5e23a`

KAO-TerraMind B is converted with the installed `aureon prepare-vllm` command
in [START-HERE.md](START-HERE.md). TerraMind uses the two pinned files directly
through `aureon-engine generation-config`. No manual operator bundle or
private scripts are required. A/C identities remain available through the
metadata CLI, with their separate earlier-artifact support scope.

Checksums are identity checks, not grants of ownership or redistribution.
AUREON ships no weights, tokenizer payloads, upstream sample imagery or private
GISTDA data. The primary metadata above was rechecked on2026-10-28 without
fetching asset content. See [source attribution](review/THIRD-PARTY-NOTICES.md).
