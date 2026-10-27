# v0.1.0 rights and attribution statement

**Approved disposition, 2026-10-28.** Norvique Company approved reliance on the
pinned KAO-TerraMind C repository's Apache-2.0 declaration for the identified
constants, class taxonomy and compatibility conventions carried by AUREON.
AUREON-owned contributions remain under [Apache-2.0](../LICENSE), with
[NOTICE](../NOTICE) and [third-party source attribution](../packaging/review/THIRD-PARTY-NOTICES.md).

## Included material covered by this disposition

The scope is `aureon/catalog.py` and `aureon/recipe.py`:

- Twelve ordered Sentinel-2 band identifiers.
- Twelve normalization means and twelve standard deviations.
- Ten ordered textual class names.
- Class-count and ignored/nodata compatibility conventions.

The ten class names are taxonomy text, not annotation rasters or training labels.
The statistics retain their stated TerraMind pre-training origin. AUREON does
not claim authorship of upstream statistics or transfer of third-party ownership.

## Primary sources and preserved discrepancy

The [KAO-TerraMind C model card](https://huggingface.co/kaopanboonyuen/KAO-TerraMind-C/blob/4e7252d3c83b3e877744da214005184a4133883f/README.md)
at revision `4e7252d3c83b3e877744da214005184a4133883f` and its
[API metadata](https://huggingface.co/api/models/kaopanboonyuen/KAO-TerraMind-C/revision/4e7252d3c83b3e877744da214005184a4133883f)
declare Apache-2.0. The [helper](https://huggingface.co/kaopanboonyuen/KAO-TerraMind-C/blob/4e7252d3c83b3e877744da214005184a4133883f/kao_terramind_inference.py)
credits Teerapong Panboonyuen but has no separate file-level license declaration.
Card and API corroborate one repository declaration, not two independent grants.

The related [GitHub LICENSE](https://github.com/kaopanboonyuen/KAO-TerraMind/blob/2d8a9c8953e9bfbec01fae4d80ce5ffcba67c777/LICENSE)
at `2d8a9c8953e9bfbec01fae4d80ce5ffcba67c777` contains Apache-2.0, while its
[README](https://github.com/kaopanboonyuen/KAO-TerraMind/blob/2d8a9c8953e9bfbec01fae4d80ce5ffcba67c777/README.md)
retains a code/weights licensing TODO. This discrepancy remains disclosed;
the owner's disposition is not an upstream clarification or endorsement.
Existing license copies, author/research attribution and source mappings are retained.

## Exclusions

This disposition does not cover model/tokenizer weight redistribution, satellite
imagery, private GISTDA data, annotation rasters or training chips, redistribution
of the complete helper, blanket clearance of unrelated code/training-data rights,
or upstream endorsement. Those materials are not included in this source release.
External model listings are identity references, not grants of additional rights.

## Exact release and historical embedded text

This statement applies to the v0.1.0 wheel SHA256
`a11b67e859ce0e2e89277bf5e35e1b8d917bfa0aa577617bf9e66d8860c2c1bf`,
source archive SHA256
`44b3d4955673bbda95bea4d0ee88c7105a7840174f8cb49dc3e33c846b27dfa0`,
and kit SHA256
`3e844b14ba7a4a8138a3386ab2dd928e426ffcbf018f3183acf91a854297db07`.

The frozen archives were built before the owner decision and retain their
historical pre-decision review wording. This public statement supersedes **only
the pending owner-disposition status**. It does not rewrite embedded text, alter
license terms or credits, resolve the upstream discrepancy, or enlarge numerical,
hardware, model/data or hosting qualification. Keep this statement with the release.
No runtime or dependency change, rebuild or new benchmark was needed for approval.

**AUREON Foundation by Norvique** denotes a project under **Norvique Company**,
not a separately incorporated foundation. Contact and private security reporting:
**norviccompanylimited@gmail.com**. No response-time SLA is promised.
