# AUREON

**Runtime selection:** use the [locked integrated Best package](https://github.com/LeChiffreVol2/aureon/blob/main/packaging/CANDIDATE-QUICKSTART.md), `e09e710c...`, v0.1.0. Original `0fa5ef2b` measurements remain historical evidence. The installed package and current examples do not imply new performance or hosted qualification.

AUREON Foundation by Norvique provides resident inference and geospatial
delivery for multimodal and generative Earth models.

## Start here

- [Install the locked package](https://github.com/LeChiffreVol2/aureon/blob/main/packaging/CANDIDATE-QUICKSTART.md)
- [Prepare your input and get a first result](https://github.com/LeChiffreVol2/aureon/blob/main/packaging/START-HERE.md)
- [Whole scenes and repeated jobs](https://github.com/LeChiffreVol2/aureon/blob/main/packaging/BEST-WORKFLOWS.md)
- [Local HTTP and download recovery](https://github.com/LeChiffreVol2/aureon/blob/main/packaging/ASYNC-SERVICE.md)
- [Operator checks](https://github.com/LeChiffreVol2/aureon/blob/main/packaging/SERVICE-OPERATIONS.md)
- [Model support](https://github.com/LeChiffreVol2/aureon/blob/main/packaging/MODEL-SUPPORT.md) and [troubleshooting](https://github.com/LeChiffreVol2/aureon/blob/main/packaging/SUPPORT.md)

The local main package is `e09e710c...`; the published GitHub alpha
`a11b67e8...` is unchanged. PyPI and hosted access remain closed. No weights
or imagery ship in the library. Generation uses its pinned Linux/Python 3.12
environment; other model routes keep their declared dependency profiles.

## Current artifact qualification ledger

The locked integrated package passed 26 exact FP32 calls including a four-tile
scene and local HTTP/Cloud delivery, with seven zero-CUDA resident closures.
The separate 14-call diagnostic retained graph/reuse and exact parity; timing
pairs changed direction. Neither result creates a new speedup or universal
non-regression claim. Other families, sustained load, scientific seam quality
and physical multi-GPU retain their separate qualification limits.

AUREON-owned code is Apache-2.0 by Norvique Company; preserve LICENSE, NOTICE
and third-party notices. Models and datasets keep their own terms.
Contact/security: norviccompanylimited@gmail.com.

