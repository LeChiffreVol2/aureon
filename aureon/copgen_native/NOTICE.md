# COP-GEN integration notices

This AUREON adapter loads separately supplied, hash-pinned upstream source and
weights. It does not embed or download COP-GEN source, checkpoints, VAEs or data.
Do not package the internal acceptance assets or evidence directories with it.

- COP-GEN code: Miguel Espinosa and contributors,
  https://github.com/miquel-espinosa/COP-GEN,
  revision `453f829234abd7f1cde8a6ad2b51968c37865204`.
  Its repository LICENSE declares Apache-2.0. The retained source contains vendored
  dependencies and unrelated legacy packaging metadata; this notice is not a
  substitute for their license/notice review if upstream code is redistributed.
- COP-GEN backbone: https://huggingface.co/mespinosami/copgen-base,
  revision `2d087d47d973494ec281f8093967d9d70d5f8682`.
  The model card declares CC-BY-4.0.
- COP-GEN VAEs: https://huggingface.co/mespinosami/copgen-vaes,
  revision `13f55445a0e25dda791c1691249d9d913ba554b4`.
  The model card declares CC-BY-4.0. This route uses only the four-band S2L2A EMA
  VAE and the VV,VH S1RTC EMA VAE.
- Model paper: Espinosa, Gmelich Meijling, Marsocci, Crowley and Czerkawski (2026),
  *COP-GEN: Latent Diffusion Transformer for Copernicus Earth Observation Data*,
  https://arxiv.org/abs/2603.03239.

Preserve upstream attribution and applicable notices when separately sharing
assets or derivatives. COP-GEN-Beta is not this model. Generated S1RTC is a
synthetic prediction, not a Sentinel-1 observation or a TerraMind S1GRD output.
