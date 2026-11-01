"""TerraMind 1.0 identities and modes; listing does not import a model or download.

Official repository metadata was inspected on 2026-10-29. These identities say
which files to load, not that every variant has passed numerical qualification.
The existing base and S1GRD pins remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from .common import MODEL_SHA, TOKENIZER_SHA

TerraMindSize = Literal["tiny", "small", "base", "large"]
TerraMindMode = Literal["generation", "backbone", "tim"]
SIZES: tuple[TerraMindSize, ...] = ("tiny", "small", "base", "large")


@dataclass(frozen=True)
class AssetPin:
    repo_id: str
    revision: str
    filename: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class TerraMindSpec:
    model_id: str
    size: TerraMindSize
    mode: TerraMindMode
    upstream_name: str
    checkpoint: AssetPin


CHECKPOINTS: Mapping[str, AssetPin] = MappingProxyType({
    "tiny": AssetPin("ibm-esa-geospatial/TerraMind-1.0-tiny",
        "2b5ac0a3ed7dd7e922ccfd595b56607f342df343", "TerraMind_v1_tiny.pt",
        "e56ea9ebcd4451078b9ca4893d5cd8a89bbee376ae16829c3e7fbbbc76de0eba", 211873402),
    "small": AssetPin("ibm-esa-geospatial/TerraMind-1.0-small",
        "960f7549bfe9d7a08946860042f83badd604c779", "TerraMind_v1_small.pt",
        "755e9cce9483fd61334ef66c79f805406db5151a8b44a685c8fbbe023c684701", 504231662),
    "base": AssetPin("ibm-esa-geospatial/TerraMind-1.0-base",
        "fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a", "TerraMind_v1_base.pt",
        MODEL_SHA, 1518713958),
    "large": AssetPin("ibm-esa-geospatial/TerraMind-1.0-large",
        "3ca567c5494310528269402a96ba4f13ceefc87c", "TerraMind_v1_large.pt",
        "a1c6b567ce6862c7ac07181551add840edce52f652abccfe5f17d23544060f81", 3787890978),
})

PINNED_S1GRD = AssetPin("ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1GRD",
    "38937fd35012c2dcf81987b400cec725b5eb0466", "TerraMind_Tokenizer_S1GRD.pt",
    TOKENIZER_SHA, 1146593994)
TOKENIZER_ASSETS: Mapping[str, AssetPin] = MappingProxyType({
    "S1GRD": PINNED_S1GRD,
    "S2L2A": AssetPin("ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S2L2A",
        "5a3f80664e0e701232c9526ca21eb7442a40f4be", "TerraMind_Tokenizer_S2L2A.pt",
        "fbec7e8fb689b28482e19a0b727f6adcef7ca4f229aa8bd3f2b3e9da66cf0263", 1157408074),
    "S1RTC": AssetPin("ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1RTC",
        "ea54d732d904c804b01e04cb5d85a462598bb9d0", "TerraMind_Tokenizer_S1RTC.pt",
        "d86e2bce7827f804f5a648aeb87a074cdf4b91126616e8fc1b66bebaec779ad0", 1146593994),
    "DEM": AssetPin("ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM",
        "aba0a5650a19eb82283e52ce59320c38c41f1e56", "TerraMind_Tokenizer_DEM.pt",
        "3d25bdf589391b6d15bb54177af7894c0d19a056e345500334f8db63dff1471b", 1145511376),
    "LULC": AssetPin("ibm-esa-geospatial/TerraMind-1.0-Tokenizer-LULC",
        "af6b7c73a36994e25df156e793901969efe88ec9", "TerraMind_Tokenizer_LULC.pt",
        "0e318c41490b472897b328256227c049aba8c1671a88c1ef3a1fa5026b3e0fd3", 735693536),
    "NDVI": AssetPin("ibm-esa-geospatial/TerraMind-1.0-Tokenizer-NDVI",
        "e1bd3b4d18023a7438221dbd6953e000cdc2dea5", "TerraMind_Tokenizer_NDVI.pt",
        "f574bfd0be0f38215c7bd02e7f763527a37028bc3f61086cb28a2c813ca52c5d", 1145511981),
})


def generation_model_id(size: TerraMindSize) -> str:
    if size not in SIZES:
        raise ValueError("TerraMind size must be tiny, small, base or large")
    return "terramind-v1-" + size


GENERATION_MODELS = tuple(generation_model_id(size) for size in SIZES)
BACKBONE_MODELS = tuple(model + "-backbone" for model in GENERATION_MODELS)
TIM_MODELS = tuple(model + "-tim" for model in GENERATION_MODELS)


def list_terramind_specs() -> tuple[TerraMindSpec, ...]:
    result = []
    modes: tuple[TerraMindMode, ...] = ("generation", "backbone", "tim")
    for mode in modes:
        for size in SIZES:
            suffix = "" if mode == "generation" else "-" + mode
            upstream_suffix = {"generation": "_generate", "backbone": "", "tim": "_tim"}[mode]
            result.append(TerraMindSpec(generation_model_id(size) + suffix, size, mode,
                "terramind_v1_" + size + upstream_suffix, CHECKPOINTS[size]))
    return tuple(result)


def get_terramind_spec(model_id: str) -> TerraMindSpec:
    for spec in list_terramind_specs():
        if spec.model_id == model_id:
            return spec
    raise ValueError("Unknown TerraMind 1.0 model_id: " + str(model_id))
