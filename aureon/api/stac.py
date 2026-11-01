"""Deterministic offline STAC asset selection; never a network or raster reader."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import cast
from urllib.parse import unquote, urljoin, urlsplit

from .eo import ConditioningSpec, _label, _utc


@dataclass(frozen=True)
class AssetBand:
    """Explicit asset key/band and optional expected whole-content identity."""
    asset_key: str
    band_index: int
    sha256: str | None = None

    def __post_init__(self) -> None:
        _label(self.asset_key, 'asset_key')
        if type(self.band_index) is not int or not 1 <= self.band_index <= 64:
            raise ValueError('band_index must be an integer in [1,64]')
        if self.sha256 is not None and (not isinstance(self.sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.sha256)):
            raise ValueError('Expected SHA-256 must be lowercase when supplied; no bytes are verified here')


@dataclass(frozen=True)
class STACSelection:
    observation: ConditioningSpec
    bands: tuple[AssetBand, ...]
    valid_mask: AssetBand
    mask_meaning: str

    def __post_init__(self) -> None:
        if (type(self.observation) is not ConditioningSpec or type(self.bands) is not tuple
                or any(type(v) is not AssetBand for v in self.bands)
                or len(self.bands) != len(self.observation.bands)):
            raise ValueError('Select exactly one asset band for each declared observation band, in order')
        if len({(v.asset_key, v.band_index) for v in self.bands}) != len(self.bands):
            raise ValueError('Duplicate selected raster band')
        if type(self.valid_mask) is not AssetBand or self.mask_meaning != 'one_is_valid_zero_is_invalid':
            raise ValueError('Supply a separate explicit binary validity mask selection and meaning')
        if self.valid_mask.asset_key in {v.asset_key for v in self.bands}:
            raise ValueError('Validity requires a distinct asset; embedded masks are not resolved by this profile')


@dataclass(frozen=True)
class ResolvedAssetBand:
    asset_key: str
    band_index: int
    expected_sha256: str | None
    href: str = field(repr=False)
    identity_verified: bool = False
    reader_validator_required: bool = True


@dataclass(frozen=True)
class ResolvedObservation:
    observation: ConditioningSpec
    item_id: str
    item_sha256: str
    bands: tuple[ResolvedAssetBand, ...]
    valid_mask: ResolvedAssetBand
    mask_meaning: str
    asset_bytes_read: int = 0


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(name + ' must be a JSON object')
    return cast(dict[str, object], value)


def _url(value: str) -> str:
    parts = urlsplit(value)
    if (parts.scheme != 'https' or not parts.hostname or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment or '\\' in value
            or any(ord(c) <= 32 or ord(c) == 127 for c in value)):
        raise ValueError('Only unsigned HTTPS asset references are admitted; credentials must stay outside provenance')
    # This is syntax validation only, not DNS/host authorization for a future fetch.
    try:
        parts.port
    except ValueError as error:
        raise ValueError('Invalid HTTPS authority') from error
    return value


def _href(value: object, *, base_href: str | None, local_root: Path | None) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError('Selected asset must have a bounded href')
    decoded = unquote(value)
    if ('\\' in decoded or any(ord(c) <= 32 or ord(c) == 127 for c in decoded)
            or urlsplit(decoded).query or urlsplit(decoded).fragment):
        raise ValueError('Ambiguous or credential-bearing asset href')
    parsed = urlsplit(value)
    if parsed.scheme:
        return _url(value)
    if parsed.netloc or value.startswith('/'):
        raise ValueError('Asset references must be explicit HTTPS URLs or contained relative paths')
    if base_href is not None:
        return _url(urljoin(_url(base_href), value))
    if local_root is None:
        raise ValueError('Relative href requires an explicit HTTPS Item URL or local asset root')
    # Lexical containment only; a reader must additionally reject symlinks/reparse points.
    parts = decoded.split('/')
    if any(part in ('', '.', '..') or ':' in part for part in parts):
        raise ValueError('Local asset href must be a normalized contained relative path')
    root = local_root.absolute()
    return str(root.joinpath(*parts))


def _projection(asset: dict[str, object], properties: dict[str, object], spec: ConditioningSpec) -> None:
    if any(record.get(key) is not None for record in (asset, properties) for key in ('proj:wkt2', 'proj:projjson')):
        raise ValueError('This resolver does not interpret WKT2/PROJJSON; explicitly reconcile projection metadata first')
    for key in ('proj:shape', 'proj:transform', 'proj:code', 'proj:epsg'):
        if key not in asset and key not in properties:
            continue
        value = asset.get(key, properties.get(key))
        if value is None:
            raise ValueError('Explicit null projection cannot validate a georeferenced observation')
        if key == 'proj:shape':
            if not isinstance(value, list) or any(type(n) is not int for n in value) or value != list(spec.grid.shape):
                raise ValueError('Declared asset shape conflicts with the selected grid')
        if key == 'proj:transform':
            if (not isinstance(value, list) or any(type(n) not in (int, float) for n in value)
                    or (value != list(spec.grid.transform) and value != [*spec.grid.transform, 0, 0, 1])):
                raise ValueError('Declared asset affine transform conflicts with the selected grid')
        if key == 'proj:code' and value != spec.grid.crs:
            raise ValueError('Declared asset CRS conflicts with the selected grid')
        if key == 'proj:epsg' and (type(value) is not int or 'EPSG:' + str(value) != spec.grid.crs):
            raise ValueError('Declared asset EPSG conflicts with the selected grid')


def _snapshot(item: Mapping[str, object]) -> tuple[dict[str, object], str]:
    """Bound plain structure and individual strings before incremental encoding."""
    if not isinstance(item, Mapping) or len(item) > 4096:
        raise ValueError('Item must be a bounded mapping')
    document = dict(item)
    pending: list[tuple[object, int]] = [(document, 0)]
    nodes = 0
    string_bytes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if depth > 32 or nodes > 50000:
            raise ValueError('Item nesting/node budget exceeded')
        if isinstance(value, str):
            if len(value) > 65536:
                raise ValueError('Individual metadata string exceeds 64 Ki characters')
            string_bytes += len(value.encode('utf-8'))
            if string_bytes > 4 * 1024 * 1024:
                raise ValueError('Item exceeds metadata budget')
        elif type(value) is dict:
            if len(value) > 4096 or any(type(key) is not str for key in value):
                raise ValueError('JSON object exceeds key budget or has non-string keys')
            pending.extend((entry, depth + 1) for pair in value.items() for entry in pair)
        elif type(value) is list:
            if len(value) > 4096:
                raise ValueError('JSON array exceeds element budget')
            pending.extend((entry, depth + 1) for entry in value)
        elif value is not None and type(value) not in (bool, int, float):
            raise ValueError('Item must contain plain JSON values')
        elif type(value) is int and value.bit_length() > 64:
            raise ValueError('Metadata integer exceeds 64 bits')
    chunks = []
    total = 0
    for chunk in json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False).iterencode(document):
        total += len(chunk.encode('utf-8'))
        if total > 4 * 1024 * 1024:
            raise ValueError('Item exceeds the 4 MiB encoded metadata budget')
        chunks.append(chunk)
    encoded = ''.join(chunks)
    return _object(json.loads(encoded), 'Item'), hashlib.sha256(encoded.encode()).hexdigest()


def resolve_stac_item(item: Mapping[str, object], *, selections: tuple[STACSelection, ...],
                      base_href: str | None = None, local_root: Path | None = None) -> tuple[ResolvedObservation, ...]:
    """Select a bounded STAC 1.0/1.1 Item subset without reading any asset.

    Supports explicit single-instant Items and binary mask assets. This is not a
    complete STAC/extension validator, catalog crawler, cloud mask decoder or COG
    loader. Projection declarations are checked against the caller's grid.
    Expected hashes are assertions until verified by a separate bounded reader.
    """
    if ((base_href is None) == (local_root is None)
            or (local_root is not None and not isinstance(local_root, Path))):
        raise ValueError('Supply exactly one explicit Item base URL or local asset root')
    if base_href is not None:
        _url(base_href)
    if (type(selections) is not tuple or not 1 <= len(selections) <= 16
            or any(type(selection) is not STACSelection for selection in selections)):
        raise ValueError('Supply a bounded tuple of typed STAC selections')
    if len({s.observation.modality for s in selections}) != len(selections):
        raise ValueError('Duplicate observation modality')
    try:
        document, item_digest = _snapshot(item)
    except (TypeError, RecursionError, OverflowError) as error:
        raise ValueError('Item must contain bounded plain JSON data') from error
    if (document.get('type') != 'Feature' or document.get('stac_version') not in ('1.0.0', '1.1.0')
            or 'geometry' not in document or not isinstance(document.get('links'), list)):
        raise ValueError('Expected an explicit STAC 1.0/1.1 Item')
    item_id = document.get('id')
    if not isinstance(item_id, str):
        raise ValueError('Item id is required')
    _label(item_id, 'Item id')
    if document['geometry'] is None:
        if 'bbox' in document:
            raise ValueError('A null STAC geometry must not have a bbox')
    else:
        geometry = _object(document['geometry'], 'geometry')
        if geometry.get('type') not in ('Point', 'MultiPoint', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon'):
            raise ValueError('Unsupported Item geometry')
        bbox = document.get('bbox')
        if not isinstance(geometry.get('coordinates'), list) or not isinstance(bbox, list) or len(bbox) not in (4, 6):
            raise ValueError('Non-null geometry needs coordinates and a bounding box')
    properties = _object(document.get('properties'), 'properties')
    if 'bands' in properties or 'raster:bands' in properties:
        raise ValueError('Item-level band arrays require explicit reconciliation into selected assets')
    instant = properties.get('datetime')
    if not isinstance(instant, str):
        raise ValueError('This resolver requires a single explicit acquisition instant; intervals are unsupported')
    _utc(instant)
    assets = _object(document.get('assets'), 'assets')
    if not 1 <= len(assets) <= 1024:
        raise ValueError('Item requires 1..1024 assets')
    identities: dict[str, str | None] = {}
    for selection in selections:
        for binding in (*selection.bands, selection.valid_mask):
            if binding.asset_key in identities and identities[binding.asset_key] != binding.sha256:
                raise ValueError('Conflicting expected identities for the same asset')
            identities[binding.asset_key] = binding.sha256
    results = []
    for selection in selections:
        spec = selection.observation
        if spec.acquired_at != instant:
            raise ValueError('Observation acquisition time must explicitly match the selected Item')

        def resolve(binding: AssetBand, band_name: str | None) -> ResolvedAssetBand:
            asset = _object(assets.get(binding.asset_key), 'selected asset')
            if asset.get('type') not in ('image/tiff', 'image/tiff; application=geotiff',
                                       'image/tiff; application=geotiff; profile=cloud-optimized'):
                raise ValueError('Selected asset must explicitly declare a GeoTIFF media type')
            _projection(asset, properties, spec)
            # Raster2 fields may be inherited at Item/asset level. Reject
            # non-identity declarations at either level rather than guessing
            # which conversion the caller intended to apply.
            for record in (properties, asset):
                scale, offset = record.get('raster:scale', 1), record.get('raster:offset', 0)
                if type(scale) not in (int, float) or type(offset) not in (int, float) or scale != 1 or offset != 0:
                    raise ValueError('Item/asset scale or offset requires explicit preprocessing')
                if band_name is not None and 'unit' in record and record['unit'] != spec.units:
                    raise ValueError('Item/asset unit conflicts with observation units')
            for collection, scale_key, offset_key in (('raster:bands', 'scale', 'offset'),
                                                       ('bands', 'raster:scale', 'raster:offset')):
                raster_bands = asset.get(collection)
                if raster_bands is None:
                    continue
                if not isinstance(raster_bands, list) or binding.band_index > len(raster_bands):
                    raise ValueError('Selected band is outside declared raster bands')
                band = _object(raster_bands[binding.band_index - 1], 'raster band')
                scale, offset = band.get(scale_key, 1), band.get(offset_key, 0)
                if type(scale) not in (int, float) or type(offset) not in (int, float) or scale != 1 or offset != 0:
                    raise ValueError('Non-identity scale/offset requires explicit preprocessing outside this resolver')
                if band_name is not None and 'unit' in band and band['unit'] != spec.units:
                    raise ValueError('Declared band units conflict with observation units')
            for collection in ('eo:bands', 'bands'):
                eo_bands = asset.get(collection)
                if eo_bands is None:
                    continue
                if not isinstance(eo_bands, list) or binding.band_index > len(eo_bands):
                    raise ValueError('Selected band is outside declared EO bands')
                band = _object(eo_bands[binding.band_index - 1], 'EO band')
                if band_name is not None and 'name' in band and band['name'] != band_name:
                    raise ValueError('Declared band name conflicts with explicit band selection')
            return ResolvedAssetBand(binding.asset_key, binding.band_index, binding.sha256,
                                     _href(asset.get('href'), base_href=base_href, local_root=local_root))

        bands = tuple(resolve(binding, name) for binding, name in zip(selection.bands, spec.bands, strict=True))
        results.append(ResolvedObservation(spec, item_id, item_digest, bands,
                                          resolve(selection.valid_mask, None), selection.mask_meaning))
    return tuple(results)
