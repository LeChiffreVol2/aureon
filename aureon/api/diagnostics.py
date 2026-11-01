"""Read-only dependency/asset diagnostics; never import a neural runtime."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import shutil
from typing import Literal

from .assets import list_assets
from .models import inspect_model
from .profiles import inspect_profile


@dataclass(frozen=True)
class DiagnosticIssue:
    code: str
    severity: Literal['error', 'warning', 'info']
    message: str
    remedy: str


def _identity(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Asset changed while hashing')
    return digest.hexdigest()


def diagnose_runtime(model_id: str, *, assets_dir: str | Path | None = None,
                     environment: str | None = None, verify_assets: bool = False,
                     modalities: Sequence[str] = ('S2L2A',),
                     tim_modalities: Sequence[str] | None = None,
                     work: str | Path | None = None, min_free_bytes: int = 0) -> dict[str, object]:
    """Inspect pinned prerequisites without downloads, mutations or CUDA calls.

    Asset filenames are resolved under the explicitly supplied directory. Hashing
    is opt-in because checkpoints are large. A size-only check is never reported
    as identity verification or production/hardware qualification. Work inspection
    reads the nearest existing parent's free space without creating directories.
    """
    if type(verify_assets) is not bool or type(min_free_bytes) is not int or min_free_bytes < 0:
        raise ValueError('verify_assets must be bool and min_free_bytes a nonnegative integer')
    if isinstance(modalities, (str, bytes)) or not modalities or len(set(modalities)) != len(modalities):
        raise ValueError('modalities must be a nonempty sequence of unique names')
    model = inspect_model(model_id)
    mode = str(model.get('mode', 'scene'))
    selected = environment or ('generation' if mode != 'scene' else 'scene' if model_id.endswith('-b') else 'scene-legacy')
    permitted = ('generation',) if mode != 'scene' else tuple(model['full_reference_environments'])
    if selected not in permitted:
        raise ValueError('Environment does not support the selected model mode')
    targets = tuple(tim_modalities) if tim_modalities is not None else ('LULC',) if mode == 'tim' else ()
    if isinstance(tim_modalities, (str, bytes)) or len(set(targets)) != len(targets):
        raise ValueError('tim_modalities must be unique names')
    if mode != 'tim' and targets:
        raise ValueError('tim_modalities requires a TiM model')
    if mode == 'tim' and not targets:
        raise ValueError('TiM requires at least one target modality')
    if any(name not in ('S2L2A', 'S2L1C', 'S2RGB', 'S1GRD', 'S1RTC', 'DEM', 'NDVI', 'LULC') for name in modalities):
        raise ValueError('Unsupported input modality')
    if mode in ('generation', 'scene') and tuple(modalities) != ('S2L2A',):
        raise ValueError('This workflow requires S2L2A input')
    if any(name not in ('S2L2A', 'S1GRD', 'S1RTC', 'DEM', 'LULC', 'NDVI') for name in targets):
        raise ValueError('Unsupported TiM target')
    if (set(modalities) & set(targets)) - {'LULC', 'NDVI'}:
        raise ValueError('TiM input/target overlap requires a tokenized input modality (LULC or NDVI)')
    tokenizers = {'S1GRD'} if mode == 'generation' else set(targets) | (set(modalities) & {'LULC', 'NDVI'})
    environment_report = inspect_profile(selected)
    issues: list[DiagnosticIssue] = []
    if environment_report['status'] != 'passed':
        issues.append(DiagnosticIssue('environment_mismatch', 'error',
            'Python/platform or pinned dependencies differ from the selected profile.',
            str(environment_report['remedy']) + ' Use the reviewed wheel/constraints; PyPI availability is separate.'))
    expected = [row for row in list_assets(model_id)
                if row['role'] == 'checkpoint' or row.get('modality') in tokenizers]
    directory = Path(assets_dir).expanduser().absolute() if assets_dir is not None else None
    checks: list[dict[str, object]] = []
    for pin in expected:
        filename = str(pin['filename'])
        path = directory / filename if directory is not None else None
        check: dict[str, object] = {'role': pin['role'], 'modality': pin.get('modality'),
            'filename': filename, 'expected_bytes': pin['bytes'], 'expected_sha256': pin['sha256'],
            'source_url': pin['source_url'], 'path': str(path) if path is not None else None,
            'size_matches': False, 'identity_verified': False}
        try:
            if path is None or not path.is_file():
                issues.append(DiagnosticIssue('asset_missing', 'error', filename + ' is not available.',
                    'Supply the separately acquired pinned asset under --assets-dir; doctor never downloads it.'))
            else:
                check['actual_bytes'] = path.stat().st_size
                check['size_matches'] = check['actual_bytes'] == pin['bytes']
                if not check['size_matches']:
                    issues.append(DiagnosticIssue('asset_size_mismatch', 'error', filename + ' has an unexpected size.',
                        'Use the exact revision/file shown in this report; do not rename a different checkpoint.'))
                elif verify_assets:
                    actual = _identity(path)
                    check['actual_sha256'] = actual
                    check['identity_verified'] = actual == pin['sha256']
                    if not check['identity_verified']:
                        issues.append(DiagnosticIssue('asset_hash_mismatch', 'error', filename + ' does not match its pin.',
                            'Replace this asset from the declared immutable source; do not bypass identity checks.'))
                else:
                    issues.append(DiagnosticIssue('asset_hash_not_checked', 'warning', filename + ' size matches; identity is unchecked.',
                        'Add --verify-assets to read and hash the file before model execution.'))
        except (OSError, ValueError) as error:
            issues.append(DiagnosticIssue('asset_unreadable', 'error', filename + ': ' + str(error),
                'Check file access and stop concurrent writes, then inspect again.'))
        checks.append(check)
    disk: dict[str, object] = {'status': 'not_requested'}
    if work is not None:
        target = Path(work).expanduser().absolute()
        parent = target
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        try:
            if not parent.is_dir():
                raise ValueError('Existing work ancestor is not a directory')
            free = shutil.disk_usage(parent).free
            disk = {'status': 'passed' if free >= min_free_bytes else 'failed',
                    'ancestor': str(parent), 'free_bytes': free, 'required_free_bytes': min_free_bytes,
                    'directories_created': False, 'write_permission_probed': False}
            if free < min_free_bytes:
                issues.append(DiagnosticIssue('disk_budget', 'error', 'Free disk space is below the requested floor.',
                    'Choose another work volume or reduce the declared output budget; no files were removed.'))
        except (OSError, ValueError) as error:
            disk = {'status': 'failed', 'error': str(error)}
            issues.append(DiagnosticIssue('work_path', 'error', str(error), 'Choose a work path with an accessible directory ancestor.'))
    status = 'blocked' if any(issue.severity == 'error' for issue in issues) else 'incomplete' if any(issue.severity == 'warning' for issue in issues) else 'passed'
    return {'schema': 'aureon.runtime_diagnostics.v1', 'model_id': model_id, 'mode': mode,
            'environment': environment_report, 'status': status, 'assets': checks,
            'modalities': list(modalities), 'tim_modalities': list(targets), 'disk': disk,
            'issues': [asdict(issue) for issue in issues], 'cuda': 'not_probed',
            'model_imported': False, 'model_executed': False, 'files_created': False,
            'scope': 'Pinned prerequisites only; not GPU capacity, input quality, runtime or production qualification'}
