"""Bounded private bundle storage. No model execution or automatic discovery.

Reflink when the filesystem supports isolated copy-on-write bytes, otherwise
copy. Never use writable hardlinks. Retirement is limited to one verified
closed worker's recoverable weight file; all original evidence is retained.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from datetime import datetime, timezone
from time import perf_counter


MEMBERS = ('config.json', 'pytorch_model.bin', 'bundle_receipt.json')


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _regular(path):
    # Walk from an absolute, not pre-resolved path so parent links are rejected.
    path = Path(path).absolute()
    for entry in (path, *path.parents):
        value = entry.lstat()
        if stat.S_ISLNK(value.st_mode) or getattr(value, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Bundle storage paths cannot traverse links/reparse points')
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Expected a regular bundle storage file')
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _read(path):
    _regular(path)
    with path.open('rb') as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('Storage receipt exceeds metadata budget')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Expected a storage receipt object')
    return value


def _private_copy(source, target):
    """Fresh target only; FICLONE failure falls back to bounded byte copying."""
    with source.open('rb') as src, target.open('xb') as dst:
        if sys.platform == 'linux':
            import fcntl
            try:
                fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # Linux FICLONE
                return 'reflink'
            except OSError as error:
                if error.errno not in {errno.EXDEV, errno.EOPNOTSUPP, errno.ENOTTY,
                                      errno.EINVAL, errno.ENOSYS, errno.EPERM}:
                    raise
                dst.seek(0)
                dst.truncate()
                src.seek(0)
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return 'copy'


def snapshot_bundle(source, destination, *, model_id=None):
    """Return (verified bundle, storage receipt); caller owns worker lifecycle."""
    from .runtime import verify_bundle
    source, destination = Path(source).absolute(), Path(destination).absolute()
    stamps = {name: _regular(source/name) for name in MEMBERS}
    original = verify_bundle(source)
    if model_id is not None and original['model_id'] != model_id:
        raise ValueError('Bundle identity differs from the explicitly selected model')
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError('Runtime bundle destination must be fresh')
    # Parent checking also rejects a linked output ancestor before any write.
    for entry in (destination.parent, *destination.parent.parents):
        value = entry.lstat()
        if stat.S_ISLNK(value.st_mode) or getattr(value, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Runtime bundle parent cannot traverse links/reparse points')
    destination.mkdir()
    methods = {}
    for name in MEMBERS:
        methods[name] = _private_copy(source/name, destination/name)
        if _regular(source/name) != stamps[name]:
            raise ValueError('Source bundle changed during snapshot')
        if os.path.samefile(source/name, destination/name):
            raise ValueError('Private runtime bundle must not alias source storage')
    copied = verify_bundle(destination)
    if (copied['files_sha256'] != original['files_sha256']
            or copied['receipt_sha256'] != original['receipt_sha256']
            or any(_regular(source/name) != stamp for name, stamp in stamps.items())):
        raise ValueError('Private runtime bundle differs from verified source')
    return copied, {'schema': 'aureon.bundle_storage.v1', 'canonical_bundle': str(source),
            'runtime_bundle': str(destination), 'methods': methods,
            'logical_bytes': sum(stamp[2] for stamp in stamps.values()),
            'physical_bytes_saved': None, 'files_sha256': copied['files_sha256'],
            'receipt_sha256': copied['receipt_sha256'],
            'policy': 'Private files; retire only after successful shutdown and canonical re-verification'}


def retire_bundle_copy(worker_folder, canonical_bundle):
    """Retire exactly bundle/pytorch_model.bin after persisted successful close.

Call after all model handles are closed. Exceptions preserve the weight file
unless unlink already completed; the fsynced intent then supplies recovery.
No directory scanning, recursive deletion, retention timer or raw-data GC.
"""
    from .runtime import verify_bundle
    started = perf_counter()
    folder, canonical = Path(worker_folder).absolute(), Path(canonical_bundle).absolute()
    target = folder/'bundle'/'pytorch_model.bin'
    report_path, intent_path = folder/'bundle-retirement.json', folder/'bundle-retirement-intent.json'
    if report_path.exists() or intent_path.exists():
        raise FileExistsError('Retirement evidence already exists; reconcile it explicitly')
    target_stamp = _regular(target)
    source_stamps = {name: _regular(canonical/name) for name in MEMBERS}
    if os.path.samefile(target, canonical/'pytorch_model.bin'):
        raise ValueError('Refusing to retire canonical or aliased model storage')
    startup_path, shutdown_path = folder/'startup.json', folder/'shutdown.json'
    startup, shutdown = _read(startup_path), _read(shutdown_path)
    if (startup.get('status') != 'ready' or not startup.get('worker_id')
            or startup['worker_id'] != shutdown.get('worker_id')
            or shutdown.get('schema') not in {'aureon.resident_scene_worker_shutdown.v1',
                                              'aureon.precision_worker_shutdown.v1'}
            or shutdown.get('full_identity_check_passed') is not True
            or shutdown.get('cleanup_passed') is not True or shutdown.get('errors') != []
            or shutdown.get('cuda_allocated_bytes_after_cleanup') != 0
            or shutdown.get('cuda_reserved_bytes_after_cleanup') != 0
            or any(shutdown.get('owned_objects_alive_after_cleanup', {}).values())):
        raise ValueError('Successful matching worker shutdown required for retirement')
    receipts = {name: _hash(folder/name) for name in ('startup.json', 'shutdown.json')}
    original, copied = verify_bundle(canonical), verify_bundle(target.parent)
    declared = startup.get('identity', startup)
    if (copied['files_sha256'] != original['files_sha256']
            or copied['receipt_sha256'] != original['receipt_sha256']
            or copied['files_sha256'] != declared.get('bundle_files_sha256')
            or copied['receipt_sha256'] != declared.get('bundle_receipt_sha256')):
        raise ValueError('Canonical, private and startup bundle identities must match')
    intent = {'schema': 'aureon.bundle_retirement.v1', 'status': 'verified_intent',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'worker_id': startup['worker_id'], 'path': str(target),
              'restore_from': str(canonical/'pytorch_model.bin'),
              'sha256': copied['files_sha256']['pytorch_model.bin'], 'bytes': target_stamp[2],
              'receipt_sha256': receipts,
              'restore': 'Verify canonical SHA256; copy only to the absent original path; verify the restored SHA256. Do not rerun a historical experiment in place.',
              'preserved': 'config, bundle receipt, startup/shutdown, source snapshots, raw numerical evidence, COG/provenance'}
    _write(intent_path, intent)
    if (_regular(target) != target_stamp
            or any(_regular(canonical/name) != stamp for name, stamp in source_stamps.items())
            or any(_hash(folder/name) != digest for name, digest in receipts.items())):
        raise ValueError('Storage or lifecycle identity changed before retirement')
    target.unlink()
    result = {**intent, 'status': 'retired', 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'logical_bytes_removed': target_stamp[2], 'physical_bytes_reclaimed': None,
              'storage_retirement_seconds': perf_counter()-started}
    _write(report_path, result)
    return result


def finish_bundle_storage(worker_folder, canonical_bundle, shutdown):
    """Post-shutdown cleanup; retain failed-worker bytes and explicit diagnostics."""
    folder = Path(worker_folder)
    if (shutdown.get('full_identity_check_passed') is not True
            or shutdown.get('cleanup_passed') is not True or shutdown.get('errors')):
        result = {'status': 'retained', 'reason': 'Worker identity/resource cleanup did not pass'}
    else:
        try:
            return retire_bundle_copy(folder, canonical_bundle)
        except Exception as error:
            result = {'status': 'retained' if (folder/'bundle/pytorch_model.bin').exists() else 'reconciliation_required',
                      'reason': str(error), 'error_type': type(error).__name__}
    # Leave the original shutdown evidence immutable. Storage is a separate
    # operation; no failed hash/cleanup is relabelled successful.
    _write(folder/'bundle-retention.json', result)
    return result


def verify_retirement(worker_folder, canonical_bundle):
    """Read-only check for an experiment whose worker already retired its copy."""
    from .runtime import verify_bundle
    folder, canonical = Path(worker_folder).absolute(), Path(canonical_bundle).absolute()
    target = folder/'bundle/pytorch_model.bin'
    if target.exists() or target.is_symlink():
        raise ValueError('Runtime weight is not retired')
    intent = _read(folder/'bundle-retirement-intent.json')
    report = _read(folder/'bundle-retirement.json')
    verified = verify_bundle(canonical)
    if (intent.get('schema') != 'aureon.bundle_retirement.v1'
            or intent.get('status') != 'verified_intent' or report.get('status') != 'retired'
            or any(report.get(key) != value for key, value in intent.items() if key != 'status')
            or intent.get('path') != str(target)
            or intent.get('restore_from') != str(canonical/'pytorch_model.bin')
            or intent.get('sha256') != verified['files_sha256']['pytorch_model.bin']
            or intent.get('bytes') != (canonical/'pytorch_model.bin').stat().st_size
            or set(intent.get('receipt_sha256', {})) != {'startup.json', 'shutdown.json'}
            or any(_hash(folder/name) != digest for name, digest in intent['receipt_sha256'].items())):
        raise ValueError('Retirement evidence no longer matches the retained canonical/lifecycle')
    return report
