"""Explicit single-job client for an already started local AUREON scene API.

No service startup, acquisition, installation, inference or file writes occur
on import. Model submission requires --execute-model. Failed requests are not
retried automatically; a lost response does not prove the job was not admitted.
"""
import argparse
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import stat
import time
from urllib.parse import urlsplit
import uuid


ARTIFACTS = ('job.json', 'provenance.json', 'class_map.tif', 'valid_mask.tif')
FILE_KEYS = {'provenance.json': 'artifacts/receipt.json',
             'class_map.tif': 'artifacts/class_map.tif',
             'valid_mask.tif': 'artifacts/valid_mask.tif'}
MAX_BYTES = 16 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def origin(url):
    parsed = urlsplit(url)
    require(parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost')
            and parsed.username is None and parsed.password is None
            and parsed.path in ('', '/') and not parsed.query and not parsed.fragment
            and parsed.port is not None and 1 <= parsed.port <= 65535,
            'Use an explicit http://127.0.0.1:PORT or http://localhost:PORT origin')
    # Connect to the literal IPv4 address instead of resolving localhost.
    return '127.0.0.1', parsed.port


def token_from_file(path):
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and 1 <= info.st_size <= 4096,
            'Token file must be a bounded regular file, not a symbolic link')
    with path.open('rb') as stream:
        raw = stream.read(4097)
    require(len(raw) <= 4096, 'Token file exceeds its size limit')
    token = raw.decode('ascii').strip()
    require(re.fullmatch(r'[A-Za-z0-9_-]{32,512}', token) is not None,
            'Token file must contain only the generated bearer token')
    return token


def fresh_directory(path):
    path = Path(path).absolute()
    require(not path.is_symlink() and not path.exists() and path.parent.is_dir(),
            'Output must be a new directory under an existing parent')
    path.mkdir(exist_ok=False)
    return path


def write_json(path, value, *, token=None):
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    require(token is None or token not in data, 'Refusing to write credential-bearing evidence')
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(data)


class Client:
    def __init__(self, url, token, timeout):
        self.host, self.port = origin(url)
        require(math.isfinite(timeout) and 0 < timeout <= 3600, 'Timeout must be in (0,3600] seconds')
        self.token, self.deadline = token, time.monotonic() + timeout

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Finite client deadline exceeded')
        return remaining

    def request(self, method, path, *, payload=None, key=None, limit=MAX_BYTES):
        require(path.startswith('/v1/') and not any(c in path for c in ('\r', '\n', '?', '#', '\\')),
                'Only constructed local API paths are allowed')
        headers = {'Authorization': 'Bearer ' + self.token, 'Connection': 'close'}
        body = None
        if payload is not None:
            body = json.dumps(payload, allow_nan=False).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        if key is not None:
            headers['Idempotency-Key'] = key
        connection = http.client.HTTPConnection(self.host, self.port, timeout=min(30, self.remaining()))
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            lengths = response.headers.get_all('Content-Length', [])
            require(len(lengths) == 1 and lengths[0].isdigit()
                    and int(lengths[0]) <= limit, 'Missing, duplicate or oversized response length')
            data = response.read(limit + 1)
            require(len(data) == int(lengths[0]), 'Response byte length mismatch')
            # No proxy, redirect, cookie store, or server-supplied target URL.
            return response.status, {'etag': response.getheader('ETag')}, data
        finally:
            connection.close()

    def json(self, method, path, **kwargs):
        status, headers, data = self.request(method, path, limit=1024 * 1024, **kwargs)
        return status, headers, json.loads(data)


def save_download(folder, job_id, name, descriptor, status, headers, data):
    require(name in ARTIFACTS and re.fullmatch(r'[0-9a-f]{32}', job_id) is not None,
            'Unexpected artifact name or job identity')
    expected_path = f'/v1/jobs/{job_id}/artifacts/{name}'
    require(isinstance(descriptor, dict) and descriptor.get('url') == expected_path
            and isinstance(descriptor.get('sha256'), str)
            and re.fullmatch(r'[0-9a-f]{64}', descriptor['sha256']) is not None
            and type(descriptor.get('bytes')) is int and 0 < descriptor['bytes'] <= MAX_BYTES,
            'Unsafe or invalid artifact descriptor')
    require(status == 200 and len(data) == descriptor['bytes']
            and sha256(data) == descriptor['sha256']
            and headers.get('etag') == '"' + descriptor['sha256'] + '"',
            'Artifact status, bytes, hash or ETag mismatch')
    with (Path(folder) / name).open('xb') as stream:
        stream.write(data)


def verify_saved_downloads(folder, descriptors):
    require(set(descriptors) == set(ARTIFACTS), 'Unexpected saved artifact set')
    for name in ARTIFACTS:
        path = Path(folder) / name
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_BYTES,
                'Saved artifacts must be bounded regular files')
        data = path.read_bytes()
        require(len(data) == descriptors[name]['bytes']
                and sha256(data) == descriptors[name]['sha256'], 'Saved download changed')


def verify_delivery(folder, *, scene_id, manifest_hash, bundle_hash, worker):
    """Verify delivered bytes/receipts and COG pixels; never load a model.

    This checks the recorded parity contract; saved logits and source snapshots
    stay on the server and require the separate installed resident audit.
    """
    import numpy as np
    import rasterio
    from aureon.catalog import CLASS_NAMES

    folder = Path(folder)
    job = json.loads((folder / 'job.json').read_bytes())
    require(job.get('schema') == 'aureon.scene_job.v1' and job.get('status') == 'completed'
            and job.get('model_name') == 'KAO-TerraMind B' and job.get('scene_id') == scene_id
            and job.get('source_manifest_sha256') == manifest_hash
            and job.get('bundle_receipt_sha256') == bundle_hash, 'Job/source/model identity mismatch')
    precision = job.get('precision', {})
    require(precision.get('float32_matmul_precision') == 'highest'
            and all(precision.get(k) is False for k in ('matmul_allow_tf32', 'cudnn_allow_tf32',
                                                       'cudnn_sdp_enabled', 'cudnn_benchmark')),
            'Highest FP32 precision policy mismatch')
    parity = job.get('parity', {})
    require(parity.get('passed') is True and parity.get('allclose') is True
            and parity.get('atol') == 1e-5 and parity.get('rtol') == 1e-5
            and parity.get('class_agreement') == 1.0
            and parity.get('scene_class_map_equal') is True
            and parity.get('scene_mask_equal') is True, 'Recorded FP32 parity gate failed')
    identity = job.get('worker', {})
    require(identity.get('model_load_count') == worker.get('model_load_count') == 1
            and identity.get('worker_id') == worker.get('worker_id')
            and identity.get('worker_startup_receipt_sha256') == worker.get('startup_receipt_sha256'),
            'Retained worker identity mismatch')
    delivery = json.loads((folder / 'provenance.json').read_bytes())
    require(delivery.get('schema') == 'aureon.georaster_delivery.v1'
            and delivery.get('status') == 'completed' and delivery.get('format') == 'COG'
            and delivery.get('class_names') == list(CLASS_NAMES), 'Delivery contract mismatch')
    provenance = delivery.get('provenance', {})
    require(provenance.get('job_id') == job.get('job_id')
            and provenance.get('scene_id') == scene_id
            and provenance.get('source_manifest_sha256') == manifest_hash
            and provenance.get('bundle_receipt_sha256') == bundle_hash,
            'Delivery provenance differs from the completed job')
    for name, key in FILE_KEYS.items():
        data = (folder / name).read_bytes()
        require(job['files_sha256'][key] == sha256(data), 'Saved artifact differs from job hash')
        if name.endswith('.tif'):
            record = delivery['files'][name]
            require(record['sha256'] == sha256(data) and record['bytes'] == len(data),
                    'Saved COG differs from delivery hash or size')
    expected = delivery['grid']
    with rasterio.open(folder / 'class_map.tif') as classes, rasterio.open(folder / 'valid_mask.tif') as mask:
        for ds in (classes, mask):
            require(ds.count == 1 and ds.dtypes == ('uint8',)
                    and 1 <= ds.width <= 1024 and 1 <= ds.height <= 1024
                    and str(ds.crs) == expected['crs']
                    and list(ds.transform)[:6] == expected['transform']
                    and ds.height == expected['height'] and ds.width == expected['width']
                    and ds.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') == 'COG', 'COG grid/type/layout mismatch')
        require(classes.nodata == 255 and mask.nodata == 0, 'COG nodata mismatch')
        values, valid_values = classes.read(1), mask.read(1)
        require(np.isin(valid_values, (0, 1)).all(), 'Validity mask is not binary')
        valid = valid_values == 1
        require(np.all(values[valid] <= 9) and np.all(values[~valid] == 255)
                and np.array_equal(classes.read_masks(1) > 0, valid)
                and np.array_equal(mask.read_masks(1) > 0, valid), 'Class/validity pixels disagree')
    return {'job_id': job['job_id'], 'parity': parity, 'cog_values_grid_mask_verified': True,
            'recorded_parity_only': True, 'task_accuracy_evaluated': False,
            'worker_id': identity['worker_id']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-model', action='store_true')
    parser.add_argument('--url', required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args(argv)
    if not args.execute_model:
        parser.error('Explicit --execute-model is required before any job submission')
    origin(args.url)
    token = token_from_file(args.token_file)
    client = Client(args.url, token, args.timeout)
    # Read-only preflight: requires the installed package, never model loading.
    from aureon.runtime import verify_bundle
    from aureon.scene_pipeline.job import load_manifest
    manifest, _, manifest_hash = load_manifest(args.manifest.resolve(strict=True))
    bundle = verify_bundle(args.bundle.resolve(strict=True))
    require(bundle['model_id'] == 'kao-terramind-b', 'This example requires KAO-TerraMind B')
    scene_id = manifest['scene_id']
    require(isinstance(scene_id, str) and 1 <= len(scene_id) <= 256, 'Invalid prepared scene ID')
    output = fresh_directory(args.output)
    report = {'schema': 'aureon.packaged_scene_example.v1', 'status': 'failed', 'scene_id': scene_id,
              'source_manifest_sha256': manifest_hash, 'bundle_receipt_sha256': bundle['receipt_sha256'],
              'server_started_by_client': False, 'task_accuracy_evaluated': False}
    try:
        status, _, info = client.json('GET', '/v1/info')
        require(status == 200 and info.get('state') == 'ready' and scene_id in info.get('scene_ids', []),
                'Service must be ready and register the supplied scene')
        worker = info.get('worker', {})
        require(worker.get('model_load_count') == 1 and isinstance(worker.get('worker_id'), str)
                and bool(worker['worker_id'])
                and re.fullmatch(r'[0-9a-f]{64}', worker.get('startup_receipt_sha256', '')) is not None,
                'Service must report one identified retained model load')
        key = uuid.uuid4().hex
        # UUID is an operation identity, not the bearer credential. Persist it
        # before submission so an ambiguous response can be investigated safely.
        write_json(output / 'submission.json', {'scene_id': scene_id, 'idempotency_key': key}, token=token)
        status, _, reply = client.json('POST', '/v1/jobs', payload={'scene_id': scene_id}, key=key)
        require(status == 202 and reply.get('reused') is False,
                'Submission not admitted as new work; inspect HTTP state before retrying')
        snapshot = reply.get('job', {})
        job_id = snapshot.get('job_id', '')
        require(re.fullmatch(r'[0-9a-f]{32}', job_id) is not None, 'Invalid returned job identity')
        report['job_id'] = job_id
        while True:
            require(snapshot.get('job_id') == job_id and snapshot.get('scene_id') == scene_id,
                    'Status belongs to another job or scene')
            state = snapshot.get('status')
            require(state in ('queued', 'running', 'completed'), 'Job failed, was cancelled, or returned invalid state')
            if state == 'completed':
                break
            time.sleep(min(.5, client.remaining()))
            status, _, reply = client.json('GET', '/v1/jobs/' + job_id)
            require(status == 200, 'Job observation failed')
            snapshot = reply.get('job', {})
        descriptors = snapshot.get('artifacts', {})
        require(set(descriptors) == set(ARTIFACTS), 'Unexpected completed artifact set')
        for name in ARTIFACTS:
            path = f'/v1/jobs/{job_id}/artifacts/{name}'
            status, headers, data = client.request('GET', path)
            save_download(output, job_id, name, descriptors[name], status, headers, data)
        verify_saved_downloads(output, descriptors)
        checks = verify_delivery(output, scene_id=scene_id, manifest_hash=manifest_hash,
                                 bundle_hash=bundle['receipt_sha256'], worker=worker)
        require(checks['job_id'] == job_id, 'Downloaded receipt is for another job')
        status, _, final = client.json('GET', '/v1/info')
        require(status == 200 and all(final.get('worker', {}).get(k) == worker[k]
                    for k in ('worker_id', 'startup_receipt_sha256', 'model_load_count')),
                'Service worker changed during observation')
        verify_saved_downloads(output, descriptors)
        report.update(status='passed', checks=checks, artifact_descriptors=descriptors,
                      independent_server_audit_required=True)
    except Exception as error:
        report['error_type'] = type(error).__name__
        raise
    finally:
        write_json(output / 'example-result.json', report, token=token)
    print(json.dumps({'status': report['status'], 'job_id': report['job_id'],
                      'output': str(output), 'independent_server_audit_required': True}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
