"""Read-only local console for the AUREON KAO-TerraMind inference engine."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
import ipaddress
import json
from pathlib import Path
import platform
import sys
from urllib.parse import parse_qs, unquote, urlsplit

from .catalog import get_catalog, plan_inference


LOCAL_HOST = '127.0.0.1'
ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / 'static'
BENCHMARK_DIRECTORY = ROOT / 'runs' / 'benchmarks'
RUNTIME_SNAPSHOT_DIRECTORY = ROOT / 'runs' / 'runtime'
ACQUISITION_DIRECTORY = ROOT / 'runs' / 'downloads'
MAX_RECEIPTS = 50
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_BENCHMARK_RESPONSE_BYTES = 4 * 1024 * 1024


def _saved_json_files(directory, pattern):
    """Yield at most 50 bounded immediate JSON objects without following links."""
    folder = Path(directory)
    if folder.is_symlink() or not folder.is_dir():
        return
    resolved = folder.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        return
    for path in sorted(folder.glob(pattern), key=lambda item: item.name, reverse=True)[:MAX_RECEIPTS]:
        try:
            if path.is_symlink() or not path.is_file() or path.resolve().parent != resolved or path.stat().st_size > MAX_RECEIPT_BYTES:
                continue
            with path.open('rb') as stream:
                raw = stream.read(MAX_RECEIPT_BYTES + 1)
            if len(raw) > MAX_RECEIPT_BYTES:
                continue
            value = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
            if isinstance(value, dict):
                yield path, value, hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, UnicodeError, RecursionError):
            continue


def read_runtime_evidence():
    """Inspect saved installation/acquisition snapshots, never their live state."""
    result = {'wsl_runtime': None, 'checkpoint_acquisition': None,
              'live_service_verified': False, 'checkpoint_files_rehashed': False}
    for path, record, digest in _saved_json_files(RUNTIME_SNAPSHOT_DIRECTORY, 'wsl-installed-*.json'):
        packages = record.get('packages')
        if (not isinstance(packages, dict) or not isinstance(record.get('platform'), str)
                or 'wsl' not in record['platform'].lower() or type(record.get('cuda_available')) is not bool
                or not isinstance(record.get('created_utc'), str)):
            continue
        safe_packages = {name: value for name, value in packages.items()
                         if name in ('torch', 'torchvision', 'terratorch', 'vllm', 'transformers', 'numpy')
                         and isinstance(value, str) and len(value) <= 100}
        if not safe_packages.get('vllm') or not safe_packages.get('terratorch'):
            continue
        result['wsl_runtime'] = {
            'status': 'recorded_snapshot', 'file': path.relative_to(ROOT).as_posix(), 'sha256': digest,
            'recorded_utc': record['created_utc'], 'platform': record['platform'], 'packages': safe_packages,
            'python': record.get('python'), 'executable': record.get('executable'),
            'cuda_build': record.get('cuda_build'), 'cuda_available_at_inspection': record['cuda_available'],
            'cuda_sanity_sum_at_inspection': record.get('cuda_sanity_sum'), 'gpu_name': record.get('gpu_name'),
            'live_service_verified': False,
        }
        break
    models = {row['id']: row for row in get_catalog()['models']}
    identity_keys = ('repo_id', 'revision', 'checkpoint', 'published_checkpoint_sha256', 'published_checkpoint_bytes')
    for path, record, digest in _saved_json_files(ACQUISITION_DIRECTORY, '*.json'):
        if record.get('schema') != 'aureon.checkpoint_acquisition.v1' or not isinstance(record.get('models'), list):
            continue
        verified = []
        seen = set()
        for row in record['models'][:16]:
            if not isinstance(row, dict) or row.get('model_id') not in models or row.get('model_id') in seen:
                continue
            model = models[row['model_id']]
            expected = model['identity']
            claimed = row.get('identity')
            if (not isinstance(claimed, dict) or any(claimed.get(key) != expected[key] for key in identity_keys)
                    or row.get('verified_sha256') != expected['published_checkpoint_sha256']
                    or row.get('bytes') != expected['published_checkpoint_bytes']
                    or not isinstance(row.get('verified_utc'), str)):
                continue
            seen.add(row['model_id'])
            verified.append({'model_id': row['model_id'], 'name': model['name'],
                             'verified_sha256': row['verified_sha256'], 'bytes': row['bytes'],
                             'verified_utc': row['verified_utc'], 'status': 'hash_verified_at_acquisition',
                             'catalog_identity_matches': True})
        if not verified:
            continue
        result['checkpoint_acquisition'] = {
            'status': 'recorded_snapshot', 'file': path.relative_to(ROOT).as_posix(), 'sha256': digest,
            'models': verified, 'all_candidates_recorded': seen == set(models),
            'checkpoint_files_rehashed': False,
        }
        break
    return result


def inspect_runtime():
    """Read client metadata and saved snapshots; never probe a neural runtime."""
    versions = {}
    for name in ('torch', 'terratorch', 'vllm'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    operating_system = platform.system()
    saved = read_runtime_evidence()
    reasons = []
    if operating_system == 'Windows':
        reasons.append('The console runs in the Windows client environment. WSL installation evidence is shown separately as a recorded snapshot.'
                       if saved['wsl_runtime'] else 'The local console runs on Windows. Prepare a compatible Linux or WSL2 vLLM runtime before serving.')
    for name, version in versions.items():
        if version is None:
            reasons.append(f'{name} is not installed in this client Python environment.')
    if saved['wsl_runtime']:
        reasons.append('Saved WSL versions and CUDA checks describe their inspection time. This console does not check whether a model service is currently running.')
    if saved['checkpoint_acquisition']:
        reasons.append('Saved checkpoint download hashes match the catalog. Checkpoint files are not rehashed by this read-only inspection.')
    reasons.append('Checkpoint loading, numerical parity, endpoint availability and speed are not verified by runtime inspection; review separate benchmark receipts.')
    return {
        'schema': 'aureon.runtime_inspection.v1',
        'status': 'setup_required' if operating_system == 'Windows' or any(value is None for value in versions.values()) else 'validation_required',
        'platform': operating_system, 'python_version': platform.python_version(),
        'python_executable': sys.executable, 'packages': versions, 'reasons': reasons,
        'ready': False, 'inspection_only': True, 'model_executed': False,
        'endpoint_probed': False, 'downloads_started': False,
        'saved_evidence': saved,
    }


def read_benchmarks(directory):
    """Read bounded immediate JSON receipts; skip links and malformed artifacts."""
    root = Path(directory)
    result = {'schema': 'aureon.benchmark_listing.v1', 'benchmarks': [],
              'skipped': 0, 'truncated': False, 'directory': 'runs/benchmarks'}
    if not root.exists():
        return result
    if root.is_symlink() or not root.is_dir():
        result['skipped'] = 1
        return result
    resolved_root = root.resolve()
    # Keep the explicitly confirmed comparisons visible before exploratory runs
    # when the bounded listing fills up; every original receipt stays on disk.
    files = sorted(root.glob('*.json'), key=lambda path: (path.name.startswith('confirmed-'), path.name), reverse=True)
    result['truncated'] = len(files) > MAX_RECEIPTS
    total_bytes = 0
    for path in files[:MAX_RECEIPTS]:
        try:
            if path.is_symlink() or not path.is_file() or path.resolve().parent != resolved_root:
                result['skipped'] += 1
                continue
            size = path.stat().st_size
            if size > MAX_RECEIPT_BYTES:
                result['skipped'] += 1
                continue
            if total_bytes + size > MAX_BENCHMARK_RESPONSE_BYTES:
                result['truncated'] = True
                break
            with path.open('rb') as stream:
                raw = stream.read(MAX_RECEIPT_BYTES + 1)
            if len(raw) > MAX_RECEIPT_BYTES:
                result['skipped'] += 1
                continue
            artifact = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
            if not isinstance(artifact, dict):
                raise ValueError('receipt must be an object')
            result['benchmarks'].append({'file': path.name, 'artifact': artifact})
            total_bytes += len(raw)
        except (OSError, ValueError, UnicodeError, RecursionError):
            result['skipped'] += 1
    return result


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = 'AUREONLocalConsole/1'
    sys_version = ''

    def log_message(self, format, *args):
        return

    def _send(self, status, payload, content_type='application/json; charset=utf-8'):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _local_request(self):
        hosts = self.headers.get_all('Host') or []
        port = self.server.server_port
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if port == 80:
            allowed |= {'127.0.0.1', 'localhost'}
        if not ipaddress.ip_address(self.client_address[0]).is_loopback or len(hosts) != 1 or hosts[0].lower() not in allowed:
            self._send(403, {'error': 'This console accepts loopback requests with a local Host only.'})
            return False
        return True

    def do_GET(self):
        if not self._local_request():
            return
        if len(self.path) > 2048:
            self._send(414, {'error': 'Request URI is too long.'})
            return
        try:
            uri = urlsplit(self.path)
            if uri.netloc or uri.scheme or uri.fragment:
                raise ValueError('Only local relative request paths are accepted.')
            path = unquote(uri.path, errors='strict')
            query = parse_qs(uri.query, keep_blank_values=True, strict_parsing=True, max_num_fields=8)
            if any(len(values) != 1 for values in query.values()):
                raise ValueError('Duplicate query parameters are not accepted.')
            allowed_parameters = {'model_id', 'backend'} if path == '/api/plan' else set()
            if set(query) - allowed_parameters:
                raise ValueError('Unknown query parameter.')
            if path == '/api/models':
                catalog = get_catalog()
                catalog['models'] = [row for row in catalog['models'] if row.get('id') in {'kao-terramind-a', 'kao-terramind-b', 'kao-terramind-c'}]
                self._send(200, catalog)
            elif path == '/api/plan':
                if 'model_id' not in query:
                    raise ValueError('model_id is required.')
                if query['model_id'][0] not in {'kao-terramind-a', 'kao-terramind-b', 'kao-terramind-c'}:
                    raise ValueError('Select a catalog KAO-TerraMind candidate.')
                options = {'backend': query['backend'][0]} if 'backend' in query else {}
                self._send(200, plan_inference(query['model_id'][0], **options))
            elif path == '/api/runtime':
                self._send(200, inspect_runtime())
            elif path == '/api/benchmarks':
                self._send(200, read_benchmarks(self.server.benchmark_directory))
            elif path in {'/', '/index.html', '/app.js', '/style.css'}:
                name = 'index.html' if path == '/' else path[1:]
                mime = {'index.html': 'text/html', 'app.js': 'text/javascript', 'style.css': 'text/css'}[name]
                self._send(200, (STATIC / name).read_bytes(), mime + '; charset=utf-8')
            else:
                self._send(404, {'error': 'Route not found.'})
        except (ValueError, UnicodeError) as exc:
            self._send(400, {'error': str(exc)})
        except OSError:
            self._send(500, {'error': 'A local console resource could not be read.'})

    do_HEAD = do_GET

    def do_POST(self):
        if self._local_request():
            self._send(405, {'error': 'The console is read-only. Use the explicit CLI for preparation and benchmarks.'})

    do_PUT = do_POST
    do_DELETE = do_POST
    do_PATCH = do_POST
    do_OPTIONS = do_POST


def create_server(port=4324, *, benchmark_directory=None):
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('port must be an integer from 0 to 65535')
    server = ThreadingHTTPServer((LOCAL_HOST, port), ConsoleHandler)
    server.daemon_threads = True
    server.benchmark_directory = BENCHMARK_DIRECTORY if benchmark_directory is None else Path(benchmark_directory)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description='AUREON — KAO-TerraMind inference engine local console')
    parser.add_argument('--port', type=int, default=4324)
    args = parser.parse_args(argv)
    with create_server(args.port) as server:
        print(f'AUREON — KAO-TerraMind inference engine: http://{LOCAL_HOST}:{server.server_port}/', flush=True)
        print('Read-only console. Model execution requires an explicit CLI command.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
