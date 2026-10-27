"""Bounded immutable snapshots of the three local scene-console assets."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import stat


ROUTES = {
    '/console': 'scene-console.html',
    '/console/': 'scene-console.html',
    '/console/scene-console.js': 'scene-console.js',
    '/console/scene-console.css': 'scene-console.css',
}
MIME = {'scene-console.html': 'text/html; charset=utf-8',
        'scene-console.js': 'text/javascript; charset=utf-8',
        'scene-console.css': 'text/css; charset=utf-8'}
MAX_ASSET_BYTES = 256 * 1024
HEADERS = {
    'Content-Security-Policy': "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' blob: data:; font-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; object-src 'none'",
    'Referrer-Policy': 'no-referrer',
    'X-Frame-Options': 'DENY',
    'Cross-Origin-Resource-Policy': 'same-origin',
}


@dataclass(frozen=True)
class ConsoleAssets:
    # Only immutable strings/bytes/tuples are retained by the HTTP listener.
    files: tuple

    def get(self, route):
        name = ROUTES.get(route)
        for filename, body, mime, digest in self.files:
            if filename == name:
                return body, mime, digest
        return None

    def manifest(self):
        return {'schema': 'aureon.local_scene_console_assets.v1',
                'scope': 'Exact public asset bytes snapshotted at explicit service startup; no credentials or model execution',
                'files': {name: {'bytes': len(body), 'sha256': digest, 'content_type': mime}
                          for name, body, mime, digest in self.files}}


def load_console_assets(directory):
    folder = Path(directory)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Console directory must be an existing ordinary directory')
    folder = folder.resolve(strict=True)
    members = []
    for name, mime in MIME.items():
        path = folder/name
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            raise ValueError('Console assets must be regular files, not links')
        if not 1 <= before.st_size <= MAX_ASSET_BYTES:
            raise ValueError('Console asset size must be 1..262144 bytes')
        with path.open('rb') as stream:
            body = stream.read(MAX_ASSET_BYTES+1)
        after = path.lstat()
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)
        if len(body) != before.st_size or stamp(before) != stamp(after) or path.is_symlink():
            raise ValueError('Console asset changed while making its startup snapshot')
        body.decode('utf-8')
        members.append((name, body, mime, hashlib.sha256(body).hexdigest()))
    return ConsoleAssets(tuple(members))
