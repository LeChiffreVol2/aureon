"""Bounded single-thread loopback HTTP adapter for the native in-process engine.

No background inference or remote deployment. A disconnect cannot cancel a
running synchronous call: completion drains before the next request, and the
client's existing failure poisoning forbids buffer reuse after a timeout.
"""
import json
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer

from .resident_protocol import PATH
from .shm_files import decode_descriptor, MAX_CONTROL_BYTES, SCHEMA


def control_headers(items):
    headers = {}
    for key, value in items:
        key = key.lower()
        if key in headers:
            raise ValueError('Duplicate control header')
        headers[key] = value
    if 'origin' in headers or 'authorization' in headers or 'transfer-encoding' in headers:
        raise ValueError('Unsupported origin, authentication or transfer encoding')
    if headers.get('content-type', '').split(';', 1)[0].strip().lower() != 'application/json':
        raise ValueError('Expected JSON control data')
    length = headers.get('content-length', '')
    if not length.isascii() or not length.isdecimal() or not 0 < int(length) <= MAX_CONTROL_BYTES:
        raise ValueError('Require a bounded positive control Content-Length')
    return int(length)


class PoolingHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'AUREON-local-inproc/1'
    def log_message(self, *unused):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(10)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def reply(self, status, value):
        body = json.dumps(value, separators=(',', ':'), allow_nan=False).encode('ascii')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self):
        if self.path != '/v1/models':
            self.close_connection = True
            return self.reply(404, {'error': 'Unknown local endpoint'})
        if self.server.engine.failed:
            self.close_connection = True
            return self.reply(503, {'error': 'Native engine is poisoned'})
        return self.reply(200, {'object': 'list', 'data': [{'id': self.server.engine.bundle['model_id']}]})

    def do_POST(self):
        try:
            if self.path != PATH or self.client_address[0] != '127.0.0.1':
                raise ValueError('Use the exact controlled loopback resident route')
            length = control_headers(self.headers.items())
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('Truncated control request')
            descriptor = decode_descriptor(body)
            if descriptor['model'] != self.server.engine.bundle['model_id']:
                raise ValueError('Control model differs from the resident model')
        except (ValueError, TypeError, UnicodeError, OSError):
            self.close_connection = True
            return self.reply(400, {'error': 'Invalid bounded local control request'})
        try:
            # Engine and this handler execute on the same thread. No next
            # request is admitted until model completion and ACK validation.
            self.server.engine.execute(descriptor)
            self.reply(200, {'schema': SCHEMA, 'request': descriptor['request'],
                'client': descriptor['client'], 'model': descriptor['model'], 'count': 8, 'status': 'completed'})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            self.close_connection = True
            self.reply(502, {'error': 'Native engine failed; this service is poisoned'})


class PoolingServer(HTTPServer):
    allow_reuse_address = False
    request_queue_size = 4
    def __init__(self, port, engine):
        self.engine = engine
        super().__init__(('127.0.0.1', port), PoolingHandler)
