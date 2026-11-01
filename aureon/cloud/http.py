"""Bounded LOOPBACK HTTP adapter. Public HTTPS deployment is not implemented."""
from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import cast
from .contracts import UploadBody, StreamBody
from ..serving.contracts import BaseRequest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
from urllib.parse import urlsplit, parse_qs

from .control import Control, Error, require, encoded, MODEL, PROFILE, CONTRACT
from ..client import _decode, ClientError


class Handler(BaseHTTPRequestHandler):
    server: Server

    protocol_version = 'HTTP/1.1'

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)
        def deadline() -> None:
            try:self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:pass
        self.request_deadline=threading.Timer(self.server.control.contract.get('request_deadline_seconds',45),deadline)
        self.request_deadline.daemon=True;self.request_deadline.start()

    def finish(self) -> None:
        if hasattr(self,'request_deadline'):self.request_deadline.cancel()
        super().finish()

    def log_message(self, *args: object) -> None:
        pass  # Do not log authorization headers or signed URL query strings.

    def handle_expect_100(self) -> bool:
        self.respond(417, {'error':{'code':'expect_not_supported'}})
        return False

    def respond(self, code: int, body: object, binary: bool = False) -> None:
        if isinstance(body,StreamBody):
            import hashlib
            try:
                self.send_response(code)
                self.send_header('Content-Type','application/octet-stream')
                self.send_header('Content-Length',str(body.bytes))
                self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Connection','close');self.end_headers();self.close_connection=True
                remaining=body.bytes;hasher=hashlib.sha256()
                while remaining:
                    chunk=body.stream.read(min(1024**2,remaining))
                    if not chunk:raise OSError('Artifact stream truncated')
                    self.wfile.write(chunk);hasher.update(chunk);remaining-=len(chunk)
                if body.stream.read(1) or hasher.hexdigest()!=body.sha256:
                    raise OSError('Artifact stream changed')
            except OSError:
                # Headers may already be committed. Close this response; never
                # append a second JSON response to partially delivered bytes.
                self.close_connection=True
            finally:body.stream.close()
            return
        data = cast(bytes, body) if binary else encoded(body)
        self.send_response(code)
        self.send_header('Content-Type', 'application/octet-stream' if binary else 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Connection', 'close')
        if code == 429:
            self.send_header('Retry-After', '5')
        self.end_headers()
        self.close_connection = True
        self.wfile.write(data)

    def length(self, limit: int) -> int:
        values = self.headers.get_all('Content-Length') or []
        require(len(values)==1 and values[0].isascii() and values[0].isdigit(), 'content_length_required', 411)
        length = int(values[0])
        require(0 <= length <= limit, 'body_too_large', 413)
        return length

    def json(self) -> object:
        require(self.headers.get('Content-Type') == 'application/json', 'json_required', 415)
        length = self.length(65536)
        body = self.rfile.read(length)
        require(len(body)==length, 'incomplete_body')
        return _decode(body)

    def route(self) -> tuple[int, object, bool]:
        require(len(self.path) <= 2048 and not self.path.startswith('//'), 'invalid_target')
        require(sum(len(k)+len(v) for k,v in self.headers.items())<=32768,'headers_too_large',431)
        require(self.headers.get_all('Transfer-Encoding') is None, 'transfer_encoding_not_supported')
        allowed = getattr(self.server, 'allowed_hosts', {f'127.0.0.1:{self.server.server_port}'})
        require(len(self.headers.get_all('Host') or [])==1 and self.headers['Host'] in allowed, 'invalid_host')
        require(self.headers.get('Origin') is None, 'browser_origin_not_supported', 403)
        parsed = urlsplit(self.path)
        require(not parsed.scheme and not parsed.netloc and not parsed.fragment, 'invalid_target')
        headers = self.headers.get_all('Authorization') or []
        require(len(headers)==1 and headers[0].startswith('Bearer '), 'unauthorized', 401)
        control = self.server.control
        tenant = control.authenticate(headers[0][7:])
        admission=cast(Callable[[str], None] | None, getattr(control,'request_admission',None))
        if admission is not None:admission(tenant)
        parts = parsed.path.strip('/').split('/')
        query = parse_qs(parsed.query, strict_parsing=True) if parsed.query else {}
        ticket = query.get('ticket', [''])[0]
        require(not query or set(query)=={'ticket'} and len(query['ticket'])==1, 'invalid_query')
        method = self.command
        extra = cast(Callable[[Handler, str, str, list[str], dict[str, list[str]]], tuple[int, object, bool] | None] | None, getattr(control, 'http_route', None))
        if extra is not None:
            extension_result = extra(self, tenant, method, parts, query)
            if extension_result is not None:
                return extension_result
        if method=='GET' and parts==['v1','status']:
            require(not query,'invalid_query')
            return 200, control.status_snapshot(tenant), False
        if method=='GET' and parts==['v1','health']:
            require(not query,'invalid_query')
            health=control.health()
            return (200 if health['control_plane']['ready'] else 503), health, False
        if method=='GET' and parts==['v1','models']:
            return 200, {'schema':'aureon.cloud.models.v1','mode':'local_'+control.execution,'models':[
                {'id':MODEL,'name':'KAO-TerraMind B','profiles':[{'id':control.profile,
                 'execution':control.execution,'qualification_sha256':control.contract.get('qualification_sha256'),
                 'scope':'api_plumbing_only' if control.execution=='fake' else 'qualified_fp32_guarded_local'}]}],
                 'limits':control.contract,'capabilities':control.capabilities()}, False
        if method=='POST' and parts==['v1','inputs']:
            return 201, control.create_input(tenant,cast(UploadBody,self.json())), False
        if len(parts)>=3 and parts[:2]==['v1','inputs']:
            identifier = parts[2]
            if method=='GET' and len(parts)==3:
                return 200, control.input(tenant,identifier), False
            if method=='POST' and len(parts)==4 and parts[3]=='complete':
                require(self.json()=={}, 'empty_object_required')
                return 202, control.complete_input(tenant,identifier), False
            if method=='PUT' and len(parts)==5 and parts[3]=='files':
                require(self.headers.get('Content-Type')=='application/octet-stream', 'binary_required', 415)
                return 200, control.upload(tenant,identifier,parts[4],ticket,self.rfile,self.length(CONTRACT['max_file_bytes'])), False
        if method=='POST' and parts==['v1','jobs']:
            keys = self.headers.get_all('Idempotency-Key') or []
            require(len(keys)==1, 'idempotency_key_required')
            result = control.submit(tenant,cast(BaseRequest,self.json()),keys[0])
            return 200 if result['reused'] else 202, result, False
        if len(parts)>=3 and parts[:2]==['v1','jobs']:
            identifier = parts[2]
            if method=='GET' and len(parts)==3:
                return 200, control.job(tenant,identifier), False
            if method=='POST' and len(parts)==4 and parts[3]=='cancel':
                require(self.json()=={}, 'empty_object_required')
                return 200, control.cancel(tenant,identifier), False
            if method=='GET' and parts[3:]==['artifacts']:
                return 200, control.artifacts(tenant,identifier), False
            if method=='GET' and len(parts)==5 and parts[3]=='artifacts':
                return 200, control.download(tenant,identifier,parts[4],ticket), True
        raise Error(404,'not_found','not found')

    def execute(self) -> None:
        try:
            code, body, binary = self.route()
            self.respond(code,body,binary)
        except Error as error:
            actions={'queue_full':'Wait for queue capacity, then explicitly replay the same request and idempotency key.',
                     'storage_capacity':'Ask the operator to restore the disk reserve; do not retry uploads continuously.',
                     'input_count_quota':'Ask the operator to review the bounded input-record retention policy.',
                     'job_count_quota':'Ask the operator to review the bounded job-record retention policy.',
                     'cannot_cancel_running_or_terminal':'Only queued jobs can be cancelled; poll a running job for its result.',
                     'profile_not_available':'Read GET /v1/models and submit the state-bound supported profile.',
                     'source_byte_quota':'Reduce the declared input to the supported source-byte bound before uploading.'}
            value: dict[str, object]={'code':error.code,'message':str(error)}
            if error.code in actions: value['action']=actions[error.code]
            if error.details is not None: value['details']=error.details
            self.respond(error.status, {'error':value})
        except (ValueError, TypeError, KeyError, ClientError):
            self.respond(400, {'error':{'code':'invalid_request'}})
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            self.close_connection = True
        except Exception:
            self.respond(503, {'error':{'code':'service_unavailable'}})

    do_GET = do_POST = do_PUT = execute


class Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    request_queue_size = 8

    def __init__(self, control: Control, port: int = 0, *, host: str = '127.0.0.1', allowed_hosts: Sequence[str] | None = None) -> None:
        if host != '127.0.0.1' and not allowed_hosts:
            raise ValueError('Non-loopback binding requires explicit allowed Host values and an operator TLS proxy')
        self.control = control
        self.slots = threading.BoundedSemaphore(4)
        super().__init__((host,port), Handler)
        self.allowed_hosts = set(allowed_hosts or [f'127.0.0.1:{self.server_port}'])

    def process_request(self, request: socket.socket | tuple[bytes, socket.socket], address: tuple[str, int]) -> None:
        if not self.slots.acquire(blocking=False):
            cast(socket.socket, request).close()
            return
        try:
            super().process_request(request,address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request: socket.socket | tuple[bytes, socket.socket], address: tuple[str, int]) -> None:
        try:
            super().process_request_thread(request,address)
        finally:
            self.slots.release()

    def handle_error(self, request: socket.socket | tuple[bytes, socket.socket], address: tuple[str, int]) -> None:
        # Socket/thread errors must not dump request material or exception strings.
        pass
