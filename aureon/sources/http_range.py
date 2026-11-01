"""Explicit bounded HTTP byte ranges, independent of STAC/model preprocessing.

No implicit HEAD, full download, retry, proxy, cookie jar or automatic redirect.
Only a strong ETag with an exact Content-Range/Length is accepted. DNS addresses
are checked and the socket connects directly to one checked address (TLS still
verifies the original hostname). Budgets count reserved response-body bytes;
headers/TLS overhead are separate. Stalled response reads poll cancellation every
100 ms. OS DNS resolution cannot be interrupted; connect/TLS use socket timeouts.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from concurrent.futures import CancelledError
import hashlib
import http.client
import io
import ipaddress
import math
import re
import socket
import ssl
import threading
import time
from typing import BinaryIO, Literal, Self, TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer


class RangeSourceError(ValueError):
    """Sanitized policy/protocol/transport error; never contains a signed URL."""


def _host(value: str) -> str:
    if not isinstance(value,str) or not value or any(c.isspace() for c in value):
        raise RangeSourceError('Invalid allowed host')
    try:
        return ipaddress.ip_address(value).compressed
    except ValueError:
        try:result=value.encode('idna').decode('ascii').lower()
        except UnicodeError:raise RangeSourceError('Invalid host encoding') from None
        if len(result)>253 or not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?',result):
            raise RangeSourceError('Host must be an exact DNS name or IP address')
        if any(not label or len(label)>63 or label.startswith('-') or label.endswith('-') for label in result.split('.')):
            raise RangeSourceError('Invalid DNS label')
        return result


@dataclass(frozen=True)
class RangePolicy:
    allowed_hosts: frozenset[str]
    allowed_ports: frozenset[int] = frozenset({443})
    allow_http: bool = False
    allowed_private_networks: tuple[str,...] = ()
    redirects: Literal['none','same-origin','allowlisted'] = 'none'
    max_redirects: int = 2
    max_range_bytes: int = 1024*1024
    max_total_bytes: int = 16*1024*1024
    max_resource_bytes: int = 64*1024**3
    max_requests: int = 64
    chunk_bytes: int = 64*1024
    timeout_seconds: float = 10.

    def __post_init__(self) -> None:
        object.__setattr__(self,'allowed_hosts',frozenset(_host(h) for h in self.allowed_hosts))
        object.__setattr__(self,'allowed_ports',frozenset(self.allowed_ports))
        object.__setattr__(self,'allowed_private_networks',tuple(self.allowed_private_networks))
        if not self.allowed_hosts or not self.allowed_ports or any(type(p) is not int or not 1<=p<=65535 for p in self.allowed_ports):
            raise RangeSourceError('Explicit nonempty hosts and valid ports are required')
        if type(self.allow_http) is not bool or self.redirects not in ('none','same-origin','allowlisted'):
            raise RangeSourceError('Invalid scheme/redirect policy')
        for name in ('max_range_bytes','max_total_bytes','max_resource_bytes','max_requests','chunk_bytes'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:
                raise RangeSourceError('Byte/request bounds must be positive integers')
        if type(self.max_redirects) is not int or not 0<=self.max_redirects<=8:
            raise RangeSourceError('Redirect bound must be between zero and eight')
        if self.chunk_bytes>self.max_range_bytes or self.max_range_bytes>self.max_total_bytes:
            raise RangeSourceError('Chunk/range/total bounds are inconsistent')
        if isinstance(self.timeout_seconds,bool) or not math.isfinite(self.timeout_seconds) or not 0<self.timeout_seconds<=60:
            raise RangeSourceError('Socket timeout must be finite and at most sixty seconds')
        for network in self.allowed_private_networks:
            try:ipaddress.ip_network(network,strict=True)
            except ValueError:raise RangeSourceError('Invalid explicit private network') from None


@dataclass(frozen=True,repr=False)
class SourceIdentity:
    etag: str = field(repr=False)
    size: int
    def __post_init__(self) -> None:
        if type(self.size) is not int or self.size<=0 or not isinstance(self.etag,str) or not re.fullmatch(r'"[\x21\x23-\x7e]*"',self.etag):
            raise RangeSourceError('Identity requires a strong quoted ETag and positive size')
    def __repr__(self) -> str:return f'SourceIdentity(size={self.size}, etag=<redacted>)'


@dataclass(frozen=True,repr=False)
class _Target:
    scheme: str
    host: str
    port: int
    path_query: str = field(repr=False)
    path: str = field(repr=False)
    @property
    def origin(self) -> str:
        host=f'[{self.host}]' if ':' in self.host else self.host
        return f'{self.scheme}://{host}:{self.port}'


def _target(url: str,policy: RangePolicy) -> _Target:
    try:
        if not isinstance(url,str) or len(url)>16384 or any(ord(c)<33 or ord(c)>126 for c in url):
            raise RangeSourceError('URL must be bounded ASCII with encoded path characters')
        parts=urlsplit(url)
        if parts.scheme not in ('https','http') or (parts.scheme=='http' and not policy.allow_http):
            raise RangeSourceError('URL scheme is not permitted')
        if parts.username is not None or parts.password is not None or parts.fragment or not parts.hostname:
            raise RangeSourceError('URL userinfo/fragments or missing hosts are not permitted')
        host=_host(parts.hostname);port=parts.port or (443 if parts.scheme=='https' else 80)
        if host not in policy.allowed_hosts or port not in policy.allowed_ports:
            raise RangeSourceError('Destination host or port is not allowlisted')
        path=parts.path or '/'
        if not path.startswith('/') or '\\' in path:
            raise RangeSourceError('Invalid HTTP path')
        return _Target(parts.scheme,host,port,path+('?' + parts.query if parts.query else ''),path)
    except RangeSourceError:raise
    except (ValueError,UnicodeError):raise RangeSourceError('Invalid URL') from None


def _addresses(target: _Target,policy: RangePolicy) -> list[tuple[int,tuple[str,int] | tuple[str,int,int,int]]]:
    try:infos=socket.getaddrinfo(target.host,target.port,type=socket.SOCK_STREAM)
    except OSError:raise RangeSourceError('Destination resolution failed') from None
    result: list[tuple[int,tuple[str,int] | tuple[str,int,int,int]]]=[]
    allowed=[ipaddress.ip_network(n) for n in policy.allowed_private_networks]
    for family,_,_,_,address in infos:
        if family not in (socket.AF_INET,socket.AF_INET6):continue
        ip=ipaddress.ip_address(address[0])
        effective=ip.ipv4_mapped if isinstance(ip,ipaddress.IPv6Address) and ip.ipv4_mapped is not None else ip
        if ip.is_multicast or ip.is_unspecified or effective.is_multicast or effective.is_unspecified or (isinstance(ip,ipaddress.IPv6Address) and ip.scope_id is not None):
            raise RangeSourceError('Unsafe destination address')
        if not effective.is_global and not any(effective in net for net in allowed):
            raise RangeSourceError('Private/reserved network is not permitted')
        item=(family,address)
        if item not in result:result.append(item)
    if not result:raise RangeSourceError('No permitted destination address')
    return result


class _PollingReader(io.RawIOBase):
    """Poll recv without poisoning SocketIO's buffered reader on a timeout.

    The original makefile is retained solely for socket lifetime accounting.
    No reads go through it. Inactivity retains the configured timeout, while
    cancellation is checked at least every 100 ms of a stalled socket read.
    """
    def __init__(self,sock: socket.socket,keepalive: BinaryIO,timeout: float,check: Callable[[],None]) -> None:
        self._socket=sock;self._keepalive=keepalive;self._timeout=timeout;self._check=check
        sock.settimeout(min(.1,timeout))

    def readable(self) -> bool:return True

    def readinto(self,buffer: WriteableBuffer) -> int:
        deadline=time.monotonic()+self._timeout
        while True:
            self._check()
            try:
                count=self._socket.recv_into(buffer)
                self._check()
                return count
            except socket.timeout:
                self._check()
                if time.monotonic()>=deadline:raise

    def close(self) -> None:
        try:self._keepalive.close()
        finally:super().close()


class _PinnedConnection(http.client.HTTPConnection):
    def __init__(self,target: _Target,address: tuple[int,tuple[str,int] | tuple[str,int,int,int]],timeout: float,check: Callable[[],None]=lambda: None) -> None:
        super().__init__(target.host,target.port,timeout=timeout)
        self._target=target;self._address=address
        self._interrupt: socket.socket | None=None
        self._aborted=threading.Event()
        def check_cancel() -> None:
            if self._aborted.is_set():raise CancelledError('Range read cancelled')
            check()
        self._check=check_cancel
        class Response(http.client.HTTPResponse):
            def __init__(response_self,sock: socket.socket,debuglevel: int=0,method: str | None=None,url: str | None=None) -> None:
                super().__init__(sock,debuglevel=debuglevel,method=method,url=url)
                assert response_self.fp is not None
                # A one-byte buffer prevents header parsing from prefetching an
                # unvalidated body, or a short range read from fetching extra data.
                response_self.fp=io.BufferedReader(_PollingReader(sock,response_self.fp,timeout,check_cancel),buffer_size=1)
        self.response_class=Response
    def connect(self) -> None:
        family,address=self._address
        sock=socket.socket(family,socket.SOCK_STREAM);sock.settimeout(self.timeout)
        try:
            sock.connect(address)
            self._check()
            if ipaddress.ip_address(sock.getpeername()[0])!=ipaddress.ip_address(address[0]):
                raise RangeSourceError('Connected peer identity differs')
            if self._target.scheme=='https':
                sock=ssl.create_default_context().wrap_socket(sock,server_hostname=self._target.host)
            self._check()
            # Keep the actual socket used by HTTPResponse's makefile. A duplicated
            # Windows socket does not reliably interrupt recv on the original.
            # HTTPConnection.close() can detach it, but makefile keeps its handle
            # alive until the body closes; shutdown on this object wakes that read.
            self._interrupt=sock
            self.sock=sock
        except BaseException:
            sock.close();self.finish();raise

    def abort(self) -> None:
        self._aborted.set()
        if self._interrupt is not None:
            try:self._interrupt.shutdown(socket.SHUT_RDWR)
            except OSError:pass
        self.close()

    def finish(self) -> None:
        self.close()
        if self._interrupt is not None:
            self._interrupt.close();self._interrupt=None


class HTTPRangeSource:
    """Single active read; total budget and resource validator live until close.

    Arbitrary read-all/file interfaces are intentionally absent. Credentials are
    request-only; custom headers are sent only to the initial origin. Provenance
    omits URL query/path, all headers and raw ETag. No transport/model default is
    changed by constructing this internal primitive.
    """
    def __init__(self,url: str,*,policy: RangePolicy,identity: SourceIdentity | None=None,headers: Mapping[str,str] | None=None) -> None:
        self._url=url;self.policy=policy;self._initial=_target(url,policy)
        if identity is not None and identity.size>policy.max_resource_bytes:
            raise RangeSourceError('Expected resource exceeds size bound')
        self._identity=identity;self._headers=dict(headers or {})
        forbidden={'host','range','if-match','if-range','accept-encoding','connection','transfer-encoding','content-length','proxy-authorization','cookie'}
        for k,v in self._headers.items():
            if (not isinstance(k,str) or not re.fullmatch(r'[A-Za-z0-9-]+',k) or k.lower() in forbidden
                    or not isinstance(v,str) or len(v)>8192 or any(ord(c)<32 or ord(c)>126 for c in v)):
                raise RangeSourceError('Invalid or controlled custom header')
        if len({k.lower() for k in self._headers})!=len(self._headers) or sum(len(k)+len(v) for k,v in self._headers.items())>16384:
            raise RangeSourceError('Custom headers are duplicate or oversized')
        self._lock=threading.Lock();self._state_lock=threading.Lock();self._cancelled=threading.Event()
        self._active: _PinnedConnection | None=None
        self._closed=False;self._failed=False;self._reserved=0;self._received=0;self._requests=0;self._reads=0

    def __repr__(self) -> str:return f'HTTPRangeSource(origin={self._initial.origin!r}, credentials=<redacted>)'

    @property
    def identity(self) -> SourceIdentity | None:return self._identity

    def _check_cancel(self,event: threading.Event | None) -> None:
        if self._cancelled.is_set() or (event is not None and event.is_set()):
            raise CancelledError('Range read cancelled')

    @staticmethod
    def _header(response: http.client.HTTPResponse,name: str,*,required: bool=True) -> str | None:
        values=response.headers.get_all(name,[])
        if len(values)>1 or (required and len(values)!=1):raise RangeSourceError('Missing or duplicate response identity/framing header')
        return str(values[0]) if values else None

    def _validate(self,response: http.client.HTTPResponse,offset: int,length: int) -> SourceIdentity:
        if response.status!=206:raise RangeSourceError('Server did not return a partial response (206 required)')
        if self._header(response,'Transfer-Encoding',required=False) is not None:
            raise RangeSourceError('Chunked/ambiguous range framing is unsupported')
        if self._header(response,'Content-Encoding',required=False) not in (None,'identity'):
            raise RangeSourceError('Encoded range payload is unsupported')
        content_range=self._header(response,'Content-Range') or ''
        match=re.fullmatch(r'bytes ([0-9]+)-([0-9]+)/([0-9]+)',content_range)
        declared=self._header(response,'Content-Length') or ''
        if not match or not re.fullmatch(r'[0-9]+',declared):raise RangeSourceError('Malformed range framing')
        start,end,total=map(int,match.groups())
        if (start!=offset or end!=offset+length-1 or total<=end or total>self.policy.max_resource_bytes or int(declared)!=length):
            raise RangeSourceError('Range extent, length or resource size differs')
        identity=SourceIdentity(self._header(response,'ETag') or '',total)
        if self._identity is not None and identity!=self._identity:raise RangeSourceError('Resource validator or total size changed')
        return identity

    def read_range(self,offset: int,length: int,*,cancel_event: threading.Event | None=None) -> bytes:
        if type(offset) is not int or offset<0 or type(length) is not int or not 0<length<=self.policy.max_range_bytes:
            raise RangeSourceError('Invalid explicit range')
        if offset+length>self.policy.max_resource_bytes or (self._identity is not None and offset+length>self._identity.size):
            raise RangeSourceError('Range exceeds resource extent')
        if not self._lock.acquire(blocking=False):raise RangeSourceError('One range read is already active')
        try:
            self._check_cancel(cancel_event)
            if self._closed or self._failed:raise RangeSourceError('Range source is closed or failed')
            if self._reserved+length>self.policy.max_total_bytes:raise RangeSourceError('Total range byte budget exceeded')
            self._reserved+=length
            url=self._url;previous=self._initial
            for hop in range(self.policy.max_redirects+1):
                self._check_cancel(cancel_event);target=_target(url,self.policy)
                if previous.scheme=='https' and target.scheme!='https':raise RangeSourceError('HTTPS redirect downgrade is forbidden')
                if hop and (self.policy.redirects=='none' or (self.policy.redirects=='same-origin' and target.origin!=self._initial.origin)):
                    raise RangeSourceError('Redirect destination is not permitted')
                addresses=_addresses(target,self.policy);self._check_cancel(cancel_event)
                if self._requests>=self.policy.max_requests:raise RangeSourceError('Request budget exceeded')
                self._requests+=1
                connection=_PinnedConnection(target,addresses[0],self.policy.timeout_seconds,lambda: self._check_cancel(cancel_event))
                with self._state_lock:self._active=connection
                response: http.client.HTTPResponse | None=None
                try:
                    self._check_cancel(cancel_event)
                    headers=dict(self._headers) if target.origin==self._initial.origin else {}
                    headers.update({'Range':f'bytes={offset}-{offset+length-1}','Accept-Encoding':'identity','Connection':'close'})
                    if self._identity is not None:headers['If-Match']=self._identity.etag
                    connection.request('GET',target.path_query,headers=headers)
                    response=connection.getresponse();self._check_cancel(cancel_event)
                    if sum(len(k)+len(v) for k,v in response.getheaders())>16384:
                        raise RangeSourceError('Response headers exceed bound')
                    if response.status in (301,302,303,307,308):
                        if hop==self.policy.max_redirects or self.policy.redirects=='none':raise RangeSourceError('Redirect budget or policy rejected response')
                        location=self._header(response,'Location') or ''
                        url=urljoin(url,location);previous=target;continue
                    identity=self._validate(response,offset,length)
                    chunks=[];remaining=length
                    while remaining:
                        self._check_cancel(cancel_event)
                        block=response.read1(min(remaining,self.policy.chunk_bytes))
                        self._check_cancel(cancel_event)
                        if not block:raise RangeSourceError('Truncated partial response')
                        self._received+=len(block);remaining-=len(block);chunks.append(block)
                    self._check_cancel(cancel_event)
                    self._identity=identity;self._reads+=1
                    return b''.join(chunks)
                finally:
                    if response is not None:response.close()
                    connection.finish()
                    with self._state_lock:self._active=None
            raise RangeSourceError('Redirect budget exhausted')
        except CancelledError:
            self._failed=True;raise
        except RangeSourceError:
            self._failed=True;raise
        except (OSError,http.client.HTTPException,ValueError):
            self._failed=True;self._check_cancel(cancel_event)
            raise RangeSourceError('Range transport failed') from None
        finally:self._lock.release()

    def provenance(self) -> dict[str,object]:
        identity=self._identity
        return {'origin':self._initial.origin,
                'path_sha256':hashlib.sha256(self._initial.path.encode()).hexdigest(),
                'size':None if identity is None else identity.size,
                'etag_sha256':None if identity is None else hashlib.sha256(identity.etag.encode()).hexdigest(),
                'identity_scope':'server strong ETag plus total length; not a content checksum',
                'reserved_payload_bytes':self._reserved,'received_payload_bytes':self._received,
                'requests':self._requests,'completed_ranges':self._reads,'closed':self._closed,'failed':self._failed,
                'credentials':'URL path/query and request/response headers omitted',
                'bounds_scope':'payload bytes; HTTP parser has separate header limits'}

    def cancel(self) -> None:
        self._cancelled.set()
        with self._state_lock:
            connection=self._active
            if connection is not None:
                connection.abort()

    def close(self) -> None:
        self._closed=True;self.cancel()

    def __enter__(self) -> Self:return self
    def __exit__(self,*args: object) -> None:self.close()
