"""Owned fixed-eight HTTP client with admission before a large tensor upload."""
import http.client
import re
import socket
import time

from .queued_http import INPUT_BYTES, OUTPUT_BYTES, PATH, read_exact


class BinaryQueueClient:
    def __init__(self, host='127.0.0.1', port=8002, timeout=15.):
        self.host, self.port, self.timeout = host, port, timeout
        self.socket = self.reader = None
        self.output = bytearray(OUTPUT_BYTES)

    def close(self):
        if self.reader is not None:
            self.reader.close()
        if self.socket is not None:
            self.socket.close()
        self.reader = self.socket = None

    def connect(self):
        if self.socket is None:
            self.socket = socket.create_connection((self.host, self.port), self.timeout)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.reader = self.socket.makefile('rb')

    def _headers(self):
        line = self.reader.readline(65537)
        if len(line)>65536 or not line:
            raise ValueError('Missing or oversized HTTP status')
        parts = line.decode('ascii').split(None, 2)
        if len(parts)<2 or parts[0] != 'HTTP/1.1' or not parts[1].isdigit():
            raise ValueError('Invalid HTTP status')
        return int(parts[1]), http.client.parse_headers(self.reader)

    def request(self, body, request_id):
        if len(body)!=INPUT_BYTES or not re.fullmatch('[0-9a-f]{32}', request_id):
            raise ValueError('Require a fixed FP32 group and request ID')
        start = time.perf_counter()
        sent_body = False
        try:
            self.connect()
            headers=(f'POST {PATH} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n'
                f'Content-Length: {INPUT_BYTES}\r\nContent-Type: application/octet-stream\r\n'
                f'X-Request-ID: {request_id}\r\nExpect: 100-continue\r\nConnection: keep-alive\r\n\r\n').encode('ascii')
            self.socket.sendall(headers)
            status, response = self._headers()
            admission = time.perf_counter()
            if status == 100:
                self.socket.sendall(body)
                sent_body = True
                status, response = self._headers()
            if response.get('X-Request-ID') != request_id or response.get('Transfer-Encoding') is not None:
                raise ValueError('Response identity/framing differs')
            lengths=response.get_all('Content-Length', [])
            if len(lengths)!=1 or not lengths[0].isdigit():
                raise ValueError('Missing response length')
            length=int(lengths[0])
            if status==200:
                if (not sent_body or length!=OUTPUT_BYTES or response.get('X-Dtype')!='<f4'
                        or response.get('X-Shape')!='8,10,224,224'):
                    raise ValueError('Logit response contract differs')
                read_exact(self.reader, self.output)
                result={'queue_ms':float(response['X-Queue-Ms']),
                        'execution_ms':float(response['X-Execution-Ms'])}
            else:
                if length>2048:
                    raise ValueError('Error body exceeds bound')
                error=bytearray(length)
                read_exact(self.reader, error)
                result={'error':error.decode('utf-8',errors='replace')}
            result.update(request_id=request_id,start=start,end=time.perf_counter(),status=status,
                          input_body_sent=sent_body,admission_ms=(admission-start)*1000)
            if response.get('Connection','').lower()=='close':
                self.close()
            return result
        except BaseException:
            self.close()
            raise
