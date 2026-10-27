"""Bounded binary HTTP front door, shared by both explicit GPU backends.

Eight reusable request/output slots outlive queue, execution and socket send.
Only the creating thread calls the adapter. HTTP threads never touch CUDA.
Importing this module performs no model execution.
"""
from dataclasses import dataclass, field
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import re
import socket
import threading
import time

import numpy as np

INPUT_SHAPE = (8, 12, 224, 224)
OUTPUT_SHAPE = (8, 10, 224, 224)
INPUT_BYTES = int(np.prod(INPUT_SHAPE)) * 4
OUTPUT_BYTES = int(np.prod(OUTPUT_SHAPE)) * 4
PATH = '/v1/kao-terramind-b/logits'


@dataclass
class Job:
    values: np.ndarray
    output: np.ndarray
    request_id: str = ''
    received: float = 0.
    enqueued: float = 0.
    started: float = 0.
    finished: float = 0.
    error: str | None = None
    status: int = 200
    stages: dict = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)


def read_exact(stream, target):
    """Read directly into owned storage, rejecting truncated request bodies."""
    view = memoryview(target).cast('B')
    offset = 0
    while offset < len(view):
        n = stream.readinto(view[offset:])
        if not n:
            raise EOFError('Truncated tensor body')
        offset += n


class QueuedServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(self, address, *, capacity=8, queue_timeout=10., socket_timeout=15., event_sink=None, jobs=None):
        if type(capacity) is not int or not 1 <= capacity <= 8:
            raise ValueError('Capacity must be 1..8')
        self.owner = threading.get_ident()
        self.jobs = queue.Queue(maxsize=capacity)
        self.free = queue.Queue(maxsize=capacity)
        if jobs is None:
            jobs=[Job(np.zeros(INPUT_SHAPE,np.float32),np.zeros(OUTPUT_SHAPE,np.float32)) for _ in range(capacity)]
        if len(jobs)!=capacity:
            raise ValueError('One exclusive input/output pair per admission slot required')
        arrays=[]
        for job in jobs:
            for values,shape in ((job.values,INPUT_SHAPE),(job.output,OUTPUT_SHAPE)):
                if (not isinstance(values,np.ndarray) or values.shape!=shape or values.dtype!=np.dtype('<f4')
                        or not values.flags.c_contiguous or not values.flags.writeable
                        or any(np.shares_memory(values,previous) for previous in arrays)):
                    raise ValueError('Admission buffers must be distinct writable contiguous FP32 arrays')
                arrays.append(values)
            self.free.put(job)
        self.capacity, self.queue_timeout, self.socket_timeout = capacity, queue_timeout, socket_timeout
        self.connections = set()
        self.connection_limit = threading.BoundedSemaphore(16)
        self.guard = threading.Lock()
        self.active_ids = set()
        self.closing = False
        self.failed = False
        self.records = deque(maxlen=4096)
        self.record_count = 0
        self.rejected = 0
        self.event_sink = event_sink
        super().__init__(address, Handler)

    def process_request(self, request, address):
        if not self.connection_limit.acquire(blocking=False):
            request.close()
            return
        with self.guard:
            self.connections.add(request)
        try:
            super().process_request(request, address)
        except BaseException:
            with self.guard:
                self.connections.discard(request)
            self.connection_limit.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            with self.guard:
                self.connections.discard(request)
            self.connection_limit.release()

    def execute_one(self, adapter, timeout=.1):
        if threading.get_ident() != self.owner:
            raise RuntimeError('GPU execution requires the original owner thread')
        try:
            job = self.jobs.get(timeout=timeout)
        except queue.Empty:
            return False
        job.started = time.perf_counter()
        try:
            if self.failed:
                job.status = 503
                raise RuntimeError('Backend unavailable after execution failure')
            if job.started - job.enqueued > self.queue_timeout:
                job.status = 504
                raise TimeoutError('Queue deadline expired before GPU execution')
            job.stages = adapter.predict_into(job.values, job.output, job.request_id)
        except Exception as exc:
            job.error = type(exc).__name__ + ': ' + str(exc)
            if job.status == 200:
                job.status = 502
                self.failed = True
        finally:
            job.finished = time.perf_counter()
            record = {'event': 'execution', 'request_id': job.request_id, 'received': job.received,
                'enqueued': job.enqueued, 'started': job.started, 'finished': job.finished,
                'status': job.status, 'error': job.error, 'stages': job.stages}
            self.records.append(record)
            self.record_count += 1
            try:
                if self.event_sink:
                    self.event_sink(record)
            except Exception:
                self.failed = True
                job.status, job.error = 502, 'Completion journal failed'
                raise
            finally:
                job.done.set()
                self.jobs.task_done()
        return True

    def stop_accepting(self):
        with self.guard:
            self.closing = True
        self.shutdown()

    def close_connections(self):
        # Call only after the owner has drained all admitted work. A handler
        # releases its buffer only after GPU completion and socket send.
        with self.guard:
            sockets = list(self.connections)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.server_close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        self.request.settimeout(self.server.socket_timeout)
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        super().setup()

    def log_message(self, *args):
        pass

    def reply_error(self, status, message):
        self.close_connection = True
        body = json.dumps({'error': message}).encode()
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Content-Type', 'application/json')
        self.send_header('Connection', 'close')
        rid = self.headers.get('X-Request-ID', '')
        if re.fullmatch('[0-9a-f]{32}', rid):
            self.send_header('X-Request-ID', rid)
        if status == 429:
            self.send_header('Retry-After', '1')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != '/health':
            return self.reply_error(404, 'Unknown endpoint')
        body = json.dumps({'ready': not self.server.failed and not self.server.closing,
                           'model': 'KAO-TerraMind B'}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _admit(self):
        server = self.server
        request_id = self.headers.get('X-Request-ID', '')
        if (self.path != PATH or not re.fullmatch('[0-9a-f]{32}', request_id)
                or self.headers.get_all('Content-Length') != [str(INPUT_BYTES)]
                or self.headers.get('Transfer-Encoding') is not None
                or self.headers.get('Expect', '').lower() not in ('', '100-continue')
                or self.headers.get('Content-Type') != 'application/octet-stream'):
            self.reply_error(400, 'Require fixed little-endian FP32 body and unique request ID')
            return None
        job = None
        with server.guard:
            if server.closing or server.failed:
                status = 503
            elif request_id in server.active_ids:
                status = 409
            else:
                try:
                    job = server.free.get_nowait()
                    server.active_ids.add(request_id)
                    status = 200
                except queue.Empty:
                    status = 429
            if status != 200:
                server.rejected += 1
        if job is None:
            if server.event_sink:
                server.event_sink({'event':'rejection','request_id':request_id,'status':status,'at':time.perf_counter()})
            self.reply_error(status, 'Admission unavailable')
            return None
        job.request_id, job.received = request_id, time.perf_counter()
        job.done.clear()
        job.error, job.status, job.stages = None, 200, {}
        return job

    def _release(self, job):
        with self.server.guard:
            self.server.active_ids.remove(job.request_id)
        self.server.free.put_nowait(job)
        self.pending_job = None

    def handle_expect_100(self):
        if self.command != 'POST':
            self.reply_error(417, 'Expect is supported only for tensor POST')
            return False
        job = self._admit()
        if job is None:
            return False
        self.pending_job = job
        try:
            self.send_response_only(100)
            self.end_headers()
        except BaseException:
            self._release(job)
            raise
        return True

    def do_POST(self):
        server = self.server
        job = getattr(self, 'pending_job', None) or self._admit()
        if job is None:
            return
        admitted = False
        try:
            read_exact(self.rfile, job.values)
            with server.guard:
                if server.closing:
                    return self.reply_error(503, 'Service draining')
                job.enqueued = time.perf_counter()
                server.jobs.put_nowait(job)
                admitted = True
            # Never reuse the slot because a caller timed out or disconnected.
            job.done.wait()
            if job.error:
                return self.reply_error(job.status, job.error)
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(OUTPUT_BYTES))
            self.send_header('X-Request-ID', job.request_id)
            self.send_header('X-Dtype', '<f4')
            self.send_header('X-Shape', '8,10,224,224')
            self.send_header('X-Queue-Ms', str((job.started-job.enqueued)*1000))
            self.send_header('X-Execution-Ms', str((job.finished-job.started)*1000))
            self.end_headers()
            self.wfile.write(memoryview(job.output).cast('B'))
        except (EOFError, OSError):
            self.close_connection = True
        finally:
            if admitted:
                job.done.wait()
            self._release(job)
