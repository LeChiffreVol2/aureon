"""Bounded-memory event journal and owner-thread resource observations."""
import json
import os
from pathlib import Path
import threading
import time


class Journal:
    def __init__(self, path):
        self.stream = Path(path).open('x', buffering=65536)
        self.lock = threading.Lock()
        self.count = 0

    def __call__(self, value):
        encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
        with self.lock:
            self.stream.write(encoded)
            self.count += 1

    def flush(self):
        with self.lock:
            self.stream.flush()

    def close(self):
        with self.lock:
            self.stream.close()


class ResourceObserver:
    def __init__(self, adapter, server, path):
        self.adapter, self.server = adapter, server
        self.torch = adapter.engine.torch if adapter.engine else adapter.graph.torch
        self.journal = Journal(path)
        self.next_sample = 0.

    def sample(self, *, force=False):
        now = time.perf_counter()
        if not force and now < self.next_sample:
            return False
        self.next_sample = now + 1.
        status = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS','RssAnon','RssFile','RssShmem','VmSize','VmHWM'):
                status[key+'_bytes'] = int(value.split()[0])*1024
            elif key == 'THP_enabled':
                status[key] = int(value.strip())
        with self.server.guard:
            ownership = {'connections':len(self.server.connections), 'active_ids':len(self.server.active_ids)}
        self.journal({'at':now,'pid':os.getpid(),**status,**ownership,
            'fd_count':len(os.listdir('/proc/self/fd')), 'os_threads':len(os.listdir('/proc/self/task')),
            'python_threads':threading.active_count(), 'queued':self.server.jobs.qsize(),
            'free_slots':self.server.free.qsize(), 'retained_records':len(self.server.records),
            'completed':self.server.record_count,'rejected':self.server.rejected,
            'cuda_allocated_bytes':self.torch.cuda.memory_allocated(),
            'cuda_reserved_bytes':self.torch.cuda.memory_reserved(),
            'cuda_max_allocated_bytes':self.torch.cuda.max_memory_allocated(),
            'cuda_max_reserved_bytes':self.torch.cuda.max_memory_reserved()})
        self.journal.flush()
        return True

    def close(self):
        self.journal.close()
