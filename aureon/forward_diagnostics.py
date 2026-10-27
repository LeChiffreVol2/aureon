"""Opt-in, bounded model timing; synchronization makes this diagnostic only."""
import json
from pathlib import Path
from time import perf_counter


def install_forward_diagnostics(model, folder, limit=64):
    import torch

    if type(limit) is not int or limit <= 0:
        raise ValueError('Forward diagnostics limit must be a positive integer')
    records = []
    pending = []
    path = Path(folder) / 'forward-diagnostics.jsonl'
    stream = path.open('x')

    def before(module, args):
        if len(records) >= limit:
            return
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        pending.append((perf_counter(), start, end))

    def after(module, args, output):
        if not pending:
            return
        began, start, end = pending.pop()
        # PyTorch calls this hook with no output if forward raised. Discard
        # that interval rather than leaking it into a later successful call.
        if output is None or len(records) >= limit:
            return
        queued = perf_counter()
        end.record()
        end.synchronize()
        record = {'index': len(records), 'forward_host_enqueue_ms': (queued-began)*1000,
                  'forward_synchronized_ms': (perf_counter()-began)*1000,
                  'cuda_event_elapsed_ms': start.elapsed_time(end),
                  'scope': 'diagnostic synchronization; CUDA interval includes host launch gaps, not kernel-only sum'}
        records.append(record)
        stream.write(json.dumps(record) + '\n')
        stream.flush()
        if len(records) == limit:
            stream.close()

    return model.register_forward_pre_hook(before), model.register_forward_hook(after, always_call=True)
