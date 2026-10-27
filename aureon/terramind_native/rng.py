"""Serial request capsule for unchanged upstream global RNG calls."""
from contextlib import contextmanager
import hashlib
import pickle
import random
import threading

_LOCK = threading.Lock()


def snapshot(torch, np, devices):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": [torch.cuda.get_rng_state(d).clone() for d in devices]}


def state_digest(state):
    h = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    for tensor in [state["torch"], *state["cuda"]]: h.update(tensor.numpy().tobytes())
    return h.hexdigest()


@contextmanager
def rng_capsule(seed, *, devices=()):
    import numpy as np
    import torch
    if type(seed) is not int or not 0 <= seed < 2**63: raise ValueError("Invalid seed")
    if not _LOCK.acquire(blocking=False): raise RuntimeError("Native RNG capsule already in use")
    before = None
    receipt = {}
    try:
        before = snapshot(torch, np, devices)
        receipt["global_before_sha256"] = state_digest(before)
        random.seed(seed); np.random.seed(seed & 0xffffffff)
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            with torch.cuda.device(device): torch.cuda.manual_seed(seed)
        yield receipt
    finally:
        try:
            if before is not None:
                receipt["request_post_rng_sha256"] = state_digest(snapshot(torch, np, devices))
                random.setstate(before["python"]); np.random.set_state(before["numpy"])
                torch.set_rng_state(before["torch"])
                for device, state in zip(devices, before["cuda"]): torch.cuda.set_rng_state(state, device)
                receipt["global_after_sha256"] = state_digest(snapshot(torch, np, devices))
                receipt["restored"] = receipt["global_after_sha256"] == receipt["global_before_sha256"]
                if not receipt["restored"]: raise RuntimeError("RNG restoration failed")
        finally:
            _LOCK.release()
