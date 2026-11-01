"""TerraMind adapter over a reusable single-signature tensor graph runtime.

Public TerraMindSession native/graph choices are unchanged. Eligibility and
numerical operations live in denoiser_binding; graph/buffer/owner lifecycle lives
in graph_runtime. This candidate requires separate real-model qualification.
"""
from contextlib import contextmanager
import threading
import types
import torch
from .denoiser_binding import (PROFILE_ID, TARGET, S1GRDDecoderBinding, eligibility,
                               native_frequency_cpu, staged_embedding)
from .graph_runtime import FixedTensorGraph

# The pinned decoder calls a module-global embedding function. Do not allow
# overlapping scoped replacements, including separate adapter instances.
_PATCH_LOCK = threading.Lock()
_MISSING = object()


class DenoiserGraph:
    def __init__(self, model):
        self.binding = S1GRDDecoderBinding(model)
        self.runtime = FixedTensorGraph(self.binding)
        self.total = self.runtime.total

    def _memory(self):
        return {'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved()}

    def validate(self, x, t, cond, cond_mask, kwargs):
        if cond_mask is not None or kwargs != {'orig_res': None}:
            raise ValueError('Unsupported decoder optional inputs')
        self.binding.validate({'image': x, 'timestep': t, 'conditioning': cond})

    def call(self, x, t, cond, cond_mask=None, **kwargs):
        self.runtime.owner()
        self.validate(x, t, cond, cond_mask, kwargs)
        return self.runtime.call({'image': x, 'timestep': t, 'conditioning': cond})

    @contextmanager
    def request(self, model):
        self.runtime.owner()
        if model.tokenizer[TARGET].decoder is not self.binding.module:
            raise ValueError('Graph belongs to a different decoder instance')
        if not _PATCH_LOCK.acquire(blocking=False):
            raise RuntimeError('Decoder embedding binding is already in use')
        try:
            with self.runtime.request():
                module = self.binding.module
                if module.forward != self.binding.original:
                    raise ValueError('Decoder forward binding changed')
                old = module.__dict__.get('forward', _MISSING)
                namespace = self.binding.namespace()
                old_embedding = namespace['timestep_embedding']
                start = self.total['replay_stage_forwards']
                setup_before = self.total['setup_seconds']
                receipt = {'profile_id': PROFILE_ID, 'enabled': True, 'restored': False,
                           'fallback': False, 'memory_before': self._memory(),
                           'runtime_policy': 'one_capture_no_recapture',
                           'qualification': 'external_candidate_evidence_required'}

                def wrapped(this, x, t, cond, cond_mask=None, **kwargs):
                    if receipt['fallback']:
                        return self.binding.original(x, t, cond, cond_mask=cond_mask, **kwargs)
                    try:
                        self.validate(x, t, cond, cond_mask, kwargs)
                    except ValueError as exc:
                        if self.runtime.started:
                            raise
                        receipt.update(enabled=False, fallback=True, reason=str(exc))
                        namespace['timestep_embedding'] = old_embedding
                        return self.binding.original(x, t, cond, cond_mask=cond_mask, **kwargs)
                    return self.runtime.call({'image': x, 'timestep': t, 'conditioning': cond})

                try:
                    module.forward = types.MethodType(wrapped, module)
                    namespace['timestep_embedding'] = self.binding.embedding
                    yield receipt
                except BaseException:
                    self.runtime.failed = True
                    self.total['failed'] = True
                    raise
                finally:
                    if old is _MISSING:
                        if 'forward' in module.__dict__:
                            delattr(module, 'forward')
                    else:
                        module.forward = old
                    namespace['timestep_embedding'] = old_embedding
                    receipt.update(restored=True, replays=self.total['replay_stage_forwards']-start,
                                   setup_seconds_this_request=self.total['setup_seconds']-setup_before,
                                   memory_after=self._memory(), total=dict(self.total))
        finally:
            _PATCH_LOCK.release()

    def close(self):
        return self.runtime.close()
