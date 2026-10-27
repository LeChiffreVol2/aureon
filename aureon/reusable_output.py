"""Explicit caller-owned reusable FP32 output, without an intermediate snapshot.

The caller owns the destination exclusively until return and must not consume
it on any exception. A later call cannot overwrite it unless the caller passes
the same destination again. Ordinary __call__ retains independent-output semantics.
Input snapshots, native ACK validation, finite scans and poison rules are unchanged.
"""
import hashlib
from pathlib import Path
import numpy as np
from .finite_profile import FiniteResidentPredictor
from .graph_pipeline import OUTPUT_SHAPE

class ReusableOutputPredictor(FiniteResidentPredictor):
    def _validate_destination(self, values, destination):
        if destination is None:return
        if (not isinstance(destination,np.ndarray) or destination.dtype!=np.dtype('<f4')
                or destination.shape!=OUTPUT_SHAPE or not destination.flags.c_contiguous
                or not destination.flags.writeable or np.shares_memory(destination,values)):
            raise ValueError('Output must be exclusive writable contiguous float32 [8,10,224,224], independent of input')
        if self._slots is not None:
            for kind in ('input','output'):
                view=self._slots._array(kind)
                try:
                    if np.shares_memory(destination,view):
                        raise ValueError('Caller output cannot alias transport mappings')
                finally:del view

    def _read_output_into(self, destination):
        view=self._slots._array('output')
        try:
            if not self.scanner.copy(destination,view):
                raise ValueError('Engine shared-memory output contains nonfinite logits; destination is invalid')
            return destination
        finally:del view

    def predict_into(self, values, destination):
        """Fill exactly this buffer after native completion; invalidate on exception.

        Passing None is rejected: choose __call__ for a new independent array.
        Reusing a destination intentionally replaces its previous prediction.
        Keep another destination (or explicitly copy) if the prior result is needed.
        """
        if destination is None:raise ValueError('Supply an explicit caller-owned destination')
        return self._predict(values,destination)

    def metadata(self):
        return {**super().metadata(),'reusable_output':True,
            'reusable_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'predict_into_contract':'caller owns destination exclusively through return; contents invalid on any exception; explicit reuse overwrites prior result',
            'predict_into_client_output_copies':1,'predict_into_intermediate_output_allocations':0,
            'predict_into_exposes_transport_mapping':False,
            'copies_saved_scope':'one intermediate full-logit copy versus __call__ followed by caller copy; not zero-copy and not removal of input snapshot'}
