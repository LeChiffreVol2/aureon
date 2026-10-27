"""Opt-in finite-scan candidate; retains all ownership and completion guards."""
from .finite_scan import FiniteScanner
from .registered_graph import RegisteredGroupGraph
from .resident_transport import ResidentSharedMemoryPredictor
import numpy as np

class FiniteRegisteredGraph(RegisteredGroupGraph):
    scanner_type = FiniteScanner
    def __init__(self, *args, **kwargs):
        self.scanner = self.scanner_type()
        super().__init__(*args, **kwargs)

    def _all_finite(self, values): return self.scanner(values)

    def _event(self, event, **extra):
        return super()._event(event, finite_scan=self.scanner.metadata(), **extra)

    def metadata(self):
        return {**super().metadata(), 'finite_scan': self.scanner.metadata()}

class FiniteResidentPredictor(ResidentSharedMemoryPredictor):
    scanner_type = FiniteScanner
    def __init__(self, *args, **kwargs):
        self.scanner = self.scanner_type()
        super().__init__(*args, **kwargs)

    def _all_finite(self, values): return self.scanner(values)

    def _write_input(self, values):
        # Preserve support for reverse/non-contiguous probes. Only the
        # ordinary contiguous path fuses its owned copy and finite check.
        if not values.flags.c_contiguous: return super()._write_input(values)
        view = self._slots._array('input')
        try: return self.scanner.copy(view, values)
        finally: del view

    def _read_output(self):
        view = self._slots._array('output')
        try:
            result = np.empty(view.shape, dtype=np.float32)
            if not self.scanner.copy(result, view):
                raise ValueError('Engine shared-memory output contains nonfinite logits')
            return result
        finally: del view

    def metadata(self):
        return {**super().metadata(), 'finite_scan': self.scanner.metadata(),
                'client_copy_validation_fused': True, 'independent_output': True}
