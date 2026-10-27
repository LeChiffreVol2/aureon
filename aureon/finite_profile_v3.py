"""Explicit SIMD-unrolled validation candidate; unchanged numerical model."""
from .finite_profile import FiniteRegisteredGraph, FiniteResidentPredictor
from .finite_scan_v3 import FiniteScannerV3

class FiniteV3RegisteredGraph(FiniteRegisteredGraph):
    scanner_type = FiniteScannerV3

class FiniteV3ResidentPredictor(FiniteResidentPredictor):
    scanner_type = FiniteScannerV3
