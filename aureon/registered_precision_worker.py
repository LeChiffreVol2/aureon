"""Explicit native worker extension, loaded only for registered-buffer runs."""
import os
from .vllm_worker import PrecisionWorker


class RegisteredPrecisionWorker(PrecisionWorker):
    def compile_or_warm_up_model(self):
        if os.environ.get('AUREON_REGISTERED_SLOTS')!='1':
            raise ValueError('Registered worker requires explicit enablement')
        result=super().compile_or_warm_up_model()
        from .registered_worker import install_registered_worker
        controller = install_registered_worker(self)
        if os.environ.get('AUREON_FINITE_SCAN') == '1':
            from .finite_profile import FiniteRegisteredGraph
            from .finite_scan import FiniteScanner
            # Verify the explicit local build before readiness, not the first request.
            identity = FiniteScanner().metadata()
            controller._pipeline_factory = FiniteRegisteredGraph.from_verified_worker
            controller.finite_scan = identity
            controller._event('finite-scan-installed')
        if os.environ.get('AUREON_FINITE_SIMD') == '1':
            if os.environ.get('AUREON_FINITE_SCAN') != '1':
                raise ValueError('SIMD profile requires finite validation')
            from .finite_profile_v3 import FiniteV3RegisteredGraph
            from .finite_scan_v3 import FiniteScannerV3
            controller._pipeline_factory = FiniteV3RegisteredGraph.from_verified_worker
            controller.finite_scan = FiniteScannerV3().metadata()
            controller._event('finite-simd-installed')
        if os.environ.get('AUREON_CPU_ACK') == '1':
            from .control_ack import install_cpu_ack
            install_cpu_ack(self,controller)
        if os.environ.get('AUREON_CPU_CONTROL') == '1':
            from .cpu_control import install_cpu_descriptor
            install_cpu_descriptor(controller)
        return result
