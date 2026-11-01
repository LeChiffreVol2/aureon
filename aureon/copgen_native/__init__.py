"""Experimental, explicit COP-GEN S2L2A192 -> synthetic S1RTC acceptance adapter.

No public route registration, acquisition or model loading occurs on import.
"""
from .contracts import CopgenInput, CopgenPlan, plan_copgen
from .engine import CopgenEngine, export_copgen

__all__ = ['CopgenInput', 'CopgenPlan', 'plan_copgen', 'CopgenEngine', 'export_copgen']
