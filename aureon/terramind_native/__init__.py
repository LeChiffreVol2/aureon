"""Resident TerraMind S2L2A -> synthetic S1GRD implementation.

Artifact-specific support and qualification are documented in the support matrix.
Importing this package does not import a model runtime.
"""
from .session import TerraMindSession, SessionBusy, SessionClosed, SessionFailed

__all__ = ["TerraMindSession", "SessionBusy", "SessionClosed", "SessionFailed"]
