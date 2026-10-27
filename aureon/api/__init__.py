"""Versioned local Engine facade. Imports are stdlib-only and never execute models."""
from .v1 import SceneRequest, SceneResult, ExecutionRequired, run_scene
from .profiles import ProfileError, inspect_profile, require_profile
from .session import EngineSession, SessionBusy, SessionClosed, SessionFailed
from .models import GenerationRequest, list_models, inspect_model, run_model, open_model_session
from ..client import SceneClient, ClientError, APIError, SubmissionUncertain, JobNotCompleted

__all__ = ["SceneRequest", "SceneResult", "ExecutionRequired", "run_scene",
           "ProfileError", "inspect_profile", "require_profile", "SceneClient",
           "ClientError", "APIError", "SubmissionUncertain", "JobNotCompleted"]
__all__ += ["EngineSession", "SessionBusy", "SessionClosed", "SessionFailed"]
__all__ += ["GenerationRequest", "list_models", "inspect_model", "run_model", "open_model_session"]
API_VERSION = "aureon.engine.v1"
