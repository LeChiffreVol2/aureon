from .v1 import SceneRequest as SceneRequest, SceneResult as SceneResult, ExecutionRequired as ExecutionRequired, run_scene as run_scene
from .profiles import ProfileError as ProfileError, inspect_profile as inspect_profile, require_profile as require_profile
from .session import EngineSession as EngineSession, SessionBusy as SessionBusy, SessionClosed as SessionClosed, SessionFailed as SessionFailed
from .models import GenerationRequest as GenerationRequest, list_models as list_models, inspect_model as inspect_model, run_model as run_model, open_model_session as open_model_session
from ..terramind_native.execution import plan_generation_execution as plan_generation_execution
from .memory_input import GenerationInput as GenerationInput
from ..terramind_native.memory_output import export_generation as export_generation
from .generation_engine import GenerationEngine as GenerationEngine
from .generation_queue import GenerationHandle as GenerationHandle, GenerationStream as GenerationStream
from .generation_bridges import GenerationRecipe as GenerationRecipe, GenerationJob as GenerationJob, TorchGeoAdapter as TorchGeoAdapter, generation_capabilities as generation_capabilities, generation_compatibility as generation_compatibility
from ..terramind_native.session import SessionBusy as GenerationBusy, SessionClosed as GenerationClosed, SessionFailed as GenerationFailed
from ..client import SceneClient as SceneClient, ClientError as ClientError, APIError as APIError, SubmissionUncertain as SubmissionUncertain, JobNotCompleted as JobNotCompleted
from .spatial import GenerationWindow as GenerationWindow, generation_windows as generation_windows, generation_tile_seed as generation_tile_seed, PreparedGenerationRaster as PreparedGenerationRaster, generate_raster_scene as generate_raster_scene
from .routing import ExecutionKey as ExecutionKey, PlacementRequest as PlacementRequest, WorkerSnapshot as WorkerSnapshot, plan_placement as plan_placement
from .contracts import ExecutionProfile as ExecutionProfile, Parallelism as Parallelism, ParallelFallback as ParallelFallback, EngineState as EngineState, HandleState as HandleState, QueueLimits as QueueLimits, ParallelPlan as ParallelPlan, RecipeValues as RecipeValues, GenerationResult as GenerationResult, QueueMetadata as QueueMetadata, EngineMetadata as EngineMetadata
API_VERSION: str
from .model_pool import ModelRoute as ModelRoute, PoolInput as PoolInput

from .terramind_encoder import TerraMindEncoder as TerraMindEncoder, TerraMindFeatures as TerraMindFeatures

from .diagnostics import diagnose_runtime as diagnose_runtime, DiagnosticIssue as DiagnosticIssue

from .generation_bridges import TorchGeoBatchAdapter as TorchGeoBatchAdapter
from .scene_generation import SceneGenerationRequest as SceneGenerationRequest, inspect_generation_scene as inspect_generation_scene, run_generation_scene as run_generation_scene

from .encoder_io import encode_file as encode_file, encode_file_with_engine as encode_file_with_engine, inspect_encoder_input as inspect_encoder_input
from .scene_generation import run_generation_scene_with_engine as run_generation_scene_with_engine

from .model_pool import (ModelPool as ModelPool, DeviceSlot as DeviceSlot, PoolResources as PoolResources, GenerationRoute as GenerationRoute, EncoderRoute as EncoderRoute, EncoderInput as EncoderInput, PoolResult as PoolResult, PoolHandle as PoolHandle, PooledFeatures as PooledFeatures, PoolError as PoolError, PoolBusy as PoolBusy, PoolClosed as PoolClosed, PoolExecutionError as PoolExecutionError, PoolCleanupError as PoolCleanupError)

from .gpu_devices import NvidiaDevice as NvidiaDevice, NvidiaInventory as NvidiaInventory, probe_nvidia_gpus as probe_nvidia_gpus, cuda_device_slots as cuda_device_slots

from .precision import ComputePrecision as ComputePrecision

from .model_bridge import ModelBridge as ModelBridge, BridgeCapabilities as BridgeCapabilities, BridgeCancelled as BridgeCancelled, ForwardAdapter as ForwardAdapter, OutputAdapter as OutputAdapter
from .preprocessing import PreprocessingRecipe as PreprocessingRecipe, PreparedTensor as PreparedTensor

from .terratorch_task import TerraTorchTaskConfig as TerraTorchTaskConfig, TerraTorchSegmentation as TerraTorchSegmentation, SegmentationPrediction as SegmentationPrediction, build_terratorch_task as build_terratorch_task

from .upstream_task import UpstreamTaskBridge as UpstreamTaskBridge, TaskConstructor as TaskConstructor, TaskKind as TaskKind, TaskPrediction as TaskPrediction

from .eo import (EOGrid as EOGrid, ConditioningSpec as ConditioningSpec, GeneratedTarget as GeneratedTarget, SamplingRecipe as SamplingRecipe, EORequest as EORequest, EOCapability as EOCapability, EORoutePlan as EORoutePlan, UnsupportedRoute as UnsupportedRoute, eo_capabilities as eo_capabilities, plan_eo as plan_eo, generate_eo as generate_eo)
from .stac import (AssetBand as AssetBand, STACSelection as STACSelection, ResolvedAssetBand as ResolvedAssetBand, ResolvedObservation as ResolvedObservation, resolve_stac_item as resolve_stac_item)

from .borrowed_model import BorrowedModelBridge as BorrowedModelBridge, BorrowedMutation as BorrowedMutation, DeviceTensorPolicy as DeviceTensorPolicy

from .temporal import GeographicConditioning as GeographicConditioning, TemporalSpec as TemporalSpec, TemporalInput as TemporalInput, TemporalFeatures as TemporalFeatures, TemporalBridge as TemporalBridge, Pooling as Pooling

from .adapter_registry import AdapterRegistry as AdapterRegistry, AdapterSpec as AdapterSpec, AdapterRequest as AdapterRequest, AdapterEligibility as AdapterEligibility, DependencyPin as DependencyPin

from .spatial import GenerationRasterSource as GenerationRasterSource

from .resource_policy import TensorLayout as TensorLayout, WorkloadMemory as WorkloadMemory, MemoryEstimate as MemoryEstimate, DeviceCapability as DeviceCapability, MemoryCredits as MemoryCredits, MemoryBudget as MemoryBudget

from .parallel_policy import ParallelPolicy as ParallelPolicy

from .borrowed_model import BorrowedValidationPolicy as BorrowedValidationPolicy

from .compatibility import BackendCompatibility as BackendCompatibility, BackendProfile as BackendProfile, inspect_backend_compatibility as inspect_backend_compatibility

from .adapter_conformance import ServingAdapter as ServingAdapter, AdapterConformance as AdapterConformance, check_adapter_conformance as check_adapter_conformance

from .best import BestTerraMindSession as BestTerraMindSession, BestProfile as BestProfile, BestSessionBusy as BestSessionBusy

from .best_workflows import (BestSceneRequest as _PreparedSceneRequest,
                            BestWorkQueue as _GenerationWorkQueue,
                            run_best_scene as _run_prepared_scene,
                            inspect_best_scene as _inspect_prepared_scene,
                            verify_best_scene as _verify_prepared_scene)
GenerationSession = BestTerraMindSession
PreparedSceneRequest = _PreparedSceneRequest
GenerationWorkQueue = _GenerationWorkQueue
run_prepared_scene = _run_prepared_scene
inspect_prepared_scene = _inspect_prepared_scene
verify_prepared_scene = _verify_prepared_scene

from .contracts import NumericalCoreReceipt as NumericalCoreReceipt
