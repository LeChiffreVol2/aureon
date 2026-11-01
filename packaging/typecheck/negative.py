"""Every deliberate consumer error must be detected at its marked line."""
from torch import nn
from aureon.api import GenerationEngine, GenerationInput, GenerationRecipe, GenerationResult, SceneRequest, SceneResult, run_model
from aureon.api import TerraMindEncoder
from aureon.api import SceneGenerationRequest, TorchGeoBatchAdapter, diagnose_runtime


def invalid(engine: GenerationEngine, item: GenerationInput, result: GenerationResult, scene: SceneRequest) -> None:
    engine.generate("not an input")  # EXPECT: arg-type
    engine.generate(item, execution_profile="fastest")  # EXPECT: arg-type
    engine.generate(item, recipe={"seed": 1})  # EXPECT: arg-type
    engine.submit(item).result(timeout="soon")  # EXPECT: arg-type
    engine.close(timeout="soon")  # EXPECT: arg-type
    GenerationEngine(model_file="m", tokenizer_file="t", work="w", parallelism="many")  # EXPECT: arg-type
    GenerationRecipe(token_steps="eight")  # EXPECT: arg-type
    result["array"] = "pixels"  # EXPECT: typeddict-item
    result["missing"]  # EXPECT: typeddict-item
    wrong: str = run_model(scene)  # EXPECT: assignment
    wrong_result: SceneResult = engine.generate(item)  # EXPECT: assignment
    engine.generate_many([item], adapter=lambda value: "invalid")  # EXPECT: arg-type # EXPECT: return-value


def invalid_encoder(engine: TerraMindEncoder) -> None:
    engine.predict("not modality tensors")  # EXPECT: arg-type
    TerraMindEncoder("terramind-v1-tiny-backbone", model_file="m", modalities=["S2L2A"], merge_method="median")  # EXPECT: arg-type


def invalid_workflows(adapter: TorchGeoBatchAdapter) -> None:
    diagnose_runtime('terramind-v1-base', verify_assets='yes')  # EXPECT: arg-type
    adapter.iter_jobs([], seed_policy='random')  # EXPECT: arg-type
    SceneGenerationRequest('i', 'v', 'm', 't', 'w', 'o', invalid_policy='fill')  # EXPECT: arg-type


def invalid_resident(engine: GenerationEngine, encoder: TerraMindEncoder) -> None:
    from aureon.api import encode_file_with_engine, inspect_encoder_input
    encode_file_with_engine(engine, input_file="i", output="o")  # EXPECT: arg-type
    inspect_encoder_input("i", modalities=42)  # EXPECT: arg-type
    encode_file_with_engine(encoder, input_file="i", output="o", seed="seventeen")  # EXPECT: arg-type


def invalid_pool(item: GenerationInput) -> None:
    from aureon.api import ModelPool, DeviceSlot
    pool = ModelPool([], devices=[], work="w", host_memory_bytes=1024)
    pool.submit("sar", item, seed="seventeen")  # EXPECT: arg-type
    pool.submit("sar", "pixels")  # EXPECT: arg-type
    DeviceSlot("gpu", "8 GiB")  # EXPECT: arg-type


def invalid_bridge() -> None:
    from aureon.api import PreprocessingRecipe
    PreprocessingRecipe(('red',), 42, 'dn')  # EXPECT: arg-type
    PreprocessingRecipe(('red',), 'dn', 'dn', workflow='guess')  # EXPECT: arg-type


def invalid_upstream_task() -> None:
    from aureon.api import UpstreamTaskBridge, TerraTorchTaskConfig, TerraTorchSegmentation, PreprocessingRecipe
    recipe = PreprocessingRecipe(('red',), 'dn', 'dn')
    TerraTorchTaskConfig(['red'], 2)  # EXPECT: arg-type
    config = TerraTorchTaskConfig(('red',), 2)
    TerraTorchSegmentation.from_checkpoint(config, 'model', expected_sha256=123, preprocessing=recipe)  # EXPECT: arg-type
    UpstreamTaskBridge(None, {}, kind='segmentation', channels=2, preprocessing=recipe)  # EXPECT: arg-type
    UpstreamTaskBridge(lambda args: nn.Identity(), {}, kind='guess', channels=2, preprocessing=recipe)  # EXPECT: arg-type
    UpstreamTaskBridge(lambda args: nn.Identity(), {}, kind='features', channels='two', preprocessing=recipe)  # EXPECT: arg-type

from aureon.api import SamplingRecipe, AssetBand, EOGrid
SamplingRecipe('adapter', 'seed', ())  # EXPECT: arg-type
AssetBand('image', 'first')  # EXPECT: arg-type
EOGrid('EPSG:32647', [1., 0., 0., 0., -1., 0.], (224, 224))  # EXPECT: arg-type

from aureon.api import DeviceTensorPolicy, TemporalSpec, GeographicConditioning, AdapterRequest, DependencyPin
DeviceTensorPolicy(device=7)  # EXPECT: arg-type
TemporalSpec(('red',), 'dn', ['2026-01-01T00:00:00Z'])  # EXPECT: arg-type
GeographicConditioning(((13., 100.),), 'EPSG:4326')  # EXPECT: arg-type
AdapterRequest('image', 'features', graph_fallback='automatic')  # EXPECT: arg-type
DependencyPin('torch', ['2.10.0'])  # EXPECT: arg-type

from aureon.api import BorrowedValidationPolicy, WorkloadMemory, TensorLayout, MemoryBudget, MemoryCredits, inspect_backend_compatibility
BorrowedValidationPolicy('unchecked')  # EXPECT: arg-type
TensorLayout((3, 224, 224), 'fp8')  # EXPECT: arg-type
WorkloadMemory(batch_size='automatic')  # EXPECT: arg-type
MemoryBudget(MemoryCredits()).reserve({'device': 1})  # EXPECT: arg-type
inspect_backend_compatibility('universal')  # EXPECT: arg-type

GenerationEngine(model_file="m", tokenizer_file="t", work="w", neural_batch=None)  # EXPECT: call-arg

from aureon.api import BestTerraMindSession
BestTerraMindSession("config", "work", execute_model=True, profile="fp16")  # EXPECT: arg-type
best_session = BestTerraMindSession("config", "work", execute_model=True)
best_session.run(job_id=7, attempt_id="1")  # EXPECT: arg-type
best_session.run(job_id="j", attempt_id="1", decoder_context=None)  # EXPECT: call-arg
