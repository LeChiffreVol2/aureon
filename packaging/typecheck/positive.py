"""Typechecked only: these functions never execute models or network requests."""
from pathlib import Path
from typing import Any, assert_type
import numpy as np
from numpy.typing import NDArray
from torch import Tensor, nn
from collections.abc import Mapping
from aureon.api import (
    EngineMetadata, EngineSession, EngineState, ExecutionProfile, GenerationEngine,
    GenerationHandle, GenerationInput, GenerationJob, GenerationRecipe,
    GenerationRequest, GenerationResult, GenerationStream, HandleState,
    ParallelFallback, Parallelism, ParallelPlan, QueueLimits, QueueMetadata,
    RecipeValues, SceneClient, SceneRequest, SceneResult, TorchGeoAdapter,
    export_generation, generation_capabilities, generation_compatibility,
    open_model_session, plan_generation_execution, run_model, run_scene,
    TerraMindEncoder, TerraMindFeatures,
    TorchGeoBatchAdapter, SceneGenerationRequest, inspect_generation_scene,
    run_generation_scene, diagnose_runtime,
)


def consumer(engine: GenerationEngine, item: GenerationInput, scene: SceneRequest,
             client: SceneClient, legacy: EngineSession) -> None:
    recipe = GenerationRecipe(seed=17)
    assert_type(recipe.native(), RecipeValues)
    assert_type(item.snapshot()[0], NDArray[np.float32])
    assert_type(item.snapshot()[1], NDArray[np.bool_])
    result = engine.generate(item, recipe=recipe, execution_profile="reference")
    assert_type(result, GenerationResult)
    assert_type(result["array"], NDArray[np.float32])
    assert_type(result["valid_mask"], NDArray[np.bool_])
    handle = engine.submit(GenerationJob(item, recipe=recipe))
    assert_type(handle, GenerationHandle)
    assert_type(handle.state, HandleState)
    assert_type(handle.result(timeout=1.5), GenerationResult)
    assert_type(handle.cancel(), bool)
    assert_type(handle.done(), bool)
    assert_type(engine.generate_many([item], max_in_flight=2), GenerationStream)
    with engine.generate_many([item]) as stream:
        assert_type(next(stream), GenerationResult)
    metadata = engine.metadata()
    assert_type(metadata, EngineMetadata)
    assert_type(metadata["state"], EngineState)
    assert_type(metadata["parallelism"], ParallelPlan)
    if metadata["queue"] is not None:
        assert_type(metadata["queue"], QueueMetadata)
        assert_type(metadata["queue"]["pending"], int)
    assert_type(export_generation(result, Path("out.tif")), dict[str, Any])
    assert_type(run_scene(scene), SceneResult)
    assert_type(run_model(scene), SceneResult)
    assert_type(legacy.run("input.json"), SceneResult)
    assert_type(client.get("job").job_id, str)
    assert_type(client.download("job", "out").receipt, Path)
    request = GenerationRequest("config.json", "out", "job", "1", input_data=item,
                                recipe=recipe.native())
    assert_type(run_model(request), dict[str, Any])
    assert_type(open_model_session("terramind-v1-base", config="assets.json", work="new", input_kind="array").generate(item), GenerationResult)
    assert_type(open_model_session("kao-terramind-b", bundle="bundle", work="new", profile_id="fp32-full-reference-v1"), EngineSession)
    assert_type(plan_generation_execution(recipe=recipe.native()), dict[str, Any])
    assert_type(generation_capabilities(), dict[str, Any])
    assert_type(generation_compatibility(item), dict[str, Any])
    adapter = TorchGeoAdapter(bands=["B01"], units="reflectance_x10000", valid_mask=lambda sample: sample["valid"])
    assert_type(adapter({"image": result["array"]}), GenerationInput)


def construct(profile: ExecutionProfile, lanes: Parallelism, fallback: ParallelFallback,
              limits: QueueLimits) -> GenerationEngine:
    return GenerationEngine(model_file="model", tokenizer_file="tokenizer", work="new",
                            execution_profile=profile, parallelism=lanes,
                            parallel_fallback=fallback, **limits)


def encoder_consumer(engine: TerraMindEncoder, inputs: Mapping[str, Tensor], item: GenerationInput) -> None:
    assert_type(engine.predict(inputs), TerraMindFeatures)
    assert_type(open_model_session("terramind-v1-small-backbone", model_file="m", modalities=["S2L2A"]), TerraMindEncoder)
    assert_type(open_model_session("terramind-v1-large-tim", model_file="m", modalities=["S2L2A"],
                                  tokenizer_files={"LULC": "t"}, tim_modalities=["LULC"]), TerraMindEncoder)
    assert_type(open_model_session("terramind-v1-tiny", config="c", work="w").generate(item), GenerationResult)


def scene_consumer(request: SceneGenerationRequest, batch: Mapping[str, Any]) -> None:
    assert_type(diagnose_runtime('terramind-v1-base'), dict[str, object])
    assert_type(inspect_generation_scene(request), dict[str, Any])
    assert_type(run_generation_scene(request, execute_model=True), dict[str, Any])
    adapter = TorchGeoBatchAdapter(bands=['B01'], units='reflectance_x10000',
                                  valid_mask=lambda sample, index: sample['valid'][index])
    assert_type(next(adapter.iter_inputs([batch])), GenerationInput)
    assert_type(next(adapter.iter_jobs([batch], seed_policy='input_identity')), GenerationJob)


def resident_consumer(engine: GenerationEngine, encoder: TerraMindEncoder,
                      request: SceneGenerationRequest, pixels: NDArray[np.float32]) -> None:
    from aureon.api import (encode_file_with_engine, inspect_encoder_input,
                            run_generation_scene_with_engine, EncoderInput)
    assert_type(encode_file_with_engine(encoder, input_file="in.npz", output="out", execute_model=True), dict[str, object])
    assert_type(inspect_encoder_input("in.npz"), dict[str, object])
    assert_type(run_generation_scene_with_engine(engine, request, execute_model=True), dict[str, Any])
    assert_type(EncoderInput({"S2L2A": pixels}).identity, str)


def pool_consumer(item: GenerationInput) -> None:
    from aureon.api import (ModelPool, DeviceSlot, PoolResources, GenerationRoute,
                            PoolHandle, PoolResult)
    budget = PoolResources(1, 1, 1, 1, 4000000, 4000000)
    route = GenerationRoute("sar", "terramind-v1-base", "model", "tokenizer", ("gpu",), budget)
    with ModelPool([route], devices=[DeviceSlot("gpu", 8000000000, "0")], work="work",
                   host_memory_bytes=16000000000, execute_model=True) as pool:
        handle = pool.submit("sar", item, seed=17, tenant="developer")
        assert_type(handle, PoolHandle)
        assert_type(handle.result(timeout=60), PoolResult)
        assert_type(pool.metadata(), dict[str, Any])


def bridge_forward(module: nn.Module, values: Mapping[str, Tensor]) -> object:
    return module(values['image'])


def bridge_output(value: object) -> Mapping[str, Tensor]:
    if not isinstance(value, Tensor):
        raise TypeError('Expected one output tensor')
    return {'logits': value}


def bridge_consumer(module: nn.Module, image: Tensor) -> None:
    from aureon.api import ModelBridge, PreparedTensor, PreprocessingRecipe
    recipe = PreprocessingRecipe(('red',), 'dn', 'standardized', mean=(1.,), std=(2.,))
    prepared = recipe.apply(image, bands=('red',), units='dn')
    assert_type(prepared, PreparedTensor)
    with ModelBridge(module, forward=bridge_forward, output=bridge_output) as bridge:
        assert_type(bridge.predict({'image': prepared.tensor}), dict[str, Tensor])
        assert_type(bridge.metadata(), dict[str, object])


def upstream_constructor(options: dict[str, object]) -> nn.Module:
    return nn.Conv2d(2, 3, 1)


def upstream_task_consumer(image: Tensor) -> None:
    from aureon.api import (UpstreamTaskBridge, TaskPrediction, PreprocessingRecipe,
                            TerraTorchTaskConfig, TerraTorchSegmentation, SegmentationPrediction)
    recipe = PreprocessingRecipe(('red', 'nir'), 'reflectance', 'reflectance')
    config = TerraTorchTaskConfig(('red', 'nir'), 3)
    assert_type(config.to_dict(), dict[str, object])
    with UpstreamTaskBridge(upstream_constructor, {}, kind='segmentation', channels=3,
                            preprocessing=recipe, output_container='tensor') as bridge:
        assert_type(bridge.predict(image, bands=recipe.bands, units='reflectance'), TaskPrediction)
        assert_type(bridge.metadata(), dict[str, object])
    with TerraTorchSegmentation.from_checkpoint(config, 'local.ckpt', expected_sha256='hash',
                                                preprocessing=recipe) as fixed:
        assert_type(fixed.predict(image, bands=config.bands, units='reflectance'), SegmentationPrediction)


def eo_consumer(engine: GenerationEngine, data: GenerationInput, request: "EORequest") -> None:
    from aureon.api import EORoutePlan, GenerationResult, plan_eo, generate_eo
    assert_type(plan_eo(request), EORoutePlan)
    assert_type(generate_eo(engine, request, data), GenerationResult)

from aureon.api import EORequest

def borrowed_temporal_consumer(module: nn.Module, image: Tensor, mask: Tensor) -> None:
    from aureon.api import (BorrowedModelBridge, DeviceTensorPolicy, TemporalSpec, TemporalInput,
                            TemporalBridge, TemporalFeatures, GeographicConditioning,
                            AdapterRegistry, AdapterRequest, AdapterSpec, AdapterEligibility)
    with BorrowedModelBridge(module, forward=bridge_forward, output=bridge_output,
                             policy=DeviceTensorPolicy('cpu')) as borrowed:
        assert_type(borrowed.predict({'image': image}), dict[str, Tensor])
    spec = TemporalSpec(('red',), 'reflectance', ('2026-01-01T00:00:00Z',))
    with TemporalBridge(module, spec=spec, pooling='mean') as temporal:
        assert_type(temporal.predict(TemporalInput(image, mask, spec)), TemporalFeatures)
    assert_type(GeographicConditioning(((13., 100.),), 'unit_sphere_xyz').encoded(), tuple[tuple[float, ...], ...])
    registry: AdapterRegistry[nn.Module, BorrowedModelBridge] = AdapterRegistry()
    def create(request: AdapterRequest, model: nn.Module) -> BorrowedModelBridge:
        return BorrowedModelBridge(model, forward=bridge_forward, output=bridge_output,
                                   policy=DeviceTensorPolicy(request.device))
    registry.register(AdapterSpec('owned', ('image',), ('features',), ('cpu',), ()), create)
    request = AdapterRequest('image', 'features')
    assert_type(registry.plan('owned', request), AdapterEligibility)
    assert_type(registry.create('owned', request, module), BorrowedModelBridge)


def scalable_consumer(module: nn.Module, image: Tensor) -> None:
    from aureon.api import (BorrowedModelBridge, BorrowedValidationPolicy, WorkloadMemory,
                            MemoryEstimate, TensorLayout, MemoryCredits, MemoryBudget,
                            DeviceCapability, ParallelPolicy, BackendCompatibility,
                            inspect_backend_compatibility, ServingAdapter, AdapterConformance,
                            check_adapter_conformance)
    workload = WorkloadMemory(inputs=TensorLayout((12, 224, 224)), batch_size=1)
    assert_type(workload.estimate(), MemoryEstimate)
    policy = ParallelPolicy('auto', workload=workload)
    assert_type(policy.estimate, MemoryEstimate)
    assert_type(DeviceCapability('cuda', 4096, 2048, (8, 6), graph=True).lane_rejection(), str | None)
    budget = MemoryBudget(MemoryCredits(device=4096))
    token = budget.reserve(MemoryCredits(device=1024))
    assert_type(token, int)
    budget.release(token)
    assert_type(inspect_backend_compatibility('borrowed-cpu'), BackendCompatibility)
    def factory() -> ServingAdapter:
        return BorrowedModelBridge(module, forward=bridge_forward, output=bridge_output,
            validation=BorrowedValidationPolicy('serving', True, 'example-v1'))
    assert_type(check_adapter_conformance(factory, {'image': image}, {'features': image}), AdapterConformance)



def selected_best_contract(config: str, work: str) -> None:
    from aureon.api import BestTerraMindSession
    from aureon._reference.integrity import CoreFingerprint
    with BestTerraMindSession(config, work, execute_model=True) as best:
        assert_type(best.run(job_id="scene", attempt_id="1"), dict[str, object])
        assert_type(best.metadata(), dict[str, object])
        assert_type(best.core_fingerprint, CoreFingerprint)
        assert_type(best.close(), dict[str, object])


def canonical_file_session(config: str, work: str) -> None:
    from aureon.api import GenerationSession, PreparedSceneRequest, GenerationWorkQueue, run_prepared_scene
    session = open_model_session("terramind-v1-base", config=config, work=work, execute_model=True)
    assert_type(session, GenerationSession)
    assert_type(session.run(job_id="j", attempt_id="1"), dict[str, object])


def selected_core_result(result: GenerationResult) -> None:
    from aureon.api import NumericalCoreReceipt
    if "numerical_core" in result:
        assert_type(result["numerical_core"], NumericalCoreReceipt)
        assert_type(result["numerical_core"]["manifest_sha256"], str)
