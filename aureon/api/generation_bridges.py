"""Explicit upstream workflow contracts; no upstream imports, guessing or patching."""
from __future__ import annotations
from collections.abc import Callable, Generator, Iterable, Mapping
from dataclasses import dataclass, field, replace
import hashlib
from typing import Any, Literal, cast

from .contracts import ExecutionProfile, RecipeValues
from .memory_input import GenerationInput
from ..terramind_native.common import INPUT_BANDS, MODEL_SHA, TOKENIZER_SHA, digest, validate_recipe
from ..terramind_native.execution import plan_generation_execution
from ..terramind_native.families import GENERATION_MODELS, get_terramind_spec


@dataclass(frozen=True)
class GenerationRecipe:
    """The existing native sampling contract, detached from upstream model objects."""
    token_steps: int = 8
    diffusion_steps: int = 10
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 0
    seed: int = 17

    def __post_init__(self) -> None:
        self.native()

    def native(self) -> RecipeValues:
        return cast(RecipeValues, validate_recipe({name: getattr(self, name) for name in
                    ("token_steps", "diffusion_steps", "temperature", "top_p", "top_k", "seed")}))

    @classmethod
    def from_terratorch(cls, settings: Mapping[str, Any], *, seed: int,
                       model_id: str, terratorch_version: str) -> GenerationRecipe:
        """Map explicit TerraTorch1.2.13 generation factory settings.

        This is a configuration bridge, not adoption of an arbitrary module,
        checkpoint, transformed DataLoader or upstream training task. The actual
        installed profile and pinned assets are still verified on engine open.
        """
        required = {"modalities", "output_modalities", "standardize", "img_size",
                    "decoding_steps", "temps", "top_p", "top_k", "timesteps"}
        if model_id not in GENERATION_MODELS or terratorch_version != "1.2.13":
            raise ValueError("Recipe bridge requires a TerraMind generation model_id / TerraTorch1.2.13")
        if not isinstance(settings, Mapping) or set(settings) - (required | {"pretrained"}) or not required <= set(settings):
            raise ValueError("Supply exactly the supported explicit TerraTorch generation factory settings")
        if (settings["modalities"] != ["S2L2A"] or settings["output_modalities"] != ["S1GRD"]
                or settings["standardize"] is not True or type(settings["img_size"]) is not int
                or settings["img_size"] != 224 or settings.get("pretrained", False) is not False):
            raise ValueError("Only S2L2A224 to S1GRD with model-owned standardization and explicit local assets is supported")
        return cls(token_steps=settings["decoding_steps"], diffusion_steps=settings["timesteps"],
                   temperature=settings["temps"], top_p=settings["top_p"], top_k=settings["top_k"], seed=seed)


class TorchGeoAdapter:
    """Convert one sample lazily using explicit per-sample metadata callbacks.

    Never interpret sample['mask'] as validity or infer an affine from bounds.
    Callbacks execute on the input consumer, before admission to model execution.
    """
    def __init__(self, *, bands: list[str] | tuple[str, ...], units: str, valid_mask: Callable[[Mapping[str, Any]], Any], normalization: str="none", grid: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None=None, source_id: Callable[[Mapping[str, Any]], str | None] | None=None) -> None:
        if not isinstance(bands, (list, tuple)) or list(bands) != INPUT_BANDS:
            raise ValueError("Declare the supported S2L2A band order")
        if units not in ("surface_reflectance_x10000", "reflectance_x10000") or normalization != "none":
            raise ValueError("Adapter requires unstandardized physical reflectance x10000")
        if not callable(valid_mask) or any(value is not None and not callable(value) for value in (grid, source_id)):
            raise TypeError("valid_mask and supplied grid/source_id must be per-sample callables")
        self._bands, self._units, self._normalization = tuple(bands), units, normalization
        self._valid_mask, self._grid, self._source_id = valid_mask, grid, source_id

    def __call__(self, sample: Mapping[str, Any]) -> GenerationInput:
        return GenerationInput.from_torchgeo(sample, bands=self._bands, units=self._units,
            normalization=self._normalization, valid_mask=self._valid_mask(sample),
            grid=self._grid(sample) if self._grid else None,
            source_id=self._source_id(sample) if self._source_id else None)


class TorchGeoBatchAdapter:
    """Unbatch CPU DataLoader output with explicit metadata and bounded admission.

    Images must be FP32 BCHW, with each item having the existing S2L2A224
    contract. The image callback can select a TerraTorch modality dictionary;
    no nested schema, labels, grid or normalization is guessed. Metadata
    callbacks receive the original batch and its local item index. Only one
    upstream batch and one newly owned input are retained by the iterator.

    The image-byte limit covers the selected batched image, not arbitrary
    DataLoader fields, worker prefetch or inputs/results admitted by an Engine.
    Set those independent budgets on the loader and ``GenerationEngine``.
    """

    def __init__(self, *, bands: list[str] | tuple[str, ...], units: str,
                 valid_mask: Callable[[Mapping[str, Any], int], Any],
                 normalization: str = "none",
                 image: Callable[[Mapping[str, Any]], Any] | None = None,
                 grid: Callable[[Mapping[str, Any], int], Mapping[str, Any] | None] | None = None,
                 source_id: Callable[[Mapping[str, Any], int], str | None] | None = None,
                 max_batch_size: int = 32,
                 max_batch_image_bytes: int = 128 * 1024**2) -> None:
        # Reuse the single-item semantic contract without constructing an input.
        TorchGeoAdapter(bands=bands, units=units, normalization=normalization,
                        valid_mask=lambda sample: None)
        if not callable(valid_mask) or any(value is not None and not callable(value)
                                          for value in (image, grid, source_id)):
            raise TypeError("Batch image/metadata selectors must be callable")
        from .contracts import positive
        self._max_batch_size = positive(max_batch_size, "max_batch_size")
        self._max_batch_image_bytes = positive(max_batch_image_bytes, "max_batch_image_bytes")
        self._bands, self._units, self._normalization = tuple(bands), units, normalization
        self._image, self._valid_mask, self._grid, self._source_id = image, valid_mask, grid, source_id

    def iter_inputs(self, batches: Iterable[Mapping[str, Any]], *,
                    max_samples: int | None = None,
                    stop: Callable[[], bool] | None = None) -> Generator[GenerationInput, None, None]:
        """Yield owned inputs lazily; close on early exit to release the batch.

        Validation is per consumed batch/item, so a later invalid item may follow
        already completed work. ``stop`` is checked before fetching a batch and
        before each item; it does not interrupt a DataLoader read or model call.
        No more than ``max_samples`` items are read/adapted. A public upstream
        iterator ``close`` method is called on exit; DataLoader worker lifetime
        remains the caller's responsibility (no private worker API is used).
        """
        from .contracts import positive
        from .lifecycle import close_preserving_error
        from .memory_input import _array
        import numpy as np

        if max_samples is not None:
            positive(max_samples, "max_samples")
        if stop is not None and not callable(stop):
            raise TypeError("stop must be a callable")
        # DataLoader.__iter__ itself may start prefetch workers. A request that
        # is already cancelled must not acquire that iterator in the first place.
        if stop is not None and stop():
            return
        iterator = iter(batches)
        primary: BaseException | None = None
        batch: Mapping[str, Any] | None = None
        selected: Any = None
        image: Any = None
        value: GenerationInput | None = None
        emitted = 0
        batch_index = -1
        try:
            while max_samples is None or emitted < max_samples:
                if stop is not None and stop():
                    return
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                batch_index += 1
                if not isinstance(batch, Mapping):
                    raise TypeError("DataLoader batch must be a mapping")
                if self._image is None and "image" not in batch:
                    raise ValueError("DataLoader batch needs 'image' or an explicit image selector")
                selected = self._image(batch) if self._image is not None else batch["image"]
                shape: Any = getattr(selected, "shape", ())
                if (not isinstance(shape, (tuple, list)) or len(shape) != 4 or tuple(shape[1:]) != (12, 224, 224)
                        or not 1 <= shape[0] <= self._max_batch_size):
                    raise ValueError("Batch image needs BCHW (B,12,224,224) within max_batch_size")
                # _array rejects masked arrays, non-CPU/sparse Torch tensors and
                # arbitrary array coercion. It takes a view; each yield snapshots.
                image = _array(selected, "batch image", tuple(shape))
                if image.dtype != np.float32:
                    raise ValueError("Batch image must be float32; no implicit casting")
                if image.nbytes > self._max_batch_image_bytes:
                    raise ValueError("Batch image exceeds max_batch_image_bytes")
                for index in range(shape[0]):
                    if max_samples is not None and emitted >= max_samples:
                        return
                    if stop is not None and stop():
                        return
                    try:
                        value = GenerationInput(image[index], bands=self._bands, units=self._units,
                            normalization=self._normalization, valid_mask=self._valid_mask(batch, index),
                            grid=self._grid(batch, index) if self._grid is not None else None,
                            source_id=self._source_id(batch, index) if self._source_id is not None else None)
                    except Exception as error:
                        error.add_note(f"While adapting DataLoader batch {batch_index}, item {index}")
                        raise
                    emitted += 1
                    yield value
                    value = None
                batch = selected = image = None
        except BaseException as error:
            primary = error
            raise
        finally:
            batch = selected = image = value = None
            close = getattr(iterator, "close", None)
            if callable(close):
                close_preserving_error(close, primary, context="DataLoader input iterator cleanup failed")

    def iter_jobs(self, batches: Iterable[Mapping[str, Any]], *,
                  recipe: GenerationRecipe | None = None,
                  seed_policy: Literal["same", "input_identity"] = "same",
                  execution_profile: ExecutionProfile | None = None,
                  job_id: Callable[[GenerationInput, int], str | None] | None = None,
                  max_samples: int | None = None,
                  stop: Callable[[], bool] | None = None) -> Generator[GenerationJob, None, None]:
        """Yield requests usable directly by ``engine.generate_many``.

        ``same`` retains the recipe seed for each input. ``input_identity``
        derives a 63-bit seed from that seed and the complete input identity,
        independent of batch grouping/order. Equal inputs receive equal seeds;
        this is not a uniqueness promise. Seed derivation does not use global
        RNG. The caller's DataLoader/transforms/callbacks retain their own RNG
        behavior; use an explicit loader generator when isolation is required.
        """
        from .lifecycle import close_preserving_error
        from ..terramind_native.execution import validate_selection

        recipe = GenerationRecipe() if recipe is None else recipe
        if not isinstance(recipe, GenerationRecipe):
            raise TypeError("recipe requires GenerationRecipe")
        if seed_policy not in ("same", "input_identity"):
            raise ValueError("seed_policy must be same or input_identity")
        if job_id is not None and not callable(job_id):
            raise TypeError("job_id must be callable")
        validate_selection(execution_profile)
        inputs = self.iter_inputs(batches, max_samples=max_samples, stop=stop)
        primary: BaseException | None = None
        value: GenerationInput | None = None
        try:
            for index, value in enumerate(inputs):
                if not isinstance(value, GenerationInput):
                    raise TypeError("Input iterator must yield GenerationInput")
                selected_recipe = recipe
                if seed_policy == "input_identity":
                    material = f"aureon.dataloader_seed.v1:{recipe.seed}:{value.identity['sha256']}"
                    seed = int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big") % 2**63
                    selected_recipe = replace(recipe, seed=seed)
                yield GenerationJob(value, recipe=selected_recipe, execution_profile=execution_profile,
                                    job_id=job_id(value, index) if job_id is not None else None)
                value = None
        except BaseException as error:
            primary = error
            raise
        finally:
            value = None
            close_preserving_error(inputs.close, primary, context="DataLoader job iterator cleanup failed")


@dataclass(frozen=True)
class GenerationJob:
    """One immutable request descriptor for lazy mixed-seed/tiled workloads."""
    input_data: GenerationInput
    recipe: GenerationRecipe = field(default_factory=GenerationRecipe)
    execution_profile: ExecutionProfile | None = None
    job_id: str | None = None
    attempt_id: str = "1"

    def __post_init__(self) -> None:
        from ..terramind_native.execution import validate_selection
        from ..terramind_native.session import _identifier
        if not isinstance(self.input_data, GenerationInput): raise TypeError("GenerationJob requires GenerationInput")
        if not isinstance(self.recipe, GenerationRecipe): raise TypeError("GenerationJob requires GenerationRecipe")
        self.recipe.native()
        validate_selection(self.execution_profile)
        if self.job_id is not None: _identifier(self.job_id, "job_id")
        _identifier(self.attempt_id, "attempt_id")

    def options(self) -> dict[str, Any]:
        return dict(recipe=self.recipe, execution_profile=self.execution_profile,
                    job_id=self.job_id, attempt_id=self.attempt_id)


def generation_capabilities(*, model_id: str="terramind-v1-base") -> dict[str, Any]:
    """CPU-visible implementation boundaries, not hardware/parity qualification."""
    if model_id not in GENERATION_MODELS:
        raise ValueError("Capabilities require a TerraMind generation model_id")
    return {"schema": "aureon.generation_capabilities.v1", "model_id": model_id,
            "task": "modality-generation", "input": {"modality": "S2L2A", "shape": [12,224,224],
                "dtype": "float32", "bands": list(INPUT_BANDS), "units": "surface_reflectance_x10000",
                "normalization": "none", "validity": "all_valid", "device": "cpu"},
            "output": {"modality": "S1GRD", "shape": [2,224,224], "dtype": "float32",
                       "units": "dB", "synthetic": True, "geospatial_export": "explicit_with_grid"},
            "precision": "highest_fp32", "execution_profiles": ["recommended", "reference"],
            "batching": {"max_execution_batch_size": 1, "fused_batching": False,
                         "step_interleaving": False, "concurrent_submission": True,
                         "gpu_ownership": "serial_existing_resident_session"},
            "stages": [{"name": "prepare", "owner": "caller_cpu", "overlap_with_execute": True},
                       {"name": "execute", "owner": "resident_session", "atomic_request": True,
                        "inner_stage_scheduling": False},
                       {"name": "deliver", "owner": "caller_cpu", "overlap_with_execute": True}],
            "cancel": "queued_only; running_discard_finishes_execution",
            "per_item_request": "GenerationJob: explicit recipe/seed, execution_profile, job_id, attempt_id",
            "qualification": "implementation_contract_only; concurrent_source_not_gpu_qualified",
            "cuda": "not_probed", "model_loaded": False}


def generation_compatibility(input_data: GenerationInput, *, recipe: GenerationRecipe | None = None,
                             execution_profile: ExecutionProfile = "recommended", model_id: str="terramind-v1-base") -> dict[str, Any]:
    """Describe admitted batch-one execution; matching keys never authorize batching."""
    if not isinstance(input_data, GenerationInput):
        raise TypeError("Compatibility requires GenerationInput")
    if recipe is None: recipe = GenerationRecipe()
    if not isinstance(recipe, GenerationRecipe): raise TypeError("recipe requires GenerationRecipe")
    native = recipe.native()
    plan = plan_generation_execution(execution_profile=execution_profile, recipe=native, model_id=model_id)
    key = {"model_id": model_id, "model_sha256": get_terramind_spec(model_id).checkpoint.sha256,
           "tokenizer_sha256": TOKENIZER_SHA, "input_modality": "S2L2A", "input_shape": [12,224,224],
           "output_modality": "S1GRD", "precision": "highest_fp32", "recipe": native,
           "execution_profile": execution_profile, "planned": plan["planned"],
           "bands": list(INPUT_BANDS), "units": "surface_reflectance_x10000", "normalization": "none"}
    return {"schema": "aureon.generation_compatibility.v1", "key": key, "sha256": digest(key),
            "input_identity": input_data.identity, "delivery_grid": input_data.metadata["grid"],
            "max_execution_batch_size": 1, "runtime_eligibility": "not_checked",
            "same_key_means": "same declared execution contract, not shared input or permitted fused execution"}
