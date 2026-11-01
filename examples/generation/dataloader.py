"""Stream an existing TorchGeo/TerraTorch DataLoader into verified COG outputs.

The caller owns the DataLoader, dataset/transforms and its worker lifetime.
Each source sample must contain an unstandardized FP32 ``image`` (12,224,224),
an explicit all-valid bool ``valid`` mask (224,224), ``source_id`` and ``crs``
strings, and a six-float list ``transform`` in affine a,b,c,d,e,f order.
This example expects Torch's default collation: BCHW images, BHW validity,
string lists and six batch-length transform tensors. Adapt the selectors for
other collators. A training label called ``mask`` is never used as validity.

Importing this module does not import Torch, load weights or open files.
Supply a provisioned pinned environment and model files; no download occurs.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def generate_files(loader: Iterable[Mapping[str, Any]], *, model_file: str | Path,
                   tokenizer_file: str | Path, work: str | Path, output: str | Path,
                   model: str = 'terramind-v1-base', max_samples: int | None = None,
                   execute_model: bool = False) -> int:
    """Generate and export incrementally; return a count, never accumulate images.

    ``work`` and ``output`` must be new directories with existing parents.
    Input-identity seeds are explicit and independent of DataLoader batching.
    A maximum of two requests/results is admitted by the engine; image batches
    are additionally limited to eight crops and 32 MiB by the adapter. These
    budgets do not bound memory used by caller-configured loader prefetch.
    """
    from aureon.api import GenerationEngine, GenerationRecipe, export_generation
    from aureon.api.generation_bridges import TorchGeoBatchAdapter
    from aureon.terramind_native.common import INPUT_BANDS

    bridge = TorchGeoBatchAdapter(
        bands=INPUT_BANDS, units='surface_reflectance_x10000',
        valid_mask=lambda batch, index: batch['valid'][index],
        source_id=lambda batch, index: batch['source_id'][index],
        grid=lambda batch, index: {
            'crs': batch['crs'][index],
            'transform': [float(component[index]) for component in batch['transform']],
        },
        max_batch_size=8, max_batch_image_bytes=32 * 1024**2,
    )
    count = 0
    with GenerationEngine(model, model_file=model_file, tokenizer_file=tokenizer_file,
                          work=work, max_pending=2, max_results=2,
                          execute_model=execute_model) as engine:
        destination = Path(output)
        destination.mkdir(exist_ok=False)
        jobs = bridge.iter_jobs(loader, recipe=GenerationRecipe(seed=17),
                                seed_policy='input_identity', max_samples=max_samples)
        with engine.generate_many(jobs, max_in_flight=2) as results:
            for index, result in enumerate(results):
                export_generation(result, destination / f'{index:06d}-synthetic-s1grd.tif')
                count += 1
    return count
