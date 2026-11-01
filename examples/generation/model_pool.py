"""Explicit resident routes; run only with local pinned assets and qualified budgets.

This example does not download weights. Numerical execution remains FP32.
Device/host byte budgets are operator inputs, not universal hardware recipes.
One GPU can evict between tiny/base; multiple distinct GPU slots can run routes
concurrently. Add separate EncoderRoute registrations for backbone/TiM features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from aureon.api.model_pool import DeviceSlot, GenerationRoute, ModelPool, PoolResources
from aureon.api.memory_input import GenerationInput
from aureon.terramind_native.common import INPUT_BANDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiny-model', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--input-npy', type=Path, required=True, help='All-valid FP32 S2L2A CHW224, reflectance x10000')
    parser.add_argument('--work', type=Path, required=True, help='Fresh directory under an existing parent')
    parser.add_argument('--cuda-device', required=True, help='Explicit GPU UUID or one physical ordinal; no comma list')
    parser.add_argument('--device-bytes', type=int, required=True)
    parser.add_argument('--host-bytes', type=int, required=True)
    parser.add_argument('--resident-device-bytes', type=int, required=True)
    parser.add_argument('--resident-host-bytes', type=int, required=True)
    parser.add_argument('--transient-device-bytes', type=int, required=True)
    parser.add_argument('--transient-host-bytes', type=int, required=True)
    parser.add_argument('--execute-model', action='store_true', required=True)
    args = parser.parse_args()
    import numpy as np

    image = np.load(args.input_npy, allow_pickle=False, mmap_mode='r')
    data = GenerationInput(image, bands=INPUT_BANDS, units='surface_reflectance_x10000',
                           valid_mask=np.ones((224, 224), bool))
    resources = PoolResources(args.resident_device_bytes, args.resident_host_bytes,
                              args.transient_device_bytes, args.transient_host_bytes,
                              3*1024**2, 1024**2)
    routes = [GenerationRoute(name, model_id, model, args.tokenizer, ('gpu',), resources)
              for name, model_id, model in [('tiny','terramind-v1-tiny',args.tiny_model),
                                           ('base','terramind-v1-base',args.base_model)]]
    with ModelPool(routes, [DeviceSlot('gpu', args.device_bytes, args.cuda_device)],
                   work=args.work, host_memory_bytes=args.host_bytes,
                   execute_model=args.execute_model) as pool:
        for name in ('tiny','tiny','base','tiny'):
            result = pool.submit(name, data, seed=17).result()
            print({'route':result.route,'worker_generation':result.worker_generation,
                   'seed':result.seed,'execution_key':result.execution_key,
                   'kind':result.kind})
        print(pool.metadata())


if __name__ == '__main__':
    # Required for spawned ownership on Windows and safe Torch/CUDA isolation.
    main()
