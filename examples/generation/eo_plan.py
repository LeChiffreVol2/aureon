"""Plan an explicit current EO route; no imagery, weights, network or model execution."""
from dataclasses import asdict
import json
from aureon.api import (EOGrid, ConditioningSpec, GeneratedTarget, SamplingRecipe,
                        EORequest, GenerationRecipe, plan_eo)

grid = EOGrid('EPSG:32647', (10., 0., 500000., 0., -10., 1500000.), (224, 224))
observed = ConditioningSpec('S2L2A', ('B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'),
                            'surface_reflectance_x10000', 'none', 'explicit_all_valid', grid,
                            'example-observation', '2026-10-30T00:00:00Z')
target = GeneratedTarget('S1GRD', ('VV','VH'), 'dB', grid)
request = EORequest('terramind-v1-base', (observed,), (target,),
                    SamplingRecipe.terramind(GenerationRecipe(seed=17)))
print(json.dumps(asdict(plan_eo(request)), indent=2))
