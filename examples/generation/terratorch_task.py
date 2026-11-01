"""Offline random-weight example. Start with CUDA_VISIBLE_DEVICES empty."""
import hashlib
import json
from pathlib import Path
import tempfile
import torch
from aureon.api import TerraTorchTaskConfig, TerraTorchSegmentation, PreprocessingRecipe, build_terratorch_task

torch.set_num_threads(1)
config=TerraTorchTaskConfig.from_dict(json.loads(json.dumps(TerraTorchTaskConfig(('red','green','blue'),2).to_dict())))
task=build_terratorch_task(config)
recipe=PreprocessingRecipe(config.bands,'reflectance','reflectance')
with tempfile.TemporaryDirectory() as directory:
    checkpoint=Path(directory)/'random-task.ckpt'
    torch.save({'state_dict':task.state_dict()},checkpoint)
    with TerraTorchSegmentation.from_checkpoint(config,checkpoint,
            expected_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),preprocessing=recipe) as engine:
        result=engine.predict(torch.ones(1,3,32,32),bands=config.bands,units='reflectance')
        print('Random-weight CPU logits:',tuple(result['logits'].shape),'labels:',tuple(result['labels'].shape))
