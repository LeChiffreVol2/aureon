import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from aureon.api import TerraTorchTaskConfig, TerraTorchSegmentation, build_terratorch_task, PreprocessingRecipe
from aureon.terramind_native.encoder_loader import fp32_policy


class TaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = TerraTorchTaskConfig(('red','green','blue'), 2)
        cls.task = build_terratorch_task(cls.config)
        cls.recipe = PreprocessingRecipe(cls.config.bands,'dn','standardized',scale=.01,
                                        mean=(.1,.2,.3),std=(.5,.6,.7))

    def test_actual_upstream_lightning_roundtrip_preprocessed_parity(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.ckpt'
            torch.save({'state_dict': self.task.state_dict(), 'epoch':5,
                        'hyper_parameters':{'class_path':'never.execute.this'}},path)
            before=torch.get_rng_state().clone()
            with TerraTorchSegmentation.from_checkpoint(self.config,path,
                    expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),preprocessing=self.recipe) as engine:
                x=torch.arange(3*32*32,dtype=torch.float32).reshape(1,3,32,32)/100
                rng=torch.get_rng_state().clone()
                with fp32_policy(),torch.inference_mode(),torch.autocast('cpu',enabled=False):
                    normalized=(x[:,[2,1,0]]*.01-torch.tensor([.1,.2,.3]).reshape(1,3,1,1))/torch.tensor([.5,.6,.7]).reshape(1,3,1,1)
                    reference=self.task(normalized).output.clone()
                result=engine.predict(x,bands=('blue','green','red'),units='dn')
                self.assertTrue(torch.equal(reference,result['logits']))
                self.assertTrue(torch.equal(reference.argmax(1),result['labels']))
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                metadata=engine.metadata();json.dumps(metadata)
                self.assertEqual(len(metadata['checkpoint']['tensor_identity']),len(self.task.state_dict()))
                self.assertFalse(metadata['checkpoint']['hyper_parameters_used'])
                result['logits'].zero_();result['labels'].zero_()
            # The input fixture construction consumes no random draws.
            self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_explicit_raw_task_state_dict(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.pt';torch.save(self.task.state_dict(),path)
            with TerraTorchSegmentation.from_checkpoint(self.config,path,
                    expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    preprocessing=self.recipe,checkpoint_format='state_dict') as engine:
                x=torch.ones(1,3,16,16)
                prepared=self.recipe.apply(x,bands=self.config.bands,units='dn')
                with fp32_policy(),torch.no_grad():reference=self.task(prepared.tensor).output
                self.assertTrue(torch.equal(reference,engine.predict(x,bands=self.config.bands,units='dn')['logits']))
                self.assertEqual(engine.metadata()['checkpoint']['format'],'state_dict')

    def test_strict_checkpoint_mismatches(self):
        state=self.task.state_dict();key=next(k for k,v in state.items() if v.is_floating_point())
        cases=[{k.removeprefix('model.'):v for k,v in state.items()},
               {**state,key:state[key].double()}, {**state,key:torch.zeros(1)},
               {**state,key:torch.full_like(state[key],float('nan'))}]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.ckpt'
            for malformed in cases:
                torch.save({'state_dict':malformed},path)
                with self.assertRaises(ValueError):
                    TerraTorchSegmentation.from_checkpoint(self.config,path,
                        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),preprocessing=self.recipe)
            with patch('aureon.api.terratorch_task.build_terratorch_task') as factory:
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    TerraTorchSegmentation.from_checkpoint(self.config,path,expected_sha256='0'*64,preprocessing=self.recipe)
                factory.assert_not_called()

    def test_exact_schema_and_no_implicit_acquisition(self):
        self.assertEqual(TerraTorchTaskConfig.from_dict(self.config.to_dict()),self.config)
        for key,value in [('pretrained',True),('backbone','remote/model'),('class_path','os.system'),('encoder_depth',5)]:
            with self.assertRaises(ValueError):TerraTorchTaskConfig.from_dict({**self.config.to_dict(),key:value})
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(ValueError,'CPU-only'):build_terratorch_task(self.config)

    def test_workflow_validation_and_closed_lifecycle(self):
        engine=TerraTorchSegmentation(self.task,self.config,self.recipe)
        with self.assertRaises(ValueError):engine.predict(torch.ones(1,3,31,32),bands=self.config.bands,units='dn')
        with self.assertRaises(ValueError):engine.predict(torch.ones(1,3,32,32),bands=self.config.bands,units='standardized')
        engine.close()
        with self.assertRaises(RuntimeError):engine.predict(torch.ones(1,3,32,32),bands=self.config.bands,units='dn')


if __name__=='__main__':unittest.main()
