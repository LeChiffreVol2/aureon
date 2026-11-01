"""Optional upstream scheduling integration: use the pinned generation environment.

Tiny CPU decoder only; no downloaded weights, GPU execution or performance claim.
Kept separate from the dependency-light core CI suite.
"""
import unittest
from unittest.mock import patch
import torch
from test_generation_batch import Decoder, batch, rng_capsule

class UpstreamSchedulingContracts(unittest.TestCase):
    def test_installed_pipeline_noise_boundary_and_ddim_steps_on_cpu(self):
        from terratorch.models.backbones.terramind.tokenizer.scheduling.diffusion_pipeline import PipelineCond
        from terratorch.models.backbones.terramind.tokenizer.scheduling.scheduling_ddim import DDIMScheduler
        decoder=Decoder();decoder.device=torch.device('cpu')
        cond=torch.arange(5*14*14,dtype=torch.float32).reshape(1,5,14,14)/1000
        def scheduler():
            return DDIMScheduler(num_train_timesteps=1000, prediction_type='sample',
                                 thresholding=True, clip_sample=False, zero_terminal_snr=True)
        self.assertEqual(batch.scheduler_identity(scheduler())['eta'],0)
        states=[]; references=[]; post_rng=[]
        for seed in (17,101):
            with torch.no_grad(),rng_capsule(seed) as rng:
                references.append(PipelineCond(decoder,scheduler(),2)(cond,timesteps=10,image_size=(224,224),verbose=False))
            post_rng.append(rng['request_post_rng_sha256'])
            owned=scheduler();captured={}
            def boundary(image,timestep,conditioning,**kw):
                captured.update(image=image.clone(),conditioning=conditioning.clone(),scheduler=owned)
                raise batch._PreparedBoundary()
            with torch.no_grad(),rng_capsule(seed) as prep_rng,patch.object(decoder,'forward',side_effect=boundary):
                try:PipelineCond(decoder,owned,2)(cond,timesteps=10,image_size=(224,224),verbose=False)
                except batch._PreparedBoundary:pass
            self.assertEqual(prep_rng['request_post_rng_sha256'],post_rng[-1])
            states.append(captured)
        decoder.calls.clear()
        with torch.no_grad(),batch._unchanged_rng(torch,()) as unchanged:
            result=batch.denoise_pair(torch,decoder,states,cancel=lambda:None)
        self.assertTrue(unchanged['unchanged'])
        self.assertEqual(result['observed_decoder_batch_sizes'],[2]*10)
        for state,reference in zip(states,references):self.assertTrue(torch.equal(state['image'],reference))


if __name__ == "__main__":
    unittest.main()
