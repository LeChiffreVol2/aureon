"""CPU checks of the native boundaries, with CUDA hidden by the runner."""
from contextlib import contextmanager
import copy
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from aureon.terramind_native.common import validate_recipe
from aureon.terramind_native.geo import write_cog
from aureon.terramind_native.loader import select_state, strict_load, verify_embedding_aliases
from aureon.terramind_native.reuse import exact_conditioning_reuse
from aureon.terramind_native.rng import rng_capsule, snapshot, state_digest


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_embeddings = {"untok_sen2l2a@224": None}
        self.weight = torch.nn.Parameter(torch.tensor(2.))
        self.calls = 0
    def forward_encoder(self, x, mask):
        self.calls += 1
        return x * self.weight + mask.to(x.dtype)


def fixture():
    core = Core().eval()
    model = SimpleNamespace(sampler=SimpleNamespace(model=core),training=False,
                            modalities=["S2L2A"],output_modalities=["tok_sen1grd@224"])
    return model, torch.arange(6.).reshape(1,2,3), torch.zeros((1,2,3),dtype=torch.bool)


class NativeTests(unittest.TestCase):
    def test_rng_success_retry_and_exception_restore(self):
        before = state_digest(snapshot(torch,np,()))
        draws = []
        for _ in range(2):
            with rng_capsule(17) as receipt:
                draws.append((random.random(),np.random.random(),torch.rand(3).tolist()))
            self.assertTrue(receipt["restored"])
            self.assertEqual(state_digest(snapshot(torch,np,())), before)
        self.assertEqual(draws[0],draws[1])
        with self.assertRaisesRegex(RuntimeError,"injected"):
            with rng_capsule(18):
                torch.rand(6); np.random.random(); random.random()
                raise RuntimeError("injected")
        self.assertEqual(state_digest(snapshot(torch,np,())), before)

    def test_rng_overlap_fails_without_leaking_lock(self):
        with rng_capsule(17):
            with self.assertRaisesRegex(RuntimeError,"already in use"):
                with rng_capsule(17): pass
        with rng_capsule(18) as receipt: pass
        self.assertTrue(receipt["restored"])

    def test_exact_reuse_eight_steps_and_one_step_control(self):
        for n in (1,8):
            model,x,mask = fixture(); core = model.sampler.model
            with torch.no_grad(), exact_conditioning_reuse(model,enabled=True) as receipt:
                outputs = [core.forward_encoder(x.clone(),mask.clone()).clone() for _ in range(n)]
            self.assertEqual(receipt["encoder_calls"],1)
            self.assertEqual(receipt["cache_hits"],n-1)
            self.assertTrue(receipt["cache_released"])
            self.assertNotIn("forward_encoder",core.__dict__)
            self.assertTrue(all(torch.equal(outputs[0],x) for x in outputs))

    def test_changed_conditioning_mask_or_parameters_falls_back(self):
        for change in ("input","mask","parameter"):
            model,x,mask = fixture(); core = model.sampler.model
            with torch.no_grad(), exact_conditioning_reuse(model,enabled=True) as receipt:
                first = core.forward_encoder(x,mask).clone()
                if change == "input": x = x+1
                elif change == "mask": mask = ~mask
                else: core.weight.add_(1)
                second = core.forward_encoder(x,mask)
                self.assertFalse(torch.equal(first,second))
            self.assertEqual(receipt["encoder_calls"],2)
            self.assertEqual(receipt["cache_hits"],0)

    def test_unsupported_encoder_or_multiple_outputs_does_not_cache(self):
        for change in ("encoder","output","training"):
            model,x,mask = fixture(); core = model.sampler.model
            if change == "encoder": core.encoder_embeddings["tok_sen1grd@224"] = None
            elif change == "output": model.output_modalities.append("tok_dem@224")
            else: model.training = True
            with torch.no_grad(), exact_conditioning_reuse(model,enabled=True) as receipt:
                for _ in range(2): core.forward_encoder(x,mask)
            self.assertFalse(receipt["eligible"]); self.assertEqual(receipt["cache_hits"],0)

    def test_cache_exception_restores_and_new_request_recomputes(self):
        model,x,mask = fixture(); core = model.sampler.model
        with self.assertRaisesRegex(RuntimeError,"injected"):
            with torch.no_grad(), exact_conditioning_reuse(model,enabled=True) as receipt:
                core.forward_encoder(x,mask); raise RuntimeError("injected")
        self.assertTrue(receipt["cache_released"]); self.assertNotIn("forward_encoder",core.__dict__)
        with torch.no_grad(), exact_conditioning_reuse(model,enabled=True) as next_receipt:
            core.forward_encoder(x,mask)
        self.assertEqual(next_receipt["encoder_calls"],1); self.assertEqual(core.calls,2)

    def test_inference_mode_fails_before_installing_hook(self):
        model,_,_ = fixture()
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError,"no_grad"):
            with exact_conditioning_reuse(model,enabled=True): pass
        self.assertNotIn("forward_encoder",model.sampler.model.__dict__)

    def test_checkpoint_subset_requires_complete_active_weights(self):
        expected = {"encoder_embeddings.untok_sen2l2a@224.w":torch.ones(2),
                    "decoder_embeddings.tok_sen1grd@224.w":torch.ones(3)}
        checkpoint = {**expected,"encoder_embeddings.tok_sen1grd@224.w":torch.zeros(4)}
        selected,unused = select_state(expected,checkpoint,allow_inactive_embeddings=True)
        self.assertEqual(set(selected),set(expected)); self.assertEqual(len(unused),1)
        for bad in ({}, {**checkpoint,"unrecognized":torch.ones(1)},
                    {**checkpoint,"decoder_embeddings.tok_sen1grd@224.extra":torch.ones(1)},
                    {**checkpoint,"decoder_embeddings.tok_sen1grd@224.w":torch.ones(3,dtype=torch.float64)}):
            with self.assertRaises(ValueError): select_state(expected,bad,allow_inactive_embeddings=True)

    def test_actual_cpu_strict_loader_checks_values(self):
        with tempfile.TemporaryDirectory() as td:
            module = torch.nn.Linear(2,3)
            expected = copy.deepcopy(module.state_dict())
            path = Path(td)/"state.pt"; torch.save(expected,path)
            with torch.no_grad(): module.weight.zero_()
            receipt = strict_load(module,path)
            self.assertTrue(receipt["matched_all_active_tensor_values"])
            self.assertTrue(torch.equal(module.weight,expected["weight"]))

    def test_fresh_embedding_containers_must_share_actual_modules(self):
        encoder = torch.nn.ModuleDict({"s2":torch.nn.Linear(2,3)})
        decoder = torch.nn.ModuleDict({"s1":torch.nn.Linear(3,2)})
        core = SimpleNamespace(encoder_embeddings=torch.nn.ModuleDict(encoder),
                               decoder_embeddings=torch.nn.ModuleDict(decoder))
        model = SimpleNamespace(encoder_embeddings=encoder,decoder_embeddings=decoder,sampler=SimpleNamespace(model=core))
        verify_embedding_aliases(model)
        core.encoder_embeddings["s2"] = torch.nn.Linear(2,3)
        with self.assertRaises(ValueError): verify_embedding_aliases(model)

    def test_continuous_cog_round_trip_and_synthetic_guard(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"out.tif"
            values = np.arange(2*224*224,dtype=np.float32).reshape(2,224,224)/1000-20
            mask = np.ones((224,224),dtype=bool)
            grid = {"shape":[224,224],"crs":"EPSG:32647","transform":[10,0,632260,0,-10,1583830]}
            with self.assertRaises(ValueError): write_cog(path,values,mask,grid,{})
            result = write_cog(path,values,mask,grid,{"synthetic":True,"sensor_observation":False})
            self.assertTrue(result["decoded_values_exact"]); self.assertEqual(result["units"],"dB")
            values[0,0,0] = np.nan
            with self.assertRaises(ValueError): write_cog(Path(td)/"bad.tif",values,mask,grid,{"synthetic":True,"sensor_observation":False})

    def test_recipe_is_explicit_and_bounded(self):
        recipe = {"token_steps":8,"diffusion_steps":10,"temperature":1.,"top_p":.8,"top_k":0,"seed":17}
        self.assertEqual(validate_recipe(recipe),recipe)
        for change in ({"token_steps":4},{"diffusion_steps":0},{"seed":True},{"top_p":.9}):
            with self.assertRaises(ValueError):validate_recipe({**recipe,**change})

    def test_cpu_only_test_process(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(torch.cuda.is_available())


if __name__ == "__main__": unittest.main()
