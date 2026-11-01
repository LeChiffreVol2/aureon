"""CPU tensor doubles for true batch-stage ownership; no TerraMind model/weights."""
from concurrent.futures import CancelledError
from copy import deepcopy
from pathlib import Path
import os
import random
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from aureon.api import GenerationInput, GenerationJob, GenerationRecipe
from aureon.api.batch_generation import _PairSession, PairCredits
from aureon.api.generation_queue import INPUT_BYTES, OUTPUT_BYTES
from aureon.terramind_native import denoiser_batch as batch
from aureon.terramind_native.common import INPUT_BANDS, write_json
from aureon.terramind_native.rng import snapshot, state_digest, rng_capsule
from aureon.terramind_native.session import SessionBusy, SessionFailed

TARGET = batch.TARGET
RECIPE = GenerationRecipe(seed=17).native()


class Scheduler:
    def __init__(self):
        self.config = {'prediction_type': 'sample', 'num_train_timesteps': 1000}
        self.timesteps = torch.arange(9, -1, -1, dtype=torch.int64)
        self.calls = []
    def step(self, model_output, timestep, sample, eta=0.0, generator=None):
        assert eta == 0 and generator is None
        self.calls.append(int(timestep))
        return types.SimpleNamespace(prev_sample=sample*0.75 + model_output*0.25)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.calls = []; self.random = False; self.fail = False
    def forward(self, image, timestep, cond, *, orig_res=None):
        self.calls.append(tuple(image.shape))
        if self.fail: raise RuntimeError('synthetic stage failure')
        if self.random: torch.rand(1)
        return image*0.25 + cond.mean(dim=(1,2,3))[:,None,None,None] + timestep.float()/100


class Sampler:
    def __init__(self):
        self.model = types.SimpleNamespace(encoder_embeddings={'untok_sen2l2a@224': object()})
    def generate(self, mod_dict, schedule, *, seed=None, num_tokens=None, **kw):
        assert num_tokens is None
        for i, step in enumerate(schedule):
            # Reproduce seed+step resets and draws, not the neural token model.
            torch.manual_seed(seed+i)
            torch.rand(3)
        mod_dict[TARGET]['tensor'] = torch.randint(0, 50, (1, 196))
        return mod_dict


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.sampler = Sampler()
        self.tokenizer = {TARGET: types.SimpleNamespace(decoder=Decoder(), noise_scheduler=Scheduler())}
        self.modalities = ['S2L2A']; self.output_modalities = [TARGET]; self.decoding_steps = 8
        self.mod_name_mapping = {'S1GRD': TARGET}
        self.pretraining_mean = {TARGET: torch.tensor(-7.)}
        self.pretraining_std = {TARGET: torch.tensor(3.)}
        self.eval()
    def forward(self, data, **kwargs):
        counts = [25]*4 + [24]*4
        schedule = [dict(num_tokens=n, target_domain=TARGET, scheme='roar', cfg_scale=1.,temperature=1.) for n in counts]
        state = {TARGET: {'tensor': torch.zeros((1,196),dtype=torch.long)}}
        tokens = self.sampler.generate(state, schedule, num_tokens=196,
                    seed=random.randint(-(2**31),2**31-1))[TARGET]['tensor']
        cond = tokens.float().reshape(1,1,14,14).expand(1,5,14,14).clone()/50
        image = torch.randn((1,2,224,224), generator=None)  # Native draw point/device.
        tok = self.tokenizer[TARGET]
        for timestep in tok.noise_scheduler.timesteps:
            prediction = tok.decoder(image, timestep, cond, orig_res=None)
            image = tok.noise_scheduler.step(prediction.float(), timestep, image, generator=None).prev_sample
        return {'S1GRD': image*3.-7.}


class Runtime:
    device = 'cpu'
    numerical_core = None
    def __init__(self):
        self.torch = torch; self.model = FakeModel(); self.versions = ()
        self.config = dict(model={}, tokenizer={}, source_manifest={}, dependencies={})
        self.loaded_identity = {'test_double': True}
    def _versions(self): return ()
    def _policy(self): pass
    def _assert_policy(self): assert not torch.is_grad_enabled()
    def load(self, config, work): pass
    def close(self): self.model = None; return {'passed':True, 'cuda_allocated_bytes':0, 'cuda_reserved_bytes':0}


def data(value=1):
    return GenerationInput(np.full((12,224,224),value,np.float32), bands=INPUT_BANDS,
        units='surface_reflectance_x10000', valid_mask=np.ones((224,224),bool))


class BatchContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.runtime = Runtime()
    def test_compatible_seeds_remain_distinct_and_other_recipes_reject(self):
        pair=batch.compatible_recipes([RECIPE,{**RECIPE,'seed':101}])
        self.assertEqual([r['seed'] for r in pair],[17,101])
        for other in ({**RECIPE,'token_steps':1},{**RECIPE,'diffusion_steps':9}):
            with self.assertRaises(ValueError): batch.compatible_recipes([RECIPE,other])
        with self.assertRaises(ValueError): batch.compatible_recipes([RECIPE])
    def test_ddpm_and_instance_overrides_are_rejected(self):
        with patch.object(batch.importlib,'import_module',return_value=types.SimpleNamespace(DDIMScheduler=Scheduler)):
            self.assertEqual(batch.scheduler_identity(Scheduler())['eta'],0)
            class Other(Scheduler): pass
            with self.assertRaises(batch.BatchIneligible): batch.scheduler_identity(Other())
            scheduler=Scheduler(); scheduler.step=lambda *a: None
            with self.assertRaises(batch.BatchIneligible): batch.scheduler_identity(scheduler)
    def test_graph_and_workspace_reject_before_cuda_access(self):
        for options in (dict(graph=True,max_workspace_bytes=batch.WORKSPACE_BYTES),dict(max_workspace_bytes=1)):
            with self.assertRaises(batch.BatchIneligible):
                batch.eligibility(None,[RECIPE,RECIPE],min_free_cuda_bytes=0,**options)
    def test_preparation_uses_native_draw_point_and_restores_rng_methods_scheduler(self):
        from aureon.terramind_native.compat import schedule_compatibility
        model=self.runtime.model; tok=model.tokenizer[TARGET]; original=tok.noise_scheduler
        before=state_digest(snapshot(torch,np,()))
        with torch.no_grad():
            prepared=batch.prepare_request(self.runtime,data().snapshot()[0],RECIPE,reuse='off',cancel=lambda:None)
        self.assertEqual(model.tokenizer[TARGET].decoder.calls, [])
        self.assertIs(tok.noise_scheduler,original)
        self.assertIsNot(prepared['scheduler'],original)
        self.assertNotIn('forward',vars(tok.decoder))
        self.assertEqual(before,state_digest(snapshot(torch,np,())))
        with torch.no_grad(),rng_capsule(17) as rng,schedule_compatibility(model,token_steps=8):
            expected=model({'S2L2A':torch.ones((1,12,224,224))})['S1GRD']
        self.assertEqual(prepared['rng']['request_post_rng_sha256'],rng['request_post_rng_sha256'])
        # Same prepared noise/schedule with batch-one decoder must reproduce the native fixture.
        image=prepared['image']
        for timestep in prepared['scheduler'].timesteps:
            pred=tok.decoder(image,timestep,prepared['conditioning'],orig_res=None)
            image=prepared['scheduler'].step(pred,timestep,image,generator=None).prev_sample
        self.assertTrue(torch.equal(image*3.-7.,expected))
    def test_real_batch_dimension_two_once_per_step_and_independent_scheduler_state(self):
        with torch.no_grad():
            states=[batch.prepare_request(self.runtime,data(v).snapshot()[0],{**RECIPE,'seed':s},reuse='off',cancel=lambda:None)
                    for v,s in ((1,17),(2,101))]
            expected=deepcopy(states)
            decoder=self.runtime.model.tokenizer[TARGET].decoder
            for state in expected:
                for timestep in state['scheduler'].timesteps:
                    pred=decoder(state['image'],timestep,state['conditioning'],orig_res=None)
                    state['image']=state['scheduler'].step(pred,timestep,state['image']).prev_sample
            decoder.calls.clear()
            receipt=batch.denoise_pair(torch,decoder,states,cancel=lambda:None)
        self.assertEqual(decoder.calls,[(2,2,224,224)]*10)
        self.assertEqual(receipt['decoder_calls'],10)
        for actual,reference in zip(states,expected):
            self.assertTrue(torch.equal(actual['image'],reference['image']))
            self.assertEqual(actual['scheduler'].calls,list(range(9,-1,-1)))
        self.assertNotEqual(states[0]['image'].data_ptr(),states[1]['image'].data_ptr())
    def test_rng_consumption_during_decoder_rejects_and_restores_global_state(self):
        before=state_digest(snapshot(torch,np,()))
        with self.assertRaisesRegex(RuntimeError,'consumed RNG'):
            with batch._unchanged_rng(torch,()): torch.rand(1)
        self.assertEqual(before,state_digest(snapshot(torch,np,())))
    def test_shared_scheduler_and_bad_shape_reject(self):
        s=dict(image=torch.zeros(1,2,224,224),conditioning=torch.zeros(1,5,14,14),scheduler=Scheduler())
        with self.assertRaises(ValueError): batch.denoise_pair(torch,Decoder(),[s,s],cancel=lambda:None)
        second=deepcopy(s);second['image']=torch.zeros(2,2,224,224)
        with self.assertRaises(ValueError): batch.denoise_pair(torch,Decoder(),[s,second],cancel=lambda:None)
    def test_cancellation_between_steps_stops_without_another_decoder_call(self):
        s=dict(image=torch.zeros(1,2,224,224),conditioning=torch.zeros(1,5,14,14),scheduler=Scheduler())
        states=[s,deepcopy(s)];decoder=Decoder(); checkpoints=[]
        def cancel():
            checkpoints.append(1)
            if len(checkpoints)==3: raise CancelledError()
        with self.assertRaises(CancelledError): batch.denoise_pair(torch,decoder,states,cancel=cancel)
        self.assertEqual(len(decoder.calls),2)
        self.assertEqual([len(x['scheduler'].calls) for x in states],[2,2])
    def test_pair_credits_exhaust_release_and_reject_underbudget(self):
        with self.assertRaises(ValueError): PairCredits(INPUT_BYTES,2*OUTPUT_BYTES)
        credits=PairCredits(2*INPUT_BYTES,2*OUTPUT_BYTES);credits.reserve()
        self.assertEqual(credits.metadata()['input_bytes_reserved'],2*INPUT_BYTES)
        with self.assertRaises(SessionBusy): credits.reserve()
        credits.release();self.assertEqual(credits.metadata()['active_pairs'],0)
        with self.assertRaises(RuntimeError): credits.release()
    def test_installed_pipeline_noise_boundary_and_ddim_steps_on_cpu(self):
        if os.environ.get('AUREON_BLOCK_FOUNDATION_IMPORTS') == '1':
            self.skipTest('Upstream pipeline imports prohibited by CPU contract runner; run this test in the dependency environment without AUREON_BLOCK_FOUNDATION_IMPORTS.')
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


class PairSessionFixture(_PairSession):
    """CPU-only executor for shared ownership tests, not a shipped B2 API."""
    def _execute_pair(self, snapshots, recipes, ids, cancel, admission):
        return batch.run_pair(self._runtime, snapshots, recipes, ids, reuse=self._reuse,
            cancel=cancel, session_identity=self._identity,
            max_workspace_bytes=self._pair_workspace, min_free_cuda_bytes=self._pair_free,
            admission=admission)


class SessionContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        write_json(self.root/'config.json',{'recipe':RECIPE})
        self.patches=[patch('aureon.api.batch_generation._NativeRuntime',Runtime),
                      patch.object(batch,'eligibility',return_value={'profile_id':batch.PROFILE,'qualified':False})]
        for p in self.patches:p.start()
        self.sessions=[]
    def tearDown(self):
        for session in self.sessions:
            try:session.close()
            except SessionFailed:pass
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def session(self):
        value=PairSessionFixture(self.root/'config.json',self.root/f'work-{len(self.sessions)}',
              execute_model=True,allow_unqualified_batch=True,reuse='off')
        self.sessions.append(value);return value
    def jobs(self):return [GenerationJob(data(i),recipe=GenerationRecipe(seed=s),job_id=f'job-{s}') for i,s in ((1,17),(2,101))]
    def test_order_owned_outputs_receipts_and_once_only_attempts(self):
        session=self.session();result=session.generate_pair(self.jobs())
        self.assertEqual([r['job_id'] for r in result],['job-17','job-101'])
        self.assertEqual([r['provenance']['recipe']['seed'] for r in result],[17,101])
        self.assertFalse(np.shares_memory(result[0]['array'],result[1]['array']))
        self.assertTrue(all(r['provenance']['deterministic_stage_rng']['unchanged'] for r in result))
        self.assertEqual(session.metadata()['jobs_completed'],2)
        self.assertEqual(session.metadata()['batch_candidate']['credits']['active_pairs'],0)
        with self.assertRaises(FileExistsError):session.generate_pair(self.jobs())
    def test_pre_cancel_no_load_no_attempts_no_credit_leak(self):
        session=self.session();event=threading.Event();event.set()
        with self.assertRaises(CancelledError):session.generate_pair(self.jobs(),cancel_event=event)
        self.assertEqual(session.metadata()['model_load_count'],0)
        self.assertEqual(session.metadata()['batch_candidate']['credits']['active_pairs'],0)
        self.assertEqual(session._attempts,set())
    def test_invalid_pair_and_profile_fail_before_load(self):
        session=self.session()
        with self.assertRaises(TypeError):session.generate_pair(iter(self.jobs()))
        jobs=self.jobs();jobs[0]=GenerationJob(data(),execution_profile='recommended')
        with self.assertRaises(batch.BatchIneligible):session.generate_pair(jobs)
        self.assertEqual(session.metadata()['model_load_count'],0)
    def test_mid_preparation_cancel_restores_owner_and_releases_pair_credits(self):
        session=self.session();event=threading.Event();original=batch.prepare_request
        before=state_digest(snapshot(torch,np,()))
        def prepare(*a,**kw):
            value=original(*a,**kw);event.set();return value
        with patch.object(batch,'prepare_request',side_effect=prepare):
            with self.assertRaises(CancelledError):session.generate_pair(self.jobs(),cancel_event=event)
        self.assertEqual(session.metadata()['state'],'ready')
        self.assertEqual(session.metadata()['jobs_completed'],0)
        self.assertEqual(session.metadata()['batch_candidate']['credits']['active_pairs'],0)
        self.assertEqual(before,state_digest(snapshot(torch,np,())))
        self.assertEqual(len(list(session.work.glob('batch-*.json'))),0)
        fresh=[GenerationJob(j.input_data,recipe=j.recipe,job_id=j.job_id+'-retry') for j in self.jobs()]
        self.assertEqual(len(session.generate_pair(fresh)),2)
    def test_overlapping_pair_and_close_reject_without_extra_admission(self):
        session=self.session();entered=threading.Event();release=threading.Event();errors=[]
        original=batch.run_pair
        def blocked(*a,**kw):
            entered.set()
            if not release.wait(5):raise RuntimeError('fixture wait timeout')
            return original(*a,**kw)
        def run():
            try:session.generate_pair(self.jobs())
            except BaseException as exc:errors.append(exc)
        with patch.object(batch,'run_pair',side_effect=blocked):
            thread=threading.Thread(target=run);thread.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(session.metadata()['batch_candidate']['credits']['active_pairs'],1)
                with self.assertRaises(SessionBusy):session.generate_pair(self.jobs())
                with self.assertRaises(SessionBusy):session.close()
            finally:release.set();thread.join(5)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
        self.assertEqual(session.metadata()['jobs_completed'],2)
    def test_failed_pair_commit_never_marks_prepared_job_success(self):
        session=self.session();original=write_json
        def fail_commit(path,value):
            if Path(path).name.startswith('batch-'):raise OSError('synthetic commit failure')
            return original(path,value)
        with patch('aureon.api.batch_generation.write_json',side_effect=fail_commit):
            with self.assertRaises(SessionFailed):session.generate_pair(self.jobs())
        import json
        saved=[json.loads(p.read_text()) for p in session.work.glob('jobs/*/*/receipt.json')]
        self.assertEqual([x['status'] for x in saved],['prepared','prepared'])
        self.assertEqual(len(list(session.work.glob('batch-*.json'))),0)
        self.assertEqual(session.metadata()['jobs_completed'],0)
    def test_execution_failure_is_terminal_releases_credits_and_process_lock(self):
        session=self.session()
        with patch.object(batch,'run_pair',side_effect=RuntimeError('fixture failure')):
            with self.assertRaises(SessionFailed):session.generate_pair(self.jobs())
        self.assertEqual(session.metadata()['state'],'failed')
        self.assertEqual(session.metadata()['batch_candidate']['credits']['active_pairs'],0)
        self.assertEqual(len(self.session().generate_pair(self.jobs())),2)
    def test_shared_pair_base_has_no_executable_fused_route(self):
        with self.assertRaises(TypeError):
            _PairSession(self.root/'config.json', self.root/'abstract',
                         execute_model=True, allow_unqualified_batch=True)
        self.assertFalse((self.root/'abstract').exists())

    def test_opt_in_required_without_creating_work(self):
        with self.assertRaisesRegex(ValueError,'allow_unqualified_batch'):
            PairSessionFixture(self.root/'config.json',self.root/'no-work',execute_model=True)
        self.assertFalse((self.root/'no-work').exists())


if __name__=='__main__':unittest.main()
