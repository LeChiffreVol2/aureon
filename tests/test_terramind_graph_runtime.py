"""CPU execution double checks: ownership and lifecycle, never GPU speed/parity."""
from contextlib import ExitStack, nullcontext
import threading
import types
import unittest
from unittest.mock import patch
import torch
from aureon.terramind_native.graph_runtime import FixedTensorGraph, CudaGraphBackend
from aureon.terramind_native.denoiser_binding import eligibility, MODULE
from aureon.terramind_native.denoiser_graph import DenoiserGraph


class Binding:
    def __init__(self): self.closed=False;self.stages=0
    def validate(self, inputs):
        if set(inputs)!={'image','timestep','conditioning'}:raise ValueError('invalid names')
    def stage(self,inputs,stats):
        self.stages+=1
        return {k:v.clone() for k,v in inputs.items()}
    def forward(self,values): return values['image']+values['timestep']+values['conditioning']
    def close(self): self.closed=True


class Backend:
    def __init__(self):self.captures=0;self.resets=0
    def capture(self,forward,buffers,stats):
        self.captures+=1;output=forward(buffers)
        stats['warmup_stage_forwards']+=1;stats['capture_stage_forwards']+=1
        def reset():self.resets+=1
        return types.SimpleNamespace(replay=lambda:output.copy_(forward(buffers)),reset=reset),output


class RuntimeTests(unittest.TestCase):
    def values(self,x=1,t=3,c=4):
        return {'image':torch.tensor([x,x+1.]),'timestep':torch.tensor(t),
                'conditioning':torch.tensor([c,c+1.])}
    def make(self):
        b=Binding();backend=Backend();return FixedTensorGraph(b,backend=backend),b,backend
    def test_all_inputs_refreshed_independent_output_and_single_allocation_across_requests(self):
        graph,b,backend=self.make()
        with graph.request():first=graph.call(self.values())
        pointers={k:v.data_ptr() for k,v in graph.buffers.items()}
        with graph.request():second=graph.call(self.values(6,8,9))
        self.assertTrue(torch.equal(first,torch.tensor([8.,10.])))
        self.assertTrue(torch.equal(second,torch.tensor([23.,25.])))
        self.assertEqual(pointers,{k:v.data_ptr() for k,v in graph.buffers.items()})
        self.assertNotEqual(first.data_ptr(),second.data_ptr())
        self.assertNotEqual(second.data_ptr(),graph.output.data_ptr())
        self.assertEqual((b.stages,backend.captures),(1,1));graph.close()
        self.assertTrue(b.closed);self.assertEqual(backend.resets,1)
        self.assertFalse(graph.buffers);self.assertIsNone(graph.output)
    def test_changed_signature_no_recapture_and_failure_is_sticky(self):
        g,b,backend=self.make();g.call(self.values())
        changed=self.values();changed['image']=torch.zeros(3)
        with self.assertRaisesRegex(ValueError,'signature'):g.call(changed)
        with self.assertRaisesRegex(RuntimeError,'Failed'):g.call(self.values())
        self.assertEqual(backend.captures,1);g.close()
    def test_failed_setup_never_retried_and_releases_staged_buffers(self):
        g,b,backend=self.make()
        with patch.object(backend,'capture',side_effect=RuntimeError('capture failure')):
            with self.assertRaisesRegex(RuntimeError,'capture failure'):g.call(self.values())
        self.assertTrue(g.started and g.failed)
        with self.assertRaisesRegex(RuntimeError,'Failed'):g.call(self.values())
        g.close();self.assertFalse(g.buffers);self.assertTrue(b.closed)
    def test_replay_failure_terminal_and_reset_failure_still_releases(self):
        g,b,_=self.make();g.call(self.values())
        g.graph.replay=lambda:(_ for _ in ()).throw(RuntimeError('replay failure'))
        g.graph.reset=lambda:(_ for _ in ()).throw(RuntimeError('reset failure'))
        with self.assertRaisesRegex(RuntimeError,'replay failure'):g.call(self.values())
        with self.assertRaisesRegex(RuntimeError,'reset failure'):g.close()
        self.assertTrue(g.closed and b.closed);self.assertFalse(g.buffers);self.assertIsNone(g.binding)
        self.assertTrue(g.close()['released'])
    def test_request_nesting_close_and_cross_thread_are_rejected(self):
        g,_,_=self.make()
        with g.request():
            with self.assertRaisesRegex(RuntimeError,'already active'):
                with g.request():pass
            with self.assertRaisesRegex(RuntimeError,'active'):g.close()
        errors=[]
        def other():
            for fn in (lambda:g.call(self.values()),g.close):
                try:fn()
                except RuntimeError as exc:errors.append(str(exc))
        thread=threading.Thread(target=other);thread.start();thread.join()
        self.assertEqual(len(errors),2);self.assertTrue(all('owner thread' in e for e in errors));g.close()


class AdmissionTests(unittest.TestCase):
    def model(self):
        cls=type('PatchedUNetCondCat',(),{'__module__':MODULE,'training':False,'model_channels':256,
            'parameters':lambda self:[types.SimpleNamespace(dtype=torch.float32,device=torch.device('cuda:0'))]})
        return types.SimpleNamespace(training=False,modalities=['S2L2A'],output_modalities=['tok_sen1grd@224'],
            tokenizer={'tok_sen1grd@224':types.SimpleNamespace(decoder=cls())})
    def test_only_one_same_asset_recipe_extension_no_diffusion_expansion(self):
        model=self.model()
        for steps in (1,8):self.assertTrue(eligibility(model,{'token_steps':steps,'diffusion_steps':10},'cuda:0')[0])
        for token,diffusion in ((2,10),(8,20),(1,9),(True,10)):
            self.assertFalse(eligibility(model,{'token_steps':token,'diffusion_steps':diffusion},'cuda:0')[0])
        self.assertFalse(eligibility(model,{'token_steps':1,'diffusion_steps':10},'cpu')[0])
        model.output_modalities=['other'];self.assertFalse(eligibility(model,{'token_steps':1,'diffusion_steps':10},'cuda:0')[0])


class CaptureBoundaryTests(unittest.TestCase):
    def test_setup_rng_restored_after_rng_drift_and_after_forward_failure(self):
        for failure in (False,True):
            state={'cuda':torch.tensor([17],dtype=torch.uint8),'resets':0}
            cpu_before=torch.get_rng_state().clone();cuda_before=state['cuda'].clone()
            stream=types.SimpleNamespace(wait_stream=lambda other:None)
            graph=types.SimpleNamespace(reset=lambda:state.update(resets=state['resets']+1))
            def forward(values):
                torch.rand(1);state['cuda'].add_(1)
                if failure:raise RuntimeError('warmup failure')
                return values['image'].clone()
            replacements={'device':lambda *a:nullcontext(),'Stream':lambda **kw:stream,
                'current_stream':lambda *a:stream,'stream':lambda *a:nullcontext(),
                'synchronize':lambda *a:None,'CUDAGraph':lambda:graph,'graph':lambda *a,**kw:nullcontext(),
                'get_rng_state':lambda *a:state['cuda'].clone(),
                'set_rng_state':lambda value,*a:state.update(cuda=value.clone())}
            stats={'warmup_stage_forwards':0,'capture_stage_forwards':0}
            with ExitStack() as stack:
                for name,value in replacements.items():stack.enter_context(patch.object(torch.cuda,name,value))
                with self.assertRaisesRegex(RuntimeError,'warmup failure' if failure else 'consumed RNG'):
                    CudaGraphBackend().capture(forward,{'image':torch.zeros(1)},stats)
            self.assertTrue(torch.equal(cpu_before,torch.get_rng_state()))
            self.assertTrue(torch.equal(cuda_before,state['cuda']))
            self.assertEqual(state['resets'],0 if failure else 1)


if __name__=='__main__':unittest.main()
