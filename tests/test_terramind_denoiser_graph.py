"""CPU-only graph contract, scoped state, math and public-session regressions."""
from contextlib import contextmanager
import math
import threading
import types
import unittest
from unittest.mock import patch
import weakref
import torch

from aureon.terramind_native.denoiser_graph import DenoiserGraph,native_frequency_cpu,staged_embedding
from aureon.terramind_native.session import SessionFailed,SessionClosed
import test_terramind_session as fixtures
Model=fixtures.Model
RECIPE=fixtures.RECIPE
TARGET=fixtures.TARGET

class Decoder(torch.nn.Module):
    model_channels=256
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.eval()
    def forward(self,x,t,cond,**kwargs):return x+t+cond

class GraphTests(unittest.TestCase):
    def make(self):
        model=types.SimpleNamespace(tokenizer={TARGET:types.SimpleNamespace(decoder=Decoder())})
        graph=DenoiserGraph(model)
        graph.binding.embedding_namespace={'timestep_embedding':lambda *a:None}
        graph._memory=lambda:{'allocated_bytes':0,'reserved_bytes':0}
        return model,graph

    def test_exact_native_cpu_embedding_math_all_training_timesteps(self):
        t=torch.arange(1000,dtype=torch.int64);half=128
        native=torch.exp(-math.log(10000)*torch.arange(start=0,end=half,dtype=torch.float)/half)
        actual=native_frequency_cpu(256)
        self.assertEqual(native.numpy().tobytes(),actual.numpy().tobytes())
        args=t[:,None].float()*native[None]
        expected=torch.cat([torch.cos(args),torch.sin(args)],dim=-1)
        self.assertTrue(torch.equal(expected,staged_embedding(t,256,10000,actual,(256,10000))))
        with self.assertRaises(ValueError):staged_embedding(t,256,9000,actual,(256,10000))

    def test_pre_capture_fallback_restores_native_method_and_global(self):
        model,g=self.make();old=g.binding.module.forward;embedding=g.binding.embedding_namespace['timestep_embedding']
        def reject(*a):raise ValueError('unsupported signature')
        g.validate=reject
        with g.request(model) as receipt:
            value=g.binding.module(torch.tensor(2.),torch.tensor(3),torch.tensor(4.),orig_res=None)
        self.assertEqual(value.item(),9);self.assertTrue(receipt['fallback']);self.assertFalse(receipt['enabled'])
        self.assertEqual(g.binding.module.forward,old);self.assertIs(g.binding.embedding_namespace['timestep_embedding'],embedding)
        self.assertEqual(g.total['capture_stage_forwards'],0);g.close()

    def test_post_capture_rejection_is_terminal_for_the_call_and_restores_bindings(self):
        model,g=self.make();old=g.binding.module.forward;embedding=g.binding.embedding_namespace['timestep_embedding']
        g.runtime.graph=types.SimpleNamespace(reset=lambda:None);g.total['capture_stage_forwards']=1;g.runtime.started=True
        def reject(*a):raise ValueError('changed signature')
        g.validate=reject
        with self.assertRaisesRegex(ValueError,'changed signature'):
            with g.request(model) as receipt:g.binding.module(torch.tensor(2.),torch.tensor(3),torch.tensor(4.),orig_res=None)
        self.assertFalse(receipt['fallback']);self.assertTrue(receipt['restored'])
        self.assertEqual(g.binding.module.forward,old);self.assertIs(g.binding.embedding_namespace['timestep_embedding'],embedding);g.close()

    def test_frequency_mutation_and_owner_thread_are_rejected(self):
        _,g=self.make();device=torch.device('cuda:0')
        x=types.SimpleNamespace(shape=(1,2,224,224),dtype=torch.float32,device=device)
        cond=types.SimpleNamespace(shape=(1,5,14,14),dtype=torch.float32,device=device)
        g.binding.frequency=torch.ones(128);g.binding.frequency_version=g.binding.frequency._version
        policy=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
                                     cudnn=types.SimpleNamespace(allow_tf32=False))
        with torch.no_grad(),patch.object(torch,'backends',policy),patch.object(torch,'get_float32_matmul_precision',return_value='highest'):
            g.validate(x,torch.tensor(1),cond,None,{'orig_res':None})
            g.binding.frequency.add_(1)
            with self.assertRaisesRegex(ValueError,'Frequency buffer'):g.validate(x,torch.tensor(1),cond,None,{'orig_res':None})
        errors=[]
        def other():
            try:g.runtime.owner()
            except RuntimeError as e:errors.append(str(e))
        thread=threading.Thread(target=other);thread.start();thread.join()
        self.assertIn('owner thread',errors[0]);g.close()

    def test_close_releases_references_even_if_graph_reset_fails(self):
        model,g=self.make();ref=weakref.ref(g.binding.module)
        def fail():raise RuntimeError('invalid capture')
        g.runtime.graph=types.SimpleNamespace(reset=fail);model=None
        with self.assertRaisesRegex(RuntimeError,'invalid capture'):g.close()
        self.assertIsNone(g.binding.module);self.assertIsNone(g.binding.original);self.assertTrue(g.runtime.closed);self.assertIsNone(ref())
        self.assertTrue(g.close()['released'])

    def test_global_binding_is_exclusive_and_restores_after_body_failure(self):
        model,g=self.make();other_model,other=self.make()
        original=g.binding.module.forward;embedding=g.binding.embedding_namespace['timestep_embedding']
        with self.assertRaisesRegex(RuntimeError,'body failure'):
            with g.request(model):
                with self.assertRaisesRegex(RuntimeError,'already in use'):
                    with other.request(other_model):pass
                raise RuntimeError('body failure')
        self.assertTrue(g.runtime.failed)
        self.assertEqual(g.binding.module.forward,original)
        self.assertIs(g.binding.embedding_namespace['timestep_embedding'],embedding)
        with other.request(other_model):pass  # Global binding released, not poisoned.
        g.close();other.close()

    def test_parameter_storage_and_registered_buffer_mutation_are_sealed(self):
        model,g=self.make()
        previous=g.binding.versions
        g.binding.module.weight.data=torch.tensor(7.)
        self.assertNotEqual(previous,g.binding._versions())
        g.close()
        model,g=self.make();g.binding.module.register_buffer('state',torch.tensor(1.))
        self.assertNotEqual(g.binding.versions,g.binding._versions());g.close()


class PublicGraphTests(fixtures.SessionTests):
    def test_one_step_request_restores_decoder_steps_scheduler_rng_and_keeps_reuse_off(self):
        session=self.session(denoiser='graph')
        first=session.run(job_id='one',attempt_id='one',recipe={**RECIPE,'token_steps':1})
        self.assertTrue(first['decoder']['fallback'])  # CPU fallback, no CUDA claim.
        self.assertFalse(first['reuse_decision']['enabled'])
        self.assertTrue(first['scheduler_restored'] and first['decoding_steps_restored'])
        self.assertTrue(first['rng']['restored'])
        self.assertEqual(session._executor.submit(lambda:session._runtime.model.decoding_steps).result(),8)
        second=session.run(job_id='eight',attempt_id='one',recipe=RECIPE)
        self.assertTrue(second['reuse_decision']['enabled'])
        self.assertTrue(second['decoding_steps_restored'])
        self.assertTrue(session.close()['passed'])

    def test_graph_pre_capture_cpu_fallback_and_native_override(self):
        session=self.session(denoiser='graph')
        first=session.run(job_id='graph',attempt_id='cpu-fallback')
        self.assertTrue(first['decoder']['fallback']);self.assertEqual(first['denoiser_mode'],'graph')
        second=session.run(job_id='native',attempt_id='off',denoiser='native')
        self.assertEqual(second['decoder']['profile_id'],'native')
        self.assertEqual(first['raw']['sha256'],second['raw']['sha256'])
        self.assertTrue(session.close()['passed'])

    def test_graph_option_and_hook_conflict_reject_before_loading(self):
        with self.assertRaises(ValueError):self.session(denoiser='invalid')
        with self.assertRaises(ValueError):self.session(denoiser='graph',decoder_context=lambda *a:None)
        session=self.session()
        with self.assertRaises(ValueError):session.run(job_id='scene',attempt_id='one',denoiser='graph',decoder_context=lambda *a:None)
        self.assertEqual(self.events,[])
        session.run(job_id='scene',attempt_id='one')

    def test_failed_graph_terminates_session_releases_request_tensor_and_graph(self):
        tensors=[];graph_instances=[]
        original=Model.forward
        def forward(model,data,**kwargs):
            tensors.append(weakref.ref(data['S2L2A']))
            raise RuntimeError('simulated capture failure')
        class FakeGraph:
            def __init__(self,model):self.model=model;self.total={'failed':False};graph_instances.append(self)
            @contextmanager
            def request(self,model):
                receipt={'profile_id':'fake-capture','restored':False}
                try:yield receipt
                finally:receipt['restored']=True
            def close(self):self.model=None;return {'released':True,'failed':True}
        # Base sessions bind the frozen provider at load; inject failure at that boundary.
        with patch('aureon._reference.terramind.denoiser_graph.eligibility',new=lambda *a:(True,'CPU seam')),\
             patch('aureon._reference.terramind.denoiser_graph.DenoiserGraph',FakeGraph),patch.object(Model,'forward',forward):
            session=self.session(denoiser='graph')
            with self.assertRaisesRegex(SessionFailed,'simulated capture failure'):session.run(job_id='scene',attempt_id='one')
        self.assertEqual(session.metadata()['state'],'failed');self.assertIsNone(tensors[0]())
        self.assertIsNone(graph_instances[0].model);self.assertTrue(session.close()['passed'])
        with self.assertRaises(SessionClosed):session.run(job_id='scene',attempt_id='two')

    def test_graph_close_failure_still_releases_model_and_restores_policy(self):
        from aureon.terramind_native import session as native
        session=self.session();session.run(job_id='scene',attempt_id='one')
        class BrokenClose:
            total={'released':True,'failed':True}
            def close(self):raise RuntimeError('injected graph close error')
        def install():
            session._runtime.denoiser_graph=BrokenClose()
            return dict(session._runtime.before)
        previous=session._executor.submit(install).result()
        with self.assertRaises(SessionFailed):session.close()
        cleanup=native.read_json(session.work/'session-cleanup.json')
        self.assertFalse(cleanup['passed']);self.assertIn('injected graph close error',cleanup['denoiser_graph']['cleanup_error'])
        self.assertIsNone(self.refs[0]());self.assertIsNone(session._runtime)
        self.assertEqual(torch.get_float32_matmul_precision(),previous['precision'])
        self.assertEqual(torch.get_num_threads(),previous['threads'])
        self.assertEqual([torch.backends.cuda.cudnn_sdp_enabled(),torch.backends.cuda.flash_sdp_enabled(),
                          torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.math_sdp_enabled()],previous['sdp'])

if __name__=='__main__':unittest.main()
