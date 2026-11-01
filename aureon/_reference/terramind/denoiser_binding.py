"""Exact numerical binding for the pinned TerraMind base S1GRD224 decoder.

The candidate admits one/eight token steps with ten diffusion steps. Only this
decoder class, FP32 shapes and asset contract are implemented; broader shapes,
modalities, diffusion schedules and model identities remain native fallbacks.
"""
import hashlib
import importlib
import math
import torch

TARGET = 'tok_sen1grd@224'
MODULE = 'terratorch.models.backbones.terramind.tokenizer.models.unet.unet'
PROFILE_ID = 'terramind_s1grd224_fp32_graph_v3'


def native_frequency_cpu(dim, max_period=10000):
    # Preserve native CPU exp/arange order, including division AFTER the
    # multiplication. CUDA exp/arange is not substituted for these FP32 bytes.
    half = dim//2
    return torch.exp(-math.log(max_period)*torch.arange(start=0, end=half, dtype=torch.float, device='cpu')/half)


def staged_embedding(timesteps, dim, max_period, frequency, signature):
    if (dim, max_period, timesteps.device) != (signature[0], signature[1], frequency.device):
        raise ValueError('Timestep embedding signature changed')
    if timesteps.ndim != 1 or timesteps.dtype != torch.int64:
        raise ValueError('Unsupported timestep type/shape')
    args = timesteps[:, None].float()*frequency[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def eligibility(model, recipe, device):
    """Admission-only fallback; no forward, CUDA allocation or qualification claim."""
    if device != 'cuda:0':
        return False, 'CUDA profile required'
    if (type(recipe.get('token_steps')) is not int or recipe['token_steps'] not in (1, 8)
            or type(recipe.get('diffusion_steps')) is not int or recipe['diffusion_steps'] != 10):
        return False, 'Graph candidate supports one/eight token steps and ten diffusion steps only'
    if model.training or list(model.modalities) != ['S2L2A'] or list(model.output_modalities) != [TARGET]:
        return False, 'Unsupported native model mode or modalities'
    tokenizer = model.tokenizer[TARGET]
    decoder = getattr(tokenizer, 'decoder', None)
    if (decoder is None or type(decoder).__name__ != 'PatchedUNetCondCat'
            or type(decoder).__module__ != MODULE or decoder.training
            or getattr(decoder, 'model_channels', None) != 256):
        return False, 'Unsupported denoiser class or configuration'
    if any(p.dtype != torch.float32 or p.device != torch.device(device) for p in decoder.parameters()):
        return False, 'Denoiser requires resident CUDA FP32 parameters'
    return True, 'Candidate fixed-shape S1GRD profile; execution qualification is external'


class S1GRDDecoderBinding:
    def __init__(self, model):
        self.module = model.tokenizer[TARGET].decoder
        self.original = self.module.forward
        self.signature = (getattr(self.module, 'model_channels', 256), 10000)
        self.frequency = self.frequency_version = None
        self.embedding_namespace = None
        self.versions = self._versions()

    def _versions(self):
        # Include registered buffers and storage identity as well as in-place
        # version counters: replacing tensor.data must not reuse a stale graph.
        return tuple((name, id(t), t._version, t.data_ptr(), t.dtype, t.device, tuple(t.shape))
                     for kind, values in (('parameter', self.module.named_parameters()), ('buffer', self.module.named_buffers()))
                     for key, t in values for name in (kind + ':' + key,))

    def validate(self, inputs):
        x, t, cond = (inputs[k] for k in ('image', 'timestep', 'conditioning'))
        if (set(inputs) != {'image', 'timestep', 'conditioning'}
                or self.module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.get_float32_matmul_precision() != 'highest'
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
                or x.shape != (1, 2, 224, 224) or x.dtype != torch.float32 or x.device != torch.device('cuda:0')
                or cond.dtype != torch.float32 or cond.device != x.device or cond.shape != (1, 5, 14, 14)
                or not isinstance(t, torch.Tensor) or t.ndim != 0 or t.dtype != torch.int64
                or t.device.type not in ('cpu', 'cuda') or (t.device.type == 'cuda' and t.device != x.device)
                or self._versions() != self.versions):
            raise ValueError('Unsupported or mutated exact denoiser graph contract')
        if getattr(self.module, 'model_channels', 256) != self.signature[0]:
            raise ValueError('Denoiser channels changed')
        if self.frequency is not None and self.frequency._version != self.frequency_version:
            raise ValueError('Frequency buffer changed')

    def stage(self, inputs, stats):
        device = inputs['image'].device
        buffers = {name: value.to(device=device).clone() for name, value in inputs.items()}
        cpu = native_frequency_cpu(*self.signature)
        self.frequency = cpu.to(device)
        self.frequency_version = self.frequency._version
        stats['frequency_cpu_sha256'] = hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
        stats['frequency_bytes'] = cpu.numel()*cpu.element_size()
        return buffers

    def forward(self, buffers):
        return self.original(buffers['image'], buffers['timestep'], buffers['conditioning'], orig_res=None)

    def namespace(self):
        if self.embedding_namespace is not None:
            return self.embedding_namespace
        return vars(importlib.import_module(self.module.__class__.__module__))

    def embedding(self, timesteps, dim, max_period=10000):
        if self.frequency is None:
            raise RuntimeError('Frequency staging must precede denoiser')
        return staged_embedding(timesteps, dim, max_period, self.frequency, self.signature)

    def close(self):
        self.module = self.original = self.frequency = self.embedding_namespace = None
        self.versions = ()
