"""Causal Wan2.1 VAE decoding with gradients only through one latent chunk.

All preceding chunks are decoded under no_grad to reproduce the exact causal
history. Their gradients are deliberately truncated. Each call owns its cache;
no mutable cache is shared with another forward or checkpoint recomputation.
"""
import torch
from torch import nn
from diffsynth.models.wan_video_vae import Decoder3d, CausalConv3d, count_conv3d
from diffsynth.core.loader.file import load_state_dict

MEAN = [-.7571, -.7089, -.9113, .1075, -.1745, .9653, -.1517, 1.5508,
        .4134, -.0715, .5517, -.3632, -.1922, -.9497, .2503, -.2921]
STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
       3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


class FaceIDDecoder(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.conv2 = CausalConv3d(16, 16, 1)
        self.decoder = Decoder3d(dim, 16, [1, 2, 4, 4], 2, [], [True, True, False], 0.)
        self.register_buffer('latent_mean', torch.tensor(MEAN).view(1, 16, 1, 1, 1), persistent=False)
        self.register_buffer('latent_std', torch.tensor(STD).view(1, 16, 1, 1, 1), persistent=False)

    @classmethod
    def from_checkpoint(cls, path, device, dtype=torch.bfloat16):
        state = load_state_dict(path)
        state = {k.removeprefix('model.'): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if k.startswith(('decoder.', 'conv2.'))}
        model = cls()
        model.load_state_dict(state, strict=True)
        del state
        return model.eval().requires_grad_(False).to(device=device, dtype=dtype)

    def decode_chunk(self, latents, chunk_index, save_on_cpu=True):
        """chunk 0 -> frame 0; chunk j>0 -> frames 4*j-3 .. 4*j.

        No spatial latent crop or interpolation: those alter decoder semantics.
        No clamp: saturating estimates must not lose their FaceID gradient.
        """
        if latents.ndim != 5 or latents.shape[1] != 16 or not 0 <= chunk_index < latents.shape[2]:
            raise ValueError('Expected Wan2.1 [B,16,T,H,W] latents and a valid chunk index')
        cache = [None] * count_conv3d(self.decoder)
        for index in range(chunk_index + 1):
            selected = index == chunk_index
            with torch.set_grad_enabled(torch.is_grad_enabled() and selected):
                z = latents[:, :, index:index + 1]
                z = z * self.latent_std + self.latent_mean
                # conv2 has temporal kernel 1, so chunkwise application is exact.
                if selected and save_on_cpu:
                    with torch.autograd.graph.save_on_cpu():
                        result, cache, _ = self.decoder(self.conv2(z), feat_cache=cache, feat_idx=[0])
                else:
                    result, cache, _ = self.decoder(self.conv2(z), feat_cache=cache, feat_idx=[0])
            if not selected:
                del result
        expected = 1 if chunk_index == 0 else 4
        if result.shape[2] != expected:
            raise RuntimeError(f'Unexpected Wan VAE temporal mapping: {result.shape[2]} != {expected}')
        return result
