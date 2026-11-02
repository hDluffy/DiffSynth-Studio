"""Explicit real-weight smoke test before an expensive training job.

Run with PYTHONPATH=. This validates math/gradients, not recognition quality:
random latents are not faces. No weights are changed or downloaded.
"""
import argparse
import json
import torch
from diffsynth.extensions.faceid.arcface import load_arcface, identity_embedding
from diffsynth.extensions.faceid.decoder import FaceIDDecoder
from diffsynth.extensions.faceid.alignment import align_faces

parser = argparse.ArgumentParser()
parser.add_argument('--faceid_weights', required=True)
parser.add_argument('--faceid_vae_path', required=True)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
device = torch.device(args.device)
dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
decoder = FaceIDDecoder.from_checkpoint(args.faceid_vae_path, device, dtype)
encoder = load_arcface(args.faceid_weights, device)
z = torch.randn(1, 16, 3, 8, 8, device=device, dtype=dtype, requires_grad=True)
video = decoder.decode_chunk(z, 2)
frames = video[0, :, [1, 3]].permute(1, 0, 2, 3)
matrix = torch.tensor([[64/112, 0, 0], [0, 64/112, 0]], device=device).repeat(2, 1, 1)
faces = align_faces(frames, matrix)
embedding = identity_embedding(encoder, faces)
reference = torch.nn.functional.normalize(torch.randn(1, 512, device=device), dim=-1)
loss = (1 - (embedding * reference).sum(-1)).mean()
loss.backward()
assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
assert z.grad[:, :, :2].abs().sum() == 0
assert all(p.grad is None for p in decoder.parameters())
assert all(p.grad is None for p in encoder.parameters())
print(json.dumps({'loss': loss.item(), 'selected_latent_grad_norm': z.grad.float().norm().item(),
                  'teachers_frozen': True, 'embedding_shape': list(embedding.shape)}, indent=2))
