"""Flow matching + identity supervision; independent frozen teachers, no collectives."""
import random
import time
import torch
from .alignment import align_faces
from .arcface import identity_embedding
from .sampling import FaceIDSigmaSampler


def cosine_identity_loss(embeddings, reference, quality):
    similarities = (embeddings * reference).sum(-1)
    loss = ((1 - similarities) * quality).sum() / quality.sum().clamp_min(1e-8)
    return loss, similarities.mean()


def flow_match_faceid_loss(pipe, auxiliary, annotation, config, step, **inputs):
    sampler = getattr(pipe, 'faceid_sigma_sampler', None)
    if sampler is None:
        sampler = FaceIDSigmaSampler(pipe.scheduler, pipe.torch_dtype, config,
                                     inputs.get('min_timestep_boundary', 0.),
                                     inputs.get('max_timestep_boundary', 1.))
    timestep, sigma, focused = sampler.sample(pipe.device)
    clean = inputs['input_latents']
    if clean.ndim != 5 or clean.shape[:3] != (1, 16, 21) or 'first_frame_latents' not in inputs:
        raise ValueError('FaceID requires 81-frame S2V latents [B,16,21,H,W] with a reference first latent')
    noise = torch.randn_like(clean) * inputs.get('noise_scale', 1.)
    noisy = pipe.scheduler.add_noise(clean, noise, timestep)
    noisy[:, :, :1] = inputs['first_frame_latents']
    inputs['latents'] = noisy
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
    target = pipe.scheduler.training_target(clean, noise, timestep)
    fm = torch.nn.functional.mse_loss(prediction[:, :, 1:].float(), target[:, :, 1:].float())
    fm = fm * pipe.scheduler.training_weight(timestep).to(fm.device)
    zero = fm.detach().new_zeros(())
    identity, cosine = zero, zero
    scheduled = step % config.faceid_every == 0
    sigma_valid = config.faceid_sigma_min <= sigma.item() <= config.faceid_sigma_max
    valid = annotation['valid'].bool().cpu()
    chunks = [j for j in range(1, 21) if valid[4*j-3:4*j+1].any()]
    ramp = min(1., step / config.faceid_warmup_steps) if config.faceid_warmup_steps else 1.
    weight = config.faceid_weight * ramp
    active = scheduled and sigma_valid and bool(chunks) and annotation['reference_valid'] and weight > 0
    frames_used, chunk_index = 0, 0
    started = time.perf_counter()
    if active:
        chunk_index = random.choice(chunks)
        candidates = [i for i in range(4*chunk_index-3, 4*chunk_index+1) if valid[i]]
        selected = sorted(random.sample(candidates, min(config.faceid_frames, len(candidates))))
        # Only the selected chunk has gradients. The fixed first latent is kept
        # as causal decoding history, NOT removed (otherwise frames shift by 4).
        estimated = noisy.float() - sigma * prediction.float()
        estimated = torch.cat((inputs['first_frame_latents'].float(), estimated[:, :, 1:]), dim=2)
        decoded = auxiliary.decoder.decode_chunk(estimated.to(auxiliary.dtype), chunk_index,
                                                  save_on_cpu=config.faceid_activation_offload)
        frames = decoded[0, :, [i-(4*chunk_index-3) for i in selected]].permute(1, 0, 2, 3)
        if tuple(frames.shape[-2:]) != (annotation['height'], annotation['width']):
            raise ValueError('FaceID cache geometry differs from decoded frames; rebuild the feature cache')
        matrices = annotation['inverse_matrices'][selected].to(device=frames.device, dtype=torch.float32)
        faces = align_faces(frames, matrices)
        with torch.autograd.graph.save_on_cpu() if config.faceid_activation_offload else torch.enable_grad():
            embeddings = identity_embedding(auxiliary.encoder, faces)
        reference = annotation['reference_embedding'].to(device=frames.device, dtype=torch.float32)
        quality = annotation['quality'][selected].to(device=frames.device, dtype=torch.float32)
        identity, cosine = cosine_identity_loss(embeddings, reference, quality)
        frames_used = len(selected)
    total = fm + weight * identity
    metrics = {
        'loss/fm': fm.detach(), 'loss/faceid': identity.detach(),
        'loss/faceid_weighted': (weight * identity).detach(), 'loss/total': total.detach(),
        'faceid/cosine': cosine.detach(), 'faceid/active': float(active),
        'faceid/eligible': float(sigma_valid), 'faceid/weight': weight,
        'sampling/interval_branch': float(focused),
        'sampling/in_interval': float(sampler.lower <= sigma.item() <= sampler.upper),
        'sampling/interval_probability': sampler.interval_probability,
        'faceid/valid_frame_fraction': valid[1:].float().mean().item(),
        'faceid/reference_valid': float(annotation['reference_valid']),
        'faceid/frames_used': frames_used, 'faceid/chunk': chunk_index,
        'faceid/sigma': sigma.detach(), 'faceid/forward_host_seconds': time.perf_counter()-started,
    }
    return {'loss': total, 'metrics': metrics}
