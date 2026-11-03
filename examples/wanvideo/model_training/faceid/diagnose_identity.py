"""Three-way identity diagnostic using the actual training scheduler and cached geometry.
No optimization. See diagnostic output README.md and run.sh for reproduction.
"""
import argparse
import csv
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/wanvideo/model_training'))
from train import WanTrainingModule
from diffsynth.core import UnifiedDataset
from diffsynth.extensions.faceid.arcface import file_sha256, identity_embedding
from diffsynth.extensions.faceid.alignment import align_faces
from diffsynth.extensions.faceid.decoder import FaceIDDecoder
from diffsynth.extensions.faceid.preprocess import FaceIDAnnotator
from diffsynth.models.wan_video_vae import count_conv3d


def pixels(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1


def pil(tensor):
    return Image.fromarray(((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy())


@torch.no_grad()
def decode_selected(decoder, latents, frames):
    # Same causal operations as decode_chunk; reuse prefix work across requested chunks.
    cache = [None] * count_conv3d(decoder.decoder)
    result = {}
    for j in range((max(frames) + 3) // 4 + 1):
        z = latents[:, :, j:j+1] * decoder.latent_std + decoder.latent_mean
        out, cache, _ = decoder.decoder(decoder.conv2(z), feat_cache=cache, feat_idx=[0])
        start = 0 if j == 0 else 4*j-3
        for frame in frames:
            if start <= frame < start + out.shape[2]:
                result[frame] = out[0, :, frame-start].float().cpu()
        del out
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--base-checkpoint', required=True)
    p.add_argument('--lora-checkpoint', required=True)
    p.add_argument('--detector', default=str(ROOT / 'faceid_model/det_10g.onnx'))
    p.add_argument('--samples', type=int, nargs='+', default=[923, 2145])
    p.add_argument('--frames', type=int, nargs='+', default=[2, 40, 80])
    p.add_argument('--sigmas', type=float, nargs='+', default=[.1, .3, .5])
    p.add_argument('--seed', type=int, default=20261103)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if not all(0 < f < 81 for f in args.frames) or not all(0 <= s <= 1 for s in args.sigmas):
        p.error('Frames must be in 1..80 and sigma in [0,1]')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    cfg = json.loads(Path(args.config).read_text())
    cache_root = Path(cfg['dataset_base_path'])
    manifest = json.loads((cache_root / '_cache_manifest.json').read_text())
    mc = manifest['config']
    metadata = Path(mc['dataset_metadata_path'])
    assert file_sha256(metadata) == mc['metadata_sha256'], 'Metadata changed since caching'
    assert file_sha256(cfg['faceid_weights']) == mc['faceid']['model_sha256']
    assert file_sha256(cfg['faceid_vae_path']) == mc['faceid']['vae_sha256']
    assert file_sha256(args.detector) == mc['faceid']['detector_sha256']
    rows = list(csv.DictReader(metadata.open()))
    operator = UnifiedDataset.default_video_operator(**{k: mc[k] for k in (
        'max_pixels', 'height', 'width', 'num_frames', 'frame_rate', 'fix_frame_rate',
        'frame_count_stride', 'frame_count_remainder', 'frame_count_rounding', 'min_num_frames', 'max_frame_padding')},
        base_path=mc['dataset_base_path'], height_division_factor=16, width_division_factor=16)
    annotator = FaceIDAnnotator(SimpleNamespace(faceid_detector_path=args.detector,
        faceid_weights=cfg['faceid_weights'], faceid_det_threshold=mc['faceid']['det_threshold'],
        faceid_min_face_size=mc['faceid']['min_face_size']), args.device)
    decoder = FaceIDDecoder.from_checkpoint(cfg['faceid_vae_path'], args.device)
    kwargs = {k: v for k, v in cfg.items() if k in inspect.signature(WanTrainingModule.__init__).parameters}
    kwargs.update(device='cpu', task='sft:train', resume_from_checkpoint=args.base_checkpoint,
                  lora_checkpoint=args.lora_checkpoint, use_gradient_checkpointing=True,
                  use_gradient_checkpointing_offload=False)
    print('Loading base and LoRA:', args.base_checkpoint, args.lora_checkpoint, flush=True)
    model = WanTrainingModule(**kwargs).eval().requires_grad_(False)
    model.to(args.device)
    pipe = model.pipe
    pipe.device = torch.device(args.device)
    provenance = dict(arguments=vars(args), training_config=cfg, cache_manifest=manifest,
        base_sha256=file_sha256(args.base_checkpoint), lora_sha256=file_sha256(args.lora_checkpoint),
        torch_version=torch.__version__, samples=[])
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    scores = []
    for sid in args.samples:
        paths = list(cache_root.rglob(f'{sid:08d}__*.pth'))
        assert len(paths) == 1, (sid, paths)
        shared, positive, negative = torch.load(paths[0], map_location='cpu', weights_only=True)
        ann = shared.pop('faceid')
        assert ann['reference_valid']
        selected = [f for f in args.frames if 0 < f < 81 and ann['valid'][f]]
        assert selected, 'No valid requested frames'
        frames = operator(rows[sid]['video'])
        assert (frames[0].height, frames[0].width) == (ann['height'], ann['width'])
        refpath = rows[sid].get('input_image')
        reference = Image.open(Path(mc['dataset_base_path']) / refpath).convert('RGB') if refpath else frames[0]
        ref = annotator.face(reference)
        assert ref is not None
        ref_cos = float(ref[1] @ ann['reference_embedding'])
        assert ref_cos > .999, f'Reference mismatch: {ref_cos}'
        folder = output / f'sample_{sid:05d}'
        folder.mkdir()
        reference.save(folder / 'reference.png')
        refcrop = align_faces(pixels(reference)[None].to(args.device), ref[0][None].to(args.device))[0]
        pil(refcrop).save(folder / 'reference_crop.png')
        provenance['samples'].append(dict(id=sid, cache=str(paths[0]), video=rows[sid]['video'],
            selected_frames=selected, reference_recompute_cosine=ref_cos))
        sheets = {f: [('reference', pil(refcrop))] for f in selected}
        raw_embeddings = {}
        def measure(branch, images, sigma=None, requested=None):
            for f, image in images.items():
                crop = align_faces(image[None].to(args.device), ann['inverse_matrices'][f:f+1].to(args.device))
                emb = identity_embedding(annotator.encoder, crop)[0].cpu()
                if branch == 'raw': raw_embeddings[f] = emb
                similarity = float(emb @ ann['reference_embedding'])
                score = dict(sample=sid, frame=f, branch=branch, requested_sigma=requested,
                    actual_sigma=sigma, cosine_reference=similarity, identity_loss=1-similarity,
                    cosine_raw_frame=float(emb @ raw_embeddings[f]), quality=float(ann['quality'][f]),
                    pixel_min=float(image.min()), pixel_max=float(image.max()),
                    pixel_outside_fraction=float((image.abs()>1).float().mean()),
                    crop_outside_fraction=float((crop.abs()>1).float().mean()))
                scores.append(score)
                label = branch if sigma is None else f'{branch}_s{requested:.2f}'
                pil(image).save(folder / f'f{f:02d}_{label}.png')
                pil(crop[0]).save(folder / f'f{f:02d}_{label}_crop.png')
                sheets[f].append((f'{label}\ncos={similarity:.4f}', pil(crop[0])))
                # Full-frame overlay: exact source polygon sampled by the affine crop.
                matrix = ann['inverse_matrices'][f]
                corners = torch.tensor([[0.,0.,1.],[111.,0.,1.],[111.,111.,1.],[0.,111.,1.]]) @ matrix.T
                overlay = pil(image)
                draw = ImageDraw.Draw(overlay)
                poly = [tuple(v) for v in corners.tolist()]
                draw.line(poly+[poly[0]], fill='red', width=3)
                overlay.save(folder / f'f{f:02d}_{label}_overlay.png')
            with (output / 'scores.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(scores[0]))
                writer.writeheader(); writer.writerows(scores)
        measure('raw', {f:pixels(frames[f]) for f in selected})
        inputs = model.transfer_data_to_device((shared,positive,negative), pipe.device, pipe.torch_dtype)
        inputs[0]['use_gradient_checkpointing'] = False
        inputs[0]['use_gradient_checkpointing_offload'] = False
        for unit in pipe.units:
            inputs = pipe.unit_runner(unit, pipe, *inputs)
        data = {**inputs[0], **inputs[1]}
        clean = data['input_latents']
        # Match the fixed reference history used by training identity decoding.
        clean_decode = torch.cat((data['first_frame_latents'], clean[:,:,1:]), dim=2)
        print(f'Sample {sid}: decoding ground truth', flush=True)
        measure('vae_gt', decode_selected(decoder, clean_decode, selected))
        generator = torch.Generator(device=args.device).manual_seed(args.seed+sid)
        noise = torch.randn(clean.shape, device=args.device, dtype=clean.dtype, generator=generator) * data.get('noise_scale',1.)
        for requested in args.sigmas:
            index = (pipe.scheduler.sigmas-requested).abs().argmin()
            timestep = pipe.scheduler.timesteps[index:index+1].to(device=pipe.device,dtype=pipe.torch_dtype)
            sigma_index = (pipe.scheduler.timesteps-timestep.to(pipe.scheduler.timesteps.device)).abs().argmin()
            sigma = pipe.scheduler.sigmas[sigma_index].to(device=pipe.device,dtype=torch.float32)
            data['latents'] = pipe.scheduler.add_noise(clean,noise,timestep)
            data['latents'][:,:,:1] = data['first_frame_latents']
            print(f'Sample {sid}: prediction sigma={float(sigma):.6f}', flush=True)
            prediction = pipe.model_fn(**{name:getattr(pipe,name) for name in pipe.in_iteration_models}, **data, timestep=timestep)
            estimated = data['latents'].float()-sigma*prediction.float()
            estimated[:,:,:1] = data['first_frame_latents'].float()
            del prediction
            measure('predicted', decode_selected(decoder,estimated.to(torch.bfloat16),selected),float(sigma),requested)
            del estimated
        for f, tiles in sheets.items():
            sheet = Image.new('RGB',(224*len(tiles),280),'white')
            draw = ImageDraw.Draw(sheet)
            for i,(label,img) in enumerate(tiles):
                sheet.paste(img.resize((224,224)),(224*i,0))
                draw.text((224*i+5,230),label,fill='black')
            sheet.save(folder / f'f{f:02d}_comparison.png')
        (output / 'provenance.json').write_text(json.dumps(provenance,indent=2))
        print(f'Sample {sid}: complete',flush=True)
    print('Saved:',output,flush=True)


if __name__ == '__main__':
    main()
