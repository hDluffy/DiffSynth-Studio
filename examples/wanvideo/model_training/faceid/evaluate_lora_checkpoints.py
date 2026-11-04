"""Compare original training LoRA checkpoints on fixed cached 81-frame S2V samples.

Single-step denoising identity evaluation, not full video-generation evaluation.
Loads the base once; replaces every adapter tensor strictly for each checkpoint.
"""
import argparse
import csv
import glob
import html
import inspect
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from diagnose_identity import (ROOT, WanTrainingModule, UnifiedDataset, FaceIDAnnotator,
    FaceIDDecoder, file_sha256, identity_embedding, align_faces, pixels, pil, decode_selected)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_step(path):
    found = re.search(r'(?:step|epoch)-(\d+)', path.stem)
    return int(found[1]) if found else 0


def read_adapter(path, expected):
    """Reject partial/mismatched/non-adapter files before touching the model."""
    raw = load_file(str(path), device='cpu')
    state = {}
    for key, tensor in raw.items():
        key = key.removeprefix('pipe.dit.')
        key = key.replace('.lora_A.weight', '.lora_A.default.weight')
        key = key.replace('.lora_B.weight', '.lora_B.default.weight')
        if key in state:
            raise ValueError(f'Duplicate normalized adapter key: {key}')
        state[key] = tensor
    missing, extra = set(expected)-set(state), set(state)-set(expected)
    if missing or extra:
        raise ValueError(f'{path}: adapter key mismatch: missing={len(missing)}, extra={len(extra)}; '
                         f'examples={sorted(missing)[:3]}, {sorted(extra)[:3]}. Use original training LoRA, not ComfyUI/full-model weights.')
    for key, tensor in state.items():
        if tensor.shape != expected[key].shape or not torch.isfinite(tensor).all():
            raise ValueError(f'{path}: invalid shape/nonfinite tensor {key}: {tensor.shape}, expected {expected[key].shape}')
    return state


@torch.no_grad()
def replace_adapter(expected, state):
    for key, param in expected.items():
        if state is None:
            param.zero_()  # Both A/B zero: mathematically exact base-only output.
        else:
            param.copy_(state[key])


def summarize(output, rows, stages):
    baseline = {(r['sample'],r['frame'],r['requested_sigma']):r for r in rows if r['checkpoint']=='base_only'}
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['checkpoint'],str(r['requested_sigma']))].append(r)
        grouped[(r['checkpoint'],'all')].append(r)
    summary = []
    for stage in stages:
        for (name,sigma), group in grouped.items():
            if name != stage['label']:
                continue
            # Macro average per sample so samples with more valid frames do not dominate.
            by_sample = defaultdict(list)
            for r in group:
                by_sample[r['sample']].append(r)
            mean = lambda key: float(np.mean([np.mean([r[key] for r in g]) for g in by_sample.values()]))
            delta = []
            for sample_rows in by_sample.values():
                delta.append(np.mean([r['cosine_reference']-baseline[(r['sample'],r['frame'],r['requested_sigma'])]['cosine_reference'] for r in sample_rows]))
            summary.append(dict(checkpoint=name,step=stage['step'],sigma=sigma,samples=len(by_sample),observations=len(group),
                cosine_reference=mean('cosine_reference'),identity_loss=mean('identity_loss'),
                cosine_raw_frame=mean('cosine_raw_frame'),cosine_reference_p10=float(np.quantile([r['cosine_reference'] for r in group],.1)),
                delta_reference_vs_base=float(np.mean(delta)),crop_outside_fraction=mean('crop_outside_fraction')))
    write_csv(output/'summary.csv',summary)
    all_rows = [r for r in summary if r['sigma']=='all']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="460" viewBox="0 0 1000 460">',
           '<rect width="1000" height="460" fill="white"/><text x="60" y="28" font-size="18">Fixed-sample identity scores (higher is better)</text>']
    vals=[r[k] for r in all_rows for k in ('cosine_reference','cosine_raw_frame')]
    lower=max(-1,min(vals)-.05); upper=min(1,max(vals)+.05)
    y=lambda v: 360-(v-lower)/(upper-lower)*290
    x=lambda i: 90+i*820/max(1,len(all_rows)-1)
    for tick in np.linspace(lower,upper,6):
        svg.append(f'<path d="M60 {y(tick)} H950" stroke="#ddd"/><text x="8" y="{y(tick)+5}" font-size="13">{tick:.3f}</text>')
    for key,color,title in [('cosine_reference','#1765ad','Reference cosine'),('cosine_raw_frame','#ce6510','Same-frame cosine')]:
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="'+ ' '.join(f'{x(i)},{y(r[key])}' for i,r in enumerate(all_rows))+'"/>')
        for i,r in enumerate(all_rows):
            svg.append(f'<circle cx="{x(i)}" cy="{y(r[key])}" r="4" fill="{color}"/>')
        svg.append(f'<text x="{60 if key=="cosine_reference" else 320}" y="445" fill="{color}" font-size="16">{title}</text>')
    for i,r in enumerate(all_rows):
        svg.append(f'<text x="{x(i)}" y="385" text-anchor="middle" font-size="11">{html.escape(r["checkpoint"])}</text>')
    svg.append('</svg>')
    (output/'scores.svg').write_text('\n'.join(svg))
    report=['# 多阶段 LoRA 固定样本评估', '',
        '本报告是训练式单步 clean latent 估计的身份评分，不是完整多步视频生成，也不是口型同步评分。', '',
        '| checkpoint | 对参考 cosine ↑ | 1-cosine ↓ | 对原始同帧 cosine ↑ | 相对 base 增益 |',
        '|---|---:|---:|---:|---:|']
    for r in all_rows:
        report.append(f'| {r["checkpoint"]} | {r["cosine_reference"]:.6f} | {r["identity_loss"]:.6f} | {r["cosine_raw_frame"]:.6f} | {r["delta_reference_vs_base"]:+.6f} |')
    report += ['', '![评分曲线](scores.svg)', '',
        '- `summary.csv`：按 checkpoint / sigma 汇总；all 对各样本等权，样本内对有效帧和 sigma 等权。p10 是所有观测的第 10 百分位。',
        '- `scores.csv`：逐样本、帧、sigma、checkpoint 的指标，包含训练 sigma gate 是否有效。',
        '- `reference_scores.csv`：原始帧和真实 latent VAE 重建基线。',
        '- `provenance.json`：权重指纹、模型配置、噪声种子、样本选择和过滤统计。',
        '- `crops/`：指定 --save-crops 时保存同坐标裁剪图。所有数值在 clamp 之前计算。',
        '- `base_only` 禁用本次待测 LoRA；如果指定的 base 自身包含已合并的 LoRA，该部分仍属于 base。',
        '- 各阶段严格替换全部 adapter 参数，不叠加；样本、噪声、条件、有效帧一致。',
        '- 若参考分数升高而同帧分数降低，需检查细节和表情，不应自动视作改进。',
        '- 默认两个已知样本仅供诊断。挑选 checkpoint 时应使用独立验证集；不以这里的最高分自动选出最佳模型。']
    (output/'README.md').write_text('\n'.join(report)+'\n')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='Training faceid_training_config.json (architecture/cache/teachers)')
    parser.add_argument('--base-checkpoint',required=True)
    parser.add_argument('--lora-checkpoints',nargs='+',required=True,help='Original training safetensors paths or quoted glob(s)')
    parser.add_argument('--cache-path',help='Override cache directory, e.g. held-out validation cache')
    parser.add_argument('--detector',default=str(ROOT/'faceid_model/det_10g.onnx'))
    parser.add_argument('--samples',type=int,nargs='+',default=[100,200],help='Zero-based metadata row IDs')
    parser.add_argument('--frames',type=int,nargs='+',default=[2,40,80])
    parser.add_argument('--sigmas',type=float,nargs='+',default=[.15,.3,.45])
    parser.add_argument('--seed',type=int,default=20261103)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',default=f'diagnostics/faceid_checkpoints_{datetime.now():%Y%m%d_%H%M%S}')
    parser.add_argument('--save-crops',action='store_true')
    args=parser.parse_args()
    if not all(0<f<81 for f in args.frames) or not all(0<s<1 for s in args.sigmas):
        parser.error('Frames must be in 1..80 and sigmas strictly between 0 and 1')
    for attr in ('samples','frames','sigmas'):
        setattr(args,attr,list(dict.fromkeys(getattr(args,attr))))
    paths=[]
    for pattern in args.lora_checkpoints:
        matches=glob.glob(pattern)
        if not matches: parser.error(f'No matching checkpoints: {pattern}')
        paths.extend(Path(v).resolve() for v in matches)
    paths=sorted(set(paths),key=lambda p:(checkpoint_step(p),str(p)))
    if any(not p.is_file() for p in paths): parser.error('Checkpoint must be a file')
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8); torch.manual_seed(args.seed)
    cfg=json.loads(Path(args.config).read_text())
    cache=Path(args.cache_path or cfg['dataset_base_path'])
    manifest=json.loads((cache/'_cache_manifest.json').read_text()); mc=manifest['config']
    if manifest.get('status')!='complete': raise ValueError('Cache must be complete')
    for path,expected in [(mc['dataset_metadata_path'],mc['metadata_sha256']),
                          (cfg['faceid_weights'],mc['faceid']['model_sha256']),
                          (cfg['faceid_vae_path'],mc['faceid']['vae_sha256']),
                          (args.detector,mc['faceid']['detector_sha256'])]:
        if file_sha256(path)!=expected: raise ValueError(f'Cache fingerprint mismatch: {path}')
    if any(cfg[k]!=v for k,v in mc['faceid']['rope'].items()): raise ValueError('Cache/config RoPE mismatch')
    metadata=list(csv.DictReader(open(mc['dataset_metadata_path'])))
    operator=UnifiedDataset.default_video_operator(**{k:mc[k] for k in ('max_pixels','height','width','num_frames','frame_rate','fix_frame_rate','frame_count_stride','frame_count_remainder','frame_count_rounding','min_num_frames','max_frame_padding')},base_path=mc['dataset_base_path'],height_division_factor=16,width_division_factor=16)
    annotator=FaceIDAnnotator(SimpleNamespace(faceid_detector_path=args.detector,faceid_weights=cfg['faceid_weights'],faceid_det_threshold=mc['faceid']['det_threshold'],faceid_min_face_size=mc['faceid']['min_face_size']),args.device)
    decoder=FaceIDDecoder.from_checkpoint(cfg['faceid_vae_path'],args.device)
    kwargs={k:v for k,v in cfg.items() if k in inspect.signature(WanTrainingModule.__init__).parameters}
    kwargs.update(device='cpu',task='sft:train',resume_from_checkpoint=args.base_checkpoint,lora_checkpoint=None,
                  preset_lora_path=None,preset_lora_model=None,use_gradient_checkpointing=True,use_gradient_checkpointing_offload=False)
    if cfg.get('fp8_models') or cfg.get('offload_models'): raise ValueError('Use normal BF16 config without custom offload/FP8')
    model=WanTrainingModule(**kwargs).eval().requires_grad_(False).to(args.device)
    pipe=model.pipe; pipe.device=torch.device(args.device)
    expected={k:v for k,v in pipe.dit.named_parameters() if '.lora_A.default.' in k or '.lora_B.default.' in k}
    if not expected: raise ValueError('No LoRA adapters created: check config')
    stages=[dict(label='base_only',path=None,step=0,sha256=None)]
    for i,path in enumerate(paths,1):
        state=read_adapter(path,expected)  # Preflight all checkpoints; do not silently score partial loads.
        del state
        stages.append(dict(label=f'{i:02d}_{path.stem}',path=str(path),step=checkpoint_step(path),sha256=file_sha256(path)))
    prov=dict(arguments=vars(args),training_config=cfg,cache_manifest=manifest,base_sha256=file_sha256(args.base_checkpoint),
              stages=stages,adapter_tensor_count=len(expected),torch_version=torch.__version__,samples=[],status='running')
    (output/'provenance.json').write_text(json.dumps(prov,indent=2))
    reference_rows=[]; sample_data=[]
    def embed(image,ann,f):
        crop=align_faces(image[None].to(args.device),ann['inverse_matrices'][f:f+1].to(args.device))
        return identity_embedding(annotator.encoder,crop)[0].cpu(),crop
    for sid in args.samples:
        if not 0<=sid<len(metadata): raise ValueError(f'Metadata row out of bounds: {sid}')
        found=list(cache.rglob(f'{sid:08d}__*.pth'))
        if len(found)!=1: raise ValueError(f'Expected one cache for {sid}, found {len(found)}')
        payload=torch.load(found[0],map_location='cpu',weights_only=True)
        ann=payload[0]['faceid']
        selected=[f for f in args.frames if bool(ann['valid'][f])]
        record=dict(sample=sid,cache=str(found[0]),reference_valid=bool(ann['reference_valid']),valid_frames=selected,
                    excluded_frames=[f for f in args.frames if f not in selected])
        prov['samples'].append(record)
        if not ann['reference_valid'] or not selected:
            record['skipped']=True; continue
        frames=operator(metadata[sid]['video'])
        if (frames[0].height,frames[0].width)!=(ann['height'],ann['width']): raise ValueError('Cache geometry mismatch')
        refpath=metadata[sid].get('input_image')
        reference=Image.open(Path(mc['dataset_base_path'])/refpath).convert('RGB') if refpath else frames[0]
        ref=annotator.face(reference)
        if ref is None or float(ref[1]@ann['reference_embedding'])<.999: raise ValueError(f'Reference mismatch: sample {sid}')
        record['reference_recompute_cosine']=float(ref[1]@ann['reference_embedding'])
        raw={}
        clean=payload[0]['input_latents']
        z=torch.cat((payload[0]['first_frame_latents'],clean[:,:,1:]),dim=2).to(device=args.device,dtype=torch.bfloat16)
        gt=decode_selected(decoder,z,selected); del z
        for f in selected:
            for branch,image in [('raw',pixels(frames[f])),('vae_gt',gt[f])]:
                e,crop=embed(image,ann,f)
                if branch=='raw': raw[f]=e
                reference_rows.append(dict(sample=sid,frame=f,branch=branch,cosine_reference=float(e@ann['reference_embedding']),cosine_raw_frame=float(e@raw[f])))
                if args.save_crops:
                    folder=output/'crops'/f'sample_{sid:05d}'; folder.mkdir(parents=True,exist_ok=True)
                    pil(crop[0]).save(folder/f'f{f:02d}_{branch}.png')
        sample_data.append(dict(sid=sid,path=found[0],ann=ann,selected=selected,raw=raw))
    write_csv(output/'reference_scores.csv',reference_rows)
    if not sample_data: raise ValueError('No valid samples/frames; inspect cache annotations')
    results=[]
    for stage in stages:
        state=None if stage['path'] is None else read_adapter(stage['path'],expected)
        replace_adapter(expected,state); del state
        print(f'Evaluating {stage["label"]}: {len(sample_data)} samples',flush=True)
        for sample in sample_data:
            sid=sample['sid']; ann=sample['ann']; selected=sample['selected']
            payload=torch.load(sample['path'],map_location='cpu',weights_only=True)
            payload[0].pop('faceid')
            inputs=model.transfer_data_to_device(payload,pipe.device,pipe.torch_dtype)
            inputs[0].update(use_gradient_checkpointing=False,use_gradient_checkpointing_offload=False)
            for unit in pipe.units: inputs=pipe.unit_runner(unit,pipe,*inputs)
            data={**inputs[0],**inputs[1]}; clean=data['input_latents']
            generator=torch.Generator(device=args.device).manual_seed(args.seed+sid)
            noise=torch.randn(clean.shape,device=args.device,dtype=clean.dtype,generator=generator)*data.get('noise_scale',1.)
            for sigma_id,requested in enumerate(args.sigmas):
                index=(pipe.scheduler.sigmas-requested).abs().argmin()
                timestep=pipe.scheduler.timesteps[index:index+1].to(device=pipe.device,dtype=pipe.torch_dtype)
                index=(pipe.scheduler.timesteps-timestep.to(pipe.scheduler.timesteps.device)).abs().argmin()
                sigma=pipe.scheduler.sigmas[index].to(device=pipe.device,dtype=torch.float32)
                data['latents']=pipe.scheduler.add_noise(clean,noise,timestep)
                data['latents'][:,:,:1]=data['first_frame_latents']
                prediction=pipe.model_fn(**{name:getattr(pipe,name) for name in pipe.in_iteration_models},**data,timestep=timestep)
                estimated=data['latents'].float()-sigma*prediction.float()
                estimated[:,:,:1]=data['first_frame_latents'].float()
                del prediction
                decoded=decode_selected(decoder,estimated.to(torch.bfloat16),selected); del estimated
                for f,image in decoded.items():
                    e,crop=embed(image,ann,f); cosine=float(e@ann['reference_embedding'])
                    results.append(dict(checkpoint=stage['label'],sample=sid,frame=f,requested_sigma=requested,actual_sigma=float(sigma),
                        eligible=cfg['faceid_sigma_min']<=float(sigma)<=cfg['faceid_sigma_max'],cosine_reference=cosine,identity_loss=1-cosine,
                        cosine_raw_frame=float(e@sample['raw'][f]),quality=float(ann['quality'][f]),crop_outside_fraction=float((crop.abs()>1).float().mean())))
                    if args.save_crops:
                        pil(crop[0]).save(output/'crops'/f'sample_{sid:05d}'/f'f{f:02d}_{stage["label"]}_sigma{sigma_id}.png')
                print(f'  sample={sid}, sigma={float(sigma):.6f} complete',flush=True)
                write_csv(output/'scores.csv',results)
        summarize(output,results,stages)
        (output/'provenance.json').write_text(json.dumps(prov,indent=2))
    # Final integrity: all checkpoints must cover exactly the same observations.
    keys=lambda label:{(r['sample'],r['frame'],r['requested_sigma']) for r in results if r['checkpoint']==label}
    if any(keys(s['label'])!=keys('base_only') for s in stages): raise RuntimeError('Unequal checkpoint coverage')
    prov['status']='complete'; (output/'provenance.json').write_text(json.dumps(prov,indent=2))
    print('Completed:',output,flush=True)


if __name__=='__main__':
    main()
