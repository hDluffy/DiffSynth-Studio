"""Render completed diagnose_identity.py outputs into Markdown and a local HTML gallery."""
import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path


def summarize(output):
    output = Path(output)
    provenance = json.loads((output / 'provenance.json').read_text())
    args = provenance['arguments']
    rows = list(csv.DictReader((output / 'scores.csv').open()))
    groups = defaultdict(list)
    for r in rows:
        groups[(r['sample'], r['branch'], r['actual_sigma'])].append(r)
    lines = ['# FaceID 三路身份诊断', '',
        '三路：原始帧、缓存真实 latent 的 VAE 重建、模型预测 clean latent 的 VAE 重建。', '',
        f"- 基模型：`{args['base_checkpoint']}`",
        f"- LoRA：`{args['lora_checkpoint']}`",
        f"- 样本（CSV 零基行号）：{args['samples']}；帧（零基）：{args['frames']}。",
        f"- 固定噪声种子：{args['seed']} + 样本 ID，同一样本各 sigma 复用同一份噪声。",
        '- 模型只做前向；不更新参数。预测路是训练式单次去噪估计，不是完整生成视频。',
        '- 缓存、ArcFace、VAE、检测器及 metadata 的指纹已校验；权重 SHA256 见 provenance.json。',
        '- 三路均使用缓存真实帧的同一裁剪矩阵。参考图重新检测，并验证特征与缓存 cosine > 0.999。',
        '- 保留固定参考首 latent 作为解码历史，不对第 0 帧评分。解码保持原始空间分辨率。',
        '- 指标使用未 clamp 的 RGB；PNG 仅为显示而 clamp 到 [-1,1]。',
        '- 下表是所选帧的算术平均，不是训练的质量加权、多 rank 聚合 loss。', '',
        '| 样本 | 分支 | 实际 sigma | 对参考 cosine ↑ | 1-cosine ↓ | 对原始同帧 cosine ↑ | 裁剪越界像素比例 |',
        '|---|---|---|---|---|---|---|']
    for (sid,branch,sigma), data in groups.items():
        means = [sum(float(r[k]) for r in data)/len(data) for k in ('cosine_reference','identity_loss','cosine_raw_frame','crop_outside_fraction')]
        lines.append(f'| {sid} | {branch} | {float(sigma):.6f}' + ' | ' + ' | '.join(f'{m:.6f}' for m in means) + ' |' if sigma else f'| {sid} | {branch} | — | ' + ' | '.join(f'{m:.6f}' for m in means) + ' |')
    lines += ['', '## 查看结果', '', '- `scores.csv`：逐样本、逐帧、逐 sigma 指标。',
        '- `index.html`：裁剪对照画廊，浏览器直接打开。',
        '- `sample_*/f*_comparison.png`：参考 / 原始 / VAE / 三个 sigma 预测裁剪。',
        '- `sample_*/f*_overlay.png`：全帧上标出缓存裁剪区域；对应不带 overlay 的 PNG 为全帧。',
        '- `provenance.json`：参数、指纹及样本来源。', '',
        '## 解读边界', '',
        '原始帧对参考的分数会受姿态、遮挡和表情影响，不是固定 1，也不是预测的严格上限。'
        '真实 VAE 重建用于隔离压缩重建误差；预测与原始同帧的分数用于辅助区分身份损失与姿态差异。'
        '两条视频是诊断样本，不代表验证集表现，也不能单独证明 LoRA 提升；证明提升需同底模、同噪声的无 LoRA 对照。', '',
        '实际 sigma 按训练 scheduler 和 BF16 timestep 回查，可能与请求值有偏差。']
    (output / 'README.md').write_text('\n'.join(lines)+'\n')
    page = ['<!doctype html><meta charset="utf-8"><title>FaceID 三路裁剪对照</title>',
        '<style>body{font:16px sans-serif;background:#eee;margin:24px}img{max-width:100%}article{background:white;padding:16px;margin:16px 0}a{margin-right:12px}</style>',
        '<h1>FaceID 三路身份对照</h1><p>参考、原始帧、真实 latent VAE 重建、三个 sigma 的预测重建。数值为未 clamp 图像的 ArcFace cosine。</p>',
        '<p><a href="README.md">实验说明</a><a href="scores.csv">逐帧评分</a><a href="provenance.json">运行参数</a></p>']
    for path in sorted(output.glob('sample_*/*comparison.png')):
        rel = path.relative_to(output).as_posix()
        prefix = path.name.split('_')[0]
        page.append(f'<article><h2>{html.escape(path.parent.name)} / {prefix}</h2><img src="{rel}"><p>全帧裁剪区域：</p>')
        for overlay in sorted(path.parent.glob(prefix+'_*_overlay.png')):
            page.append(f'<a href="{overlay.relative_to(output).as_posix()}">{html.escape(overlay.stem)}</a>')
        page.append('</article>')
    (output / 'index.html').write_text('\n'.join(page))


if __name__ == '__main__':
    summarize(sys.argv[1])
