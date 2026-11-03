# 多阶段 FaceID LoRA 评分

入口：`evaluate_lora_checkpoints.py`；便捷启动：`run_evaluate_lora_checkpoints.sh`。

在固定基模型上依次测试 base_only、step-50、step-100 等 LoRA。基模型只加载一次，各阶段完整替换全部 LoRA 参数，不合并叠加；参数缺失、多余、rank/shape 不匹配或非有限值均报错。所有阶段共用样本、帧、sigma、噪声、文本和音频条件。不会更新模型权重。

**此处评分是训练式单步去噪估计的身份表现，不是从纯噪声开始的完整多步生成评估。** 用它对比 checkpoint 在固定条件下的身份监督表现；最终还需完整生成和口型、动作、画质评估。

## 直接运行当前目录下的所有阶段

在仓库根目录执行：

```bash
bash examples/wanvideo/model_training/faceid/run_evaluate_lora_checkpoints.sh --save-crops
```

默认配置：

- 基模型：`a800_step-9000_sa_rope.safetensors`。
- LoRA 目录：`models/train/Wan2.2-S2V-14B_faceid_lora_every1_faceidw0.1`。
- 自动查找 `step-数字.safetensors`，按数字排序，排除 ComfyUI 转换文件。
- 样本：metadata 零基行号 923、2145；帧：零基 2、40、80；完整模型输入仍为 81 帧。
- sigma：0.15、0.30、0.45，避开训练 gate 边界的 BF16 舍入歧义。实际 sigma 写入 CSV。
- 噪声种子：20261103 + 样本 ID，各 checkpoint 和各 sigma 复用。
- 输出：`diagnostics/faceid_checkpoints_时间戳/`；日志在同名 `.log` 中。
- 单张空闲 GPU，默认 cuda:0；原尺寸 BF16 14B 模型需要足够显存。

## 切换训练目录、基模型和测试样本

```bash
LORA_DIR=models/train/Wan2.2-S2V-14B_faceid_lora \
BASE_CHECKPOINT=/data-training/hjq/DiffSynth-Studio/a800_step-9000_sa_rope.safetensors \
DEVICE=cuda:0 \
OUTPUT=diagnostics/faceid_compare_custom \
bash examples/wanvideo/model_training/faceid/run_evaluate_lora_checkpoints.sh \
  --samples 923 2145 --frames 2 40 80 --sigmas 0.15 0.3 0.45 --save-crops
```

`TRAIN_CONFIG` 可覆盖配置文件路径，默认使用 LoRA 目录内的 `faceid_training_config.json`。`--cache-path` 可指定单独的验证集缓存；缓存必须完整，且与教师模型、metadata、RoPE 配置匹配。样本 ID 必须属于这个缓存对应的 metadata。

## 只选择指定阶段

```bash
DIFFSYNTH_MODEL_BASE_PATH=/data-training/models \
PYTHONPATH=. \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python examples/wanvideo/model_training/faceid/evaluate_lora_checkpoints.py \
  --config models/train/Wan2.2-S2V-14B_faceid_lora_every1_faceidw0.1/faceid_training_config.json \
  --base-checkpoint a800_step-9000_sa_rope.safetensors \
  --lora-checkpoints \
    models/train/Wan2.2-S2V-14B_faceid_lora_every1_faceidw0.1/step-50.safetensors \
    models/train/Wan2.2-S2V-14B_faceid_lora_every1_faceidw0.1/step-200.safetensors \
  --samples 923 2145 --frames 2 40 80 --sigmas 0.15 0.3 0.45 \
  --device cuda:0 --save-crops --output diagnostics/faceid_compare_50_200
```

支持多个路径和带引号的通配符。使用**训练保存的原始 LoRA**，不能传完整模型或 ComfyUI 格式。各 checkpoint 必须有相同的 adapter 结构和 rank；对不同 rank 的实验分别用对应配置运行并比较 CSV。已有输出目录会被拒绝，防止混合不同实验结果。

## 输出与解读

| 文件 | 内容 |
|---|---|
| `README.md` | 各阶段汇总表及解释 |
| `summary.csv` | 各阶段、各 sigma 和 all 的均值、p10、相对 base 的增益 |
| `scores.csv` | 每样本、每帧、每 sigma、每阶段的详细分数 |
| `scores.svg` | 对参考、对原始同帧的相似度趋势图，浏览器直接打开 |
| `reference_scores.csv` | 原始帧及真实 latent VAE 重建评分 |
| `provenance.json` | 路径、SHA256、种子、配置、过滤统计、运行是否完成 |
| `crops/` | 指定 `--save-crops` 时输出 112×112 对齐裁剪图 |

- `cosine_reference`：对参考身份特征相似度，越大越好。
- `identity_loss`：`1 - cosine_reference`，越小越好；不是训练中的权重缩放 loss。
- `cosine_raw_frame`：对原始同帧人脸特征的相似度，用于监测身份/表情保真变化；它不是严格的表情或口型指标。
- `delta_reference_vs_base`：同样本、同帧、同 sigma 的参考相似度，相对未加载待测 LoRA 的增益。
- `cosine_reference_p10`：分数较低部分的表现。两个样本的 p10 不具备统计稳定性。
- `eligible`：实际 sigma 是否落在原训练的 FaceID 有效区间；本脚本即便在区间外也会评分。
- `crop_outside_fraction`：未 clamp 的裁剪图中，超出 [-1,1] 的像素分量比例。

均值先在各样本内部对有效帧/sigma 平均，再对样本等权平均。无效参考、无有效目标帧的样本会记录并跳过；所有 checkpoint 使用完全相同的有效集合，不会按模型预测分数重新筛选。缓存过滤已决定可评估帧的范围，结果可能偏向容易检测的脸，需结合过滤统计解读。

模型使用 eval/no_grad。解码沿用训练的因果历史与固定参考首 latent。沿用原始帧缓存裁剪矩阵，不对预测脸重新检测，因此位置偏移也会反映在分数中。PNG 仅显示时 clamp，评分前不 clamp。

每个 checkpoint 完成后都会更新汇总文件；中断后结果可能只包含部分阶段，**须检查 `provenance.json` 的 `status == "complete"` 才作为完整对比**。不支持断点续评，复跑请使用新目录。

建议先用默认两个样本确认运行，再固定独立验证集比较所有阶段。不要只按参考相似度选模型：如果该分数上升而同帧相似度下降，先检查图像细节、表情和口型。基模型应与 LoRA 训练时一致；历史配置不能证明其来源时，需另行核实。
