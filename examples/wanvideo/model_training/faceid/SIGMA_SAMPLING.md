# FaceID 训练的 sigma 采样设置

已支持 `original`、`interval`、`mixed` 三种模式；默认 `original` 保留旧行为。仅影响 `train_faceid.py` 的训练采样，不改变普通 S2V 训练、缓存内容或多 checkpoint 评估脚本的固定 sigma。

## 参数

| 环境变量（run.sh） | Python 参数 | 默认值 | 含义 |
|---|---|---|---|
| `SIGMA_SAMPLING_MODE` | `--sigma_sampling_mode` | `original` | 原采样 / 区间采样 / 混合采样 |
| `SIGMA_SAMPLING_MIN` | `--sigma_sampling_min` | 跟随 `FACEID_SIGMA_MIN`，默认 0.1 | 采样区间下限，含边界 |
| `SIGMA_SAMPLING_MAX` | `--sigma_sampling_max` | 跟随 `FACEID_SIGMA_MAX`，默认 0.5 | 采样区间上限，含边界 |
| `SIGMA_SAMPLING_PROB` | `--sigma_sampling_prob` | 0.5 | mixed 模式选择区间分支的概率 |

`FACEID_SIGMA_MIN/MAX` 仍是 **FaceID loss 的启用区间**。新增 `SIGMA_SAMPLING_MIN/MAX` 决定 **训练 timestep 的抽样范围**，两者可以不同。若采样范围超出 gate，那部分训练仍计算 FM，但不计算 FaceID。

- `original`：在原 `min/max_timestep_boundary` 对应索引范围内均匀抽样，与旧版相同，并保持相同 RNG 消耗。采样上下限和混合概率不参与抽样。
- `interval`：在原索引范围与指定有效 sigma 区间的交集中均匀抽取**原始索引**。不是连续 sigma 上的均匀分布。
- `mixed`：概率 p 从 interval 分支抽样，概率 1-p 从 original 分支抽样。p=0 等同 original，p=1 等同 interval。

区间依据 timestep 转成模型 dtype（如 BF16）后，scheduler 最近邻回查得到的实际 sigma 筛选。因此不会出现“名义 0.5，实际 0.5006”超出所设采样上限的问题。重复映射到同一实际 sigma 的原始索引保留原分布的重数。无候选交集会在模型初始化时明确报错，不会回退到其他区间。

## 公共设置

以下命令在仓库根目录执行，使用现有缓存，无需重新提取：

```bash
export FACEID_WEIGHTS="$PWD/faceid_model/backbone.pth"
export FACEID_VAE_PATH=/data-training/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors
export CACHE_PATH=/data-training/multi_face/cache_faceid_f81
export BASE_CHECKPOINT="$PWD/a800_step-9000_sa_rope.safetensors"
export NUM_PROCESSES=8
export FACEID_EVERY=1
export FACEID_WEIGHT=0.1
export FACEID_WARMUP_STEPS=100
export FACEID_SIGMA_MIN=0.1
export FACEID_SIGMA_MAX=0.5
```

各实验保持相同基础权重、初始 LoRA、学习率、warmup、训练预算和验证集。默认不加载初始 LoRA；若已有 `LORA_CHECKPOINT` 环境变量，请确认它指向各实验统一的初始权重。以下输出路径应使用新目录。

### S0：保留原采样（默认）

```bash
SIGMA_SAMPLING_MODE=original \
OUTPUT_PATH=models/train/faceid_sigma_S0_original \
bash examples/wanvideo/model_training/faceid/run.sh train
```

完全保留原训练其他默认参数时，不设置本节公共的 `FACEID_EVERY/WEIGHT/WARMUP` 覆盖即可；新增模式本身默认就是 original。

### S1：50% 区间 + 50% 原采样（建议先做此对照）

```bash
SIGMA_SAMPLING_MODE=mixed SIGMA_SAMPLING_PROB=0.5 \
SIGMA_SAMPLING_MIN=0.1 SIGMA_SAMPLING_MAX=0.5 \
OUTPUT_PATH=models/train/faceid_sigma_S1_mix50 \
bash examples/wanvideo/model_training/faceid/run.sh train
```

### S2：75% 区间 + 25% 原采样

```bash
SIGMA_SAMPLING_MODE=mixed SIGMA_SAMPLING_PROB=0.75 \
SIGMA_SAMPLING_MIN=0.1 SIGMA_SAMPLING_MAX=0.5 \
OUTPUT_PATH=models/train/faceid_sigma_S2_mix75 \
bash examples/wanvideo/model_training/faceid/run.sh train
```

### S3：只训练 FaceID 有效 sigma 区间

```bash
SIGMA_SAMPLING_MODE=interval \
SIGMA_SAMPLING_MIN=0.1 SIGMA_SAMPLING_MAX=0.5 \
OUTPUT_PATH=models/train/faceid_sigma_S3_interval \
bash examples/wanvideo/model_training/faceid/run.sh train
```

该设置使实际 sigma 落在 gate 内，但不保证每个样本都产生 FaceID loss：仍取决于参考/目标帧有效性、FACEID_EVERY、warmup 权重等。

### 单独对比更窄区间

```bash
SIGMA_SAMPLING_MODE=mixed SIGMA_SAMPLING_PROB=0.5 \
SIGMA_SAMPLING_MIN=0.15 SIGMA_SAMPLING_MAX=0.35 \
OUTPUT_PATH=models/train/faceid_sigma_mix50_narrow \
bash examples/wanvideo/model_training/faceid/run.sh train
```

此例保持 gate=[0.1,0.5]；原采样分支落在 [0.1,0.15) 或 (0.35,0.5] 时依然能产生身份监督。

直接调用 `train_faceid.py` 时，在现有训练参数后增加：

```text
--sigma_sampling_mode mixed --sigma_sampling_prob 0.75
--sigma_sampling_min 0.1 --sigma_sampling_max 0.5
```

## TensorBoard 和实验解释

新增指标：

- `sampling/interval_branch`：各 rank 中选择区间分支的比例，长时间均值应接近 p。
- `sampling/in_interval`：实际 sigma 落入采样目标区间的比例。
- `sampling/interval_probability`：所设分支概率，original=0、interval=1。

结合已有 `faceid/eligible`（gate 命中率）、`faceid/active`（身份监督实际激活率）、`faceid/sigma`、`loss/fm`、`loss/faceid_when_active` 观察。多卡日志是 rank 聚合后的标量，sigma 均值不能代表完整直方图。

混合采样的实际区间命中率是 `p + (1-p) × 原采样命中率`，不是 p；原采样分支也可能落在区间内。各 rank 独立抽样，单个 step 不要求严格配额为 50%/75%。

**本实现不做重要性加权，采样同时改变 FM 和 FaceID 的训练分布。** FM 原 scheduler 权重仍保留；区间内 FM 被相对强调。纯区间模式缺少区间外训练，不能靠损失加权恢复全区间覆盖。建议依次 S0→S1→S2，S3 作为对照，不直接根据更低的训练 loss 判断获益。

改采样不会重建缓存。运行配置写入原有 `faceid_training_config.json`。为公平比较，使用固定 sigma/样本的 `evaluate_lora_checkpoints.py`，并补充完整生成质量检查。不得把采样改变导致的 loss 均值降低直接视为模型提升。
