# Wan2.2 S2V：81 帧单身份 FaceID LoRA

本入口在原有 Flow Matching 损失上增加可微 ArcFace 身份损失。每条视频只能有一个身份；不同视频可以是不同人物。默认只训练 LoRA，推理不需要 ArcFace、检测器或额外节点。原 train.py 与非 FaceID 脚本仍可使用。

## 1. 准备模型和依赖

在仓库根目录执行（使用原训练环境）：

```bash
pip install -r examples/wanvideo/model_training/faceid/requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

需要显式提供以下本地文件，不会自动下载或猜测替代权重：

- `FACEID_WEIGHTS`：InsightFace **arcface_torch iResNet100** 的 PyTorch `backbone.pth`（512 维、RGB、112×112），或同结构的 state_dict。支持 `state_dict` 包装和 `module.` 前缀，严格加载全部参数。不是 ONNX、不是其他 ArcFace backbone、不是包含分类头的训练包。模型参考：<https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch>。
- `FACEID_DETECTOR_PATH`：SCRFD ONNX 人脸检测器，如 buffalo_l 配套 `det_10g.onnx`。只用于预处理，默认 ONNX Runtime CPU provider。识别用的 `w600k_r50.onnx` 不能替代 PyTorch iResNet100。
- `FACEID_VAE_PATH`：Wan2.1 16 通道 VAE，本仓库原始 `.pth` 或转换后的 `.safetensors`。预处理与训练必须用同一份文件。不是 Wan2.2 48 通道 VAE。
- 原有 S2V DiT、T5 和 wav2vec 权重仍按 `MODEL_BASE_PATH` 查找。

本机已有 VAE 的候选路径：
`/data-training/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors`。

```bash
export FACEID_WEIGHTS=/absolute/path/to/iresnet100/backbone.pth
export FACEID_DETECTOR_PATH=/absolute/path/to/det_10g.onnx
export FACEID_VAE_PATH=/data-training/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors
export MODEL_BASE_PATH=/data-training/models
export DATASET_BASE_PATH=/data-training/new_train_data
export DATASET_METADATA_PATH=/absolute/path/to/metadata_faceid.csv
export CACHE_PATH=/data-training/new_train_data/cache_faceid_f81_p327680_v1
```

CSV 延续现有 `video,prompt,input_audio` 字段。参考图默认取处理后的视频首帧；若 CSV 有 `input_image` 路径，会作为实际 S2V 参考图及身份参考。人脸需足够清晰，默认最短边至少 48 像素。不要用 ComfyUI 转换后的 LoRA 作为训练初始化文件。

先运行真实权重的小型可微链路检查（不运行 DiT、不保存模型）：

```bash
python examples/wanvideo/model_training/faceid/verify_teachers.py \
  --faceid_weights "$FACEID_WEIGHTS" --faceid_vae_path "$FACEID_VAE_PATH"
```

检查权重架构匹配、输出 512 维、VAE → 对齐 → ArcFace → latent 梯度非零且有限，以及教师无参数梯度。随机 latent 不是真实人脸，该检查不能证明识别质量。

## 2. 生成新的 FaceID 特征缓存

```bash
NUM_PROCESSES=8 bash examples/wanvideo/model_training/faceid/run.sh cache
python examples/wanvideo/model_training/faceid/audit_cache.py "$CACHE_PATH"
```

默认 81 帧、16 fps、最大 327680 像素、最多重复末帧 1 次。样本需要至少 80 个有效采样点，短视频会报错而不无限补帧。更长视频取前 5 秒并对齐音频。

可使用 `MAX_PIXELS=589824`，但训练还要负担可微解码的激活，建议先以默认像素预算验证显存。修改配置或模型后用新缓存目录。仅配置完全相同的中断续跑才使用：

```bash
RESUME_FEATURE_CACHE=1 bash examples/wanvideo/model_training/faceid/run.sh cache
```

每个视频：参考图提取身份向量；目标视频每个 4 帧解码块检查第 2、4 帧，共 40 个候选；检测结果不是恰好一张脸、脸太小或与参考身份余弦相似度低于阈值时屏蔽该帧。参考图无效则该样本只贡献 FM。不会悄悄改用视频平均身份。质量权重使用检测置信度；尚未实现独立模糊度/姿态估计器。

默认检测阈值 0.6，身份筛选阈值 0.2。可通过 `FACEID_DET_THRESHOLD`、`FACEID_MIN_FACE_SIZE`、`FACEID_MIN_SIMILARITY` 调整；阈值不是跨模型通用的质量保证，应检查 audit 的有效率。严格单脸策略可能屏蔽有镜子、海报或误检的帧。

manifest 记录 ArcFace/VAE/检测器 SHA256、对齐版本和阈值。普通旧缓存没有这些信息，不能用于此入口。audit 检查样本数量、参考 embedding、帧数及可监督样本数；若有效率很低，先排查检测和参考图，不要盲目长训。

## 3. ZeRO-3 LoRA 训练

```bash
export OUTPUT_PATH=./models/train/Wan2.2-S2V-14B_faceid_lora
bash examples/wanvideo/model_training/faceid/run.sh train
```

使用既有 `accelerate_config_zero3.yaml`：8 卡 ZeRO-3、优化器 CPU offload、允许聚合权重保存。只加载 DiT 到训练 pipeline；独立的冻结 VAE **decoder-only** 和 FP32 ArcFace 每个 rank 一份，初始化在 accelerator.prepare 之后，不进入 ZeRO-3/优化器/LoRA 导出。辅助网络与 DiT 的参数卸载机制不要混用。

如需微调后的底座或继续训练已有 LoRA：

```bash
BASE_CHECKPOINT=/absolute/path/base.safetensors \
LORA_CHECKPOINT=/absolute/path/original_lora.safetensors \
bash examples/wanvideo/model_training/faceid/run.sh train
```

这两个参数只恢复模型权重，不恢复 optimizer、训练步数或 FaceID warmup。继续训练时自行设置 `FACEID_WARMUP_STEPS`；新输出目录可避免覆盖旧结果。不要加入 `--trainable_models dit`，它会变成全量训练，本入口明确拒绝。

默认参数：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `LEARNING_RATE` | `1e-5` | LoRA 学习率 |
| `LORA_RANK` | `32` | LoRA rank |
| `NUM_EPOCHS` | `10` | 训练轮数 |
| `SAVE_STEPS` | `50` | 保存间隔 |
| `FACEID_WEIGHT` | `0.05` | 身份损失最终权重 |
| `FACEID_WARMUP_STEPS` | `100` | 权重线性 warmup，0 为立即启用 |
| `FACEID_EVERY` | `2` | 每几个 optimizer step 尝试身份监督 |
| `FACEID_SIGMA_MIN/MAX` | `0.1 / 0.5` | 实际 scheduler sigma 范围 |
| `FACEID_FRAMES` | `2` | 同一解码块最多监督帧数，可设 1 |

FM 的时间步分布不变，所有步骤都计算 FM；只有满足 sigma、调度和人脸有效条件时才计算 FaceID。因此 `FACEID_EVERY=2` 不意味着每两个步骤一定获得身份监督。建议小型试跑临时设置 `FACEID_EVERY=1 FACEID_WARMUP_STEPS=0 SAVE_STEPS=2 NUM_EPOCHS=1`，确认出现 active=1、保存成功，再恢复默认设置。`FACEID_WEIGHT=0` 可做相同入口的 FM 对照，但仍加载教师权重用于保持配置一致。

损失为 `L = L_FM + weight(step) * L_ID`；`L_ID` 是对齐人脸与参考 embedding 的加权 cosine distance。`z0_hat = zt - sigma*v_pred`，sigma 来自实际 scheduler 查表，不用 timestep/1000。首 latent 是固定参考图，只作因果历史，不参与身份监督。

### 解码与内存语义

81 帧对应 21 个 latent。chunk 0 对应首帧，chunk j>0 对应帧 `4j-3 ... 4j`。每次随机选一个有有效脸的 chunk。此前所有 chunk 在 no_grad 下顺序解码，保留真实因果历史；选中的 chunk 可微解码并保存激活到 CPU。

这是**截断历史梯度**：前向与完整因果解码对应帧匹配，但身份梯度只传到选中的 latent chunk，不传到历史 chunk。缓存是每次调用局部变量，不共享，不在有状态 decoder 外套 checkpoint。没有裁剪/缩放 latent 来冒充全图解码。选后半段需要顺序解码前缀，所以额外时间开销可能明显；当前版本没有承诺固定加速比。

FaceID 分支不执行分布式 collective，各 rank 有效脸/噪声不同不会导致教师参数通信顺序不一致。DiT 的 FM 路径每步保持执行；日志归约所有 rank 同步参与。

## 4. TensorBoard

训练默认启用，事件目录为 `$OUTPUT_PATH/tensorboard_log`：

```bash
tensorboard --logdir ./models/train/Wan2.2-S2V-14B_faceid_lora/tensorboard_log \
  --host 0.0.0.0 --port 6006
```

浏览器打开训练服务器的 6006 端口（或按现有 SSH 端口转发方式访问）。

| 曲线 | 解读 |
|---|---|
| `loss/total` | 跨 rank 平均的总损失 |
| `loss/fm` | 原有速度预测损失 |
| `loss/faceid` | 未激活 rank 记零后的平均身份损失 |
| `loss/faceid_weighted` | 实际加入总损失的身份项 |
| `loss/faceid_when_active` | 仅激活 rank 的身份损失，适合观察趋势 |
| `faceid/cosine_when_active` | 仅激活 rank 的身份相似度，通常越高越好 |
| `faceid/active` | 本步骤有身份监督的 rank 比例 |
| `faceid/eligible` | sigma 满足范围的 rank 比例 |
| `faceid/reference_valid` | 参考身份有效比例 |
| `faceid/valid_frame_fraction` | 81 帧中去掉首帧后有效候选占比；默认最多 0.5 |
| `faceid/weight` | 当前 warmup 后权重 |
| `faceid/frames_used` | 平均实际监督帧数 |
| `faceid/sigma` | 平均采样 sigma |
| `faceid/forward_host_seconds` | 身份分支主机墙钟耗时（异步 CUDA 下不是精确 GPU 计时） |
| `memory/cuda_allocated_gib` | forward 后显存分配量 |
| `memory/cuda_peak_allocated_gib` | 进程累计峰值（包含此前 backward，不是单步 profiler） |

全局无 active 的 step 不写 `*_when_active`，不要把普通 `loss/faceid=0` 理解为身份完美。原 `loss` 标签仍保留 rank0 的兼容日志，正式比较使用归约后的 `loss/total`。脚本固定梯度累积为 1，此时事件步数与 optimizer step 一致。

## 5. 测试与验收

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -p test_faceid_training.py

# 可选：两张空闲 GPU 的小模型 ZeRO-3 联调，不需要人脸模型文件
OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=2 tests/faceid_zero3_smoke.py
```

检查可微对齐、冻结教师输入梯度、因果窗口与完整解码一致性、历史梯度截断、FaceID 对可训练参数的独立贡献、warmup/sigma/无效脸屏蔽及真实 TensorBoard event 内容。

还需真实数据验收：少量样本先跑数步，确认身份分支确实激活、梯度有限、ZeRO-3 保存可恢复。然后对比 FM-only / FM+FaceID，在固定参考图、音频、seed 下评估身份、跨帧稳定性、嘴形和表情，不能仅凭训练 cosine 上升认定生成质量提高。普通 LoRA 导出不含教师；可沿用现有 ComfyUI 格式转换流程。
