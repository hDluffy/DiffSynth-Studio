# MiniMax-H3-Ref2VA 训练显存分析与降显存方案

本文针对
`bash examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA-batch-run.sh data/tmp001`
的两个阶段，给出显存峰值来源、可复核的估算方法，以及为“最多 7 张参考图 + 音频参考”预留的方案。

## 1. 先给结论

当前脚本有两个不同的 OOM 风险，处理顺序应为：

1. **先把阶段 1（cache）改成 CPU offload，或单独在多卡上运行。** 它加载文本编码器、视频 VAE 和音频 VAE，权重的 BF16 文件大小约为 66.7 + 10.4 + 0.6 = **77.7 GB**，单张 80 GB 卡几乎没有运行时余量。
2. **阶段 2 要明确使用 gradient checkpointing。** 当前脚本仅有 `--use_gradient_checkpointing_offload`；在当前实现中该选项会直接走 `torch.utils.checkpoint` 的 `save_on_cpu()` 分支，所以已经会做 checkpoint，但没有显式打开普通 checkpoint。为避免后续实现变化和配置误读，建议两个选项同时传入。
3. **先固定到 480x832、124 帧验证链路，再逐步提高到 209 帧/更高分辨率。** `MAX_PIXELS=1048576` 对目标视频是上限，不是固定分辨率；源视频未超过上限时仍会保留接近原始分辨率。
4. **参考图数量主要增加序列长度和文本编码器输入。** 7 张图本身的 latent 不是最大头，但 `ref_image_short_edge=2048` 会让 Qwen 图像 token 数快速增加。应把该值做成参数，并先用 768 或 1024。
5. **音频参考的 latent 很小，但当前 loader 会把每条参考音频读到目标视频全长。** 7 条音频在长视频上会线性增加 DiT 序列长度；应支持每条音频的最大秒数（建议 3--5 秒）并预缓存。

## 2. 代码路径和显存峰值

### 阶段 1：`sft:data_process`

batch 脚本加载：

```text
Ref2VA/text_encoder/model*.safetensors
Ref2VA/video_vae/source/model.safetensors
Ref2VA/audio_vae/model.safetensors
```

随后每个样本依次执行：

1. 视频解码、resize、`float32` 视频张量；
2. 视频 VAE 编码为 `[1, 24, T_latent, H/16, W/16]`；
3. 音频 VAE 编码为 `[2, 32, T_audio]`；
4. 每张参考图经过视频 VAE 编码；
5. 参考图/音频 token 进入文本编码器；
6. 构建 packed sequence，保存到 `.pth`。

阶段 1 在 `torch.no_grad()` 下运行，不需要反向激活，但模型权重和 VAE 编码的瞬时激活仍然占用显存。`--use_gradient_checkpointing` 对这一阶段没有决定性作用，**`--enable_model_cpu_offload` 才是首要开关**。

### 阶段 2：`sft:train`

阶段 2 从 `.pth` 读取 `prompt_embeds`、latent 和 packed index，只加载 DiT；文本编码器和 VAE 不再参与。因此阶段 2 的峰值主要是：

* ZeRO-3 分片后的 DiT 权重、LoRA 参数和 Adam 状态；
* `x=[1,S,96]`、`audio_x=[1,S,32]`；
* 每层 `hidden=[S,5376]`、QKV 和 MLP 临时张量；
* backward 时的 LoRA 梯度和未被 checkpoint 释放的激活。

这里的 `S` 是 packed sequence 长度，按 64 对齐：

```text
S = align64(
      text_len
    + target_video_rows
    + target_audio_rows
    + reference_visual_rows
    + reference_audio_rows
  )

target_video_rows = T_latent * (H_latent/2) * (W_latent/2)
target_audio_rows = 2 * round(num_frames / 24 * 40)
```

DiT 的 hidden size 是 5376、attention head 是 56x128、MLP hidden 是 14336。只看 BF16 临时张量，每层大致为：

```text
hidden:  S * 5376 * 2 bytes
QKV:     S * (56*3*128) * 2 bytes
MLP fc1: S * (2*14336) * 2 bytes
```

因此没有 gradient checkpointing 时，50 层的激活会远超单卡容量；即使有 checkpointing，当前层的 QKV/MLP 临时峰值仍随 `S` 线性增长。必须使用 memory-efficient attention（FlashAttention、SageAttention 或 PyTorch SDPA 的 flash/mem-efficient kernel），不要退化到普通显式 attention 矩阵。

## 3. 对 `data/tmp001` 的具体估算

样本 `3780908535548692415.mp4` 的视频是 1280x720、约 8.87 秒、30 fps。脚本传入 `NUM_FRAMES=362`，但 loader 按 24 fps 和 `17n+5` 规则会取到 **209 帧**（`floor(8.87*24)=212`，向下取到 209），不是 362 帧。

在动态分辨率 `MAX_PIXELS=1048576` 下，该视频原始面积约 0.92 MP，通常不会缩小；按 32 对齐后约为 **704x1280**。于是：

补充：脚本顶部 help 文案仍写着 `DATASET_REPEAT=5`、`NUM_EPOCHS=5`、`LORA_RANK=32`，但当前实际默认值是 10、50、64（以脚本变量为准）。这不会改变单步视频激活峰值，但会显著改变总训练时长/总更新次数，建议同步修正文案或显式传参。

| 项目 | 当前样本（209 帧） | 真正有 362 帧的样本 |
|---|---:|---:|
| video latent | `[1,24,62,44,80]` | `[1,24,107,44,80]` |
| target video rows | 54,560 | 94,160 |
| target audio rows | 696 | 1,206 |
| 两张参考图 visual rows（short edge 2048） | 约 5,376 | 约 5,376 |
| `prompt_embeds`（两张图时的典型量级） | 约 7,000--7,500 tokens x 5120 | 同左；取决于图像尺寸和 prompt |
| packed `S`（典型） | 约 68,160 | 约 108,288 |
| `hidden` BF16 单份 | 约 0.73 GB | 约 1.16 GB |
| QKV BF16 临时量级 | 约 2.93 GB | 约 4.65 GB |
| MLP fc1 BF16 临时量级 | 约 3.91 GB | 约 6.20 GB |

表中的 QKV/MLP 是单层临时张量的数量级，不是最终峰值；还要加上 layer norm、投影、checkpoint、LoRA backward、CUDA allocator 碎片和 ZeRO 通信 buffer。它解释了为什么长视频即使使用 8 卡 ZeRO-3，单卡仍可能在 24/32 GB 上 OOM。

模型权重（由本地 Ref2VA safetensors 的参数量和 BF16 字节数核对）：

| 模块 | 参数量 | BF16 权重 |
|---|---:|---:|
| text encoder | 33.36B | 66.7 GB |
| transformer/DiT | 33.12B | 66.3 GB |
| video VAE | 2.60B | 10.4 GB |
| audio VAE | 0.15B | 0.61 GB |

阶段 2 使用 8 个 ZeRO-3 rank 时，DiT 权重的静态分片约为 66.3/8 = **8.3 GB/rank**；运行时还会发生 layer all-gather，所以不能把 8.3 GB 当成完整显存预算。若 `accelerate_config_zero3.yaml` 的 8 个进程与实际 GPU 数量不一致，先修正 `num_processes`。

## 4. 立即可用的降显存方案

### P0：先让当前任务跑通

#### 保持 batch 脚本训练参数不变的方案

以下方案不改变 `NUM_FRAMES=362`、`MAX_PIXELS=1048576`、`DATASET_REPEAT=10`、`NUM_EPOCHS=50`、`LORA_RANK=64`、BF16、8 卡 ZeRO-3 或数据内容，只改变运行时显存管理：

1. 先用 CUDA allocator 可扩展段，减少大块连续显存申请失败：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_MODULE_LOADING=LAZY \
STAGE=train \
bash examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA-batch-run.sh data/tmp001
```

这些环境变量必须在进程启动前设置；仅在运行中的 Python 进程里设置无效。已有 stage-1 cache 时使用 `STAGE=train`，不会重复编码 cache。

2. 如果仍 OOM，使用 8 卡 ZeRO-3 参数/optimizer CPU offload，并缩小 ZeRO 通信 buffer。仓库已有可复用的低显存 DeepSpeed 配置示例：

```text
examples/qwen_image/model_training/special/low_vram_training/ds_z3_cpuoffload.json
```

关键设置是 `offload_param.device=cpu`、`offload_optimizer.device=cpu`、`overlap_comm=false`、`reduce_bucket_size=5e7`、`stage3_prefetch_bucket_size=5e7`。需要通过一个新的 Accelerate YAML 的 `deepspeed_config_file` 引用该 JSON，并保持 `num_processes: 8`，然后运行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_MODULE_LOADING=LAZY \
STAGE=train \
ZERO3_CONFIG=/tmp/minimax_h3_zero3_cpuoffload.yaml \
bash examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA-batch-run.sh data/tmp001
```

CPU offload 会降低吞吐，但不改变模型、数据、帧数、分辨率、repeat、epoch 或 LoRA rank。当前节点约有 1 TiB 主机内存，具备承接该 offload 的容量。

#### 本次 OOM 的证据

已有 stage-2 日志显示 OOM 发生在 DeepSpeed ZeRO-3 Linear wrapper 的矩阵乘阶段：

```text
Tried to allocate 6.49 GiB
GPU free: 5.73 GiB
PyTorch allocated: 55.89 GiB
reserved but unallocated: 16.59 GiB
```

这说明主要问题是大块 Linear/MLP 临时张量叠加 ZeRO 参数通信 buffer 和 CUDA allocator 碎片；不是 `DATASET_REPEAT` 或 epoch 导致的单步显存峰值。`expandable_segments:True` 应先于 CPU offload 尝试，因为它不改变训练行为。

阶段 1 建议至少加入：

```text
--enable_model_cpu_offload
--dataset_num_workers 0
```

注意：这些参数目前没有被 `MiniMax-H3-Ref2VA-batch-run.sh` 的环境变量包装；要立即生效，需要把参数补到脚本的 `accelerate launch ... train.py` 命令，或绕过 batch wrapper 直接运行 `train.py`。仅在 shell 前面设置同名环境变量不会自动传给 argparse。

阶段 2 建议同时加入（当前版本只传 offload 也会进入 checkpoint，但显式写全更稳妥）：

```text
--use_gradient_checkpointing
--use_gradient_checkpointing_offload   # 显存仍不够时再保留，代价是 CPU/GPU 拷贝
```

对于 24/32 GB 卡，先用：

```bash
STAGE=both NUM_FRAMES=124 HEIGHT=480 WIDTH=832 MAX_PIXELS=393216 \
  LORA_RANK=16 DATASET_REPEAT=1 \
  bash examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA-batch-run.sh data/tmp001
```

`HEIGHT/WIDTH` 同时设置后会关闭动态分辨率的不确定性；`124` 帧仍满足 `17n+5`，且不会超过当前 8.87 秒样本的可用帧数。确认稳定后再按顺序提高到 `209` 帧、`704x1280` 或更高配置。降低 `DATASET_REPEAT` 只改变训练时间，不改变单步峰值显存。

### P1：阶段 1 的权重处理

* 优先使用 `--enable_model_cpu_offload`，必要时再加 `--enable_optimizer_cpu_offload`（阶段 1 没有 optimizer，阶段 2 才有意义）。
* 让阶段 1 和阶段 2 分开启动，阶段 1 完成并释放文本编码器/VAE 后再启动阶段 2；不要在同一 Python 进程中复用两个阶段。
* `dataset_num_workers=0` 可避免多个 worker 预取大帧列表；它主要降低 CPU/RAM 和进程间复制，对 GPU 峰值也有间接帮助。
* 如果 CPU 内存足够，可把阶段 1 的 text encoder/VAE 做离线缓存；阶段 2 已经使用 `.pth` cache，不需要再次加载这些模块。

### P2：阶段 2 的模型和优化器

* 保持 ZeRO-3，并使 `num_processes` 与实际 GPU 数一致；单卡不能指望 ZeRO-3 把 66.3 GB DiT 变成可训练的小模型。
* 在冻结 base DiT、只训练 LoRA 的前提下，把默认脚本的 `LORA_RANK=64` 降到 16 或 32。它对峰值的收益小于降低分辨率/帧数，但可以减少 LoRA optimizer state。
* 显存仍不足时使用仓库已有的 NF4 训练路径（`examples/minimax_h3/model_training/special/quant_training/MiniMax-H3-Ref2VA-bitsandbytes_nf4.sh`）。量化时保留 `time_embedder`、patch projection、condition projection 和 final output 等模块为全精度，先做小数据集收敛对比。
* 不要把 `gradient_accumulation_steps` 当作单步显存优化；当前每个样本本来就是 micro-batch 1。它主要改变有效 batch size 和吞吐。
* 确认日志中的 attention implementation 是 FlashAttention/Sage/PyTorch flash SDPA。若使用显式 `SxS` attention，长序列一定会爆显存。

## 5. 面向 7 张参考图 + 音频参考的设计

### 5.1 参考图

当前 `train.py` 的 `get_pipeline_inputs()` 把 `ref_image_short_edge` 固定为 2048，batch 脚本也无法覆盖它；`MAX_PIXELS` 不会限制这里的图像短边。因此建议把以下参数加入 argparse、数据缓存 key 和 batch 脚本环境变量：

```text
REF_IMAGE_SHORT_EDGE=768     # 24/32 GB 卡起步
REF_IMAGE_SHORT_EDGE=1024    # 40/80 GB 卡的折中值
```

建议在 cache 阶段就把参考图缩放/编码并保存，阶段 2 只读 latent 和 `prompt_embeds`。7 张图的 visual rows 大致是 2 张图的 3.5 倍；文本 encoder 的 image token 也近似按参考图面积线性增加。若必须保留 2048，建议按“每个样本最多 2--3 张”分桶训练，7 张图采用预先聚合的 identity embedding，而不是把 7 张原图全部拼进同一序列。

### 5.2 音频参考

代码中音频 VAE 的 latent 时间长度约为：

```text
T_audio_ref ~= 40 * duration_seconds
audio rows  = 2 * T_audio_ref
```

当前 `MiniMaxH3ReferenceLoader` 对每条音频使用目标视频全长作为 `duration`。建议扩展 metadata，例如：

```json
{"type": "audio", "audio": "voice.wav", "max_seconds": 5}
```

并在 loader 中将音频裁剪到 `min(max_seconds, num_frames/24)`。这样 7 条参考音频最多增加约 `7 * 2 * 40 * 5 = 2800` rows，而不是在 15 秒视频上增加约 8400 rows。音频 latent 本身很小，主要节省的是 DiT 的序列激活和文本/packed 处理。

### 5.3 参考数量的预算公式

上线前应按实际 metadata 计算每条样本的 `S`，并按最大值而不是平均值配显存：

```text
S_max = align64(
    text_tokens(7 images + prompt)
  + video_frames/17 * 5 * (H/32) * (W/32) / 4
  + 2 * 40 * video_seconds
  + sum(reference_image_rows)
  + sum(reference_audio_rows)
)
```

建议先用一个 dry-run/缓存统计脚本输出每个样本的 `prompt_embeds.shape[0]`、`packed.seq_len`、`ref_visual_anchor.shape[0]` 和 `ref_audio_anchor.shape[0]`，按 `S` 降序取 P95/P99 样本做显存压测；不要只用平均视频长度测试。

## 6. 推荐落地顺序

1. 在不改 batch 脚本和训练参数的前提下，先设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；如果仍 OOM，再切换到 ZeRO-3 CPU offload 配置。
2. 阶段 1 若也需要 CPU offload，只能通过直接给 `train.py` 增加 `--enable_model_cpu_offload` 或使用外部启动包装实现；当前 batch 脚本没有对应环境变量，不能只设置一个同名 shell 变量。
3. 用当前 `362 / MAX_PIXELS=1048576 / rank 64` 配置记录 `torch.cuda.max_memory_allocated()` 和 `max_memory_reserved()`，确认 allocator 方案是否解决大块申请失败。
4. 需要 7 张图时优先降低参考图短边和音频参考秒数，再考虑 NF4；不要先把目标视频分辨率和帧数同时拉满。
5. 若仍 OOM，再检查 attention kernel、ZeRO rank、prefetch/reduce bucket 和 CPU offload 是否实际生效；不要把降低帧数/分辨率的测试配置当成当前生产配置。

## 7. 诊断记录模板

每次实验至少保存以下字段，便于判断是“权重峰值”还是“序列激活峰值”：

```text
GPU model / total VRAM
stage: cache | train
height, width, num_frames, actual_loaded_frames
num_references_image, num_references_audio
ref_image_short_edge, max_audio_seconds
prompt_embeds.shape
video_latents.shape, audio_latents.shape
packed.seq_len
attention implementation
max_memory_allocated / max_memory_reserved
OOM stack frame
```

## 8. `audio_latents` 缺失报错（2026-10-21）

该报错不是 OOM，也不是 DeepSpeed 的下列提示导致：

```text
Do not find activation_checkpointing config in deepspeed config,
skip initializing deepspeed gradient checkpointing.
```

当前启动参数已传入 `--use_gradient_checkpointing_offload`，项目自身会走
`torch.utils.checkpoint` 和 `save_on_cpu()` 路径；上述信息只表示没有使用
DeepSpeed 自带的 activation-checkpointing 配置。

真正的错误是：

```text
TypeError: model_fn_minimax_h3() missing 1 required positional argument: 'audio_latents'
```

对数据集 `chosen_20261020_182157_bd5b3621` 的实际检查结果：

* metadata 共 220 条，220 条 `input_audio` 都是非空 `.mp3` 路径；
* stage-1 日志对这些 MP3 报 `tuple index out of range`，并把音频返回为 `None`；
* 已生成的 224 个 `.pth` 中，包含 `audio_input_latents` 的数量为 0；
* cache 中的 `ref_audio_anchor` 存在，因为 reference audio 使用的是另一套纯音频 loader。

`input_audio` 当前由 `LoadAudioWithTorchaudio` 读取。该 loader 会先用视频
reader 取帧数和帧率，再按视频时长截取音轨。纯 MP3 没有视频帧信息，
因此在音频文件本身正常的情况下也会失败。由于 loss 只在 cache 含
`audio_input_latents` 时创建 `audio_latents`，stage 2 最终必然触发上述异常。

### 不改训练代码和训练参数的恢复方案

1. 先确认 `input_audio` 的数据语义。如果目标音频就是对应视频的内置音轨，
   将 metadata 中的 `input_audio` 指向对应 MP4；`references[].audio` 仍可保留 MP3。
2. 如果目标音频必须是该 MP3，则先将它与对应视频 mux 为带视频帧的
   MP4，再让 `input_audio` 指向该 MP4。这只是数据预处理，不改帧数、分辨率、
   epoch、LoRA rank 或 ZeRO-3 配置。示例：

   ```bash
   ffmpeg -i source.mp4 -i target.mp3 \
     -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest target_with_audio.mp4
   ```

3. 不要继续使用现有 cache。其中已有 `.cache-complete`，因此
   `SKIP_COMPLETED=1` 会跳过它。为避免删除旧数据，使用新的 `CACHE_ROOT`
   重建 stage 1：

   ```bash
   STAGE=cache \
   CACHE_ROOT="$PWD/models/train/MiniMax-H3-Ref2VA-batch-cache-audio-fixed" \
   SKIP_COMPLETED=0 \
   bash examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA-batch-run.sh DATA_ROOT
   ```

4. 在进入 stage 2 前抽检新 cache：

   ```bash
   python - <<'PY'
   import glob, torch
   files = glob.glob(
       "models/train/MiniMax-H3-Ref2VA-batch-cache-audio-fixed/**/*.pth",
       recursive=True,
   )
   assert files, "No cache file found"
   shared = torch.load(files[0], map_location="cpu", weights_only=False)[0]
   assert "audio_input_latents" in shared, sorted(shared)
   print(shared["audio_input_latents"].shape)
   PY
   ```

5. 检查通过后，训练时必须指向同一个新 `CACHE_ROOT`：

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   CUDA_MODULE_LOADING=LAZY \
   STAGE=train \
   CACHE_ROOT="$PWD/models/train/MiniMax-H3-Ref2VA-batch-cache-audio-fixed" \
   bash examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA-batch-run.sh DATA_ROOT
   ```

如果训练语义允许目标音轨为静音，`train.py` 已有 `--silent_on_missing_audio`
兜底选项；但当前 batch wrapper 没有暴露它，而且它不适用于需要真实目标
音频的数据，不应用来掩盖这次 cache 失效。
