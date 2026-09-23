# MiniMax-H3 Ref2VA 全量与 LoRA 训练指南

本文基于当前仓库代码进行静态审计，说明 MiniMax-H3 Ref2VA 的全量训练、LoRA 训练及分布式训练能力，并给出可直接改造的操作流程。

## 1. 结论

当前工程已经具备 MiniMax-H3 Ref2VA 的主要训练链路，但“全部支持”需要按下面的边界理解。

| 能力 | 结论 | 说明 |
|---|---|---|
| Ref2VA 原版 DiT 全参数训练 | 支持 | 官方脚本采用两阶段缓存，第二阶段训练 `dit` 全部可训练参数 |
| Ref2VA Pruned DiT 全参数训练 | 支持 | 可加载 Comfy-Org 的 pruned BF16 DiT |
| Ref2VA BF16 LoRA | 支持 | 两阶段训练，LoRA 注入 `attn` 和 `mlp` 线性层 |
| Ref2VA 量化 LoRA | 支持 | 仓库已有 NF4、FP8、Int8 ConvRot 及在线 bitsandbytes NF4 示例 |
| 单卡训练 | 支持但受显存限制 | LoRA 可用两阶段或量化方案；BF16 全参训练通常不现实 |
| 单机多卡 LoRA | 支持 | 可使用 Accelerate DDP；显存不足时也可使用 DeepSpeed ZeRO-3 |
| 单机多卡全参训练 | 支持 | 已提供 8 卡 DeepSpeed ZeRO-3 配置 |
| 多机多卡 LoRA/全参训练 | 代码路径支持 | Accelerate 和 DeepSpeed 均具备多机参数；仓库未提供 Ref2VA 多机配置及自动化/实机验证记录，需要在目标集群做小数据冒烟测试 |
| 文本编码器、DiT、Video VAE、Audio VAE 全栈联合全参训练 | 无开箱即用流程 | 仓库中的“full”特指 DiT 全参数训练；现有两阶段流程会冻结并缓存其他组件的输出 |
| 完整训练状态续跑 | 不支持 | 可加载 LoRA 或模型权重，但不恢复 optimizer、scheduler、epoch 和 dataloader 状态 |

因此，如果“全量训练”指社区通常所说的 **Ref2VA DiT 全参数微调**，当前工程已支持；如果指 Ref2VA 全部组件联合训练，则当前工程没有完备的官方流程。

多机多卡属于框架能力已经接通、但缺少仓库级端到端验证的状态，不应在未做集群冒烟测试前视为生产级承诺。

## 2. 判断依据

关键实现如下：

- 训练入口：[`examples/minimax_h3/model_training/train.py`](../../examples/minimax_h3/model_training/train.py)
  - 创建 `accelerate.Accelerator`，数据加载器会由 Accelerate 按进程切分。
  - 支持 `sft:data_process` 和 `sft:train` 两阶段训练。
  - Ref2VA 的 `references` 由专用加载器解析并送入 Pipeline。
  - 同时计算视频 flow-matching loss 和音频 flow-matching loss。
- Ref2VA 数据加载：[`diffsynth/utils/data/minimax_h3.py`](../../diffsynth/utils/data/minimax_h3.py)
  - 支持 `image`、`video`、`audio`、`video_audio` 四种参考块。
- Ref2VA 条件编码与序列打包：[`diffsynth/pipelines/minimax_h3_audio_video.py`](../../diffsynth/pipelines/minimax_h3_audio_video.py)
  - 参考图像/视频经 Video VAE 编码，参考音频经 Audio VAE 编码。
  - 参考 token、目标音频 token、目标视频 token 被共同打包给 DiT。
- MiniMax-H3 音视频损失：[`diffsynth/diffusion/loss.py`](../../diffsynth/diffusion/loss.py)
  - 视频和音频分别加噪、预测并计算损失。
  - 支持 `training_cfg_scale` 和 `audio_loss_weight`。
- LoRA 注入与恢复：[`diffsynth/diffusion/training_module.py`](../../diffsynth/diffusion/training_module.py)
  - 通过 PEFT 注入 LoRA，并支持 `--lora_checkpoint` 续接 LoRA 权重。
- 分布式训练与缓存：[`diffsynth/diffusion/runner.py`](../../diffsynth/diffusion/runner.py)
  - `accelerator.prepare` 负责 DDP/DeepSpeed 包装及 dataloader 分片。
  - 数据预处理缓存按全局 `process_index` 分目录保存，多进程文件名不会冲突。
- 分布式 checkpoint：[`diffsynth/diffusion/logger.py`](../../diffsynth/diffusion/logger.py)
  - 保存前执行全进程 barrier，仅全局主进程写最终 safetensors。
- 单机 8 卡 ZeRO-3 配置：[`examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml`](../../examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml)
- 官方训练与验证示例：
  - [`full/MiniMax-H3-Ref2VA.sh`](../../examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA.sh)
  - [`lora/MiniMax-H3-Ref2VA.sh`](../../examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA.sh)
  - [`validate_full/MiniMax-H3-Ref2VA.py`](../../examples/minimax_h3/model_training/validate_full/MiniMax-H3-Ref2VA.py)
  - [`validate_lora/MiniMax-H3-Ref2VA.py`](../../examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Ref2VA.py)

## 3. 环境准备

建议使用 Linux、NVIDIA GPU、支持 BF16 的 CUDA 环境和 Python 3.10 以上版本。

```bash
cd /path/to/DiffSynth-Studio
pip install -e ".[audio,training]"
```

主要依赖包括：

- PyTorch、torchvision；
- Accelerate；
- DeepSpeed，全参训练和 ZeRO 分片需要；
- PEFT，LoRA 注入需要；
- torchaudio、torchcodec、librosa、PyAV/FFmpeg，音视频读取需要；
- ModelScope，示例数据和模型下载需要。

检查环境：

```bash
python -c "import torch, accelerate, deepspeed, peft, torchaudio; print(torch.__version__, accelerate.__version__, deepspeed.__version__, peft.__version__)"
accelerate env
nvidia-smi
```

在多机环境中，所有节点应使用完全一致的代码版本、Python 环境、CUDA/NCCL/DeepSpeed 版本和模型文件。

## 4. 数据集格式

### 4.1 目录结构

```text
data/ref2va_train/
├── metadata.json
├── target_0001.mp4
├── ref_0001.png
├── ref_0002.mp4
└── ref_0003.wav
```

`metadata.json` 是一个 JSON 数组。最小的图像参考样本如下：

```json
[
  {
    "video": "target_0001.mp4",
    "input_audio": "target_0001.mp4",
    "prompt": "对目标音视频内容的描述",
    "references": [
      {"type": "image", "image": "ref_0001.png"}
    ]
  }
]
```

`references` 也可以混合以下类型：

```json
{
  "video": "target_0001.mp4",
  "input_audio": "target_0001.mp4",
  "prompt": "根据参考内容生成目标音视频",
  "references": [
    {"type": "image", "image": "ref_0001.png"},
    {"type": "video", "video": "ref_0002.mp4"},
    {"type": "audio", "audio": "ref_0003.wav"},
    {"type": "video_audio", "video": "ref_0002.mp4", "audio": "ref_0002.mp4"}
  ]
}
```

四种参考类型的语义：

| `type` | 必填字段 | 说明 |
|---|---|---|
| `image` | `image` | 单张参考图像 |
| `video` | `video` | 静音参考视频；如果需要参考音轨，应改用 `video_audio` |
| `audio` | `audio` | 纯音频参考 |
| `video_audio` | `video`、`audio` | 同时提供参考视频和参考音频；两个字段可以指向同一 MP4 |

训练命令中必须同时保留：

```bash
--data_file_keys "video,input_audio,references" \
--extra_inputs "input_audio,references"
```

前者负责读取文件，后者负责把音频和参考块送入 Pipeline。

### 4.2 数据约束

- 目标视频固定按 24 FPS 采样。
- `--num_frames` 必须满足 `17n+5`，例如 39、56、124。
- 固定分辨率训练时，`height` 和 `width` 应为 32 的倍数；官方示例使用 480×832。
- 每个样本的训练 micro-batch 固定为 1；有效全局 batch size 为：

  ```text
  world_size × gradient_accumulation_steps
  ```

- `input_audio` 应存在。若视频没有可读取的音轨，可以增加 `--silent_on_missing_audio`，加载器返回空音轨时会生成静音兜底。
- 参考图像会在 Pipeline 内按参考短边缩放；参考视频会裁剪、按 24 FPS 采样，并对齐到 H3 VAE 的时序分组。
- 两阶段训练中，改变 prompt、参考内容、分辨率、帧数、音频、`training_cfg_scale` 等预处理相关参数后，必须重建缓存。

可以先用官方样例验证环境：

```bash
modelscope download \
  --dataset DiffSynth-Studio/diffsynth_example_dataset \
  --include "minimax_h3/MiniMax-H3-Ref2VA/*" \
  --local_dir ./data/diffsynth_example_dataset
```

## 5. 为什么采用两阶段训练

标准 BF16 流程将训练拆成两个阶段：

1. `sft:data_process`：加载文本编码器、processor、Video VAE 和 Audio VAE，编码 prompt、目标视频/音频及所有参考块，把 DiT 所需输入保存为 `.pth` 缓存。
2. `sft:train`：只加载 Ref2VA DiT，从缓存读取输入并执行全参或 LoRA 训练。

这样可以避免超大的 Qwen3-VL 文本编码器与 DiT 长时间同时驻留显存，也能让多个 epoch 复用编码结果。

缓存目录结构类似：

```text
models/train/MiniMax-H3-Ref2VA-cache/
├── 0/0.pth
├── 0/1.pth
├── 1/0.pth
└── ...
```

数字目录是 Accelerate 的全局进程编号。第二阶段会递归发现所有 `.pth` 文件。

## 6. Ref2VA LoRA 训练

### 6.1 阶段一：生成缓存

以下示例先使用单进程生成缓存。把数据路径和输出路径替换为实际路径。

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/ref2va_train \
  --dataset_metadata_path data/ref2va_train/metadata.json \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:Ref2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:Ref2VA/audio_vae/model.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --output_path "./models/train/MiniMax-H3-Ref2VA-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"
```

阶段一不加载 DiT，也不会真正注入 LoRA；命令中的 LoRA 参数用于正确切分 Pipeline。缓存生成后，建议检查样本数：

```bash
find ./models/train/MiniMax-H3-Ref2VA-cache -name '*.pth' | wc -l
```

结果应与 `metadata.json` 的样本数一致。多进程数据切分可能为了补齐 shard 而重复尾部样本，因此首次验证推荐单进程生成缓存。

### 6.2 阶段二：单卡 LoRA

标准 BF16 DiT 很大，只有在单卡显存足够时才使用此方式；显存不足请使用后面的量化 LoRA 或分布式方案。

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path "./models/train/MiniMax-H3-Ref2VA-cache" \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --gradient_accumulation_steps 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Ref2VA-lora" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"
```

输出 `epoch-*.safetensors` 只包含可训练的 LoRA 参数。

### 6.3 单机多卡 LoRA（DDP）

DDP 会在每张卡复制完整 BF16 DiT，适用于每卡都能容纳底模和激活的机器。将阶段二的启动前缀替换为：

```bash
accelerate launch \
  --multi_gpu \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/minimax_h3/model_training/train.py \
  ...其余阶段二参数...
```

`--num_processes` 是本次作业总进程数。单机时通常等于参与训练的 GPU 数。

若单卡无法容纳完整 BF16 DiT，可用后文的 ZeRO-3 启动方式，或优先使用 NF4/Int8/FP8 LoRA。

### 6.4 量化 LoRA

仓库已有以下脚本：

- 预量化 NF4：[`MiniMax-H3-NF4-Ref2VA.sh`](../../examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Ref2VA.sh)
- 在线 bitsandbytes NF4：[`MiniMax-H3-Ref2VA-bitsandbytes_nf4.sh`](../../examples/minimax_h3/model_training/special/quant_training/MiniMax-H3-Ref2VA-bitsandbytes_nf4.sh)
- Int8 ConvRot：[`MiniMax-H3-Int8-ConvRot-Ref2VA.sh`](../../examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-Ref2VA.sh)
- Pruned FP8：[`MiniMax-H3-FP8-Pruned-Ref2VA.sh`](../../examples/minimax_h3/model_training/lora/MiniMax-H3-FP8-Pruned-Ref2VA.sh)

预量化 NF4 脚本是单阶段流程，所有冻结组件和 LoRA 可同时加载，适合先做单卡功能验证。量化底模保持冻结，保存的仍是浮点 LoRA 权重。

分布式量化 LoRA 的代码会跳过冻结量化权重的 DDP 广播，但仍应在目标 PyTorch、量化后端和 NCCL 版本上做小规模测试。

### 6.5 从 LoRA 权重继续训练

在阶段二命令中增加：

```bash
--lora_checkpoint "./models/train/MiniMax-H3-Ref2VA-lora/epoch-4.safetensors"
```

这只恢复 LoRA 权重，不恢复 optimizer、scheduler、epoch 或随机数状态；学习率、训练轮数和数据重复次数会按新命令重新开始计算。

## 7. Ref2VA DiT 全参数训练

### 7.1 阶段一：生成缓存

与 LoRA 阶段一类似，但使用 `--trainable_models "dit"`：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/ref2va_train \
  --dataset_metadata_path data/ref2va_train/metadata.json \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:Ref2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:Ref2VA/audio_vae/model.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --output_path "./models/train/MiniMax-H3-Ref2VA-full-cache" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --task "sft:data_process"
```

### 7.2 单机 8 卡 ZeRO-3 全参训练

仓库配置默认是 1 台机器、8 个进程、BF16、DeepSpeed ZeRO Stage 3：

```bash
accelerate launch \
  --config_file examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path "./models/train/MiniMax-H3-Ref2VA-full-cache" \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --gradient_accumulation_steps 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Ref2VA-full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"
```

输出的 `epoch-*.safetensors` 是完整 DiT 权重，不包含文本编码器和两个 VAE。

需要注意，当前入口会先构造模型，再交给 `accelerator.prepare`。因此 ZeRO-3 能降低正式训练阶段的参数、梯度和 optimizer 分片占用，但启动阶段仍可能出现较高的单进程 CPU/GPU 内存峰值。`--initialize_model_on_cpu` 只改变初始加载位置，后续仍会在包装前执行设备迁移，不能把它视为彻底解决启动显存峰值的方案。

### 7.3 从全参权重继续训练

在阶段二命令中增加：

```bash
--resume_from_checkpoint "./models/train/MiniMax-H3-Ref2VA-full/epoch-1.safetensors"
```

该参数恢复 DiT 权重，但同样不恢复 optimizer、scheduler 和训练进度。

## 8. 多机多卡训练

### 8.1 前置条件

假设两台机器、每台 8 卡：

- 主节点地址：`10.0.0.10`；
- rendezvous 端口：`29500`，所有节点之间可访问；
- `num_machines=2`；
- `num_processes=16`，这里是全局总进程数，不是每台机器的进程数；
- 节点 0 使用 `machine_rank=0`，节点 1 使用 `machine_rank=1`。

生产环境还需要满足：

1. 数据集、阶段一缓存和输出目录位于所有节点可见的共享文件系统，且绝对路径一致。
2. 如果使用节点本地盘，必须在进入阶段二前汇总所有 rank 的缓存；否则不同节点看到的缓存集合不一致。
3. 模型最好提前下载到每个节点相同路径，或放到共享只读目录。当前模型下载入口没有“仅 rank 0 下载、其他 rank 等待”的完整协调机制，不建议让所有进程同时下载同一批大文件。
4. 设置 `DIFFSYNTH_SKIP_DOWNLOAD=True`，避免正式作业再次访问远端；使用本地路径时也可通过 `--model_paths` 显式加载。
5. 所有节点同时执行启动命令，只有 `--machine_rank` 不同。

可在每个节点设置：

```bash
export DIFFSYNTH_SKIP_DOWNLOAD=True
export NCCL_DEBUG=INFO
```

若集群有多张网卡，还需要按实际网络设置 `NCCL_SOCKET_IFNAME`。IB/RDMA 相关变量应由集群管理员按硬件配置，不建议照抄通用值。

### 8.2 多机多卡 LoRA（DDP）

在节点 0 执行：

```bash
accelerate launch \
  --multi_gpu \
  --num_machines 2 \
  --num_processes 16 \
  --machine_rank 0 \
  --main_process_ip 10.0.0.10 \
  --main_process_port 29500 \
  --rdzv_backend static \
  --mixed_precision bf16 \
  examples/minimax_h3/model_training/train.py \
  ...LoRA 阶段二参数...
```

在节点 1 执行同一命令，仅修改：

```bash
--machine_rank 1
```

DDP 会在每个 GPU 复制完整底模。如果每卡放不下 BF16 DiT，可把下面的 ZeRO-3 多机启动方式用于 LoRA 阶段二，或者使用量化 LoRA。

### 8.3 多机多卡全参或 LoRA（DeepSpeed ZeRO-3）

节点 0：

```bash
accelerate launch \
  --config_file examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml \
  --num_machines 2 \
  --num_processes 16 \
  --machine_rank 0 \
  --main_process_ip 10.0.0.10 \
  --main_process_port 29500 \
  --rdzv_backend static \
  examples/minimax_h3/model_training/train.py \
  ...全参或 LoRA 阶段二参数...
```

节点 1 执行相同命令并设置 `--machine_rank 1`。

命令行的 `num_machines`、`num_processes` 和 `machine_rank` 会覆盖单机 YAML 中的对应值。首次运行建议：

- 使用 1～2 个真实样本；
- `dataset_repeat=1`、`num_epochs=1`；
- 设置较小的 `save_steps` 或按 epoch 保存；
- 检查两台机器均参与计算、loss 有限、barrier 不挂起、主节点能成功保存并重新加载 checkpoint。

ZeRO-3 保存 LoRA 时，当前 logger 会先通过 Accelerate/DeepSpeed 收集 state dict，再过滤出 LoRA 参数。因此 rank 0 的主机内存峰值可能接近收集完整 BF16 DiT 所需的内存。若保存阶段 OOM，应增加主节点内存、降低保存频率，或改用能让完整底模常驻每卡的 DDP/量化 LoRA 方案。当前代码若要从根本上消除该峰值，需要改造 LoRA 的 ZeRO-3 保存逻辑。

### 8.4 阶段一是否也要分布式

阶段一缓存只需生成一次，可以：

- 在单卡上顺序生成，最容易保证无重复；或
- 使用单机/多机 Accelerate 加速，dataloader 会按进程切分，缓存按全局 rank 分目录写入。

多机生成缓存时必须使用共享输出目录，并在所有进程完全退出后再启动阶段二。Accelerate 为对齐各 rank 的 dataloader 长度可能复制尾部样本，因此应核对缓存数量和样本唯一性；对严格不能重复的数据集，优先单进程生成缓存。

## 9. CFG 蒸馏适配与音频损失

MiniMax-H3 底模带有 CFG 蒸馏特性。训练入口提供两种相关机制。

### 9.1 `training_cfg_scale`

- 默认 `1.0`：标准 flow-matching 训练，计算开销较低。
- 大于 `1.0`：额外执行一路无梯度的无条件前向，用逆 CFG 形式拟合目标，显存和计算时间都会增加。
- 两阶段训练时，阶段一和阶段二必须使用同一个值；否则缓存中可能缺少无条件分支输入。

示例：

```bash
--training_cfg_scale 3.0
```

### 9.2 Training Adapter

LoRA 示例还提供可选的 DeCFG Training Adapter。它只在训练时融合，推理时不要再次加载：

```bash
modelscope download \
  --model DiffSynth-Studio/MiniMax-H3-TrainingAdapter \
  --include model_ref2va.safetensors \
  --local_dir ./models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter
```

Training Adapter 作用于 DiT，因此只在加载 DiT 的第二阶段增加：

```bash
--preset_lora_path "./models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model_ref2va.safetensors" \
--preset_lora_model "dit"
```

使用 ComfyUI 布局的 Pruned DiT 时，应使用示例中注明的 `model_ref2va_for_comfy_dit.safetensors` 版本。

### 9.3 音频损失权重

```bash
--audio_loss_weight 1.0
```

- `1.0`：视频和音频损失按实现中的默认权重相加。
- `0.0`：不把音频误差计入总损失，但音频流仍会加噪并经过 DiT。

## 10. 验证训练结果

### 10.1 验证 LoRA

修改验证脚本中的 LoRA 路径：

```python
pipe.load_lora(
    pipe.dit,
    "models/train/MiniMax-H3-Ref2VA-lora/epoch-4.safetensors",
)
```

然后执行：

```bash
python examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Ref2VA.py
```

### 10.2 验证全参 DiT

修改验证脚本中的完整 DiT 路径：

```python
ModelConfig(
    path="models/train/MiniMax-H3-Ref2VA-full/epoch-1.safetensors",
    **vram_config,
)
```

然后执行：

```bash
python examples/minimax_h3/model_training/validate_full/MiniMax-H3-Ref2VA.py
```

验证时应固定 prompt、references、seed、分辨率、帧数和推理步数，并同时检查：

- 参考图像/视频/音频遵循程度；
- 目标视频画质与时序稳定性；
- 音频是否存在、是否与画面同步；
- 未训练 prompt 上的泛化能力；
- 与底模相同输入的对照结果。

## 11. 重要限制与避坑

1. **“full”不是全栈联合训练。** 官方命令只训练 `dit`；文本编码器、Video VAE 和 Audio VAE 被用于缓存构建并保持冻结。
2. **不要把框架 CPU Offload 与分布式混用。** 当前 `--enable_model_cpu_offload` 分支只把 optimizer、dataloader 和 scheduler 交给 `accelerator.prepare`，模型本身由自定义 offload manager 管理，不会走 DDP/DeepSpeed 梯度同步。它适合单进程低显存训练。DeepSpeed 自身的 CPU offload 是另一条路径。
3. **checkpoint 不是完整训练状态。** `--lora_checkpoint` 和 `--resume_from_checkpoint` 都只恢复权重。
4. **多机必须使用一致可见的缓存。** 阶段二会递归扫描缓存目录；不同节点看到不同文件会造成数据不一致。
5. **避免多进程并发下载大模型。** 先下载，再设置 `DIFFSYNTH_SKIP_DOWNLOAD=True`。
6. **ZeRO-3 保存会产生主机内存峰值。** 全参 checkpoint 本身很大；LoRA 在当前实现中也可能先收集完整 state dict 再过滤。
7. **修改预处理条件后重建缓存。** 尤其是分辨率、帧数、prompt、references、音频和 `training_cfg_scale`。
8. **保留 `--find_unused_parameters`。** 官方 Ref2VA 多卡命令已启用，移除后可能触发 DDP 未使用参数错误。
9. **控制有效 batch size。** 扩大 GPU 数量后，如需保持优化行为，应相应调整 `gradient_accumulation_steps` 或学习率，并重新验证收敛。
10. **动态分辨率需谨慎。** 不同样本的序列长度和显存峰值差异很大，首次分布式验证建议固定 480×832、124 帧。

## 12. 推荐上线检查清单

- [ ] 单卡完成 1 个样本的数据缓存和 1 个训练 step。
- [ ] 单机多卡完成 1 个 epoch，并能保存、重新加载 checkpoint。
- [ ] 多机节点的代码、依赖、模型 hash 和数据路径一致。
- [ ] 主节点端口互通，NCCL 网卡配置正确。
- [ ] 数据、缓存、输出位于共享文件系统。
- [ ] 缓存数量、字段、dtype 和样本唯一性符合预期。
- [ ] 没有在分布式作业中启用框架级 `--enable_model_cpu_offload`。
- [ ] 主节点有足够 CPU 内存完成 ZeRO-3 checkpoint 收集。
- [ ] 训练中 loss 为有限值，所有 rank step 数一致。
- [ ] 全参和 LoRA checkpoint 均已通过独立推理脚本验证。

## 13. 本次审计范围

本次结论来自训练入口、Ref2VA 数据加载、条件打包、损失、LoRA 注入、Accelerate/DeepSpeed 包装和 checkpoint 逻辑的代码审计，并对相关 Python 文件进行了语法编译检查。

由于完整 MiniMax-H3 Ref2VA 训练需要大模型权重、多卡/多机硬件和真实集群网络，本次未执行全量端到端训练。因此，单机多卡能力有仓库配置和实现依据；多机多卡能力属于代码级支持，仍需按第 8 节在目标集群完成冒烟测试后确认。
