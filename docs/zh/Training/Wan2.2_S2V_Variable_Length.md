# Wan2.2-S2V 可变时长训练

本文说明如何训练包含 81、97、113 等不同帧数的视频，并保证视频采样、音频长度和预计算缓存保持一致。当前实现暂不包含长度分桶；每个 DataLoader batch 仍为单样本。

## 设计约束

`--num_frames` 表示最大采样帧数，而不是固定帧数。有效帧数由以下配置决定：

```text
frame_count_stride * n + frame_count_remainder
```

Wan2.2-S2V 推荐使用 `16n+1`：

```text
--frame_count_stride 16
--frame_count_remainder 1
```

当容器时长换算后的帧数不在该序列中时，`--frame_count_rounding` 控制归整方式：

- `nearest`：选择最近值；距离相同时优先向下，避免制造帧。
- `floor`：向下取有效值。
- `ceil`：向上取有效值。

向上归整时重复最后一个可解码视频帧。`--max_frame_padding` 限制最多允许重复多少个目标帧，超过限制会立即报错。

以目标 16 FPS、有效范围 81 到 113 为例：

| 可用帧数 | nearest 结果 | 操作 |
|---:|---:|---|
| 80 | 81 | 重复末帧 1 次 |
| 89 | 81 | 等距时向下截断 8 帧 |
| 90 | 97 | 重复末帧 7 次 |
| 104 | 97 | 向下截断 7 帧 |
| 110 | 113 | 重复末帧 3 次 |
| 140 | 113 | 受最大帧数限制，截断到 113 |

可用帧数根据最后一个可解码源帧的时间戳计算，不依赖经常存在一帧偏差的容器 `duration` 字段。

## 音视频对齐

采样后的视频时间跨度定义为：

```text
duration = (sample_num_frames - 1) / sampled_frame_rate
```

音频会被重采样到 `--audio_sample_rate`，然后精确调整为：

```text
target_audio_samples = round(duration * audio_sample_rate)
```

支持两种策略：

- `--audio_duration_policy trim_pad`：长音频截断尾部，短音频补零。适合视频使用前缀窗口的训练。
- `--audio_duration_policy strict`：只有时长差不超过 `--audio_duration_tolerance_seconds` 才允许微调，否则报错。

无论使用哪种策略，进入 S2V audio encoder 的音频长度都与视频时间跨度完全一致。短音频补零还受到 `--max_audio_padding_seconds` 限制；明显缺失的音频不会被静默接受。可选的 `--max_audio_trimming_seconds` 可限制长音频截断量。

## 推荐配置

当前 81/97/113 帧数据推荐：

```text
NUM_FRAMES=113
MIN_NUM_FRAMES=81
FRAME_RATE=16
FIX_FRAME_RATE=1
FRAME_COUNT_STRIDE=16
FRAME_COUNT_REMAINDER=1
FRAME_COUNT_ROUNDING=nearest
MAX_FRAME_PADDING=8
AUDIO_SAMPLE_RATE=16000
AUDIO_DURATION_POLICY=trim_pad
AUDIO_DURATION_TOLERANCE_SECONDS=0.05
MAX_AUDIO_PADDING_SECONDS=0.5
```

如果数据质量要求“不允许明显裁剪或补齐”，使用：

```text
AUDIO_DURATION_POLICY=strict
AUDIO_DURATION_TOLERANCE_SECONDS=0.05
MAX_AUDIO_PADDING_SECONDS=0.05
MAX_AUDIO_TRIMMING_SECONDS=0.05
```

## 阶段一：预计算特征

可变长度原始视频不能直接放入包含冻结 VAE 的 ZeRO-3 训练路径。不同 rank 的 VAE 调用次数不同，会导致参数 all-gather 次序分叉。因此第一阶段使用普通 `MULTI_GPU` 配置，预计算 VAE、T5 和 audio encoder 特征。

在 node2 单节点 8 卡执行：

```bash
cd /data-training/hjq/DiffSynth-Studio

ACCELERATE_BIN=/data-training/miniconda/bin/accelerate \
MODEL_BASE_PATH=/data-training/models \
DATASET_BASE_PATH=/data-training/train_data_5s \
DATASET_METADATA_PATH=/data-training/train_data_5s/metadata.csv \
DATA_FEATURE_CACHE_PATH=/data-training/train_data_5s/cache_s2v_f81-113_16n1_fps16_v1 \
NUM_PROCESSES=8 \
NUM_FRAMES=113 \
MIN_NUM_FRAMES=81 \
FRAME_RATE=16 \
FRAME_COUNT_STRIDE=16 \
FRAME_COUNT_REMAINDER=1 \
FRAME_COUNT_ROUNDING=nearest \
MAX_FRAME_PADDING=8 \
AUDIO_DURATION_POLICY=trim_pad \
MAX_AUDIO_PADDING_SECONDS=0.5 \
TILED=1 \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-cache-run.sh
```

缓存目录会生成：

- 每个样本的 `.pth` 特征；
- `_cache_manifest.json`：配置指纹、metadata SHA256、文件数和全局帧长分布；
- `_cache_summary_rank_*.json`：每个 rank 的保存数量和帧长分布。

缓存仅保存训练需要的 tensor 和长度元数据，不保存原始 PIL 视频帧。Tensor 在落盘前转到 CPU。

### 预计算断点续算

任务中断后使用完全相同的配置和缓存目录，并增加：

```text
RESUME_FEATURE_CACHE=1
```

配置指纹不一致时程序会拒绝混用缓存。完整缓存再次执行断点续算时会直接退出。
断点续算必须保持与首次预计算相同的进程数。预计算不会为整除进程数而重复尾部样本，
结束时还会校验缓存文件总数与源样本数完全一致。

## 阶段二：node2/node3 缓存训练

在两台节点分别启动同一脚本。共享缓存路径必须一致。

node2：

```bash
NODES="node2 node3" NODE_RANK=0 MASTER_ADDR=node2 \
ACCELERATE_BIN=/data-training/miniconda/bin/accelerate \
MODEL_BASE_PATH=/data-training/models COMM_IFNAME=bond0 \
DATASET_BASE_PATH=/data-training/train_data_5s \
DATA_FEATURE_CACHE_PATH=/data-training/train_data_5s/cache_s2v_f81-113_16n1_fps16_v1 \
OUTPUT_PATH=./models/train/Wan2.2-S2V-14B_variable_length \
LEARNING_RATE=1e-7 SAVE_STEPS=100 \
RESUME_FROM_CHECKPOINT=/data-training/hjq/DiffSynth-Studio/models/train/merge_lv2_sa_step-600.safetensors \
S2V_REF_ROPE_MODE=source_id_local ENABLE_TENSORBOARD_LOG=1 \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh
```

node3：将 `NODE_RANK=0` 改为：

```text
NODE_RANK=1
```

训练脚本默认添加 `--require_cache_manifest`。以下情况会在加载模型前失败：

- manifest 缺失；
- manifest 状态不是 `complete`；
- 实际 `.pth` 文件数与 manifest 不一致。

## 日志与排查

每个进程默认记录前 `--data_processing_log_samples` 条采样决策，包括：

- 原始帧数和 FPS；
- 目标 FPS 下的可用帧数；
- 最终选择帧数；
- 视频补帧/截断数量；
- 音频调整前后采样点数；
- 音频执行 `trim`、`pad` 或 `none`。

预计算完成后检查：

```bash
python -m json.tool /data-training/train_data_5s/cache_s2v_f81-113_16n1_fps16_v1/_cache_manifest.json
find /data-training/train_data_5s/cache_s2v_f81-113_16n1_fps16_v1 -name "*.tmp*"
```

第二条命令应无输出。建议先用小 metadata（例如 16 到 32 条）完成一次“预计算 → 两节点训练 3 step”测试，再生成完整缓存。

## 当前边界

- 暂未实现长度分桶。随机混合长度是正确的，但一个 global step 的耗时由最长样本决定。
- batch size 当前为 1，因此不同 tensor shape 不需要 stack。
- 修改帧数、FPS、VAE tiled 配置、冻结编码器版本或 metadata 后必须使用新的缓存目录。
- 缓存训练阶段只加载 DiT；不要重新加入 VAE、T5 或 audio encoder。
