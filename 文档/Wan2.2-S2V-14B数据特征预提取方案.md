# Wan2.2-S2V-14B 数据特征预提取实现方案

本文说明基于 `examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh` 新增的数据特征预提取流程。目标是把和数据强相关、但训练中会重复计算的特征先缓存到磁盘，后续训练只读取缓存并执行 DiT 训练子图。

## 1. 背景

Wan2.2-S2V 全量训练中，单条样本会经过这些数据相关处理：

1. 读取视频和音频。
2. 视频按 `height/width/num_frames/frame_rate` 采样、裁剪、缩放。
3. VAE 编码输入视频，得到 `input_latents` 等 latent 特征。
4. T5 文本编码 prompt，得到 `context`。
5. S2V 音频编码器处理 `input_audio`，得到 `audio_embeds`。
6. S2V 相关 motion/pose 条件被转换成训练所需 tensor。

这些结果只和数据、采样配置、模型编码器和部分训练配置有关，不依赖 DiT 当前参数。预提取后，正式训练阶段可以跳过 VAE/T5/音频编码，直接从 `.pth` 缓存读取 DiT 训练输入。

## 2. 实现入口

全量训练脚本已支持参数：

```bash
PRE_EXTRACT_DATA_FEATURES=0|1|only|cache
```

含义如下：

| 值 | 行为 |
| --- | --- |
| `0` | 默认行为，直接从原始数据训练。 |
| `1` | 先运行数据特征预提取，再从缓存继续训练。 |
| `only` | 只执行数据特征预提取，不启动训练。 |
| `cache` | 不重新预提取，只从已有缓存训练。 |

默认数据路径以 `./data/test_data` 为例：

```bash
DATASET_BASE_PATH=./data/test_data
DATASET_METADATA_PATH=./data/test_data/metadata.csv
```

尺寸配置支持两种方式：

```bash
# 固定分辨率
HEIGHT=576 WIDTH=1024 NUM_FRAMES=81

# 动态分辨率
MAX_PIXELS=1048576 NUM_FRAMES=81
```

脚本默认不设置 `HEIGHT/WIDTH`，走 `MAX_PIXELS` 动态分辨率路径。

如果 metadata 中还包含 `s2v_pose_video`，需要同步打开对应字段：

```bash
DATA_FILE_KEYS=video,input_audio,s2v_pose_video \
EXTRA_INPUTS=input_image,input_audio,s2v_pose_video \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
```

默认缓存目录会根据尺寸配置自动生成。固定分辨率时，例如同时设置 `HEIGHT=576 WIDTH=1024 NUM_FRAMES=81`：

```bash
./models/cache/Wan2.2-S2V-14B_full_576x1024x81_features
```

动态分辨率时，例如不设置 `HEIGHT/WIDTH`，只设置 `MAX_PIXELS=1048576 NUM_FRAMES=81`：

```bash
./models/cache/Wan2.2-S2V-14B_full_max_pixels_1048576_frames_81_features
```

也可以通过 `DATA_FEATURE_SIZE_TAG` 自定义规格片段，或者直接通过环境变量覆盖完整缓存路径：

```bash
DATA_FEATURE_CACHE_PATH=./models/cache/my_s2v_features
```

## 3. 推荐命令

直接从原始数据训练：

```bash
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
```

先预提取，再训练：

```bash
PRE_EXTRACT_DATA_FEATURES=1 DATASET_BASE_PATH=./data/test_data DATASET_METADATA_PATH=./data/test_data/metadata.csv bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
```

只预提取，不训练：

```bash
PRE_EXTRACT_DATA_FEATURES=only DATASET_BASE_PATH=./data/test_data DATASET_METADATA_PATH=./data/test_data/metadata.csv bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
```

从已有缓存训练，不重新预提取：

```bash
PRE_EXTRACT_DATA_FEATURES=cache DATA_FEATURE_CACHE_PATH=./models/cache/Wan2.2-S2V-14B_full_max_pixels_1048576_frames_81_features bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
```

如果需要完全自定义训练命令，也可以直接把 `DATASET_BASE_PATH` 指向缓存目录并调用 `train.py --task sft:train`。

## 4. 两阶段任务设计

### 4.1 预提取阶段

脚本内部执行：

```bash
--task sft:data_process
--dataset_repeat 1
--output_path ${DATA_FEATURE_CACHE_PATH}
```

`WanTrainingModule` 在 `sft:data_process` 下会调用 `split_pipeline_units`，保留 DiT 训练子图之前的数据处理单元。输出的 `.pth` 是处理后的训练输入，供第二阶段直接读取。

预提取阶段固定 `dataset_repeat=1`，避免重复缓存相同样本。

### 4.2 训练阶段

脚本内部执行：

```bash
--dataset_base_path ${DATA_FEATURE_CACHE_PATH}
--task sft:train
```

这里不传 `--dataset_metadata_path`，`UnifiedDataset` 会进入缓存读取模式，递归扫描 `${DATA_FEATURE_CACHE_PATH}` 下的 `.pth` 文件。

`task=sft:train` 会让训练只保留 DiT 相关子图，避免再次运行 VAE/T5/音频编码。

## 5. 分布式兼容方式

预提取阶段仍然通过 `accelerate launch` 启动多进程。`runner.py` 中的 `launch_data_process_task` 会把 dataloader 交给 `accelerator.prepare(dataloader)`，因此不同 rank 处理不同样本。

每个 rank 写入自己的子目录：

```text
DATA_FEATURE_CACHE_PATH/
  0/
    00000000__video=...__input_audio=...__hash.pth
  1/
    00000001__video=...__input_audio=...__hash.pth
  ...
```

写入时使用临时文件 + `os.replace`，避免进程中断时留下半写入文件。预提取结束前会调用 `accelerator.wait_for_everyone()`，确保所有 rank 完成写入。

读取缓存时，`UnifiedDataset` 会递归扫描所有 rank 子目录，并对文件列表排序，保证多进程读取顺序稳定。

## 6. 同名文件区分

新增缓存文件名由 `UnifiedDataset.build_data_cache_key` 生成，包含：

1. metadata 原始行号。
2. `data_file_keys` 中的原始相对路径。
3. 对整条 metadata 的 SHA1 hash 前缀。

例如 metadata 中存在：

```csv
video,input_audio,prompt
speaker_a/clip.mp4,speaker_a/audio.wav,hello
speaker_b/clip.mp4,speaker_b/audio.wav,hello
```

缓存文件名会类似：

```text
00000000__video=speaker_a__clip.mp4__input_audio=speaker_a__audio.wav__a1b2c3d4e5f6a7b8.pth
00000001__video=speaker_b__clip.mp4__input_audio=speaker_b__audio.wav__c9d8e7f6a5b4c3d2.pth
```

即使两个子目录下都叫 `clip.mp4`，缓存名也会带上子目录路径和 hash，不会互相覆盖。

## 7. 变更文件

本次实现涉及：

| 文件 | 作用 |
| --- | --- |
| `diffsynth/core/data/unified_dataset.py` | 生成稳定缓存 key；缓存读取时排序。 |
| `diffsynth/diffusion/runner.py` | data_process 落盘时使用缓存 key，按 rank 子目录写入，原子替换并分布式同步。 |
| `examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh` | 增加 `PRE_EXTRACT_DATA_FEATURES`，支持原始训练、只预提取、预提取后训练、仅从缓存训练四种模式。 |

## 8. 注意事项

1. 缓存和 `height/width/max_pixels/num_frames/frame_rate` 强相关。修改这些配置后应使用新的 `DATA_FEATURE_CACHE_PATH`。
2. 缓存和 VAE/T5/音频编码器版本相关。更换模型文件后建议重新预提取。
3. 预提取缓存可能很大，需要提前确认磁盘空间。
4. 如果 metadata 内容发生变化，应清理旧缓存或使用新缓存目录，避免混用不同数据版本。
5. 当前方案复用现有 `sft:data_process` 和 `sft:train` 机制，不改变默认直接训练行为。
