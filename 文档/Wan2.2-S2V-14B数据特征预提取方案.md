# Wan2.2-S2V-14B 数据特征预提取方案

本文整理 Wan2.2-S2V-14B full training 的数据特征预提取拆分方案、当前代码改动、运行命令，以及 cache 提取阶段 CUDA OOM 的处理方式。

## 1. 目标

Wan2.2-S2V 训练中，VAE 编码、T5 文本编码、音频编码、S2V motion/pose 条件处理都和数据本身强相关，不依赖 DiT 的当前训练参数。预先把这些结果保存成 `.pth` cache 后，正式训练阶段可以跳过这些重复计算，只加载 cache 并执行 DiT 训练子图。

本次调整把数据 cache 提取从训练脚本中拆出来：

| 文件 | 作用 |
| --- | --- |
| `examples/wanvideo/model_training/full/Wan2.2-S2V-14B-cache-run.sh` | 单独执行 `sft:data_process`，生成数据特征 cache。 |
| `examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh` | 单机训练入口，只保留 raw 训练和从 cache 训练。 |
| `examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh` | 多节点入口，只保留从 cache 加载训练。 |
| `examples/wanvideo/model_training/train.py` | 新增 `--tiled`、`--tile_size`、`--tile_stride` 参数，用于降低 VAE 预处理显存。 |

## 2. Cache 路径规则

默认 cache 保存到原数据集目录下：

```bash
${DATASET_BASE_PATH}/Wan2.2-S2V-14B_full_${DATA_FEATURE_SIZE_TAG}_features
```

`DATA_FEATURE_SIZE_TAG` 根据尺寸配置自动生成：

```bash
# 固定分辨率
HEIGHT=448 WIDTH=832 NUM_FRAMES=81
# cache 目录片段: 448x832x81

# 动态分辨率
MAX_PIXELS=589824 NUM_FRAMES=81
# cache 目录片段: max_pixels_589824_frames_81
```

也可以显式覆盖：

```bash
DATA_FEATURE_CACHE_PATH=/data/work/train_data_5s/s2v_cache
```

注意：cache 与 `height/width/max_pixels/num_frames/frame_rate/fix_frame_rate`、metadata、VAE/T5/音频编码器版本相关。修改这些内容后应使用新的 cache 目录。

## 3. 单机运行命令

### 3.1 提取数据 cache

默认命令：

```bash
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-cache-run.sh
```

指定数据集：

```bash
DATASET_BASE_PATH=data/train_data \
DATASET_METADATA_PATH=data/train_data/metadata.csv \
DATA_FILE_KEYS=video,input_audio,s2v_pose_video \
EXTRA_INPUTS=input_image,input_audio,s2v_pose_video \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-cache-run.sh
```

脚本内部执行的核心任务是：

```bash
--task sft:data_process
--dataset_repeat 1
--output_path "${DATA_FEATURE_CACHE_PATH}"
```

`dataset_repeat` 固定为 `1`，避免重复缓存同一条样本。

### 3.2 从 cache 训练

默认训练脚本仍可从原始数据训练：

```bash
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
```

从已提取的 cache 训练：

```bash
TRAIN_FROM_CACHE=1 \
DATASET_BASE_PATH=data/train_data \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
```

如果 cache 不在默认位置：

```bash
TRAIN_FROM_CACHE=1 \
DATA_FEATURE_CACHE_PATH=/data/work/train_data_5s/s2v_cache \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
```

cache 训练阶段不传 `--dataset_metadata_path`。`UnifiedDataset` 会在 `DATA_FEATURE_CACHE_PATH` 下递归扫描 `.pth` 文件，并进入缓存读取模式。

## 4. 多节点从 cache 训练

`Wan2.2-S2V-14B-multinode-cache-run.sh` 已只保留 cache 训练路径，核心参数为：

```bash
--dataset_base_path "${DATA_FEATURE_CACHE_PATH}"
--task sft:train
```

启动方式：

```bash
# node1
NODE_RANK=0 \
DATASET_BASE_PATH=data/train_data \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh

# node2
NODE_RANK=1 \
DATASET_BASE_PATH=data/train_data \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh
```

后台运行示例：

```bash
# node1
NODE_RANK=0 nohup bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh > train.log 2>&1 & tail -f train.log

# node2
NODE_RANK=1 nohup bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh > train.log 2>&1 & tail -f train.log
```

如果 cache 在自定义目录，各节点都需要能访问相同路径，或提前同步：

```bash
DATA_FEATURE_CACHE_PATH=/data/work/train_data_5s/s2v_cache \
NODE_RANK=0 \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh
```

脚本启动前会检查 `DATA_FEATURE_CACHE_PATH` 是否存在，不存在会提示先运行 `Wan2.2-S2V-14B-cache-run.sh` 或覆盖 cache 路径。

## 5. OOM 处理

之前执行 `Wan2.2-S2V-14B-cache-run.sh` 时的 OOM 出现在 `sft:data_process` 阶段：

```text
WanVideoUnit_S2V.process_motion_latents
pipe.vae.encode(...)
torch.OutOfMemoryError: CUDA out of memory
```

原因是训练入口原先把 `tiled` 固定为 `False`，S2V 会一次性 VAE encode 73 帧 motion video。80G GPU 上模型已占用约 78G，再申请约 1.27GiB 时失败。

当前 cache 脚本默认启用这些缓解项：

```bash
TILED=1
TILE_SIZE=30,52
TILE_STRIDE=15,26
OFFLOAD_MODELS="${MODEL_ID_WITH_ORIGIN_PATHS%%,*}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

含义：

| 参数 | 作用 |
| --- | --- |
| `TILED=1` | 开启 VAE tiled encode，避免整段视频一次进 VAE。 |
| `TILE_SIZE/TILE_STRIDE` | 控制 VAE tile 大小和步长。 |
| `OFFLOAD_MODELS` | 默认 offload 第一个模型条目，即 DiT；cache 提取前半段不需要 DiT 常驻显存。 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 减少 CUDA 内存碎片导致的分配失败。 |

如果仍然 OOM，优先减小 tile：

```bash
TILE_SIZE=20,36 TILE_STRIDE=10,18 \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-cache-run.sh
```

也可以降低提取分辨率或帧数：

```bash
MAX_PIXELS=458752 NUM_FRAMES=65 \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-cache-run.sh
```

如果想关闭默认 offload 做对比：

```bash
OFFLOAD_MODELS= \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-cache-run.sh
```

## 6. 常用环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATASET_BASE_PATH` | `./data/test_data` | 原始数据集目录；也是默认 cache 的父目录。 |
| `DATASET_METADATA_PATH` | `${DATASET_BASE_PATH}/metadata.csv` | 原始数据 metadata。 |
| `DATA_FILE_KEYS` | `video,input_audio` | 需要从 metadata 读取并做数据算子的字段。 |
| `EXTRA_INPUTS` | `input_image,input_audio` | 额外传给 Wan S2V pipeline 的字段。 |
| `DATA_FEATURE_CACHE_PATH` | `${DATASET_BASE_PATH}/Wan2.2-S2V-14B_full_${DATA_FEATURE_SIZE_TAG}_features` | cache 输出或读取目录。 |
| `TRAIN_FROM_CACHE` | `0` | 单机训练脚本中设为 `1/cache` 时从 cache 训练。 |
| `DATA_PROCESS_CONFIG_FILE` | `examples/wanvideo/model_training/full/accelerate_config_data_process.yaml` | cache 提取用 accelerate 配置。 |
| `TRAIN_CONFIG_FILE` | `examples/wanvideo/model_training/full/accelerate_config_14B.yaml` | 单机训练用 accelerate 配置。 |
| `CONFIG_FILE` | 脚本相关默认值 | 兼容旧变量；cache 脚本中可覆盖 data process config，多节点脚本中覆盖 ZeRO-3 config。 |
| `NUM_PROCESSES` | 空 | 单机 accelerate 进程数覆盖。 |

## 7. 推荐流程

1. 先提取 cache：

```bash
DATASET_BASE_PATH=data/train_data \
DATASET_METADATA_PATH=data/train_data/metadata.csv \
DATA_FILE_KEYS=video,input_audio,s2v_pose_video \
EXTRA_INPUTS=input_image,input_audio,s2v_pose_video \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-cache-run.sh
```

2. 单机从 cache 训练：

```bash
TRAIN_FROM_CACHE=1 \
DATASET_BASE_PATH=data/train_data \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
```

3. 多节点从 cache 训练：

```bash
NODE_RANK=0 DATASET_BASE_PATH=data/train_data \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh

NODE_RANK=1 DATASET_BASE_PATH=data/train_data \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh
```

## 8. 验证命令

修改后已执行过：

```bash
bash -n examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
bash -n examples/wanvideo/model_training/full/Wan2.2-S2V-14B-cache-run.sh
bash -n examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode-cache-run.sh
python3 -m py_compile examples/wanvideo/model_training/train.py
```

这些命令只验证语法，不会启动真实训练或 cache 提取。
