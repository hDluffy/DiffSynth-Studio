# Wan2.2-S2V-14B 多节点训练运行文档

本文档说明如何使用 `Wan2.2-S2V-14B-multinode.sh` 在 `node1`、`node2` 两个节点上启动 Wan2.2-S2V-14B full training。

## 运行方式

该脚本使用 `accelerate launch` 和 DeepSpeed 的 `standard` 多节点启动方式。需要在每个节点分别执行同一个脚本：

- `node1` 作为主节点，默认 `MASTER_ADDR=node1`，`NODE_RANK=0`
- `node2` 作为第二个节点，`NODE_RANK=1`

脚本默认节点列表是：

```bash
NODES="node1 node2"
```

如果当前机器的 hostname 正好是 `node1` 或 `node2`，脚本会自动推断 `NODE_RANK`。如果 hostname 不匹配，需要手动指定 `NODE_RANK`。

## 前置检查

在两个节点上确认以下条件一致：

1. 代码路径存在：

```bash
cd /app/DiffSynth-Studio
test -f examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode.sh
```

2. `accelerate` 可执行文件存在：

```bash
test -x /app/miniconda3/bin/accelerate
```

如果路径不同，运行时设置 `ACCELERATE_BIN`。

脚本会自动把 `ACCELERATE_BIN` 所在目录加入 `PATH`，这样 DeepSpeed JIT 编译 CPUAdam 时可以找到同一环境里的 `ninja`。

3. `ninja` 可执行文件存在：

```bash
test -x /app/miniconda3/bin/ninja
```

4. 数据集路径存在：

```bash
test -f data/diffsynth_example_dataset/wanvideo/Wan2.2-S2V-14B/metadata.csv
```

5. GPU 可见：

```bash
nvidia-smi
```

6. `node2` 可以访问 `node1:29500`。如果端口冲突或不可用，运行时设置 `MASTER_PORT`。

7. 两个节点的分布式通信网卡一致。当前环境默认使用 `enp94s0f0np0`，脚本会设置 `NCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME`。

8. IB/RDMA 已可用。当前脚本默认设置 `NCCL_NET=IB`、`NCCL_IB_DISABLE=0`，并使用和 nccl-tests 压测一致的 `NCCL_MIN_NCHANNELS=16`、`NCCL_MAX_NCHANNELS=16`。

## 启动命令

在 `node1` 上执行：

```bash
cd /app/DiffSynth-Studio
MASTER_ADDR=node1 NODE_RANK=0 bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode.sh
```

在 `node2` 上执行：

```bash
cd /app/DiffSynth-Studio
MASTER_ADDR=node1 NODE_RANK=1 bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode.sh
```

两个节点需要在较短时间内都启动。`node1` 会等待 `node2` 加入分布式 rendezvous。

## 常用参数

默认每个节点会自动通过 `nvidia-smi -L` 检测 GPU 数。如果需要手动指定，例如每个节点使用 8 张 GPU：

```bash
GPUS_PER_NODE=8 MASTER_ADDR=node1 NODE_RANK=0 bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode.sh
```

如果端口 `29500` 被占用：

```bash
MASTER_PORT=29501 MASTER_ADDR=node1 NODE_RANK=0 bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode.sh
```

如果 `accelerate` 不在默认路径：

```bash
ACCELERATE_BIN=/path/to/accelerate MASTER_ADDR=node1 NODE_RANK=0 bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode.sh
```

如果通信网卡不是 `enp94s0f0np0`：

```bash
COMM_IFNAME=eth0 MASTER_ADDR=node1 NODE_RANK=0 bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode.sh
```

如果需要临时关闭 IB，改走 socket 网络：

```bash
NCCL_NET=Socket NCCL_IB_DISABLE=1 MASTER_ADDR=node1 NODE_RANK=0 bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode.sh
```

## 环境变量说明

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `NODES` | `node1 node2` | 节点列表，顺序决定 rank |
| `NUM_MACHINES` | `NODES` 的数量 | 总节点数 |
| `MASTER_ADDR` | `node1` | 主节点地址 |
| `MASTER_PORT` | `29500` | 分布式通信端口 |
| `NODE_RANK` | 按 hostname 推断 | 当前节点 rank，`node1=0`，`node2=1` |
| `GPUS_PER_NODE` | 自动检测，失败时为 `8` | 每节点进程数 |
| `NUM_PROCESSES` | `GPUS_PER_NODE * NUM_MACHINES` | 全局总训练进程数 |
| `ACCELERATE_BIN` | `/app/miniconda3/bin/accelerate` | accelerate 可执行文件 |
| `CONFIG_FILE` | `examples/wanvideo/model_training/full/accelerate_config_zero3.yaml` | accelerate ZeRO-3 配置 |
| `TRAIN_SCRIPT` | `examples/wanvideo/model_training/train.py` | 训练入口 |
| `PATH` | 自动追加 `/app/miniconda3/bin` | 确保 DeepSpeed JIT 能找到 `ninja` |
| `COMM_IFNAME` | `enp94s0f0np0` | NCCL/Gloo 默认通信网卡 |
| `NCCL_SOCKET_IFNAME` | `COMM_IFNAME` | NCCL 通信网卡 |
| `GLOO_SOCKET_IFNAME` | `COMM_IFNAME` | Gloo 通信网卡 |
| `NCCL_NET` | `IB` | 强制 NCCL 使用 IB net backend |
| `NCCL_IB_DISABLE` | `0` | 启用 NCCL IB/RDMA |
| `NCCL_MIN_NCHANNELS` | `16` | NCCL 最小 channel 数 |
| `NCCL_MAX_NCHANNELS` | `16` | NCCL 最大 channel 数 |

## 验证脚本

在任意节点检查 shell 语法：

```bash
bash -n examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode.sh
```

在 `node2` 上检查文件是否已经同步：

```bash
ssh node2 bash -n /app/DiffSynth-Studio/examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode.sh
```

## 常见问题

### 卡在 OMP_NUM_THREADS 提示后

`OMP_NUM_THREADS` 是 torchrun 的正常提示。如果后续长时间没有输出，先确认两个节点都已经启动。若进程已启动但仍不前进，检查是否有进程连接到 `169.254.*` 这类链路本地地址。当前脚本默认固定 `COMM_IFNAME=enp94s0f0np0`，避免 PyTorch 选错网卡，同时设置 `NCCL_NET=IB` 让 NCCL 走 IB/RDMA。

### 无法推断 NODE_RANK

如果报错：

```text
Cannot infer NODE_RANK from hostname.
```

说明当前 hostname 不是 `node1` 或 `node2`。启动时显式传入：

```bash
NODE_RANK=0 bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-multinode.sh
```

### 端口被占用

如果 `node1` 上 `29500` 被占用，两个节点都需要使用同一个新端口：

```bash
MASTER_PORT=29501
```

### 进程数不正确

如果脚本自动检测的 GPU 数不符合预期，两个节点都显式指定同一个 `GPUS_PER_NODE`：

```bash
GPUS_PER_NODE=8
```

### DeepSpeed 报 ninja not installed

如果出现：

```text
RuntimeError: Ninja is required to load C++ extensions
```

通常不是没有安装，而是训练进程的 `PATH` 找不到 `ninja`。当前脚本会自动把 `/app/miniconda3/bin` 加入 `PATH`，并在启动前打印：

```text
ninja: /app/miniconda3/bin/ninja
```

### ZeRO-3 下 wav2vec2 weight norm 报 IndexError

如果使用 `accelerate_config_zero3.yaml` 时出现：

```text
IndexError: Dimension out of range ... torch._weight_norm(..., dim=2)
```

这是 ZeRO-3 初始化分片和 `Wav2Vec2ForCTC` 随机初始化流程冲突导致的。S2V audio encoder 会从 `model.safetensors` 加载权重，不需要随机初始化；当前代码已在构造 wav2vec2 时跳过初始化以兼容 ZeRO-3。

### 数据路径不一致

训练参数使用相对路径：

```text
data/diffsynth_example_dataset/wanvideo/Wan2.2-S2V-14B
```

两个节点都需要在 `/app/DiffSynth-Studio` 下看到相同的数据文件，否则某个节点会在加载数据时失败。
