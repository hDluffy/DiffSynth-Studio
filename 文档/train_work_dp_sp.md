# PyTorch 中 DP 和 DDP 的原理与示例

本文用最小的 PyTorch 训练例子解释两种常见多卡训练方式：

- `torch.nn.DataParallel`，简称 DP。
- `torch.nn.parallel.DistributedDataParallel`，简称 DDP。

结论先放前面：实际训练中优先使用 DDP。DP 更适合理解多卡数据并行的基本流程，DDP 才是 PyTorch 推荐的高性能多卡训练方式。

## 1. 数据并行的核心思想

假设我们有 4 张 GPU，单卡一次只能处理 `batch_size=8`，那么数据并行会让每张卡各处理 8 条样本：

```text
全局 batch: 32 条样本
GPU 0: 8 条样本
GPU 1: 8 条样本
GPU 2: 8 条样本
GPU 3: 8 条样本
```

每张 GPU 上都有一份相同的模型副本。每张卡只计算自己那部分数据的前向和反向，最后把梯度同步起来，让所有模型副本得到一致的参数更新。

可以把一次训练迭代理解成：

```text
切分 batch -> 多卡并行 forward -> 计算 loss -> 多卡 backward -> 同步梯度 -> optimizer.step()
```

DP 和 DDP 的主要区别在于：

- DP 是单进程多线程，主要由 GPU 0 负责分发数据、收集输出、汇总梯度。
- DDP 是多进程，通常每张 GPU 一个进程，每个进程独立读数据、独立 forward/backward，并在反向传播过程中同步梯度。

## 2. DP: DataParallel

### 2.1 DP 的工作流程

`DataParallel` 的核心流程如下：

```text
一个 Python 进程
    |
    |-- 把一个 batch 按第 0 维切分到多张 GPU
    |-- 每张 GPU 临时复制一份模型
    |-- 每张 GPU 用自己的 mini-batch 做 forward
    |-- 把各 GPU 的输出 gather 到主 GPU，默认是 cuda:0
    |-- 在主 GPU 上计算 loss
    |-- backward 时，把各副本的梯度汇总到原始模型
    |-- optimizer.step() 更新原始模型参数
```

DP 的优点是改代码很少；缺点是 GPU 0 压力大，模型复制和输出收集有额外开销，单进程多线程也会受到 Python 调度影响。因此大模型和多机多卡训练通常不用 DP。

### 2.2 DP 示例代码

保存为 `dp_example.py` 后运行：

```bash
python dp_example.py
```

代码如下：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # 1. 构造一个简单二分类任务。
    # x 是输入特征，y 是标签。这里用随机数据演示训练流程。
    x = torch.randn(1024, 10)
    y = (x.sum(dim=1, keepdim=True) > 0).float()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. 创建模型，并把原始模型放到主设备。
    # DataParallel 默认以 cuda:0 作为主设备。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SmallMLP().to(device)

    # 3. 如果有多张 GPU，用 DataParallel 包装模型。
    # 它会在 forward 时自动切分输入，并把模型复制到多张 GPU。
    if torch.cuda.device_count() > 1:
        print(f"Use DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # 4. 定义损失函数和优化器。
    # 注意 optimizer 接收的是包装后的 model.parameters()。
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 5. 普通训练循环不需要特殊处理。
    for epoch in range(3):
        total_loss = 0.0

        for batch_x, batch_y in loader:
            # 5.1 输入数据先移动到主设备。
            # DataParallel 会从主设备把 batch 按 dim=0 切到其他 GPU。
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 5.2 清空上一轮梯度。
            optimizer.zero_grad()

            # 5.3 forward。
            # 内部发生：scatter input -> replicate model -> parallel forward -> gather output。
            logits = model(batch_x)

            # 5.4 在主设备上计算 loss。
            loss = criterion(logits, batch_y)

            # 5.5 backward。
            # 内部发生：各 GPU 计算自己的梯度，然后把梯度累加回主模型。
            loss.backward()

            # 5.6 更新参数。
            # optimizer 更新的是主模型参数；下一次 forward 时再复制到其他 GPU。
            optimizer.step()

            total_loss += loss.item()

        print(f"epoch={epoch}, loss={total_loss / len(loader):.4f}")


if __name__ == "__main__":
    main()
```

### 2.3 DP 每一步的原理

1. 构造数据集

   `TensorDataset(x, y)` 把输入和标签封装成样本集合，`DataLoader` 每次产出一个 batch。DP 的并行维度默认是 batch 的第 0 维，所以输入张量一般要满足第 0 维是样本数。

2. 创建模型并移动到 `cuda:0`

   DP 需要一个主模型。默认情况下主模型在 `cuda:0`，也就是主 GPU。后续每次 forward 时，PyTorch 会把这个主模型复制到其他 GPU 上。

3. 使用 `nn.DataParallel(model)`

   包装后的模型仍然像普通模型一样调用 `model(batch_x)`。区别是 `DataParallel` 在调用内部模型前会自动做这些事：

   ```text
   scatter: 把 batch_x 按 batch 维切成多份
   replicate: 把模型复制到多张 GPU
   parallel_apply: 每张 GPU 独立执行 forward
   gather: 把每张 GPU 的输出收集到主 GPU
   ```

4. forward

   假设 `batch_size=64`，有 4 张 GPU，那么每张 GPU 大约处理 16 条样本。每个模型副本参数相同，但输入数据不同，所以输出也不同。

5. loss

   DP 默认把所有 GPU 的输出收集到 `cuda:0`，所以 loss 通常也在 `cuda:0` 上计算。这也是 GPU 0 容易成为瓶颈的原因之一。

6. backward

   每张 GPU 根据自己那部分数据计算梯度。DP 会把各个副本的梯度累加到主模型参数的 `.grad` 上。最终 optimizer 看到的是汇总后的梯度。

7. optimizer.step()

   优化器只更新主模型的参数。下一轮 forward 时，更新后的主模型会再次复制到其他 GPU。

## 3. DDP: DistributedDataParallel

### 3.1 DDP 的工作流程

DDP 的核心流程如下：

```text
多个 Python 进程，通常每张 GPU 一个进程

rank 0 -> cuda:0 -> 一份模型 -> 一部分数据
rank 1 -> cuda:1 -> 一份模型 -> 一部分数据
rank 2 -> cuda:2 -> 一份模型 -> 一部分数据
rank 3 -> cuda:3 -> 一份模型 -> 一部分数据

每个 rank 独立 forward/backward。
backward 过程中，DDP 自动对梯度做 all-reduce。
all-reduce 后，每个 rank 上的梯度一致。
每个 rank 各自执行 optimizer.step()，参数仍然保持一致。
```

DDP 的关键点：

- 一个进程只控制一张 GPU，减少单进程调度瓶颈。
- 每个进程都有完整模型副本。
- 每个进程只读取自己那份数据，通常通过 `DistributedSampler` 实现。
- 梯度同步发生在反向传播过程中，不需要手动收集所有输出。
- 同步后各进程梯度相同，所以每个进程各自 `optimizer.step()` 后参数仍然相同。

### 3.2 DDP 示例代码

保存为 `ddp_example.py` 后运行：

```bash
torchrun --standalone --nproc_per_node=2 ddp_example.py
```

如果有 4 张 GPU：

```bash
torchrun --standalone --nproc_per_node=4 ddp_example.py
```

代码如下：

```python
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def setup_distributed():
    # torchrun 会为每个进程设置这些环境变量。
    # LOCAL_RANK 表示当前进程使用本机第几张 GPU。
    # RANK 表示当前进程在全局所有进程中的编号。
    # WORLD_SIZE 表示总进程数，通常等于总 GPU 数。
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    # nccl 是 NVIDIA GPU 上最常用的分布式通信后端。
    dist.init_process_group(backend="nccl")

    return local_rank, rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def main():
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 1. 每个进程都构造同样的数据集。
    # 这里固定随机种子，保证所有 rank 生成同一份演示数据。
    # 真正读取哪些样本由 DistributedSampler 决定。
    torch.manual_seed(0)
    x = torch.randn(1024, 10)
    y = (x.sum(dim=1, keepdim=True) > 0).float()
    dataset = TensorDataset(x, y)

    # 2. DistributedSampler 按 rank 切分数据。
    # rank 0 读一部分，rank 1 读另一部分，避免多个进程重复训练同一批样本。
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    # 3. DDP 下 DataLoader 通常不再设置 shuffle=True。
    # 是否打乱数据交给 DistributedSampler 控制。
    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
    )

    # 4. 每个进程创建一份模型，并放到自己的 GPU。
    model = SmallMLP().to(device)

    # 5. 用 DDP 包装模型。
    # device_ids=[local_rank] 表示当前进程只操作这一张 GPU。
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(3):
        # 6. 每个 epoch 调用 set_epoch。
        # 这样 DistributedSampler 在不同 epoch 会使用不同的 shuffle 顺序，
        # 同时保证各 rank 的切分仍然互不重复。
        sampler.set_epoch(epoch)

        total_loss = 0.0

        for batch_x, batch_y in loader:
            # 7. 每个进程只把自己的 batch 移动到自己的 GPU。
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # 8. forward 只在当前进程对应的 GPU 上执行。
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            # 9. backward 时 DDP 自动同步梯度。
            # DDP 会给参数梯度注册 autograd hook。
            # 当某个梯度算出来后，它会参与 all-reduce。
            # all-reduce 结束后，每个 rank 拿到平均后的梯度。
            loss.backward()

            # 10. 每个 rank 都执行 optimizer.step()。
            # 因为每个 rank 的梯度已经一致，所以更新后的参数也一致。
            optimizer.step()

            total_loss += loss.item()

        # 11. 避免多个进程重复打印，只让 rank 0 输出日志。
        if rank == 0:
            print(f"epoch={epoch}, loss={total_loss / len(loader):.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
```

### 3.3 DDP 每一步的原理

1. `torchrun` 启动多个进程

   例如：

   ```bash
   torchrun --standalone --nproc_per_node=4 ddp_example.py
   ```

   会启动 4 个 Python 进程。每个进程执行同一份代码，但环境变量不同：

   ```text
   rank 0: LOCAL_RANK=0, RANK=0, WORLD_SIZE=4
   rank 1: LOCAL_RANK=1, RANK=1, WORLD_SIZE=4
   rank 2: LOCAL_RANK=2, RANK=2, WORLD_SIZE=4
   rank 3: LOCAL_RANK=3, RANK=3, WORLD_SIZE=4
   ```

2. `torch.cuda.set_device(local_rank)`

   让当前进程绑定到指定 GPU。这样 rank 0 只操作 `cuda:0`，rank 1 只操作 `cuda:1`。

3. `dist.init_process_group(backend="nccl")`

   初始化通信组。所有进程加入同一个通信组后，才能执行梯度同步。NVIDIA GPU 训练通常使用 `nccl` 后端。

4. `DistributedSampler`

   DDP 不是把一个 batch 从主进程切到多张 GPU，而是每个进程自己从数据集中取样本。`DistributedSampler` 的作用是按照 `rank` 和 `world_size` 切分数据。

   例如数据集有 8 条样本，`world_size=4`：

   ```text
   rank 0: 样本 0, 4
   rank 1: 样本 1, 5
   rank 2: 样本 2, 6
   rank 3: 样本 3, 7
   ```

   实际训练中还会结合 shuffle，让不同 epoch 的样本顺序变化。

5. `sampler.set_epoch(epoch)`

   如果使用 `DistributedSampler(shuffle=True)`，每个 epoch 开始前都应该调用 `set_epoch(epoch)`。否则每个 epoch 的 shuffle 顺序可能相同，影响训练随机性。

6. 创建模型并移动到当前 GPU

   每个进程都有一份完整模型。DDP 初始化时会同步参数，保证各 rank 的初始参数一致。

7. `DDP(model, device_ids=[local_rank])`

   DDP 会包装模型，并为模型参数注册梯度同步 hook。用户仍然像普通模型一样调用 `model(batch_x)`。

8. forward

   每个 rank 只处理自己的 mini-batch。假设每卡 `batch_size=32`，`world_size=4`，那么一次全局迭代实际处理的样本数是：

   ```text
   global_batch_size = per_gpu_batch_size * world_size
                     = 32 * 4
                     = 128
   ```

9. backward 与 all-reduce

   DDP 的关键发生在 `loss.backward()`。当 autograd 计算出某一层参数的梯度后，DDP 会自动触发通信，把所有 rank 上这层参数的梯度做 all-reduce。

   以 4 张 GPU 为例，每个 rank 先得到自己的梯度：

   ```text
   rank 0: grad_0
   rank 1: grad_1
   rank 2: grad_2
   rank 3: grad_3
   ```

   all-reduce 后，每个 rank 都得到同一个平均梯度：

   ```text
   grad = (grad_0 + grad_1 + grad_2 + grad_3) / 4
   ```

   所以每个 rank 虽然只看了部分数据，但同步后得到的梯度等价于使用更大的 global batch 做了一次训练。

10. `optimizer.step()`

   每个 rank 都执行参数更新。由于同步后的梯度相同，并且初始参数相同，所以更新后的模型参数仍然相同。

11. `rank == 0` 打印日志

   DDP 是多进程程序。如果每个进程都打印日志，会看到重复输出。通常只让主进程，也就是 `rank == 0`，负责打印、保存 checkpoint、写 TensorBoard 日志。

## 4. 使用 Accelerate + DeepSpeed 实现 DDP/ZeRO

Accelerate + DeepSpeed 经常被用来替代手写 DDP 训练脚本。它不是简单地把模型包成 `DistributedDataParallel`，而是：

```text
accelerate launch 负责启动多进程
Accelerator 负责识别分布式环境和当前进程 rank
accelerator.prepare(...) 负责包装 model / optimizer / dataloader
DeepSpeed 负责梯度同步、ZeRO 分片、混合精度、offload 等训练引擎逻辑
```

所以它和 DDP 的关系可以这样理解：

- 裸 DDP：自己写 `torchrun`、`init_process_group`、`DistributedSampler`、`DDP(model)`、`loss.backward()`。
- Accelerate 普通多卡：用 `accelerate launch` + `Accelerator` 自动包装成 DDP。
- Accelerate + DeepSpeed：用 `accelerate launch` + `Accelerator` 自动接入 DeepSpeed 引擎。它仍然是数据并行训练语义，但可以通过 ZeRO 把优化器状态、梯度、参数切分到不同 GPU 上，减少单卡显存。

### 4.1 Accelerate + DeepSpeed 配置

最常见方式是先运行：

```bash
accelerate config
```

在交互式配置里选择使用 DeepSpeed。下面是一个单机 2 卡、ZeRO Stage 2 的示意配置，可以保存为 `accelerate_ds_stage2.yaml`：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
mixed_precision: bf16
num_machines: 1
num_processes: 2
machine_rank: 0
main_training_function: main
use_cpu: false
deepspeed_config:
  zero_stage: 2
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
```

运行命令：

```bash
accelerate launch --config_file accelerate_ds_stage2.yaml accelerator_deepspeed_example.py
```

如果 GPU 不支持 BF16，把 `mixed_precision: bf16` 改成 `fp16` 或 `no`。

### 4.2 Accelerate + DeepSpeed 示例代码

保存为 `accelerator_deepspeed_example.py`：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # 1. 创建 Accelerator。
    # 它会根据 accelerate launch 的配置识别当前是否使用 DeepSpeed、
    # 当前进程的 rank、world_size、device 和混合精度类型。
    accelerator = Accelerator()

    # 2. 构造一个简单二分类任务。
    # 每个进程都会执行这段代码。这里固定随机种子，保证演示数据一致。
    torch.manual_seed(0)
    x = torch.randn(1024, 10)
    y = (x.sum(dim=1, keepdim=True) > 0).float()
    dataset = TensorDataset(x, y)

    # 3. 这里不需要手写 DistributedSampler。
    # accelerator.prepare(loader) 会根据进程数自动处理 dataloader sharding。
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4. 创建普通 PyTorch 模型、优化器和损失函数。
    # 注意这里不要手动 model.to(cuda)，后面交给 accelerator.prepare。
    model = SmallMLP()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # 5. 让 Accelerate 接管训练对象。
    # 在 DeepSpeed 配置下，model 会被包装成 DeepSpeed engine，
    # optimizer 会被包装成 DeepSpeed/Accelerate optimizer，
    # loader 会被包装成按 rank 切分数据的 dataloader。
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    for epoch in range(3):
        total_loss = 0.0

        for batch_x, batch_y in loader:
            # 6. batch 已经被 accelerator 放到了当前进程对应的 device。
            # 不需要写 batch_x = batch_x.to(device)。
            with accelerator.accumulate(model):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                # 7. 不再直接调用 loss.backward()。
                # accelerator.backward 会根据当前后端选择正确的 backward：
                # - 普通 DDP：触发 DDP 梯度同步
                # - DeepSpeed：调用 DeepSpeed engine 的 backward，处理 ZeRO、混合精度和梯度累积
                accelerator.backward(loss)

                # 8. optimizer.step() 仍然保留普通 PyTorch 写法。
                # 在 DeepSpeed 下，这一步会进入 DeepSpeed optimizer/engine，
                # 完成梯度规约、ZeRO 分片状态更新和参数更新。
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.detach().float().item()

        # 9. accelerator.print 只在主进程打印，避免多进程重复输出。
        accelerator.print(f"epoch={epoch}, local_loss={total_loss / len(loader):.4f}")

    # 10. 结束训练时释放分布式资源，并让 tracker 等组件正确收尾。
    accelerator.end_training()


if __name__ == "__main__":
    main()
```

### 4.3 每一步的操作和原理

1. `accelerate launch`

   类似 `torchrun`，它会启动多个 Python 进程，并为每个进程设置分布式训练所需的信息。单机 2 卡时可以理解为：

   ```text
   process 0 -> cuda:0
   process 1 -> cuda:1
   ```

   和手写 DDP 不同的是，用户通常不需要自己读取 `LOCAL_RANK`、`RANK`、`WORLD_SIZE`，也不需要自己调用 `dist.init_process_group`。

2. `Accelerator()`

   `Accelerator` 会读取启动环境和配置文件，判断当前训练类型是普通单卡、DDP、DeepSpeed、FSDP 还是其他后端。使用上面的配置时，它会进入 `distributed_type=DEEPSPEED` 的路径。

3. 构造普通 PyTorch 对象

   在 `accelerator.prepare(...)` 之前，模型、优化器、DataLoader 都是普通 PyTorch 对象。代码保持接近单卡写法，这是 Accelerate 的主要价值。

4. `accelerator.prepare(model, optimizer, loader)`

   这是最关键的一步。它会根据当前后端做不同包装：

   ```text
   model     -> DeepSpeedEngine
   optimizer -> DeepSpeed/Accelerate optimizer wrapper
   loader    -> DataLoaderShard 或类似的数据切分包装
   ```

   因此你不需要手写 `DistributedSampler`。Accelerate 会让不同进程拿到不同 batch，避免所有 GPU 重复训练同一批数据。

5. forward

   每个进程只处理自己拿到的数据。假设 `batch_size=32`、`num_processes=2`，那么每次全局训练大致处理：

   ```text
   global_batch_size = per_process_batch_size * num_processes
                     = 32 * 2
                     = 64
   ```

6. `accelerator.backward(loss)`

   这一步替代 `loss.backward()`。在 DeepSpeed 后端下，它会调用 DeepSpeed engine 的 backward。具体会根据 ZeRO stage 决定同步方式：

   - ZeRO Stage 0 接近普通 DDP，参数、梯度、优化器状态都复制在每张卡上。
   - ZeRO Stage 1 切分优化器状态。
   - ZeRO Stage 2 切分优化器状态和梯度。
   - ZeRO Stage 3 进一步切分模型参数。

7. `optimizer.step()`

   代码上仍然像普通 PyTorch，但实际执行的是被包装后的优化器逻辑。以 ZeRO Stage 2 为例，每个 rank 只保存和更新一部分优化器状态和梯度分片；更新完成后，DeepSpeed 保证各 rank 的参数状态一致。

8. `accelerator.accumulate(model)`

   这是梯度累积的统一写法。如果设置 `gradient_accumulation_steps > 1`，Accelerate 会控制什么时候同步梯度、什么时候真正 step，减少不必要的通信。普通 DDP 下类似于正确使用 `no_sync()`；DeepSpeed 下则交给 DeepSpeed engine 处理。

9. `accelerator.print`

   多进程训练中，普通 `print()` 会被每个进程执行。`accelerator.print()` 只在主进程打印，适合日志输出。

10. 保存 checkpoint

   DeepSpeed ZeRO Stage 1/2 保存模型通常比较接近普通 DDP。ZeRO Stage 3 下，模型参数本身也是分片的，直接取 `state_dict` 可能只得到占位符。常见做法是：

   ```python
   accelerator.wait_for_everyone()
   accelerator.save_state("accelerate_ds_ckpt")
   ```

   如果需要导出完整权重，ZeRO Stage 3 通常需要开启 `zero3_save_16bit_model` 或使用 DeepSpeed 提供的 `zero_to_fp32.py` 做离线合并。

### 4.4 ZeRO 为什么比普通 DDP 省显存

普通 DDP 中，每张 GPU 都保存完整训练状态：

```text
每张 GPU:
  完整参数 parameters
  完整梯度 gradients
  完整优化器状态 optimizer states
```

对 AdamW 来说，优化器状态通常包括一阶动量、二阶动量等，显存开销很大。ZeRO 的思路是把这些状态切开，分摊到不同 GPU：

| 方式 | 参数 | 梯度 | 优化器状态 | 主要收益 |
| --- | --- | --- | --- | --- |
| DDP / ZeRO-0 | 每卡完整保存 | 每卡完整保存 | 每卡完整保存 | 实现简单，显存占用高 |
| ZeRO-1 | 每卡完整保存 | 每卡完整保存 | 按 rank 切分 | 节省优化器状态显存 |
| ZeRO-2 | 每卡完整保存 | 按 rank 切分 | 按 rank 切分 | 进一步节省梯度显存 |
| ZeRO-3 | 按 rank 切分 | 按 rank 切分 | 按 rank 切分 | 显存节省最大，通信更多 |

ZeRO-3 的执行可以理解为：

```text
forward 前：临时 all-gather 当前层需要的参数
forward 后：不再需要的参数重新释放或分片保存
backward 时：计算梯度，并把梯度 reduce-scatter 成分片
step 时：每个 rank 只更新自己负责的参数/优化器状态分片
```

所以 ZeRO-3 能训练更大的模型，但会增加通信和调度开销。模型不大时，ZeRO-2 或普通 DDP 可能更快；模型大到单卡显存吃紧时，ZeRO-3 更有价值。

### 4.5 DeepSpeed 配置文件方式

上面的例子使用 Accelerate 的 DeepSpeed 简化配置。实际项目里也常用 DeepSpeed JSON 配置文件，例如 `ds_zero2.json`：

```json
{
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": "auto",
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "steps_per_print": 2000
}
```

然后在 Accelerate 配置中引用它：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: 1
num_processes: 2
machine_rank: 0
main_training_function: main
use_cpu: false
deepspeed_config:
  deepspeed_config_file: ds_zero2.json
  zero3_init_flag: false
```

注意：使用 `deepspeed_config_file` 时，ZeRO stage、混合精度、梯度累积、gradient clipping 等 DeepSpeed 相关字段应该放在 DeepSpeed JSON 里，不要在 Accelerate YAML 里重复配置。否则这些字段可能被忽略或触发配置冲突。如果 DeepSpeed JSON 里定义了 `optimizer` 或 `scheduler`，代码里可能需要使用 Accelerate 的 `DummyOptim` 或 `DummyScheduler`。如果 JSON 里不定义 optimizer/scheduler，就可以继续使用普通 PyTorch optimizer，代码改动最少。

### 4.6 什么时候用 Accelerate + DeepSpeed

适合使用 Accelerate + DeepSpeed 的场景：

1. 想保留接近单卡 PyTorch 的训练代码，但需要多卡训练。
2. 模型或 optimizer state 太大，普通 DDP 显存不够。
3. 需要 ZeRO Stage 2/3、CPU/NVMe offload、混合精度等能力。
4. 希望同一份训练代码在单卡、DDP、DeepSpeed 之间切换。

不一定需要 DeepSpeed 的场景：

1. 模型较小，普通 DDP 显存和速度都够。
2. 训练逻辑非常简单，只需要标准多卡同步梯度。
3. 不想引入 DeepSpeed 配置和 checkpoint 合并复杂度。

## 5. 使用 Accelerate + DeepSpeed 实现 USP

USP 是 `Unified Sequence Parallel`，也就是统一序列并行。它和前面的 DDP/ZeRO 不是同一层面的并行：

- DDP/ZeRO 主要解决“多个样本如何并行训练”和“参数、梯度、优化器状态如何同步或分片”。
- USP 主要解决“单个超长序列如何切到多张 GPU 上计算”。

在视频 DiT 里，一段视频会被 patchify 成很长的 token 序列：

```text
video latent: [B, C, F, H, W]
patchify 后: [B, S, D]
S = F * H * W 的 patch/token 数量
```

普通 DDP/ZeRO 下，每张 GPU 都要处理完整的 `[B, S, D]`。当 `S` 很大时，attention 的显存和计算会明显上涨。USP 的做法是把同一个样本的序列维切开：

```text
完整序列: [B, S, D]

rank 0: [B, S/4, D]
rank 1: [B, S/4, D]
rank 2: [B, S/4, D]
rank 3: [B, S/4, D]
```

每个 rank 只保留一段 token，但 attention 又需要看到全局上下文，所以 USP 会在 attention 内部做跨 rank 通信。DiffSynth 当前 Wan/MOVA 的 USP 基于 xFuser/yunchang 的 long-context attention 实现，整体思路来自 Ulysses + Ring Attention 的统一序列并行。

### 5.1 USP 和 DDP/ZeRO 的关系

三者可以同时存在，但负责的东西不同：

| 技术 | 并行维度 | 每个 rank 看到的数据 | 主要通信 | 解决的问题 |
| --- | --- | --- | --- | --- |
| DDP | batch 维 | 不同样本 | 梯度 all-reduce | 多样本并行训练 |
| ZeRO | 参数/梯度/优化器状态 | 不同样本，训练状态分片 | all-gather / reduce-scatter | 降低模型训练状态显存 |
| USP | sequence 维 | 同一样本的不同 token 分片 | attention 内通信 + 输出 all-gather | 降低超长序列激活/attention 显存 |

最容易混淆的一点是 dataloader：

- DDP/ZeRO 通常希望不同 rank 读取不同样本。
- USP 希望所有 rank 在同一个 step 读取同一条样本，然后把这条样本的 token 序列切开。

因此 `Accelerate + DeepSpeed + USP` 不能照搬普通 DDP 写法。普通写法会把 `dataloader` 交给 `accelerator.prepare(dataloader)`，Accelerate 会自动按 rank 切数据；但 USP 训练中这样做会让 rank 0、rank 1 读到不同视频，它们就不再是同一个序列的不同分片，attention 通信的语义会错。

正确思路是：

```text
accelerate launch 启动多进程
DeepSpeed/ZeRO 管理模型参数、梯度、优化器状态
USP 初始化 sequence-parallel group
所有 rank 读取同一条样本
模型内部把 token 序列按 rank 切分
attention 内部通信，保证每个 token 能看到全局上下文
末尾 all-gather 拼回完整序列
accelerator.backward(loss) 触发 DeepSpeed backward 和 ZeRO 同步
```

### 5.2 DiffSynth 中的 USP 接入点

以 WanVideo 为例，本仓库里和 USP 相关的核心位置是：

1. 启动脚本

   `examples/wanvideo/model_training/full/Wan2.2-S2V-14B-sequence_parallel.sh` 使用 `accelerate launch` 启动训练，并传入：

   ```bash
   --use_sequence_parallel
   ```

2. DeepSpeed 配置

   `examples/wanvideo/model_training/full/accelerate_config_zero3_noinit.yaml` 使用：

   ```yaml
   distributed_type: DEEPSPEED
   mixed_precision: bf16
   num_processes: 8
   deepspeed_config:
     zero_stage: 3
     train_micro_batch_size_per_gpu: 1
     gradient_accumulation_steps: 1
     zero3_init_flag: false
     zero3_save_16bit_model: true
   ```

   这里显式写 `train_micro_batch_size_per_gpu: 1` 很关键。因为 USP 分支通常不会把 dataloader 交给 `accelerator.prepare()` 做自动切分，DeepSpeed 不能从 dataloader wrapper 推断 micro batch size。

3. Pipeline monkey patch

   `WanVideoPipeline.enable_usp()` 会把 DiT/VACE 的 forward 和 self-attention forward 替换成 USP 版本：

   ```python
   for block in self.dit.blocks:
       block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
   self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
   ```

   这一步的含义是：不重写整个模型结构，只替换关键 forward，让模型在进入 DiT block 前切 sequence，在 block 内使用 sequence-parallel attention，在 block 后把 sequence 拼回来。

4. USP 进程组初始化

   `diffsynth/utils/xfuser/xdit_context_parallel.py` 中的 `initialize_usp` 会初始化 xFuser 的 sequence-parallel 拓扑：

   ```python
   initialize_model_parallel(
       sequence_parallel_degree=dist.get_world_size(),
       ring_degree=1,
       ulysses_degree=dist.get_world_size(),
   )
   ```

   当前配置表示所有进程都放进一个 sequence-parallel group，使用 Ulysses 风格的序列并行，`ring_degree=1`。

5. 序列切分和合并

   `usp_dit_forward` 的核心逻辑是：

   ```python
   x, (f, h, w) = self.patchify(x)

   chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
   chunks = [pad_to_same_length(chunk) for chunk in chunks]
   x = chunks[get_sequence_parallel_rank()]

   for block in self.blocks:
       x = block(x, context, t_mod, freqs)

   x = get_sp_group().all_gather(x, dim=1)
   x = remove_padding(x)
   x = self.unpatchify(x, (f, h, w))
   ```

### 5.3 Accelerate + DeepSpeed + USP 的训练骨架

下面是一个简化后的实现骨架，重点展示和普通 DeepSpeed DDP 写法不同的地方。真实项目里模型可以换成 `WanTrainingModule` 或其他视频 DiT 训练模块。

```python
import os

import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel


def initialize_usp_from_accelerate(accelerator):
    # accelerate launch + Accelerator() 通常已经初始化 torch.distributed。
    # 如果没有初始化，这里兜底初始化一次。
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
    torch.cuda.set_device(local_rank)

    # 初始化 xFuser 的 sequence-parallel group。
    # 单机示例中 world_size == GPU 数。
    init_distributed_environment(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    )
    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=1,
        ulysses_degree=dist.get_world_size(),
    )


def main(num_epochs=3):
    accelerator = Accelerator()
    initialize_usp_from_accelerate(accelerator)

    dataset = build_dataset()

    # 关键点：USP 下不要让 accelerate shard dataloader。
    # 所有 rank 必须在同一个 step 读到同一个样本。
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=0,
    )

    model = build_training_model(device=accelerator.device)

    # 关键点：在 DeepSpeed 包装模型前启用 USP monkey patch。
    # 对 WanVideoPipeline 来说就是 model.pipe.enable_usp()。
    model.pipe.enable_usp()

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # 和普通 DDP/DeepSpeed 不同：这里不 prepare loader。
    # model/optimizer/scheduler 交给 DeepSpeed 管，loader 保持所有 rank 同步读取同一条数据。
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    for epoch in range(num_epochs):
        for data in loader:
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(model, accelerator)


if __name__ == "__main__":
    main()
```

这段代码和普通 Accelerate + DeepSpeed 的差异集中在三处：

1. 需要初始化 USP 的 sequence-parallel group。
2. 需要在模型进入 DeepSpeed 包装前执行 `model.pipe.enable_usp()`。
3. 不把 `loader` 传给 `accelerator.prepare()`，以保证所有 rank 读取同一条样本。

### 5.4 USP forward 的计算原理

以视频 DiT 为例，普通 forward 是：

```text
完整视频 latent
  -> patchify 得到完整 token 序列 [B, S, D]
  -> 所有 DiT blocks 都处理完整 [B, S, D]
  -> unpatchify 还原视频 latent
```

USP forward 是：

```text
所有 rank 读取同一条视频
  -> 每个 rank 都得到完整 latent
  -> patchify 得到完整 token 序列 [B, S, D]
  -> 按 sequence 维切分
       rank 0: [B, S0, D]
       rank 1: [B, S1, D]
       rank 2: [B, S2, D]
       rank 3: [B, S3, D]
  -> 每个 block 只持有本 rank 的 token 分片
  -> self-attention 通过 xFuserLongContextAttention 做跨 rank 通信
  -> block 结束后仍保持 sequence 分片
  -> 最后 all-gather 拼回完整 [B, S, D]
  -> unpatchify 还原视频 latent
```

attention 是 USP 最关键的地方。局部 token 分片不能只看局部 token，否则结果会变成“每张卡只关注自己那段视频 patch”，和原始全局 attention 不等价。因此 `usp_attn_forward` 会：

```text
本地 x 分片
  -> 计算本地 q/k/v
  -> RoPE 根据全局 position offset 取本 rank 对应的位置编码
  -> xFuserLongContextAttention 内部跨 rank 通信
  -> 得到等价于全序列 attention 的本地输出分片
  -> 输出仍是 [B, S_rank, D]
```

最后再用 sequence-parallel group 做 `all_gather`，把各 rank 的 `[B, S_rank, D]` 拼回完整 `[B, S, D]`。

### 5.5 为什么 USP 下 dataloader 不能普通 shard

普通 DDP 的目标是扩大 batch：

```text
rank 0: 样本 A
rank 1: 样本 B
rank 2: 样本 C
rank 3: 样本 D
```

USP 的目标是切同一个样本的 sequence：

```text
rank 0: 样本 A 的第 0 段 token
rank 1: 样本 A 的第 1 段 token
rank 2: 样本 A 的第 2 段 token
rank 3: 样本 A 的第 3 段 token
```

如果误把 dataloader 交给 `accelerator.prepare(dataloader)`，很容易变成：

```text
rank 0: 样本 A 的第 0 段 token
rank 1: 样本 B 的第 1 段 token
rank 2: 样本 C 的第 2 段 token
rank 3: 样本 D 的第 3 段 token
```

这时 attention 通信会把不同样本的 token 混在一起，语义是错误的。更严重时，不同样本的分辨率、帧数或 sequence length 不一致，还可能导致 collective 通信 shape 不一致而 hang 住。

所以 USP 训练通常要求：

1. 每个 rank 在同一步读取同一条样本。
2. `batch_size` 通常设为 1。
3. 视频规格固定，或者至少保证每个 rank 切分后的 shape 一致。
4. DeepSpeed 配置里显式写 `train_micro_batch_size_per_gpu`。

### 5.6 和 ZeRO-3 组合时的执行顺序

`Accelerate + DeepSpeed + USP + ZeRO-3` 的一次 step 可以理解成：

```text
1. 所有 rank 读取同一条样本
2. Pipeline 前处理得到 latent / text embedding / audio embedding 等输入
3. DiT patchify 得到 [B, S, D]
4. USP 按 sequence 维切分，每个 rank 保留 [B, S_rank, D]
5. ZeRO-3 在需要某层参数时 all-gather 参数分片
6. USP attention 在 q/k/v 上做 sequence-parallel 通信
7. block 输出仍保持本地 sequence 分片
8. DiT 末尾 all-gather sequence，恢复完整输出
9. loss 计算
10. accelerator.backward(loss)
11. DeepSpeed/ZeRO 处理梯度 reduce-scatter、参数/优化器状态分片更新
12. optimizer.step()
```

这里有两类通信同时存在：

- USP 通信：发生在 forward/backward 的 attention 计算中，围绕 sequence/token 分片。
- ZeRO 通信：发生在参数 all-gather、梯度 reduce-scatter、optimizer state 更新中，围绕参数和梯度分片。

它们是正交的，但通信量会叠加。因此 USP + ZeRO-3 更省显存，但不一定比普通 DDP 更快；它主要用于“单卡放不下完整长序列”或“长视频训练 attention 显存过高”的场景。

### 5.7 常见坑

1. `--use_sequence_parallel` 需要训练脚本真正处理

   启动脚本传参只是第一步。训练代码里还需要：

   ```python
   parser.add_argument("--use_sequence_parallel", action="store_true")
   ```

   并在 launcher 中根据它切换到 USP 分支：初始化 xFuser group、调用 `model.pipe.enable_usp()`、避免 prepare dataloader。

2. 不要让不同 rank 读不同样本

   USP 不是数据并行。所有 rank 必须合作计算同一个样本的不同 sequence 分片。

3. sequence length 要能被 world size 合理切分

   有些实现会自动 padding，有些路径会直接 assert。比如 S2V 路径中存在 `x.shape[1] % world_size == 0` 的约束。固定 `height/width/num_frames` 时要提前确认 token 数能被 `num_processes` 整除。

4. 保存 checkpoint 仍按 DeepSpeed 规则处理

   USP 不改变参数保存语义。用了 ZeRO-3 时，checkpoint 仍然遵循 DeepSpeed ZeRO-3 的分片保存/合并规则。

5. USP 更偏向长序列显存优化，不是通用加速开关

   序列越长，USP 越有价值；序列较短时，跨 rank attention 通信可能抵消收益。

## 6. DP 和 DDP 的关键区别

| 对比项 | DP | DDP |
| --- | --- | --- |
| 进程模型 | 单进程多线程 | 多进程，通常一张 GPU 一个进程 |
| 数据分发 | 主 GPU scatter 一个 batch | 每个进程通过 sampler 读取自己的数据 |
| 模型副本 | 每次 forward 时复制到多卡 | 每个进程长期持有一份模型 |
| 输出处理 | gather 到主 GPU | 不需要 gather 输出 |
| 梯度同步 | 汇总到主模型 | backward 中 all-reduce |
| GPU 0 压力 | 较大 | 相对均衡 |
| 性能 | 通常较差 | 通常更好 |
| 多机训练 | 不适合 | 支持 |
| 推荐程度 | 只适合简单实验 | 推荐用于正式训练 |

## 7. 常见注意事项

1. DDP 下不要让所有 rank 训练完全相同的数据

   应该使用 `DistributedSampler` 或等价的数据切分逻辑。否则多卡只是在重复训练同一批数据，不能真正扩大 batch。

2. DDP 下保存 checkpoint 通常只在 rank 0 做

   因为所有 rank 的参数一致，没必要每个进程都保存一份。

3. DDP 的学习率通常要结合 global batch 调整

   如果单卡 batch 是 32，8 卡训练的 global batch 是 256。global batch 变大后，学习率和 warmup 策略可能需要重新设置。

4. DP/ DDP 都不是模型并行

   数据并行要求每张 GPU 能放下一份完整模型。如果单张 GPU 放不下模型，需要考虑模型并行、流水线并行、张量并行、ZeRO、FSDP 或 DeepSpeed 等方案。

5. DDP 中 `model.module`

   DDP 包装后，原始模型在 `model.module` 里。保存权重时常见写法是：

   ```python
   if rank == 0:
       torch.save(model.module.state_dict(), "model.pt")
   ```

## 8. 一句话总结

DP 是“一个进程把数据和模型临时分到多张卡，再把结果收回来”；DDP 是“每张卡一个进程，各自训练自己的数据，并在反向传播时同步梯度”；Accelerate + DeepSpeed 是“用 Accelerate 管理启动、设备、数据切分和训练循环包装，用 DeepSpeed 在数据并行基础上加入 ZeRO 分片、混合精度和 offload”；USP 是“让多张卡共同处理同一个样本的不同 sequence 分片”。实际项目中，小模型优先 DDP 或 Accelerate 普通多卡，大模型参数/优化器显存吃紧时考虑 DeepSpeed ZeRO，长视频或超长 token 序列显存吃紧时考虑 USP，并可与 DeepSpeed ZeRO 组合。
