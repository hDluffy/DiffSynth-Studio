# Wan2.2-S2V CPU、GPU 与 Feature Cache 优化说明

## 1. 文档目的

本文记录 Wan2.2-S2V-14B 多机训练排查期间完成的主要改进、问题根因、实现原理、训练语义影响和容量评估，重点包括：

1. 可变帧数采样与音视频严格对齐；
2. Feature Cache 的构建、校验和断点续算；
3. ZeRO-3 基础权重与 resume 权重的低内存加载；
4. Gradient checkpoint CPU offload 优化；
5. RoPE 在 checkpoint 中的重复保存问题；
6. Transformer Block 与 Audio Injector checkpoint 合并；
7. 分布式内存阶段日志和多节点日志隔离；
8. 113 帧实测结果以及 129 帧训练容量评估。

默认讨论环境：

    模型：Wan2.2-S2V-14B
    精度：BF16
    并行：2 节点 × 8 GPU，共 16 rank
    GPU：NVIDIA A800 80GB
    DeepSpeed：ZeRO Stage 3
    batch：每 rank 1 个样本
    空间上限：max_pixels=589824，典型 latent 为 72×128

## 2. 优化结果概览

| 问题 | 根因 | 改进 | 是否改变训练数学逻辑 |
|---|---|---|---|
| 每个 rank 加载完整 14B 权重 | ZeRO-3 分片前，每个进程实体化完整 state dict | 只有 global rank 0 读取参数，其他 rank 保留 key 骨架 | 否 |
| 加载后 CPU RSS 不下降 | Python/glibc allocator 保留已释放页面 | 删除临时 state dict，执行 GC 和 malloc trim | 否 |
| forward 时约 125 GiB/rank | non-reentrant checkpoint 与 save_on_cpu 保存层内激活 | S2V offload 改用 reentrant checkpoint | 否 |
| RoPE 约 107 GiB/rank 的重复副本 | 约 2.68 GiB RoPE 被作为 40 层显式输入保存 | RoPE 通过只读闭包共享 | 否 |
| Audio Injector 额外保存中间 hidden state | Transformer 和 Audio Injector 分成两个 checkpoint | 合并为一个复合 checkpoint | 否 |
| 多节点日志出现 NUL 或覆盖 | 共享目录中多个节点写同一文件 | 主节点和远端节点使用不同日志文件 | 否 |
| 视频长度不满足 16n+1 | 视频时长换算不一定得到合法帧数 | floor、ceil、nearest 取整与补帧上限 | 会改变样本裁剪或补帧，但符合配置 |
| 音频和视频持续时间不一致 | 抽帧后音频仍保留原始长度 | 按最终帧数和 FPS 裁剪或补齐 | 会改变输入音频长度，这是正确对齐行为 |

修复后的 113 帧训练已经经过：

    first_step_after_forward
    first_step_after_backward
    first_step_after_optimizer_step

说明 ZeRO-3 加载、cache 输入、forward、backward 和 optimizer step 已形成完整闭环。

## 3. 可变帧数采样

### 3.1 合法帧数

当前 S2V 约束为：

    frame_count_stride = 16
    frame_count_remainder = 1
    frames = 16n + 1

合法值包括：

    81、97、113、129、145……

129 是合法值：

    129 = 16 × 8 + 1

### 3.2 映射策略

视频按目标 FPS 换算后得到 available frames。系统计算：

    floor_count：不大于 available 的最近合法值
    ceil_count：不小于 available 的最近合法值

支持：

- floor：始终向下截断；
- ceil：始终向上补帧；
- nearest：选择距离最近的合法值，距离相同时向下，避免无必要的重复帧。

在合法值 113 和 129 之间：

| available | nearest 结果 | 操作 |
|---:|---:|---|
| 113 | 113 | 不处理 |
| 118 | 113 | 截断 5 帧 |
| 121 | 113 | 平局时向下 |
| 122 | 129 | 尾帧补 7 帧 |
| 128 | 129 | 尾帧补 1 帧 |
| 135 | 129 | 截断 6 帧 |

max_frame_padding 限制最大补帧数量。若需要的补帧量超过限制，数据处理会显式报错。

### 3.3 持续时间

首帧位于 0 秒，因此 N 帧覆盖：

    duration = (N - 1) / FPS

16 FPS 下：

    81 帧  -> 5 秒
    97 帧  -> 6 秒
    113 帧 -> 7 秒
    129 帧 -> 8 秒

## 4. 音视频严格对齐

音频长度根据最终训练帧数计算：

    target_duration = (num_frames - 1) / video_frame_rate
    target_samples = round(target_duration × audio_sample_rate)

129 帧、16 FPS、16 kHz 时：

    target_duration = 8 秒
    target_samples = 128000

处理规则：

- 音频过长：从尾部裁剪；
- 音频过短：从尾部补零；
- strict：超过容差报错；
- trim_pad：执行裁剪或补齐；
- max_audio_padding_seconds：限制最大补零长度；
- max_audio_trimming_seconds：限制最大裁剪长度。

cache 中记录：

    sample_num_frames
    sample_duration_seconds
    audio_num_samples
    audio_sample_rate
    video_frame_rate
    video_sampling_info

训练阶段不需要根据文件名或原始媒体时长重新推断。

## 5. Feature Cache 改进

### 5.1 预计算目的

原始数据路径包含：

    视频/音频解码
    抽帧、resize、音视频对齐
    VAE、文本和音频特征编码
    DiT 训练

Feature Cache 提前执行冻结模块，训练阶段主要执行：

    读取预计算 tensor
    采样 timestep/noise
    DiT forward/backward
    optimizer step

收益：

- 降低 CPU 解码压力；
- 避免冻结 VAE 在可变长度 ZeRO-3 rank 上执行次数不一致；
- 降低训练期 GPU 峰值；
- 在构建阶段提前暴露坏数据和对齐错误；
- 保存 sample_num_frames，为后续长度分桶提供依据。

### 5.2 Manifest 和续算

cache 目录中的 _cache_manifest.json 记录：

    构建状态、配置快照、metadata 哈希
    cache 文件数、帧数分布

使用 --require_cache_manifest 时检查：

- manifest 是否存在；
- status 是否为 complete；
- 文件数是否匹配；
- 时间和空间配置是否与 cache 构建配置一致。

断点续算只复用配置和 metadata 哈希一致的成功样本。写文件使用临时文件后原子替换，避免中断留下不完整文件。

### 5.3 当前 cache

当前完整 cache：

    /data-training/train_data_5s/cache_s2v_f81-113_16n1_fps16_10s

实际帧数分布：

    {"113": 32}

虽然配置范围是 81～113，但 smoke 数据全部为 113 帧，并非混合长度。

129 帧需要重新构建独立 cache，例如：

    /data-training/train_data_5s/cache_s2v_f81-129_16n1_fps16_10s

不能只把训练参数改为 129 后继续使用现有 113 帧 cache。

## 6. ZeRO-3 权重加载优化

### 6.1 原问题

ZeRO-3 最终分片参数、梯度和优化器状态，但旧路径在分片前让每个 rank 读取完整 checkpoint：

    state_dict = load_state_dict(path)

14B BF16 权重粗略为：

    14B × 2 bytes ≈ 28 GB

16 rank 重复加载理论上可达：

    28 GB × 16 ≈ 448 GB

resume checkpoint 还可能再次产生类似峰值。

### 6.2 初始化阶段即分片

模型在 deepspeed.zero.Init 上下文中创建。Parameter 构造时即进入 ZeRO-3 管理，而不是每个进程先创建完整 14B 模型再切分。

### 6.3 只有 global rank 0 实体化参数

当前 state dict 结构：

    global rank 0：
        {"layer.weight": Tensor(...)}

    其他 rank：
        {"layer.weight": None}

所有 rank 必须保留相同 key。Transformers 根据 key 是否存在决定是否进入对应的 GatheredParameters collective。若 rank 0 有 key 而其他 rank 没有，collective 顺序不一致，可能产生 NCCL hang 或 timeout。

### 6.4 分发过程

Transformers 按模块执行：

    1. 所有 rank 聚合当前模块的参数分片；
    2. global rank 0 从 checkpoint 写入完整参数；
    3. 离开 GatheredParameters 上下文；
    4. DeepSpeed 将参数重新分片到所有 rank。

示例：

    checkpoint W = [10, 11, 12, 13, 14, 15, 16, 17]
    world size = 4

    rank 0 -> [10, 11]
    rank 1 -> [12, 13]
    rank 2 -> [14, 15]
    rank 3 -> [16, 17]

非零 rank 不需要从磁盘读取完整 W。

### 6.5 Buffer

ZeRO-3 分片 Parameter，但普通 Buffer 由 Transformers 直接 copy。因此：

    Parameter：仅 global rank 0 实体化
    Buffer：每个 rank 实体化

Buffer 通常远小于参数，不会重新引入大峰值。

### 6.6 基础模型与 resume

基础模型和 resume 权重均使用相同加载策略。resume 时可为 checkpoint key 添加 pipe.dit. 等前缀，并检查 unexpected keys。

当前 --resume_from_checkpoint 只恢复模型权重，不恢复：

    optimizer、scheduler、global step
    DataLoader 位置、RNG 状态

因此其含义是“从指定权重继续训练”，不是完整 DeepSpeed engine 断点恢复。

### 6.7 主动归还 CPU 内存

加载完成或失败时执行：

    del state_dict
    gc.collect()
    malloc_trim(0)

作用：

- del：解除临时 checkpoint tensor 引用；
- gc.collect：处理循环引用和延迟回收；
- malloc_trim：Linux 上尽量将 glibc arena 空闲页面归还系统。

不会释放仍被模型引用的参数，也不会改变权重值。

### 6.8 训练语义

ZeRO-3 改造只改变：

    哪个 rank 读取完整 checkpoint
    完整参数临时放在哪里
    何时释放临时 state dict

不改变最终参数值、forward、backward、loss 或优化器公式。

## 7. Gradient Checkpoint Offload

### 7.1 两个选项

--use_gradient_checkpointing：

- 不保存层内大部分激活；
- backward 时重算 forward；
- checkpoint 输入主要留在 GPU。

--use_gradient_checkpointing_offload：

- 同样进行重计算；
- 额外使用 save_on_cpu；
- 将跨越 forward/backward 保存的 tensor 移到 CPU；
- 降低 GPU 压力，但增加 CPU 内存和 PCIe 传输。

### 7.2 non-reentrant 问题

旧 offload 使用：

    checkpoint(..., use_reentrant=False)

non-reentrant 的第一次 forward 建立 autograd 图。与 save_on_cpu 组合后，Attention、FFN 和 Audio Injector 为 backward 保存的层内 tensor 会被搬到 CPU。

长序列 S2V 在首个 forward 内使每 rank CPU RSS 从约 1.6 GiB 增长到约 120～125 GiB。

### 7.3 reentrant 原理

当前 S2V offload 使用：

    checkpoint(..., use_reentrant=True)

第一次 forward 在 no_grad 下运行，不保存层内 autograd 中间激活；backward 时重新执行函数并建立图。因此 save_on_cpu 主要保存显式输入。

第一次 forward 的 no_grad 不会冻结参数。参数梯度在 backward 重计算阶段生成。

## 8. RoPE 重复保存优化

### 8.1 RoPE 原理

RoPE 对 Attention 的 Query 和 Key 做位置相关旋转。将两个实数通道视为复数：

    z = x0 + i×x1
    z' = z × exp(i×theta)

视频 3D RoPE 将 head 通道分给时间、高度和宽度。S2V 还区分生成视频、参考图和 motion token。source_id_local 给参考图加入来源相位，避免不同来源但局部坐标相同的 token 完全重合。

### 8.2 113 帧 RoPE 大小

典型 latent 为 [1, 16, 29, 72, 128]。第一时间位置作为 reference，主视频有 28 个时间位置。空间 patch 为 2×2：

    每个时间位置 = 36 × 64 = 2304 tokens
    主视频 = 28 × 2304 = 64512
    reference = 2304
    motion ≈ 3456
    总计 ≈ 70272 tokens

RoPE 近似形状：

    [1, 70272, 40 heads, 64 complex channels]

complex128 每元素 16 字节：

    大小 ≈ 2.68 GiB

### 8.3 重复保存

旧调用把 RoPE 作为每层显式 checkpoint 输入。reentrant 必须保存显式输入用于重计算：

    2.68 GiB × 40 层 ≈ 107 GiB/rank

### 8.4 最终方案

RoPE 来自 token_states.detach，不需要梯度，也不会在 block 中修改，因此通过外部只读闭包引用，不再作为显式输入。

每次 forward 和 backward 重计算仍使用同一 RoPE，计算仍是：

    Block(x, context, t, RoPE)

只改变引用方式，不改变 Attention 数值。

### 8.5 带梯度条件为什么仍显式传入

context、t_mod、audio_emb_global 和 merged_audio_emb 可能来自可训练模块。

若多个 reentrant checkpoint 通过闭包共享同一带梯度的上游 tensor，可能重复遍历已释放的上游图并报：

    Trying to backward through the graph a second time

最终规则：

    无梯度、只读且很大的 RoPE：闭包共享
    可能需要梯度的条件 tensor：显式 checkpoint 输入

## 9. Audio Injector Checkpoint 合并

### 9.1 原始函数

第 i 层：

    y_i = Transformer_i(x_i, context, t_mod, RoPE)
    x_(i+1) = AudioInjector_i(y_i, audio)

整体：

    x_(i+1) = AudioInjector_i(Transformer_i(x_i))

没有音频注入的层中，AudioInjector 等价于恒等映射。

### 9.2 合并前

旧逻辑：

    y = checkpoint(Transformer, x, context, t_mod, RoPE)
    x_next = checkpoint(AudioInjector, y)

第一个 checkpoint 保存 Transformer 入口 x，第二个保存 Audio Injector 入口 y。

113 帧时单个 hidden state：

    [1, 70272, 5120] × BF16 ≈ 0.67 GiB

40 层额外保存 y，理论上可增加约 26.8 GiB/rank。

### 9.3 合并后

当前逻辑：

    def block_with_audio(x, context, t_mod, audio_global, audio):
        y = Transformer(x, context, t_mod, RoPE)
        return AudioInjector(y, audio_global, audio)

    x_next = checkpoint(block_with_audio, x, context, t_mod, audio_global, audio)

forward 只保存复合函数入口。backward 时：

    恢复 x 和条件
    重算 Transformer，临时得到 y
    立即重算 Audio Injector
    在同一计算图内完成链式反向

y 不再作为第二个 checkpoint 入口长期保存。

### 9.4 梯度等价性

设 y=T(x)，z=A(y)，则：

    z = A(T(x))
    dz/dx = dA/dT × dT/dx

拆分或合并 checkpoint 不改变模块执行顺序和链式法则，也不改变 Transformer、Audio Injector、音频编码器、context 或 timestep 的梯度。

简单例子：

    T(x)=W×x
    A(y,a)=y+U×a
    F(x,a)=W×x+U×a

无论两个 checkpoint 还是一个复合 checkpoint：

    dF/dx=W，dF/dW=x，dF/da=U，dF/dU=a

差异只在中间 y 是否跨越 forward/backward 保存。

## 10. 分阶段内存日志

当前记录：

    training_entry
    optimizer_and_dataloader_created
    before_model_to_device
    after_model_to_device
    after_accelerator_prepare
    first_step_data_loaded
    first_step_after_forward
    first_step_after_backward
    first_step_after_optimizer_step

每条包含 host、global/local rank、CPU RSS/HWM、CUDA allocated/reserved。

它们可区分权重加载、ZeRO prepare、cache 数据、forward 激活、backward 重计算和 optimizer state 问题。

### 10.1 113 帧实测

| 阶段 | CPU RSS/rank | CUDA allocated/rank | CUDA reserved/rank |
|---|---:|---:|---:|
| prepare 后 | 约 1.6 GiB | 约 8.7 GiB | 约 13.5 GiB |
| first forward 后 | 约 28.5 GiB | 约 13.4 GiB | 约 29.0 GiB |
| first backward 后 | 约 2.6 GiB | 约 19.2 GiB | 约 69.7 GiB |
| optimizer step 后 | 约 2.6 GiB | 约 19.2 GiB | 约 69.7 GiB |

稳定运行时 nvidia-smi 约 75.9～76.3 GiB/卡，剩余约 4.8～5.2 GiB。CPU forward 峰值从修复前约 125 GiB/rank 降至约 28.5 GiB/rank。

## 11. 多节点日志隔离

node2 和 node3 若共享仓库并同时重定向到 train-cache.log，会各自维护文件偏移并覆盖同一 inode，产生日志交错、NUL 稀疏区和首错丢失。

当前：

    主节点：train-cache.log
    远端节点：train-cache.log.rank1

日志隔离不影响训练通信或数值。

## 12. 129 帧容量评估

### 12.1 功能支持

129 满足 16n+1，因此：

    帧采样：支持
    VAE 时间约束：支持
    音频对齐：支持
    cache 格式：支持
    DiT 动态长度：支持

但必须先构建真实 129 帧 cache。

### 12.2 Token 增长

    113 帧：latent temporal=29，主视频时间位置=28，总 token≈70272
    129 帧：latent temporal=33，主视频时间位置=32，总 token≈79488
    增长：79488/70272≈1.131，即约 13.1%

Flash Attention 避免保存完整的二次方 Attention 矩阵，但 Q/K/V、FFN 激活、workspace 和 checkpoint 输入仍随序列增长。

### 12.3 CPU

按 113 帧 28.5 GiB/rank 线性估算：

    129 帧 ≈ 32.2 GiB/rank
    单节点 8 rank ≈ 258 GiB

1 TiB 主机预计安全，前提是继续使用 RoPE 去重和 reentrant 修复。

### 12.4 GPU

113 帧已经约 76 GiB/卡，仅余约 5 GiB。129 帧同分辨率的粗略峰值可能达到 82～85 GiB/卡。

因此在以下配置下有较高 CUDA OOM 风险：

    max_pixels=589824
    典型 576×1024
    ZeRO-3
    optimizer/parameter 不做 CPU offload
    A800 80GB

不能声明为已支持且保证不 OOM。

### 12.5 推荐

方案 A，优先降低空间分辨率：

    589824 / 1.131 ≈ 521000 pixels
    推荐 512×896 = 458752 pixels

129 帧、512×896 的 token 总量预计低于当前稳定的 113 帧、576×1024。

方案 B，保持空间分辨率，启用 ZeRO-3 optimizer CPU offload：

    zero_stage: 3
    zero3_init_flag: true
    offload_optimizer_device: cpu
    offload_param_device: none

代价是 CPU 内存、PCIe 流量增加和 optimizer step 变慢，仍需真实 smoke test。

长期方案包括紧凑 RoPE 广播存储、S2V sequence/context parallel，以及 DeepSpeed bucket 调优。这些尚未完成验证。

### 12.6 验收标准

129 帧至少完成：

    first_step_data_loaded
    first_step_after_forward
    first_step_after_backward
    first_step_after_optimizer_step
    连续第二个 step

同时观察 CUDA allocated/reserved、nvidia-smi 峰值和 CPU RSS/HWM。建议至少保留 5～8 GiB CUDA 工程余量。

## 13. 正确性与边界

### 13.1 不改变训练定义

以下属于实现级优化：

    ZeRO-3 rank 0 权重实体化
    state dict 主动释放
    reentrant checkpoint
    RoPE 改为只读闭包引用
    Transformer 与 Audio Injector checkpoint 合并
    日志隔离

它们不改变 loss、forward 顺序、模型参数、学习率、优化器公式、音频注入公式、RoPE 数值或 checkpoint 权重格式。

### 13.2 有意改变输入

以下由配置控制，并会改变输入样本：

    视频向 16n+1 取整
    有限尾帧复制
    多余视频帧截断
    音频裁剪或补齐

这是为了保证合法时间长度和严格音视频同步。

### 13.3 尚未覆盖

- 当前完整 smoke cache 全为 113 帧，尚未长期实测 81/97/113/129 混合训练；
- 长度分桶尚未实现；
- ZeRO-3 可变长度原始数据训练仍应先预计算 cache；
- 129 帧同分辨率尚未完成真实 backward/optimizer 验证；
- resume 不是完整 optimizer/scheduler 状态恢复。

## 14. 相关代码和测试

主要文件：

    diffsynth/core/data/operators.py
    diffsynth/core/data/unified_dataset.py
    diffsynth/core/gradient/gradient_checkpoint.py
    diffsynth/core/loader/file.py
    diffsynth/core/loader/model.py
    diffsynth/diffusion/runner.py
    diffsynth/diffusion/training_module.py
    diffsynth/models/wan_video_dit_s2v.py
    diffsynth/pipelines/wan_video.py
    examples/wanvideo/model_training/train.py
    script/launch_wan22_s2v_multinode.sh

测试：

    tests/test_gradient_checkpoint_offload.py
    tests/test_zero3_memory_efficient_loading.py

当前结果：

    Ran 17 tests
    OK

## 15. 最终结论

当前已完成：

    可变帧合法化与音视频严格对齐
    cache 预计算、manifest、校验和断点续算
    ZeRO-3 低 CPU 内存加载
    activation checkpoint CPU offload
    RoPE 重复保存消除
    Audio Injector checkpoint 合并
    分阶段内存诊断和多节点日志隔离

113 帧、典型 576×1024、16 卡 A800 80GB 已实测通过完整训练 step。

129 帧在代码和数据语义上受支持，CPU 内存预计安全；相同空间分辨率下 GPU 只有约 5 GiB 余量，存在较高 OOM 风险。推荐优先使用 512×896，或增加 ZeRO-3 optimizer CPU offload，并以真实 129 帧 cache 完成至少两个完整 step 后再用于正式训练。
