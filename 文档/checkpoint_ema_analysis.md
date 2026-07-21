# WanVideo 训练保存逻辑与 EMA 支持分析

本文档基于 `examples/wanvideo/model_training/train.py` 及其调用的通用训练组件分析训练过程中的模型保存逻辑，并给出增加 EMA（Exponential Moving Average）能力的实现流程。

## 当前保存调用链

`train.py` 本身不直接保存权重，而是完成训练对象组装：

1. `wan_parser()` 通过 `add_general_config()` 注入通用参数，包括 `--output_path`、`--remove_prefix_in_ckpt`、`--save_steps`、`--resume_from_checkpoint`。
2. 主流程构造 `WanTrainingModule`，其中 `resume_from_checkpoint()` 只加载已有权重文件。
3. 主流程构造 `ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, ...)`。
4. `launcher_map[args.task](...)` 对训练任务调用 `launch_training_task()`。
5. `launch_training_task()` 在每个 batch 完成 `backward -> optimizer.step -> scheduler.step -> optimizer.zero_grad` 后调用 `model_logger.on_step_end(...)`。

实际保存代码位于：

- `diffsynth/diffusion/runner.py`
- `diffsynth/diffusion/logger.py`
- `diffsynth/diffusion/training_module.py`

## 当前保存时机

训练任务保存逻辑由 `--save_steps` 控制：

- `--save_steps` 不为 `None`：`ModelLogger.on_step_end()` 每次被调用都会递增内部 `num_steps`，当 `num_steps % save_steps == 0` 时保存 `step-{num_steps}.safetensors`。
- `--save_steps` 不为 `None` 且训练结束时最后一步不是保存点：`ModelLogger.on_training_end()` 会额外保存一次 `step-{num_steps}.safetensors`。
- `--save_steps` 为 `None`：每个 epoch 结束后保存 `epoch-{epoch_id}.safetensors`，`epoch_id` 从 0 开始。

注意：`ModelLogger.num_steps` 是 logger 内部计数，不会从 checkpoint 恢复；使用梯度累积时，它按 `on_step_end()` 调用次数计数，不是显式持久化的全局 optimizer step。

## 当前保存内容

`ModelLogger.save_model()` 的核心流程：

1. `accelerator.wait_for_everyone()` 等待所有进程。
2. `accelerator.get_state_dict(model)` 获取当前模型 state dict。
3. 仅主进程继续保存。
4. `accelerator.unwrap_model(model).export_trainable_state_dict(...)` 过滤出 `requires_grad=True` 的可训练参数。
5. 如果配置了 `remove_prefix_in_ckpt`，从导出 key 前缀中移除该字符串。
6. 执行 `state_dict_converter`，默认是 identity。
7. `accelerator.save(state_dict, path, safe_serialization=True)` 保存 safetensors 文件。

因此当前 checkpoint 是“可训练参数权重快照”，不是完整训练状态。它不包含 optimizer、scheduler、随机数状态、epoch、step，也不包含 EMA shadow weights。

LoRA 训练时，可训练参数通常就是注入后的 LoRA 参数；全参或部分模块训练时，保存的是对应 `requires_grad=True` 的参数。`--remove_prefix_in_ckpt` 默认值是 `pipe.dit.`，保存时会移除此前缀，恢复时再加回。

## 当前恢复逻辑

`WanTrainingModule.__init__()` 中调用 `self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)`。其逻辑是：

1. `load_state_dict(path)` 加载权重文件。
2. 如果设置了 `remove_prefix_in_ckpt`，对 checkpoint 中所有 key 加回此前缀。
3. `self.load_state_dict(state_dict, strict=False)` 加载到当前模块。
4. 如果存在 unexpected keys，直接报错；missing keys 允许存在。

这不是完整意义上的断点续训。它只恢复模型权重，不恢复 optimizer、scheduler、logger step、epoch 或数据采样状态。

另外，`resume_from_checkpoint()` 在 `switch_pipe_to_training_mode()` 之前调用。LoRA 续训应使用 `--lora_checkpoint`，因为 LoRA adapter 是在 `switch_pipe_to_training_mode()` 内注入后再加载的。

## 是否支持 EMA

当前不支持 EMA。

依据：

- 参数解析中没有 `--use_ema`、`--ema_decay`、`--ema_checkpoint` 等 EMA 参数。
- 训练循环中没有维护 EMA shadow 参数，也没有在 `optimizer.step()` 后更新 EMA。
- 保存逻辑只保存当前模型的可训练参数，没有 EMA 分支或 `*-ema.safetensors` 输出。
- 仓库中未发现 EMA/ExponentialMovingAverage 相关实现。

## EMA 实现流程建议

建议先实现“可训练参数 EMA”，与当前保存语义保持一致：当前保存什么参数，EMA 就维护并保存同一组参数。这样可同时覆盖 LoRA 训练、单模型全参训练和部分模块训练。

### 1. 增加命令行参数

在 `diffsynth/diffusion/parsers.py` 增加 EMA 配置，例如：

- `--use_ema`：启用 EMA。
- `--ema_decay`：EMA 衰减系数，常用默认值可设为 `0.999` 或 `0.9999`。
- `--ema_update_after_step`：可选，前若干步不更新 EMA。
- `--ema_update_every`：可选，每隔多少个有效 optimizer step 更新一次。
- `--ema_device`：可选，`cpu` 或当前训练设备。WanVideo 全参训练模型很大，CPU EMA 更省显存但更新更慢。
- `--ema_checkpoint`：可选，从已有 EMA 权重初始化 shadow weights。
- `--save_ema` 或 `--save_ema_alongside`：控制是否额外保存 EMA 文件。

### 2. 增加 EMA 管理类

可新增 `diffsynth/diffusion/ema.py`，职责包括：

- 初始化时读取 unwrapped model 的可训练参数名。
- 以 fp32 shadow tensor 保存 EMA 权重。
- `update(model)` 在每次有效 optimizer step 后执行：

```python
shadow[name].mul_(decay).add_(param.detach().float(), alpha=1 - decay)
```

- `state_dict()` 返回 shadow weights，key 使用模型原始参数名。
- `load_state_dict()` 支持从 `--ema_checkpoint` 加载，并处理 `remove_prefix_in_ckpt` 的反向映射。

为了保持和现有保存逻辑一致，EMA 管理类应只维护 `requires_grad=True` 参数，不维护冻结模型参数。

### 3. 在训练循环中更新 EMA

在 `diffsynth/diffusion/runner.py` 的 `launch_training_task()` 中：

1. `accelerator.prepare(...)` 之后创建 EMA 对象，确保参数设备和包装状态已稳定。
2. 在 `optimizer.step()` 之后、保存之前更新 EMA。
3. 使用梯度累积时，建议只在 `accelerator.sync_gradients` 为真时更新 EMA，避免 micro step 重复更新。
4. 更新顺序建议为：

```python
optimizer.step()
scheduler.step()
if ema is not None and accelerator.sync_gradients:
    ema.update(accelerator.unwrap_model(model))
optimizer.zero_grad()
model_logger.on_step_end(accelerator, model, save_steps, loss=loss, ema=ema)
```

如果保留当前 `ModelLogger.num_steps` 语义，需要明确 EMA 的 step 计数是有效 optimizer step 还是 logger step。推荐 EMA 内部单独维护 `ema.num_updates`。

### 4. 扩展保存逻辑

在 `ModelLogger` 中增加 EMA 保存分支，保留当前非 EMA 权重输出：

- 当前权重继续保存为 `step-{n}.safetensors` 或 `epoch-{n}.safetensors`。
- EMA 权重额外保存为 `step-{n}-ema.safetensors` 或 `epoch-{n}-ema.safetensors`。
- EMA 保存同样执行 `remove_prefix_in_ckpt` 和 `state_dict_converter`。

可抽出一个通用保存函数，例如：

```python
def save_state_dict(self, accelerator, state_dict, model, file_name):
    if accelerator.is_main_process:
        state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
            state_dict,
            remove_prefix=self.remove_prefix_in_ckpt,
        )
        state_dict = self.state_dict_converter(state_dict)
        accelerator.save(state_dict, path, safe_serialization=True)
```

当前模型权重使用 `accelerator.get_state_dict(model)`；EMA 权重使用 `ema.state_dict()`。

### 5. 扩展恢复逻辑

保留现有 `--resume_from_checkpoint` 行为用于恢复当前模型权重。新增 `--ema_checkpoint` 用于恢复 EMA shadow weights：

- 如果传入 `--ema_checkpoint`，从该文件加载 EMA shadow。
- 如果未传入但启用了 EMA，则从当前模型可训练参数初始化 shadow。
- 如果 `remove_prefix_in_ckpt` 不为空，加载 EMA checkpoint 时需要像当前恢复逻辑一样给 key 加回前缀。

需要注意：即便增加 EMA checkpoint，当前训练仍然不是完整断点续训，除非另外保存 optimizer、scheduler、epoch、step 和 RNG 状态。

### 6. 分布式与 offload 限制

建议第一阶段明确支持普通单卡/DDP LoRA 或单模型训练；对以下场景加保护或文档限制：

- DeepSpeed ZeRO-3：参数可能分片，逐 step 维护完整 EMA 会有额外通信和内存成本。
- `enable_model_cpu_offload`：参数设备会动态变化，EMA 更新需要确认 offload 后参数仍可被一致访问。
- 全参 WanVideo 训练：EMA 会额外占用一份可训练参数内存；CPU EMA 更稳妥，但会增加 step 时间。

如果要兼容 ZeRO-3，可考虑后续做 sharded EMA 或只在保存点 gather 后输出 EMA，但后者不是严格逐 step EMA。

### 7. 测试建议

最小测试覆盖：

1. 小型 toy module 单测：验证 EMA 更新公式、只跟踪可训练参数、prefix 移除/恢复。
2. `save_steps=1` 集成测试：确认同时生成 `step-1.safetensors` 和 `step-1-ema.safetensors`。
3. LoRA 场景测试：确认 EMA key 与现有 LoRA checkpoint key 一致。
4. resume 测试：确认 `--ema_checkpoint` 能恢复 shadow weights。
5. DDP smoke test：确认多进程只由 main process 写文件，所有 rank 更新后 EMA 一致。
