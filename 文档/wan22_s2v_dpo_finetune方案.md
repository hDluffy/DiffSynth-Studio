# Wan2.2-S2V-14B DPO 微调实现方案

## 1. 结论

当前仓库没有现成的 DPO 训练实现。Wan2.2-S2V-14B 的训练入口是 `examples/wanvideo/model_training/train.py`，现有任务只支持：

- `sft` / `sft:train` / `sft:data_process`
- `direct_distill` / `direct_distill:train` / `direct_distill:data_process`

推荐新增一条 `dpo` 任务链路，优先支持 LoRA DPO，后续再扩展 full DPO。核心改造包括：

1. 增加偏好对数据格式：同一条件下包含 `chosen_video` 和 `rejected_video`。
2. 增加 Wan S2V 专用输入解析：共享音频、pose、首帧/参考图，分别编码 chosen/rejected 的 video latent。
3. 增加 `FlowMatchDPOLoss`：在同一个 timestep 和同一份 noise 下比较 chosen/rejected 的 denoising loss 差异。
4. 增加 reference 策略：推荐先实现 `reference-free DPO` 或 `reference loss 预计算`；不建议训练时同时加载两份 14B DiT。
5. 增加 DPO 训练脚本：`examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-DPO.sh`。

## 2. 现有 Wan2.2-S2V 训练链路

### 2.1 训练脚本

当前 full 训练脚本：

```bash
examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
```

关键配置：

```bash
--data_file_keys "video,input_audio,s2v_pose_video"
--model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,...,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth"
--audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/"
--trainable_models "dit"
--extra_inputs "input_image,input_audio,s2v_pose_video"
```

LoRA 训练脚本：

```bash
examples/wanvideo/model_training/lora/Wan2.2-S2V-14B.sh
```

其差异是训练 `dit` 上的 LoRA：

```bash
--lora_base_model "dit"
--lora_target_modules "q,k,v,o,ffn.0,ffn.2"
--lora_rank 32
```

DPO 阶段建议从 LoRA 开始，因为 DPO 每个样本至少要前向 chosen/rejected 两次，显存和时间成本约为 SFT 的 2 倍以上；如果再在线跑 reference model，则接近 4 倍。

### 2.2 训练模块

入口类：

```python
examples/wanvideo/model_training/train.py::WanTrainingModule
```

当前 `task_to_loss`：

```python
self.task_to_loss = {
    "sft": FlowMatchSFTLoss,
    "direct_distill": DirectDistillLoss,
}
```

`forward()` 流程：

1. `get_pipeline_inputs(data)` 从单条 metadata 构造 `inputs_shared/inputs_posi/inputs_nega`。
2. `transfer_data_to_device()` 转设备和 dtype。
3. 依次运行 `pipe.units`，包括 VAE 编码、文本编码、S2V 音频/pose 条件处理。
4. 调用 loss。

DPO 不能直接复用当前单视频 `get_pipeline_inputs()`，因为每条训练数据需要两个视频：`chosen` 和 `rejected`。

### 2.3 S2V 条件路径

相关单元：

```python
diffsynth/pipelines/wan_video.py::WanVideoUnit_S2V
```

它负责：

- `input_audio -> audio_embeds`
- `motion_video -> motion_latents`，没有 motion video 时使用 73 帧全零 motion video
- `s2v_pose_video -> s2v_pose_latents`
- 正样本 `inputs_posi["audio_embeds"]`
- 负样本 `inputs_nega["audio_embeds"] = 0`

S2V DiT forward 分支：

```python
diffsynth/pipelines/wan_video.py::model_fn_wans2v
```

关键输入：

```python
latents, timestep, context, audio_embeds, motion_latents, s2v_pose_latents
```

`latents[:, :, 0:1]` 被当作参考首帧 latent，`latents[:, :, 1:]` 是待去噪主体。因此 DPO 数据中 chosen/rejected 最好共享同一个首帧；如果首帧不同，模型会同时学习“首帧参考差异”和“视频质量偏好”，偏好信号会混杂。

## 3. DPO 数据格式设计

### 3.1 推荐 metadata 字段

建议新增独立 DPO 数据集目录，例如：

```text
data/dpo/wan22-s2v/
  metadata.csv
  videos/chosen/xxx.mp4
  videos/rejected/xxx.mp4
  audio/xxx.wav
  pose/xxx.mp4
```

`metadata.csv` 推荐字段：

| 字段 | 必需 | 含义 |
| --- | --- | --- |
| `prompt` | 是 | 文本条件 |
| `chosen_video` | 是 | 偏好样本视频 |
| `rejected_video` | 是 | 非偏好样本视频 |
| `input_audio` | 是 | S2V 音频输入 |
| `s2v_pose_video` | 否 | pose 条件视频 |
| `input_image` | 建议 | 共享首帧/参考图；若缺省，可从 chosen 第一帧取 |
| `motion_video` | 否 | S2V motion 条件；通常可缺省 |
| `preference_score_gap` | 否 | 偏好强度，可作为样本权重 |
| `pair_id` | 建议 | 方便追踪同一偏好对 |

### 3.2 为什么不用 `video` 字段

现有训练默认把 `video` 当作唯一监督目标，并从 `data["video"][0]` 取 `input_image`。DPO 需要同时保留 chosen/rejected，建议避免继续复用 `video` 字段，否则会在 pipeline 单元和 extra input 解析中产生歧义。

### 3.3 数据质量约束

DPO 对偏好数据非常敏感，建议做以下离线校验：

- chosen/rejected 的 `height/width/num_frames/fps` 一致。
- 两者对应同一 `prompt/input_audio/s2v_pose_video/input_image`。
- 两者首帧尽量一致；不一致时必须明确这是目标偏好的一部分。
- 帧数满足 Wan VAE 的 `4n+1` 约束，例如 81、121、241。
- 音频长度覆盖视频时长，避免 audio embedding padding 影响偏好判断。

## 4. DPO loss 设计

### 4.1 Flow Matching 下的 log-prob 近似

扩散/flow matching 模型通常不能直接得到标准 autoregressive log-prob。工程上可用 denoising MSE 作为负 log-prob 的代理：

```text
log p_theta(x | c, t, eps) ≈ - MSE(v_theta(x_t, t, c), target)
```

对同一条件 `c`、同一 timestep `t`、同一 noise `eps`，分别计算：

```text
loss_chosen_theta
loss_rejected_theta
```

则策略模型的偏好 logit 可写为：

```text
delta_theta = -loss_chosen_theta + loss_rejected_theta
```

如果有 reference model：

```text
delta_ref = -loss_chosen_ref + loss_rejected_ref
```

DPO loss：

```text
loss_dpo = -log sigmoid(beta * (delta_theta - delta_ref))
```

如果采用 reference-free DPO：

```text
loss_dpo = -log sigmoid(beta * delta_theta)
```

### 4.2 推荐训练目标

建议组合 DPO 和 chosen SFT 稳定项：

```text
loss = loss_dpo + lambda_sft * loss_chosen_theta
```

建议初始参数：

```text
beta = 0.05 ~ 0.2
lambda_sft = 0.05 ~ 0.2
```

原因：Wan2.2-S2V 是大视频模型，偏好数据噪声和 pair 分布偏移都可能造成过优化。SFT 稳定项可以约束模型不要为了赢过 rejected 而破坏基本生成能力。

### 4.3 共享 timestep 和 noise

DPO 对比必须公平。chosen/rejected 应使用：

- 同一个随机 timestep
- 同一份 noise，或至少同一随机种子生成的 shape 对齐 noise
- 同一条件：prompt/audio/pose/input_image/motion

推荐实现：

```python
timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
timestep = pipe.scheduler.timesteps[timestep_id]
noise = torch.randn_like(chosen_latents)

chosen_noisy = scheduler.add_noise(chosen_latents, noise, timestep)
rejected_noisy = scheduler.add_noise(rejected_latents, noise, timestep)
```

如果 chosen/rejected latent shape 不一致，应直接报错，不做自动 resize。

## 5. Reference model 策略

### 5.1 方案 A：Reference-free DPO，首选 MVP

不加载 reference model，使用：

```text
loss_dpo = -log sigmoid(beta * (-loss_chosen + loss_rejected))
```

优点：

- 改动最小。
- 显存可控。
- 适合先验证数据和训练流程。

缺点：

- 没有 KL/reference 约束，容易过优化。
- 更依赖 `lambda_sft` 和较小学习率。

建议作为第一阶段实现。

### 5.2 方案 B：预计算 reference loss，推荐生产化

离线用 base model 对每个 pair 采样若干 timestep/noise seed，保存：

```text
ref_loss_chosen
ref_loss_rejected
timestep_id
noise_seed
```

训练时复用相同 timestep/noise seed，计算策略模型 loss，再带入 DPO 公式。

优点：

- 训练时不加载第二份 DiT。
- DPO 语义完整。
- 显存接近 reference-free。

缺点：

- 需要额外数据预处理任务。
- 如果训练时重新采样 timestep/noise，预计算 ref loss 无法直接复用。

适合第二阶段。

### 5.3 方案 C：在线 reference model，不推荐 14B full 训练

同时加载 frozen base DiT 和 trainable DiT/LoRA DiT，在线计算 `delta_ref`。

优点：实现语义最直接。

缺点：14B S2V 显存和通信成本非常高，full DPO 基本不可取；LoRA DPO 也会显著增加显存。

仅建议小分辨率、小 batch 的 debug 使用。

## 6. 代码改造方案

### 6.1 新增 loss

文件：

```text
diffsynth/diffusion/loss.py
```

新增：

```python
def FlowMatchDPOLoss(
    pipe,
    chosen_inputs,
    rejected_inputs,
    beta=0.1,
    lambda_sft=0.1,
    reference_free=True,
    ref_chosen_loss=None,
    ref_rejected_loss=None,
):
    ...
```

内部步骤：

1. 从 `chosen_inputs["input_latents"]` 和 `rejected_inputs["input_latents"]` 取 latent。
2. 检查 shape 一致。
3. 采样同一个 timestep 和 noise。
4. 分别构造 `latents`、`training_target`。
5. 调用 `pipe.model_fn(**models, **inputs, timestep=timestep)` 两次。
6. 分别计算 per-sample MSE，不要立即 reduce 成全局 scalar；DPO 最好保留 batch 维。
7. 计算 `delta_theta` 和 DPO loss。
8. 加 chosen SFT 稳定项。

注意：当前 runner 的 dataloader `collate_fn=lambda x: x[0]`，实际 batch size 是 1。短期可以按 scalar 实现；后续若要真实 batch，需要重写 collate。

### 6.2 新增 Wan DPO 输入构造

文件：

```text
examples/wanvideo/model_training/train.py
```

建议新增方法：

```python
def get_dpo_pipeline_inputs(self, data):
    shared = {... prompt/audio/pose/height/width/num_frames ...}
    chosen_data = data.copy(); chosen_data["video"] = data["chosen_video"]
    rejected_data = data.copy(); rejected_data["video"] = data["rejected_video"]
    chosen_inputs = self.get_pipeline_inputs(chosen_data)
    rejected_inputs = self.get_pipeline_inputs(rejected_data)
    return chosen_inputs, rejected_inputs
```

但要注意 `input_image`：

- 如果 metadata 提供 `input_image`，chosen/rejected 都应使用同一个 `input_image`。
- 如果没有提供，可从 chosen 第一帧取，并强制 rejected 第一帧对齐；否则建议报 warning。

更清晰的实现是新增专用方法：

```python
def build_single_video_inputs(self, data, video_key):
    inputs_posi = {"prompt": data["prompt"]}
    inputs_shared = {
        "input_video": data[video_key],
        "input_image": data.get("input_image", data[video_key][0]),
        "input_audio": data["input_audio"],
        "s2v_pose_video": data.get("s2v_pose_video"),
        ...
    }
    return inputs_shared, inputs_posi, {}
```

### 6.3 运行 pipeline units 的方式

DPO 有两种实现方式。

#### 方式 1：分别跑 chosen/rejected 的 pipeline units

```python
chosen_inputs = self.transfer_data_to_device(chosen_inputs, ...)
rejected_inputs = self.transfer_data_to_device(rejected_inputs, ...)
for unit in self.pipe.units:
    chosen_inputs = self.pipe.unit_runner(unit, self.pipe, *chosen_inputs)
for unit in self.pipe.units:
    rejected_inputs = self.pipe.unit_runner(unit, self.pipe, *rejected_inputs)
loss = FlowMatchDPOLoss(self.pipe, chosen_inputs, rejected_inputs, ...)
```

优点：改动小。

缺点：文本、音频、pose 条件会重复计算一次。

#### 方式 2：共享条件，只分别处理 video latent

把 prompt/audio/pose/input_image/motion 条件抽成 shared，chosen/rejected 只分别编码 `input_latents`。

优点：效率更好。

缺点：需要更细粒度拆 pipeline units，改动更大。

建议第一阶段用方式 1，后续在 `dpo:data_process` 中缓存结果解决重复开销。

### 6.4 新增 task

在 `WanTrainingModule.task_to_loss` 加入：

```python
"dpo": lambda pipe, inputs_shared, inputs_posi, inputs_nega: ...
"dpo:train": ...
"dpo:data_process": lambda pipe, *args: args
```

实际更推荐 `forward()` 中针对 `task.startswith("dpo")` 单独分支，因为 DPO loss 需要两个输入包，不适合塞进现有三元组接口。

建议结构：

```python
def forward(self, data, inputs=None):
    if self.task.startswith("dpo"):
        return self.forward_dpo(data, inputs)
    ... existing path ...
```

### 6.5 数据集读取

当前 `UnifiedDataset` 只会对 `data_file_keys` 中的字段应用 operator。DPO 脚本应设置：

```bash
--data_file_keys "chosen_video,rejected_video,input_audio,s2v_pose_video,input_image"
```

问题：`input_audio` 需要 `LoadAudio`，`chosen_video/rejected_video/input_image/s2v_pose_video` 需要不同 operator。当前 `main_data_operator` 可以处理视频和图片，`input_audio` 已通过 `special_operator_map` 处理。可直接复用：

```python
special_operator_map={
  "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
}
```

如果 `input_image` 是图片路径，默认 video operator 会把图片转成单帧 list；训练输入最好取 `data["input_image"][0]`。需要在 DPO input builder 中处理这个细节。

## 7. 推荐新增脚本

新增：

```text
examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-DPO.sh
```

示例：

```bash
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/dpo/wan22-s2v \
  --dataset_metadata_path data/dpo/wan22-s2v/metadata.csv \
  --data_file_keys "chosen_video,rejected_video,input_audio,s2v_pose_video,input_image" \
  --height 448 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 20 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
  --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --learning_rate 2e-5 \
  --num_epochs 1 \
  --task dpo \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,input_audio,s2v_pose_video" \
  --use_gradient_checkpointing_offload \
  --output_path "./models/train/Wan2.2-S2V-14B_dpo_lora"
```

建议新增 argparse 参数：

```python
--dpo_beta 0.1
--dpo_lambda_sft 0.1
--dpo_reference_free
--dpo_ref_loss_key_chosen ref_loss_chosen
--dpo_ref_loss_key_rejected ref_loss_rejected
```

## 8. 数据预处理和缓存建议

DPO 在线处理开销大。建议分两阶段：

### 8.1 MVP：直接在线训练

适合小数据、小分辨率验证。

优点：代码少。

缺点：每 step 需要两次 VAE encode、两次文本/音频/pose 条件处理、两次 DiT forward。

### 8.2 推荐：`dpo:data_process` 缓存

类似现有 `sft:data_process`，把以下内容缓存到 `.pth`：

```python
{
  "chosen": (inputs_shared, inputs_posi, inputs_nega),
  "rejected": (inputs_shared, inputs_posi, inputs_nega),
  "pair_id": ...,
}
```

缓存中应至少包含：

- chosen/rejected `input_latents`
- prompt `context`
- `audio_embeds`
- `motion_latents`
- `s2v_pose_latents`
- `first_frame_latents` 或等价首帧条件

训练时 `metadata_path=None` 读取缓存，避免重复 VAE/audio/text 编码。

## 9. 风险和注意事项

### 9.1 DPO 目标与视频生成不完全匹配

DPO 原始形式基于文本模型 log-prob。视频 diffusion/flow matching 只能用 denoising loss 近似 log-prob。这个近似可用，但指标解释要谨慎。

### 9.2 偏好对质量比算法更重要

如果 chosen/rejected 差异来自分辨率、首帧、音频长度、pose 对齐等非目标因素，DPO 会学习错误偏好。

### 9.3 S2V 首帧条件容易污染偏好

`model_fn_wans2v` 会把 `latents[:, :, 0:1]` 作为 reference latent。chosen/rejected 首帧不同会显著影响训练目标。推荐 pair 内共享 `input_image`，并在数据校验中强制首帧一致或接近。

### 9.4 显存成本

LoRA DPO 推荐起步设置：

```text
height=448, width=832, num_frames=81
learning_rate=1e-5 ~ 2e-5
lora_rank=16 或 32
beta=0.05 ~ 0.1
lambda_sft=0.1
```

full DPO 不建议作为第一阶段目标。

## 10. 分阶段落地计划

### 阶段 1：Reference-free LoRA DPO MVP

改动：

- `diffsynth/diffusion/loss.py` 新增 `FlowMatchDPOLoss`。
- `examples/wanvideo/model_training/train.py` 新增 `forward_dpo()`。
- 新增 DPO shell 脚本。
- 使用在线处理，不做缓存。

验收：

- 单 pair 能跑通 forward/backward。
- loss 不为 NaN。
- chosen/rejected 交换后 loss 趋势相反。
- LoRA checkpoint 可保存并能用 validate 脚本加载。

### 阶段 2：DPO 数据缓存

改动：

- 支持 `dpo:data_process`。
- 缓存 chosen/rejected pipeline outputs。
- 训练时从缓存读取。

验收：

- 缓存训练和在线训练在固定 seed 下 loss 接近。
- 训练吞吐明显提升。

### 阶段 3：Reference loss 预计算

改动：

- 增加 reference loss 预计算任务。
- metadata 或缓存中写入 `ref_loss_chosen/ref_loss_rejected`。
- `FlowMatchDPOLoss` 支持非 reference-free 模式。

验收：

- DPO logit 使用 `delta_theta - delta_ref`。
- 比 reference-free 更稳定，生成结果不明显退化。

### 阶段 4：评估闭环

评估维度：

- 音频口型同步：人工或 SyncNet 类指标。
- 视频质量：清晰度、脸部稳定性、运动连续性。
- 偏好胜率：同一 prompt/audio 下 base vs DPO blind pairwise。
- 回归测试：SFT 验证集上不出现明显质量下降。

## 11. 推荐最终文件清单

建议新增或修改：

```text
diffsynth/diffusion/loss.py
  + FlowMatchDPOLoss

examples/wanvideo/model_training/train.py
  + dpo_beta/dpo_lambda_sft 参数
  + get_dpo_pipeline_inputs()
  + forward_dpo()
  + task/launcher map 增加 dpo、dpo:train、dpo:data_process

examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-DPO.sh
  + LoRA DPO 启动脚本

文档/wan22_s2v_dpo_finetune方案.md
  + 本报告
```

## 12. 推荐优先级

建议不要一开始实现 full DPO 或在线 reference DPO。更稳妥的顺序是：

1. Reference-free LoRA DPO 跑通。
2. 加 chosen SFT regularization 防止模型漂移。
3. 引入缓存降低成本。
4. 再加 reference loss 预计算，补齐严格 DPO。

这样能最快验证数据是否有效，同时避免 14B S2V 双模型在线训练带来的显存和调试成本。

## 13. 已实现执行流程

本次实现已落地阶段 1，并为阶段 2 缓存训练和阶段 3 reference loss 预计算保留接口。涉及文件：

```text
diffsynth/diffusion/loss.py
  + FlowMatchDPOLoss

diffsynth/core/data/unified_dataset.py
  + 文件列空值自动保留为 None，兼容可选 input_image/s2v_pose_video

examples/wanvideo/model_training/train.py
  + dpo / dpo:data_process / dpo:train 任务入口
  + dpo_beta / dpo_lambda_sft / reference loss 参数
  + chosen/rejected 输入构造与共享 input_image 逻辑

examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-DPO.sh
  + Wan2.2-S2V-14B LoRA DPO 默认启动脚本
```

### 13.1 数据准备

推荐 metadata 至少包含：

```csv
pair_id,prompt,chosen_video,rejected_video,input_audio,s2v_pose_video,input_image,preference_score_gap
0001,"a person singing",videos/chosen/0001.mp4,videos/rejected/0001.mp4,audio/0001.wav,pose/0001.mp4,images/0001.png,1.0
```

必需字段：`prompt`、`chosen_video`、`rejected_video`、`input_audio`。

可选字段：`s2v_pose_video`、`input_image`、`preference_score_gap`、`dpo_weight`、`ref_loss_chosen`、`ref_loss_rejected`、`pair_id`。

执行时需保证 chosen/rejected 已被同一组 dataset operator 处理后具有相同帧数和分辨率；代码会在 `forward_dpo()` 中检查帧数与每帧尺寸，不一致会直接报错。`input_image` 若缺省，会使用 chosen 第一帧作为 pair 内共享参考图，并只发出一次 warning。

### 13.2 在线 LoRA DPO 训练

默认脚本：

```bash
bash examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-DPO.sh
```

可通过环境变量覆盖常用参数：

```bash
DATASET_BASE_PATH=data/dpo/wan22-s2v \
DATASET_METADATA_PATH=data/dpo/wan22-s2v/metadata.csv \
OUTPUT_PATH=./models/train/Wan2.2-S2V-14B_dpo_lora \
HEIGHT=448 WIDTH=832 NUM_FRAMES=81 \
LEARNING_RATE=2e-5 NUM_EPOCHS=1 LORA_RANK=32 \
DPO_BETA=0.1 DPO_LAMBDA_SFT=0.1 \
bash examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-DPO.sh
```

脚本核心参数：

```bash
--task dpo
--data_file_keys "chosen_video,rejected_video,input_audio,s2v_pose_video,input_image"
--extra_inputs "input_image,input_audio,s2v_pose_video"
--dpo_reference_free
--dpo_beta 0.1
--dpo_lambda_sft 0.1
--lora_base_model "dit"
--lora_target_modules "q,k,v,o,ffn.0,ffn.2"
```

`FlowMatchDPOLoss` 会对 chosen/rejected 使用同一个 timestep 和同一份 noise，分别计算 denoising MSE：

```text
delta_theta = -loss_chosen_theta + loss_rejected_theta
loss = -log sigmoid(beta * delta_theta) + dpo_lambda_sft * loss_chosen_theta
```

如果 metadata 中存在 `preference_score_gap` 或 `dpo_weight`，该值会作为 DPO 项样本权重；两者同时存在时优先使用 `dpo_weight`。

### 13.3 缓存式 DPO 训练

当在线处理吞吐不足时，可以先缓存 chosen/rejected 的 pipeline 前处理结果。第一步生成缓存：

```bash
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/dpo/wan22-s2v \
  --dataset_metadata_path data/dpo/wan22-s2v/metadata.csv \
  --data_file_keys "chosen_video,rejected_video,input_audio,s2v_pose_video,input_image" \
  --height 448 --width 832 --num_frames 81 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
  --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --task dpo:data_process \
  --lora_base_model "dit" \
  --extra_inputs "input_image,input_audio,s2v_pose_video" \
  --output_path ./models/cache/Wan2.2-S2V-14B_dpo
```

第二步读取缓存训练 LoRA，注意不要传 `--dataset_metadata_path`：

```bash
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path ./models/cache/Wan2.2-S2V-14B_dpo \
  --height 448 --width 832 --num_frames 81 \
  --dataset_repeat 20 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
  --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --learning_rate 2e-5 --num_epochs 1 \
  --task dpo:train \
  --dpo_reference_free --dpo_beta 0.1 --dpo_lambda_sft 0.1 \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path ./models/train/Wan2.2-S2V-14B_dpo_lora
```

`dpo:data_process` 保存的 `.pth` 是一个 dict，至少包含 `chosen` 和 `rejected` 两个已处理输入包；如果 metadata 中带有 `pair_id`、`dpo_weight`、reference loss 字段，也会一并保存。

### 13.4 Reference loss 预留接口

当前不在线加载第二份 14B reference DiT。若后续离线预计算 reference loss，可在 metadata 或缓存中提供：

```text
ref_loss_chosen
ref_loss_rejected
```

训练时加 `--dpo_use_reference`，loss 会使用：

```text
loss_dpo = -log sigmoid(beta * (delta_theta - delta_ref))
```

字段名可通过以下参数覆盖：

```bash
--dpo_ref_loss_key_chosen ref_loss_chosen
--dpo_ref_loss_key_rejected ref_loss_rejected
```

### 13.5 512x1024x161 Sequence Parallel 训练

512x1024x161 规格建议使用 unified sequence parallel，先跑缓存式 DPO 或 DiT-only 训练，避免在线 DPO 同时承担视频/音频/T5/VAE 前处理带来的额外显存峰值。

full S2V sequence-parallel 启动脚本：

```bash
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-sequence_parallel.sh
```

常用覆盖参数：

```bash
DATASET_BASE_PATH=data/s2v \
DATASET_METADATA_PATH=data/s2v/metadata.csv \
OUTPUT_PATH=./models/train/Wan2.2-S2V-14B_full_sp_512x1024x161 \
HEIGHT=512 WIDTH=1024 NUM_FRAMES=161 NUM_PROCESSES=8 \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-sequence_parallel.sh
```

缓存式 DPO 的 sequence-parallel 训练脚本：

```bash
CACHE_BASE_PATH=./models/cache/Wan2.2-S2V-14B_dpo_512x1024x161 \
OUTPUT_PATH=./models/train/Wan2.2-S2V-14B_dpo_lora_512x1024x161_sequence_parallel \
HEIGHT=512 WIDTH=1024 NUM_FRAMES=161 NUM_PROCESSES=8 \
bash examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-DPO-sequence_parallel.sh
```

Sequence parallel 训练分支不会把 dataloader 交给 `accelerator.prepare()`，以保证所有 rank 读取同一条样本。因此 DeepSpeed 无法从 dataloader 推断 batch size，配置中需要显式包含 `train_micro_batch_size_per_gpu: 1`。默认 `accelerate_config_zero3_noinit.yaml` 已包含该字段，`runner.py` 也会在 `--use_sequence_parallel` 时兜底写入。

USP attention 后端默认由 `DIFFSYNTH_USP_ATTN_BACKEND=auto` 控制：环境中存在 `flash_attn` 时使用 xFuser/yunchang 的 FlashAttention；不存在时，在当前 `ring_degree=1` 的 Ulysses sequence-parallel 配置下自动回退到 PyTorch `scaled_dot_product_attention`，该 fallback 支持训练反传。可选值：`auto`、`fa`、`torch`。如果显式设置 `DIFFSYNTH_USP_ATTN_BACKEND=fa` 但环境缺少 `flash_attn`，会直接报错；如需强制 PyTorch 后端，可设置：

```bash
DIFFSYNTH_USP_ATTN_BACKEND=torch \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-sequence_parallel.sh
```

`examples/wanvideo/model_training/full/Wan2.2-S2V-14B-sequence_parallel.sh` 默认不再传 `--fp8_models`。原因是 `fp8_models` 会让 T5/VAE 进入 VRAM management 的 `module_map` 加载分支，该分支会直接调用 `model.load_state_dict(assign=True)`；在 DeepSpeed ZeRO3 空参数初始化场景下，当前模型参数可能是 `torch.Size([0])`，从而出现：

```text
size mismatch for token_embedding.weight: copying a param with shape ... from checkpoint, the shape in current model is torch.Size([0])
```

因此默认路径使用 ZeRO3-aware 的普通加载分支。只有在确认当前加载器已经兼容 FP8+ZeRO3 后，才通过环境变量显式启用：

```bash
FP8_MODELS="Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B-sequence_parallel.sh
```

如果再次遇到 `SIGKILL` 且 `dmesg` 出现 `Out of memory`，优先使用缓存式 DPO sequence-parallel 脚本，并确保训练阶段只加载 DiT：`--model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors"`。
