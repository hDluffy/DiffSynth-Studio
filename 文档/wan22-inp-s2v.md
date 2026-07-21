# Wan2.2-Fun-A14B-InP 首尾帧训练与 Wan2.2-S2V-14B 对比分析

本文基于当前仓库代码分析：

- `examples/wanvideo/model_training/full/Wan2.2-Fun-A14B-InP.sh`
- `examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh`
- `examples/wanvideo/model_training/train.py`
- `diffsynth/pipelines/wan_video.py`
- `diffsynth/models/wan_video_dit.py`
- `diffsynth/models/wan_video_dit_s2v.py`
- `diffsynth/diffusion/loss.py`
- `diffsynth/diffusion/runner.py`

## 结论

Wan2.2-Fun-A14B-InP 的首尾帧不是把首尾帧直接替换进训练目标 latent，也不是在 loss 中固定首尾帧；它的核心做法是：

1. 从训练视频中取 `video[0]` 和 `video[-1]`，作为 `input_image` 和 `end_image`。
2. 用 VAE 把“首帧 + 中间全 0 + 尾帧”编码成条件 latent。
3. 构造一个 mask，标记哪些 latent 位置是真实首尾帧条件，哪些位置是空白。
4. 把 mask 和 VAE 条件 latent 拼成 `y`。
5. 在 DiT 前向时把 noisy target latent `x` 与条件 `y` 在 channel 维拼接，作为 inpaint/first-last-frame 条件输入。
6. loss 仍然对完整视频 `input_latents` 做 flow matching MSE，模型学习在首尾帧条件约束下预测整段视频的速度/噪声目标。

Wan2.2-S2V-14B 的结构不同。S2V 把 `input_image` 编码后融合进主 `latents` 的第 0 个 latent frame，并在 S2V DiT 中把第 0 帧当作 reference token，实际预测的是 `latents[:, :, 1:]`。所以 S2V 原生是“首帧 reference + 音频 + 可选 pose/motion”的结构，不是 Fun-InP 这种 `y` 条件通道结构。

在 Wan2.2-S2V-14B 基础上可以实现首尾帧，但不能只在训练脚本加 `--extra_inputs "input_image,end_image,input_audio"`。需要改 S2V 的条件建模方式。最稳妥方案是在 S2V DiT 中增加尾帧 reference token，并相应调整位置编码、mask、输出拼接和 loss 裁剪。

## Wan2.2-Fun-A14B-InP 脚本入口

`Wan2.2-Fun-A14B-InP.sh` 分两段训练：

- high noise model：加载 `PAI/Wan2.2-Fun-A14B-InP:high_noise_model/...`，训练 timestep 边界为 `[900, 1000]`，脚本中为 `--max_timestep_boundary 0.358 --min_timestep_boundary 0`。
- low noise model：加载 `PAI/Wan2.2-Fun-A14B-InP:low_noise_model/...`，训练 timestep 边界为 `[0, 900]`，脚本中为 `--max_timestep_boundary 1 --min_timestep_boundary 0.358`。

两段共同点：

- 数据集字段默认只要求 `video` 和 `prompt`。
- `--extra_inputs "input_image,end_image"` 打开首尾帧条件。
- `--trainable_models "dit"` 表示只训练 DiT。
- `--remove_prefix_in_ckpt "pipe.dit."` 保存时去掉训练模块里的 `pipe.dit.` 前缀。

## 完整训练流程

### 1. 数据读取

训练入口是 `examples/wanvideo/model_training/train.py`。

`UnifiedDataset` 根据 metadata 读取样本。WanVideo 默认使用 `UnifiedDataset.default_video_operator(...)` 处理主视频：

- 视频路径转绝对路径。
- 读取视频或图像序列。
- crop/resize 到脚本指定的 `height=480, width=832`。
- 帧数满足 WanVideo VAE 的时间压缩约束：默认 `time_division_factor=4, time_division_remainder=1`，也就是 `4n+1` 帧。

对 InP 脚本来说，metadata 中主要有：

- `video`
- `prompt`

`train.py` 的 `WanTrainingModule.parse_extra_inputs()` 会把：

- `input_image` 映射为 `data["video"][0]`
- `end_image` 映射为 `data["video"][-1]`

因此训练时首尾帧来自同一条训练视频，不需要 metadata 单独提供首尾帧路径。

### 2. 构造 pipeline inputs

`WanTrainingModule.get_pipeline_inputs()` 生成三组输入：

- `inputs_shared`：正负 prompt 共享的输入，如 `input_video`、尺寸、帧数、cfg、timestep 边界、首尾帧等。
- `inputs_posi`：正向条件，主要是 `prompt`。
- `inputs_nega`：负向条件，训练中通常为空，因为 `cfg_scale=1`。

关键字段：

```python
inputs_shared = {
    "input_video": data["video"],
    "height": data["video"][0].size[1],
    "width": data["video"][0].size[0],
    "num_frames": len(data["video"]),
    "cfg_scale": 1,
    "tiled": False,
    "rand_device": self.pipe.device,
    "max_timestep_boundary": ...,
    "min_timestep_boundary": ...,
}
```

然后 `parse_extra_inputs()` 加入：

```python
inputs_shared["input_image"] = data["video"][0]
inputs_shared["end_image"] = data["video"][-1]
```

### 3. pipeline 单元执行

训练 forward 中会依次执行 `self.pipe.units`。对 Fun-InP 首尾帧最关键的是：

1. `WanVideoUnit_NoiseInitializer`
2. `WanVideoUnit_PromptEmbedder`
3. `WanVideoUnit_InputVideoEmbedder`
4. `WanVideoUnit_ImageEmbedderVAE`
5. `WanVideoUnit_ImageEmbedderCLIP`
6. `WanVideoUnit_CfgMerger`

#### NoiseInitializer

把 `num_frames` 转为 latent 时间长度：

```python
length = (num_frames - 1) // 4 + 1
shape = (1, z_dim, length, height // upsampling_factor, width // upsampling_factor)
noise = pipe.generate_noise(shape, ...)
```

Wan VAE 时间压缩约为 4 倍，所以 81 帧视频会变成 21 个 latent frames。

#### InputVideoEmbedder

训练时 `input_video` 不为空：

```python
input_video = pipe.preprocess_video(input_video)
input_latents = pipe.vae.encode(input_video, ...)
```

因为 `pipe.scheduler.training=True`，它返回：

```python
{"latents": noise, "input_latents": input_latents}
```

注意：这里的 `input_latents` 是完整目标视频的 VAE latent，后续 loss 会基于它加噪、构造 training target。

#### ImageEmbedderVAE：首尾帧 latent 的核心

`WanVideoUnit_ImageEmbedderVAE` 只在 `pipe.dit.require_vae_embedding=True` 时生效。Fun-InP 模型属于这类模型。

流程：

1. resize 首帧到训练尺寸。
2. 创建 mask：

```python
msk = torch.ones(1, num_frames, height//8, width//8)
msk[:, 1:] = 0
```

这表示原始像素时间轴上只有第 0 帧是条件帧。

3. 如果有 `end_image`，则把最后一帧也标成条件：

```python
msk[:, -1:] = 1
```

4. 构造 VAE 输入视频：

```python
vae_input = concat([
    first_frame,
    zeros(num_frames - 2),
    end_frame,
], time_dim)
```

也就是首帧是真实图像，中间帧全 0，尾帧是真实图像。

这里首尾帧 image 本身不需要像 mask 那样手动 repeat 或 pad 到 4 倍时间长度。原因是输入给 VAE 的不是“单独的首帧 latent”和“单独的尾帧 latent”，而是一个完整的 pixel-space 条件视频：

```text
[首帧真实图像, 0, 0, ..., 0, 尾帧真实图像]
```

这个条件视频的长度就是 `num_frames`，训练脚本和 dataset 已经保证它满足 Wan VAE 需要的 `4n+1` 时间长度，例如 81 帧。因此 VAE 编码时会自己把这段 81 帧条件视频压缩成 21 个 latent frames，不需要外部再对 image 做时间 pad。

如果只有首帧、没有尾帧，则构造方式是：

```python
vae_input = concat([
    first_frame,
    zeros(num_frames - 1),
], time_dim)
```

同样也是完整 `num_frames` 条件视频，不需要额外 pad。只有在 `num_frames` 本身不满足 `4n+1` 时，才需要在数据读取/shape check 阶段修正帧数；当前训练链路通过 `default_video_operator(..., time_division_factor=4, time_division_remainder=1)` 已经处理了这个约束。

5. 将 pixel-space mask 对齐到 Wan VAE 的时间压缩布局：

```python
msk = concat([repeat(msk[:, 0:1], 4), msk[:, 1:]], dim=1)
msk = msk.view(1, msk.shape[1] // 4, 4, h, w)
msk = msk.transpose(1, 2)[0]
```

这里的目的是让 mask 与 VAE latent 的时间打包方式一致。Wan VAE 会在时间维做 4 倍压缩，但它适配的是 `4n+1` 帧结构，不是简单地把原始帧按 `[0..3], [4..7]` 直接分组。以 81 帧为例，目标视频会被编码成 21 个 latent frames。

首帧是特殊边界帧，所以代码先把第 0 帧 mask repeat 4 次，再接上剩余的 80 帧 mask：

```text
原始 mask:        81 = 1 + 80
repeat 后 mask:   84 = 4 + 80
重排为 latent:    84 = 21 * 4
```

随后 `view(..., 21, 4, h, w).transpose(1, 2)` 把时间包里的 4 个子帧位置转成 4 个 mask channels。这样 mask 才能和 VAE 输出的 latent 时间轴对齐。

也就是说，首尾图像输入给 VAE 后，条件 latent `y` 的时间长度会自然对齐到目标 latent；额外需要手动做 4 倍时间对齐的是 mask。最终 mask shape 是：

```text
(4, latent_t, latent_h, latent_w)
```

6. VAE 编码首尾帧条件视频：

```python
y = pipe.vae.encode([vae_input], ...)[0]
```

Wan VAE latent 通常是 16 channels，因此 `y` 是：

```text
(16, latent_t, latent_h, latent_w)
```

7. 拼接 mask 和 VAE latent：

```python
y = torch.concat([msk, y])
y = y.unsqueeze(0)
```

最终 `y` 是：

```text
(1, 20, latent_t, latent_h, latent_w)
```

其中 20 = 4 个 mask channels + 16 个 VAE latent channels。

#### ImageEmbedderCLIP

如果模型要求 CLIP 图像 embedding，则会编码首帧：

```python
clip_context = pipe.image_encoder.encode_image([input_image])
```

如果有 `end_image` 且模型有 image positional embedding：

```python
clip_context = concat([first_clip, end_clip], dim=1)
```

Fun-InP 主要依赖 VAE 条件 `y`，CLIP 条件是否参与取决于加载的 DiT 配置 `require_clip_embedding` 和 `has_image_pos_emb`。

### 4. FlowMatch loss

loss 在 `diffsynth/diffusion/loss.py` 的 `FlowMatchSFTLoss()`。

核心步骤：

1. 根据脚本里的 timestep boundary 随机采样 timestep。
2. 对完整视频 latent 加噪：

```python
noise = torch.randn_like(input_latents)
latents = scheduler.add_noise(input_latents, noise, timestep)
training_target = scheduler.training_target(input_latents, noise, timestep)
```

3. 调用模型：

```python
noise_pred = pipe.model_fn(..., latents=latents, y=y, clip_feature=..., timestep=timestep)
```

4. 用 MSE 训练：

```python
loss = mse(noise_pred, training_target) * scheduler.training_weight(timestep)
```

Fun-InP 没有 `first_frame_latents`，所以 loss 不会裁掉首帧。首尾帧是条件输入，目标仍是完整视频的 flow matching 目标。

### 5. DiT 如何使用首尾帧条件

普通 Wan DiT 前向在 `model_fn_wan_video()`。

对 Fun-InP 来说关键逻辑是：

```python
x = latents
if y is not None and dit.require_vae_embedding:
    x = torch.cat([x, y], dim=1)
```

也就是说：

- `latents` 是加噪后的目标视频 latent。
- `y` 是首尾帧 inpaint 条件。
- 二者在 channel 维拼起来。
- DiT 的 `patch_embedding` 输入通道数 `in_dim` 必须和 `latents channels + y channels` 对齐。

如果 `latents` 是 16 channels，`y` 是 20 channels，则 DiT 输入可能是 36 channels。Fun-InP 的 DiT 权重就是按这种输入结构训练/发布的。

### 6. 优化与保存

训练循环在 `diffsynth/diffusion/runner.py`：

1. 构建 optimizer，默认 `torch.optim.AdamW`。
2. DataLoader 用 `collate_fn=lambda x: x[0]`，每步处理一个样本。
3. `loss = model(data)`。
4. `accelerator.backward(loss)`。
5. `optimizer.step()`、`scheduler.step()`、`optimizer.zero_grad()`。
6. `ModelLogger` 按 epoch 或 `save_steps` 保存。

`ModelLogger.save_model()` 只导出 `requires_grad=True` 的参数：

```python
state_dict = export_trainable_state_dict(...)
```

对于 `--trainable_models "dit"`，保存的是 DiT 参数；对于 LoRA 训练，保存的是 LoRA 参数。

## Wan2.2-S2V-14B 训练结构

S2V full 脚本：

```bash
--data_file_keys "video,input_audio"
--height 448
--width 832
--num_frames 81
--model_id_with_origin_paths "... diffusion ... wav2vec ... t5 ... vae ..."
--audio_processor_path "... wav2vec2-large-xlsr-53-english/"
--trainable_models "dit"
--extra_inputs "input_image,input_audio"
```

LoRA 脚本额外包含：

```bash
--data_file_keys "video,input_audio,s2v_pose_video"
--extra_inputs "input_image,input_audio,s2v_pose_video"
```

### S2V 数据输入

`train.py` 的 `parse_extra_inputs()` 对 S2V 做了同样的首帧抽取：

```python
inputs_shared["input_image"] = data["video"][0]
inputs_shared["input_audio"] = data["input_audio"]
```

如果提供 `s2v_pose_video`，也会进入 `inputs_shared`。

### S2V pipeline 条件

S2V 相关逻辑在 `WanVideoUnit_S2V`：

#### 音频

```python
audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(
    input_audio,
    audio_sample_rate,
    pipe.audio_processor,
    fps=16,
    batch_frames=num_frames-1,
)
```

正向条件拿真实音频 embedding：

```python
inputs_posi["audio_embeds"] = audio_embeds[0]
```

负向条件是全 0 音频：

```python
inputs_nega["audio_embeds"] = 0.0 * audio_embeds
```

训练时 `cfg_scale=1`，通常只用正向条件。

#### motion latent

S2V 固定构造 73 帧 motion latent：

- 如果传入 `motion_video`，编码真实 motion video，并设置 `drop_motion_frames=False`。
- 否则创建全 0 motion video，编码后设置 `drop_motion_frames=True`。

```python
motion_latents = pipe.vae.encode(motion_video_or_zeros)
```

#### pose latent

如果传入 `s2v_pose_video`：

1. 取 `num_frames - 1` 帧。
2. 不足时用 `-1` padding。
3. 前面补一帧以适配 VAE 编码。
4. VAE 编码后丢掉第 0 个 latent frame：

```python
pose_conds.append(cond_latents[:, :, 1:])
```

所以 `s2v_pose_latents` 对应的是待生成的后续帧，不含首帧 reference。

### S2V 首帧 latent 如何加载

S2V 模型配置中：

```python
fuse_vae_embedding_in_latents=True
require_vae_embedding=False
require_clip_embedding=False
```

因此它不会走 Fun-InP 的 `ImageEmbedderVAE`，而是走 `WanVideoUnit_ImageEmbedderFused`：

```python
z = pipe.vae.encode([input_image])
latents[:, :, 0:1] = z
return {
    "latents": latents,
    "fuse_vae_embedding_in_latents": True,
    "first_frame_latents": z,
}
```

训练 loss 中有特殊处理：

```python
inputs["latents"] = add_noise(input_latents, noise, timestep)
inputs["latents"][:, :, 0:1] = first_frame_latents
```

也就是完整目标视频先被加噪，然后第 0 个 latent frame 被替换为干净首帧 latent。

模型预测后：

```python
noise_pred = noise_pred[:, :, 1:]
training_target = training_target[:, :, 1:]
```

loss 不训练第 0 个 reference latent，只训练后续 latent frames。

### S2V DiT 前向结构

`model_fn_wan_video()` 检测到 `audio_embeds is not None` 后转到 `model_fn_wans2v()`。

S2V 前向第一步：

```python
origin_ref_latents = latents[:, :, 0:1]
x = latents[:, :, 1:]
```

含义：

- 第 0 个 latent frame 是首帧 reference。
- `x` 是要预测的后续视频 latent。

然后：

```python
x = patch_embedding(x) + cond_encoder(s2v_pose_latents)
ref_latents = patch_embedding(origin_ref_latents)
x = concat([x, ref_latents], dim=token_seq)
mask = concat([0 for x tokens, 1 for ref tokens])
```

再把 motion tokens 注入：

```python
x, freqs, mask = dit.inject_motion(..., motion_latents, ...)
```

mask 类型含义：

- 0：目标视频 token。
- 1：首帧 reference token。
- 2：motion token。

最后加上可训练的条件类型 embedding：

```python
x = x + dit.trainable_cond_mask(mask)
```

音频通过 `dit.after_transformer_block()` 在多个 transformer block 后以 cross-attention/AdaIN 方式注入到目标视频 token 上。

输出阶段：

```python
x = x[:, :seq_len_x]
x = head(x, t[:-1])
x = unpatchify(x)
x = concat([origin_ref_latents, x], dim=2)
```

它为了兼容 WanVideo 的 loss/推理接口，把首帧 reference latent 拼回输出最前面。随后 loss 会裁掉第 0 帧，只对后续帧算 MSE。

## Fun-InP 与 S2V 的关键差异

| 维度 | Wan2.2-Fun-A14B-InP | Wan2.2-S2V-14B |
|---|---|---|
| 条件目标 | 首帧 + 尾帧 inpaint | 首帧驱动的 speech-to-video |
| 首帧来源 | `data["video"][0]` | `data["video"][0]` |
| 尾帧来源 | `data["video"][-1]` | 原生没有 |
| 首尾帧 latent 注入 | VAE 编码成条件 `y`，与 noisy latent 在 channel 维拼接 | 首帧 VAE latent 替换 `latents[:, :, 0:1]` |
| DiT 输入通道 | `x + y`，需要更大的 `in_dim` | 主 latent 通道不变 |
| 条件 token | 没有单独 reference token；条件在 channel 维 | 首帧是 reference token，拼到 token 序列后面 |
| mask | `y` 内含 4 channel mask，标记首尾帧 | `trainable_cond_mask` 区分目标/ref/motion token |
| loss | 对完整视频 latent 计算 | 裁掉第 0 帧，只训练后续帧 |
| 音频 | 无 | wav2vec/audio encoder + block 内注入 |
| pose | 无 | 可选 `s2v_pose_latents` 经 `cond_encoder` 加到目标 token |
| high/low noise | Wan2.2 A14B 分 high/low 两个 DiT 分段训练 | S2V 脚本只有一个 DiT |

## 能否在 Wan2.2-S2V-14B 基础上实现首尾帧

可以实现，但需要改模型结构和训练流程。原因是 S2V 原生只有“首帧 reference”机制，没有“尾帧 reference”或 Fun-InP 的 `y` 条件通道。

### 不建议的简单做法

只把 S2V 脚本改成：

```bash
--extra_inputs "input_image,end_image,input_audio"
```

基本不会生效。原因：

- `end_image` 会进入 `inputs_shared`，但 S2V 的 `ImageEmbedderFused` 只读取 `input_image`。
- S2V 的 `ImageEmbedderVAE` 因为 `require_vae_embedding=False` 不会执行。
- `model_fn_wans2v()` 没有 `end_image`、`end_frame_latents` 或第二个 reference token 的逻辑。

### 方案 A：在 S2V 中加入尾帧 reference token，推荐

这是最贴合 S2V 现有结构的做法。

改造思路：

1. 新增一个 S2V 专用 pipeline unit，或扩展 `WanVideoUnit_ImageEmbedderFused`：
   - VAE 编码 `input_image` 得到 `first_frame_latents`。
   - VAE 编码 `end_image` 得到 `end_frame_latents`。
   - 训练时把 `latents[:, :, 0:1]` 替换为首帧干净 latent。
   - 可选：把最后一个 latent frame 替换为尾帧干净 latent，或者只把尾帧作为 token 条件，不替换目标 latent。

2. 扩展 `model_fn_wans2v()`：
   - 保留 `origin_ref_latents = latents[:, :, 0:1]`。
   - 新增 `end_ref_latents`。
   - `x = latents[:, :, 1:]` 是否包含最后 latent，需要根据训练目标设计决定。
   - 将 `first_ref_latents` 和 `end_ref_latents` 分别 patchify 后拼到 token 序列。

3. 扩展 mask 类型：
   - 当前 `trainable_cond_mask = nn.Embedding(3, dim)`，只有 0/1/2 三类。
   - 如果要区分尾帧，建议改成 `nn.Embedding(4, dim)`：
     - 0：目标视频 token
     - 1：首帧 reference token
     - 2：motion token
     - 3：尾帧 reference token
   - 如果复用 reference 类型，也可以首尾都用 1，但模型较难区分首尾语义，位置编码必须足够明确。

4. 调整 RoPE grid/位置编码：
   - 当前 `get_grid_sizes()` 把 ref token 放在特殊位置：
     ```python
     grid_sizes_ref = [[
         tensor([30, 0, 0]),
         tensor([31, rh, rw]),
         tensor([1, rh, rw]),
     ]]
     ```
   - 尾帧 reference 需要独立的位置段，不能简单复用首帧 reference 的位置，否则模型无法稳定学习“这是结尾约束”。

5. 调整输出和 loss：
   - 如果尾帧只是条件 token，不拼回 `noise_pred`，loss 可以继续只裁掉第 0 帧。
   - 如果尾帧也被替换到主 latent 最后一帧，则 loss 应裁掉首帧和尾帧，避免训练模型预测已经给定的干净端点：
     ```python
     noise_pred = noise_pred[:, :, 1:-1]
     training_target = training_target[:, :, 1:-1]
     ```
   - 这个需要新增 `last_frame_latents` 或更通用的 mask-based loss，避免影响其它 Wan 任务。

推荐采用“尾帧作为额外 reference token，不替换主 latent 最后一帧”的第一版方案。这样改动集中在 S2V DiT token 条件侧，loss 改动较小；但约束强度可能弱于直接固定尾帧。

### 方案 B：把 Fun-InP 的 `y` 条件通道搬到 S2V，不推荐作为第一步

这要求 S2V DiT 的 `patch_embedding` 输入通道从原始 latent channels 改成 `latent + mask + condition_latent`。这不是简单代码改动，因为：

- 现有 S2V 权重的 `patch_embedding` shape 不匹配。
- 需要初始化新增输入通道权重。
- 还要决定 `cond_encoder(s2v_pose_latents)` 与 `y` 条件的融合顺序。
- 会破坏加载官方 S2V DiT 权重的严格兼容性。

除非计划重新训练较多步，或做专门的结构初始化，否则不建议。

### 方案 C：用 pose/motion 或 prompt 间接约束尾帧，不是真正首尾帧

可以把尾帧信息转成 pose video 的最后姿态或文本描述，但这不能保证生成结果最后一帧匹配给定图像。它不是严格 FLF2V。

## 建议实现路径

最小可行改造：

1. 训练脚本增加：
   ```bash
   --extra_inputs "input_image,end_image,input_audio"
   ```

2. 在 `WanVideoUnit_ImageEmbedderFused` 或新增 `WanVideoUnit_S2VEndImageEmbedder` 中编码：
   ```python
   end_frame_latents = pipe.vae.encode([end_image])
   ```

3. 在 `FlowMatchSFTLoss` 中支持可选的 `first_frame_latents/end_frame_latents`：
   - 首帧继续替换 `latents[:, :, 0:1]`。
   - 是否替换尾帧取决于方案。
   - 如果替换尾帧，则 loss 裁掉最后 latent frame。

4. 在 `model_fn_wans2v()` 增加参数：
   ```python
   end_frame_latents: Optional[torch.Tensor] = None
   ```

5. 在 S2V DiT 中把尾帧 patchify 后作为 reference token 拼入：
   ```python
   end_ref_tokens = patchify(patch_embedding(end_frame_latents))
   x = concat([target_tokens, first_ref_tokens, end_ref_tokens], dim=1)
   ```

6. 把 `trainable_cond_mask` 从 3 类扩为 4 类，或先复用 ref 类做实验。

7. 修改 `get_grid_sizes()` / RoPE 预计算，让尾帧 reference 有明确的“末端时间位置”。

8. 验证推理：
   - 输入首帧、尾帧、音频。
   - 输出保存时仍类似 S2V 示例使用 `video[1:]`，因为第 0 帧是 reference。
   - 如果尾帧被固定进输出，需要确认最后输出帧是否来自生成结果还是给定尾帧。

## 风险点

1. S2V 音频特征与视频长度绑定。当前 `batch_frames=num_frames-1`，模型天然预测首帧之后的 80 帧。如果裁掉尾帧目标，需要重新确认音频 token 和目标帧 token 的对齐。
2. 尾帧 reference 的位置编码很重要。没有明确“末帧”位置，模型可能把尾帧当成普通参考图，无法形成终点约束。
3. 如果扩展 `trainable_cond_mask`，加载旧权重时会有新 embedding 参数缺失，需要初始化并训练。
4. 如果改 `patch_embedding` 输入通道，将不兼容官方 S2V 权重，不建议。
5. 首尾帧约束和音频口型/动作约束可能冲突。训练数据需要同时包含合理的音频、首帧、尾帧对应关系。

## 最终判断

Wan2.2-S2V-14B 基础上可以做首尾帧 S2V，但应按“额外尾帧 reference token”的方向实现，而不是照搬 Fun-InP 的 `y` 条件通道。Fun-InP 的首尾帧机制是 inpaint 条件通道；S2V 的首帧机制是主 latent 序列中的 reference token。两者底层条件注入方式不同，直接混用会遇到 DiT 输入通道、位置编码和 loss 对齐问题。

推荐第一阶段实现：

- 保持 S2V 主 latent/音频/pose 结构不变。
- 增加尾帧 reference token。
- 给尾帧单独 mask 类型和末端位置编码。
- loss 先只裁首帧，观察尾帧收敛；如果尾帧匹配不足，再尝试固定尾 latent 并裁掉尾帧 loss。
