# Bernini 3D RoPE 迁移到 Wan2.2-S2V 参考帧的设计

本文档基于当前仓库源码分析 Bernini 的 3D RoPE/source_id 位置编码方式，并给出把该思路应用到 `Wan2.2-S2V-14B` 训练的设计方案。

设计边界：

- 只改 S2V 参考帧 latent token 的位置编码。
- 目标视频 latent、pose 条件、motion frame pack、audio injection 保持当前 Wan2.2-S2V 原有位置编码和输入组织方式。
- 默认配置保持旧行为，避免已有训练脚本、cache、checkpoint 和推理结果被隐式改变。
- 通过参数显式选择参考帧使用哪种 RoPE 编码方式。

## 1. Bernini 扩散侧 3D RoPE

Bernini 相关实现位于：

```text
Bernini/bernini/models/transformer_wan.py
Bernini/docs/position_encoding_zh.md
Bernini/configs/bernini_renderer_wan22/config.json
Bernini/configs/bernini_renderer_train/train_cfg/bernini_renderer_high.yaml
```

核心类是 `WanRotaryPosEmbed`。它对 VAE latent 的三维 token 网格生成 RoPE：

```text
hidden_states: [B, C, T, H, W]
patch_size:    (1, 2, 2)
token grid:    T' = T, H' = H / 2, W' = W / 2
seq_len:       T' * H' * W'
rotary_emb:    [1, 1, seq_len, head_dim / 2]
```

Bernini 和 Wan 一样使用复数形式 RoPE。每个 attention head 的实数维度为 `attention_head_dim`，复数相位维度为 `attention_head_dim / 2`。

### 1.1 三轴拆分

Bernini 将每个 head 的旋转通道拆到时间、高、宽三个轴：

```python
h_dim = w_dim = 2 * (attention_head_dim // 6)
t_dim = attention_head_dim - h_dim - w_dim
```

以 `attention_head_dim=128` 为例：

```text
real dim:    t=44, h=42, w=42
complex dim: t=22, h=21, w=21
total:       64 = 128 / 2
```

每个 token 的基础位置相位是：

```text
rope_3d(t, h, w) = concat(rope_t(t), rope_h(h), rope_w(w))
```

### 1.2 source_id 相位

Bernini 在基础 3D RoPE 上增加了一个可选 `source_id` 相位：

```python
pos = torch.tensor([float(source_id)], dtype=torch.float64, device=hidden_states.device)
freqs_visual_id = get_1d_rotary_pos_embed(
    self.attention_head_dim,
    pos,
    self.theta,
    use_real=False,
    repeat_interleave_real=False,
    freqs_dtype=torch.float64,
)
freqs = freqs * freqs_visual_id
```

由于复数 RoPE 是单位相位，乘法等价于相位相加：

```text
final_phase(t, h, w, source_id)
  = phase_3d(t, h, w) + phase_source(source_id)
```

Bernini 的语义约定是：

```text
source_id = 0      目标 noisy latent
source_id = 1..N   条件源，例如参考图、参考视频
```

这让多个条件源即使共享相同局部三维坐标，也能在 self-attention 的 Q/K 相位中被区分。

## 2. Wan2.2-S2V 当前 RoPE 链路

当前 S2V 模型实现位于：

```text
diffsynth/models/wan_video_dit_s2v.py
diffsynth/models/wan_video_dit.py
diffsynth/pipelines/wan_video.py
examples/wanvideo/model_training/train.py
diffsynth/configs/model_configs.py
```

S2V DiT 是 `WanS2VModel`，注册在 `diffsynth/configs/model_configs.py`：

```text
model_class: diffsynth.models.wan_video_dit_s2v.WanS2VModel
dim:         5120
num_heads:   40
head_dim:    128
patch_size:  (1, 2, 2)
```

### 2.1 当前 token 组织

`WanS2VModel.forward()` 和 `diffsynth/pipelines/wan_video.py::model_fn_wans2v()` 中有一份基本相同的逻辑：

```python
origin_ref_latents = latents[:, :, 0:1]
x = latents[:, :, 1:]
```

语义是：

- `origin_ref_latents`：第 0 个 latent frame，作为 S2V 参考帧 token。
- `x`：待预测的视频主体 latent，和 `s2v_pose_latents` 融合后进入主 token 序列。

主视频 token：

```python
x, (f, h, w) = self.patchify(self.patch_embedding(x) + self.cond_encoder(pose_cond))
seq_len_x = x.shape[1]
```

参考帧 token：

```python
ref_latents, (rf, rh, rw) = self.patchify(self.patch_embedding(origin_ref_latents))
x = torch.cat([x, ref_latents], dim=1)
```

拼接顺序固定为：

```text
[target_video_tokens, reference_frame_tokens, motion_tokens]
```

其中 motion tokens 是稍后由 `inject_motion()` 追加。

### 2.2 当前参考帧位置编码

S2V 当前使用 `precompute_freqs_cis_3d()` 和 `rope_precompute()` 生成三维 RoPE。目标视频 token 的坐标从 `(0, 0, 0)` 开始：

```python
grid_sizes_x = [[zeros, [f, h, w], [f, h, w]]]
```

参考帧不是使用 source_id，而是放到一个远离目标视频的时间位置：

```python
ref_time_id = max(30, int(f) + 9)
grid_sizes_ref = [[
    torch.tensor([ref_time_id, 0, 0]).unsqueeze(0),
    torch.tensor([ref_time_id + int(rf), rh, rw]).unsqueeze(0),
    torch.tensor([rf, rh, rw]).unsqueeze(0),
]]
```

因此当前参考帧的区分方式是：

```text
reference_position = (t = max(30, target_latent_t + 9), h, w)
```

这属于时间轴偏移方案，而不是 Bernini 的 source_id 方案。

### 2.3 motion RoPE 不能一起改

S2V 的 motion frame pack 使用负时间坐标和多尺度 patch：

```text
post  scale: negative time, patch (1, 2, 2)
2x    scale: negative time, patch (2, 4, 4)
4x    scale: negative time, patch (4, 8, 8)
```

这些坐标来自 `FramePackMotioner.forward()`，服务于 motion 历史帧，不属于参考帧 source_id 设计范围。为了避免破坏 S2V 原有运动条件建模，本方案不修改 motion RoPE。

## 3. 设计目标

目标是在 Wan2.2-S2V 训练中引入 Bernini 的 source_id 思路，但只作用于参考帧 token：

```text
target_video_tokens     保持当前 3D RoPE
reference_frame_tokens  可选：当前 3D RoPE 或 source_id 增强 3D RoPE
motion_tokens           保持当前负时间/多尺度 3D RoPE
```

也就是说，source_id 不是全局替换 Wan2.2-S2V 的位置编码，而是参考帧分支的可选增强。

这样做的原因：

- Wan2.2-S2V 预训练权重已经习惯目标视频和 motion tokens 的现有坐标分布。
- 当前参考帧只有一个 source，最小改动可以先让它拥有显式 source_id 相位。
- 后续如果扩展多参考帧，可以自然把不同参考帧分配为 `source_id=1,2,...`。
- 默认保持旧行为，便于做 ablation 和回滚。

## 4. 参考帧 RoPE 模式

建议新增参数 `s2v_ref_rope_mode`，可选值如下。

### 4.1 `legacy_time_offset`

默认值，完全保持当前实现。

```text
target:    3D RoPE, t = 0..f-1
reference: 3D RoPE, t = max(30, f + 9)
motion:    原有负时间/多尺度 RoPE
```

适用场景：

- 兼容已有 checkpoint。
- 作为 baseline。
- 训练/推理不希望改变位置编码分布。

### 4.2 `source_id_time_offset`

推荐的低风险迁移模式。参考帧仍使用当前时间偏移坐标，但额外乘上 Bernini 式 source_id 相位：

```text
reference_phase
  = phase_3d(t=max(30, f+9), h, w)
  + phase_source(source_id=1)
```

目标视频和 motion tokens 不乘 source_id 相位。

优点：

- 保留当前 S2V 参考帧的时间偏移先验。
- 只在参考帧 segment 上增加 source_id 区分信息。
- 从已有 Wan2.2-S2V 权重继续训练时，分布变化小于完全改成本地时间坐标。

缺点：

- 它不是 Bernini 最纯粹的“局部坐标 + source_id”形式，而是兼容性优先的混合方案。

### 4.3 `source_id_local`

更接近 Bernini 的模式。参考帧使用局部三维坐标，再乘 source_id 相位：

```text
reference_phase
  = phase_3d(t=0..rf-1, h, w)
  + phase_source(source_id=1)
```

目标视频仍保持当前坐标，motion tokens 也保持当前坐标。

优点：

- 与 Bernini 的 source_id 思路最一致。
- 参考帧不再依赖硬编码的 `max(30, f+9)` 时间位置。
- 后续多参考帧可统一使用局部坐标，通过不同 `source_id` 区分来源。

缺点：

- 会改变当前 S2V 参考帧的时间位置分布。
- 对直接加载原始 Wan2.2-S2V 权重继续训练更激进，需要更严格 ablation。

## 5. 参数设计

建议新增模型级参数：

```python
s2v_ref_rope_mode: str = "legacy_time_offset"
s2v_ref_source_id: float = 1.0
s2v_ref_rope_theta: float = 10000.0
s2v_ref_time_base: int = 30
s2v_ref_time_margin: int = 9
```

参数含义：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `s2v_ref_rope_mode` | `"legacy_time_offset"` | 选择参考帧 RoPE 模式 |
| `s2v_ref_source_id` | `1.0` | 参考帧 source_id，相位连续，允许 float |
| `s2v_ref_rope_theta` | `10000.0` | source_id 相位使用的 RoPE theta |
| `s2v_ref_time_base` | `30` | legacy 时间偏移下限 |
| `s2v_ref_time_margin` | `9` | legacy 时间偏移相对目标长度的 margin |

`s2v_ref_rope_mode` 支持：

```text
legacy_time_offset
source_id_time_offset
source_id_local
```

训练 CLI 建议增加同名参数：

```bash
--s2v_ref_rope_mode "legacy_time_offset"
--s2v_ref_source_id 1.0
--s2v_ref_rope_theta 10000.0
--s2v_ref_time_base 30
--s2v_ref_time_margin 9
```

训练 shell 脚本建议使用环境变量透传：

```bash
S2V_REF_ROPE_MODE=${S2V_REF_ROPE_MODE:-legacy_time_offset}
S2V_REF_SOURCE_ID=${S2V_REF_SOURCE_ID:-1.0}
S2V_REF_ROPE_THETA=${S2V_REF_ROPE_THETA:-10000.0}
S2V_REF_TIME_BASE=${S2V_REF_TIME_BASE:-30}
S2V_REF_TIME_MARGIN=${S2V_REF_TIME_MARGIN:-9}

MODEL_ARGS+=(
  --s2v_ref_rope_mode "${S2V_REF_ROPE_MODE}"
  --s2v_ref_source_id "${S2V_REF_SOURCE_ID}"
  --s2v_ref_rope_theta "${S2V_REF_ROPE_THETA}"
  --s2v_ref_time_base "${S2V_REF_TIME_BASE}"
  --s2v_ref_time_margin "${S2V_REF_TIME_MARGIN}"
)
```

推荐实验顺序：

```text
1. legacy_time_offset       baseline
2. source_id_time_offset    推荐首个迁移实验
3. source_id_local          Bernini-style ablation
```

## 6. 代码落点设计

### 6.1 `WanS2VModel.__init__`

在 `diffsynth/models/wan_video_dit_s2v.py::WanS2VModel.__init__` 增加可选参数并保存为属性。

为了兼容当前 model hash 加载流程，也建议提供 setter。因为 `ModelPool` 通过 hash 找到 `model_configs.py` 的 `extra_kwargs`，训练 CLI 不方便直接覆盖构造参数。加载后调用 setter 更稳妥：

```python
def configure_ref_rope(
    self,
    mode=None,
    source_id=None,
    theta=None,
    time_base=None,
    time_margin=None,
):
    if mode is not None:
        self.s2v_ref_rope_mode = mode
    if source_id is not None:
        self.s2v_ref_source_id = float(source_id)
    if theta is not None:
        self.s2v_ref_rope_theta = float(theta)
    if time_base is not None:
        self.s2v_ref_time_base = int(time_base)
    if time_margin is not None:
        self.s2v_ref_time_margin = int(time_margin)
```

### 6.2 source_id 相位生成

当前 S2V 已经有 `precompute_freqs_cis()`，但它只预计算整数位置。Bernini 支持 float source_id，因此建议新增一个按 position 动态计算的 helper：

```python
def get_1d_rope_phase_for_position(dim, position, theta=10000.0, device=None):
    inv_freq = 1.0 / (
        theta ** (
            torch.arange(0, dim, 2, dtype=torch.float64, device=device)[: dim // 2] / dim
        )
    )
    phase = torch.outer(
        torch.as_tensor([float(position)], dtype=torch.float64, device=device),
        inv_freq,
    )
    return torch.polar(torch.ones_like(phase), phase)
```

对 S2V 来说：

```text
dim = self.dim // self.num_heads = 128
source phase shape = [1, 64]
```

应用到 `rope_precompute()` 的输出时需要 reshape 成可广播形状：

```python
source_phase = source_phase.view(1, 1, 1, -1)
pre_compute_freqs[:, seq_len_x:seq_len_x + ref_seq_len] *= source_phase
```

`pre_compute_freqs` 的实际形状是：

```text
[1, total_seq_len, num_heads, head_dim / 2]
```

因此乘法只影响参考帧 token，不影响目标视频 token 和 motion tokens。

### 6.3 参考帧 grid 生成

建议把当前 `get_grid_sizes()` 改成依赖配置：

```python
def get_ref_time_id(self, f):
    return max(self.s2v_ref_time_base, int(f) + self.s2v_ref_time_margin)

def get_grid_sizes(self, grid_size_x, grid_size_ref):
    f, h, w = grid_size_x
    rf, rh, rw = grid_size_ref

    grid_sizes_x = torch.tensor([f, h, w], dtype=torch.long).unsqueeze(0)
    grid_sizes_x = [[torch.zeros_like(grid_sizes_x), grid_sizes_x, grid_sizes_x]]

    if self.s2v_ref_rope_mode == "source_id_local":
        ref_time_id = 0
    else:
        ref_time_id = self.get_ref_time_id(f)

    grid_sizes_ref = [[
        torch.tensor([ref_time_id, 0, 0]).unsqueeze(0),
        torch.tensor([ref_time_id + int(rf), rh, rw]).unsqueeze(0),
        torch.tensor([rf, rh, rw]).unsqueeze(0),
    ]]
    return grid_sizes_x + grid_sizes_ref
```

`legacy_time_offset` 和 `source_id_time_offset` 共用现有参考帧时间偏移；`source_id_local` 才把参考帧时间坐标切回局部起点。

### 6.4 统一 RoPE 构造，避免训练/推理漂移

当前 `WanS2VModel.forward()` 和 `model_fn_wans2v()` 各有一份 RoPE 构造逻辑。新增 source_id 后，建议把这段逻辑收敛到 `WanS2VModel` 方法中，例如：

```python
def build_s2v_rope(self, token_states, grid_sizes, seq_len_x, ref_seq_len):
    freqs = rope_precompute(
        token_states.detach().view(
            1,
            token_states.size(1),
            self.num_heads,
            self.dim // self.num_heads,
        ),
        grid_sizes,
        self.freqs,
        start=None,
    )
    if self.s2v_ref_rope_mode in {"source_id_time_offset", "source_id_local"}:
        source_phase = get_1d_rope_phase_for_position(
            self.dim // self.num_heads,
            self.s2v_ref_source_id,
            theta=self.s2v_ref_rope_theta,
            device=token_states.device,
        ).view(1, 1, 1, -1)
        freqs[:, seq_len_x:seq_len_x + ref_seq_len] = (
            freqs[:, seq_len_x:seq_len_x + ref_seq_len] * source_phase
        )
    return freqs
```

然后两个前向入口都调用同一个方法：

```python
grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
x = torch.cat([x, ref_latents], dim=1)
pre_compute_freqs = dit.build_s2v_rope(
    x,
    grid_sizes,
    seq_len_x=seq_len_x,
    ref_seq_len=ref_latents.shape[1],
)
```

这样训练、验证和 pipeline 推理使用同一套参考帧 RoPE 逻辑。

### 6.5 训练脚本配置注入

在 `examples/wanvideo/model_training/train.py::wan_parser()` 增加参数：

```python
parser.add_argument("--s2v_ref_rope_mode", type=str, default="legacy_time_offset")
parser.add_argument("--s2v_ref_source_id", type=float, default=1.0)
parser.add_argument("--s2v_ref_rope_theta", type=float, default=10000.0)
parser.add_argument("--s2v_ref_time_base", type=int, default=30)
parser.add_argument("--s2v_ref_time_margin", type=int, default=9)
```

在 `WanTrainingModule.__init__` 中接收这些参数，`WanVideoPipeline.from_pretrained()` 之后配置 DiT：

```python
if hasattr(self.pipe.dit, "configure_ref_rope"):
    self.pipe.dit.configure_ref_rope(
        mode=s2v_ref_rope_mode,
        source_id=s2v_ref_source_id,
        theta=s2v_ref_rope_theta,
        time_base=s2v_ref_time_base,
        time_margin=s2v_ref_time_margin,
    )
```

如果有 `dit2`，也应同步配置：

```python
if hasattr(self.pipe, "dit2") and hasattr(self.pipe.dit2, "configure_ref_rope"):
    self.pipe.dit2.configure_ref_rope(...)
```

当前 Wan2.2-S2V 只有一个 S2V DiT，但同步 `dit2` 可以避免以后混合专家或双 DiT 配置漏配。

### 6.6 推理侧配置

训练得到的 LoRA 或 full checkpoint 如果使用了非默认 RoPE，推理必须使用同一模式。

当前实现支持两种接入方式。

1. 在推理脚本加载 pipeline 后手动配置：

```python
pipe.dit.configure_ref_rope(
    mode="source_id_time_offset",
    source_id=1.0,
    theta=10000.0,
    time_base=30,
    time_margin=9,
)
```

2. 在 `WanVideoPipeline.__call__()` 中传同名入参：

```python
video = pipe(
    prompt=prompt,
    input_image=input_image,
    input_audio=input_audio,
    s2v_ref_rope_mode="source_id_time_offset",
    s2v_ref_source_id=1.0,
    s2v_ref_rope_theta=10000.0,
    s2v_ref_time_base=30,
    s2v_ref_time_margin=9,
    ...
)
```

不传这些参数时，`__call__()` 不会覆盖当前 `pipe.dit` 上已有的配置；训练验证脚本则通过 `S2V_REF_ROPE_MODE` 等环境变量显式设置。

## 7. 与数据 cache 的关系

`Wan2.2-S2V-14B-cache-run.sh` 预提取的是 pipeline features，例如 VAE latent、文本 embedding、audio embedding、pose latent。当前 S2V RoPE 在 DiT forward 时动态生成，不写入 feature cache。

因此：

- 已有 cache 可以复用。
- `s2v_ref_rope_mode` 改变后，不需要重跑 cache。
- 训练和验证必须在 DiT 前向时使用同一 RoPE 配置。

需要注意：如果未来把 RoPE 也写入 cache，则 `s2v_ref_rope_mode` 必须进入 cache key，否则会误用旧位置编码。

## 8. Checkpoint 兼容性

新增配置不引入可学习参数，只改变 RoPE 相位生成，因此：

- 旧 checkpoint 可以正常加载。
- `legacy_time_offset` 下应与当前行为 bitwise 或近似一致。
- 使用 `source_id_time_offset` 或 `source_id_local` 训练出来的 LoRA/full checkpoint，推理时必须设置相同 `s2v_ref_rope_mode`。
- 该配置不会自动保存在当前 state_dict 中，建议训练脚本把参数打印到日志，并在 checkpoint 目录额外保存一份训练参数。

如果要把模型发布成独立模型，建议把这些字段写入模型配置或 README：

```json
{
  "s2v_ref_rope_mode": "source_id_time_offset",
  "s2v_ref_source_id": 1.0,
  "s2v_ref_rope_theta": 10000.0,
  "s2v_ref_time_base": 30,
  "s2v_ref_time_margin": 9
}
```

## 9. 验证方案

### 9.1 单元验证

建议增加最小测试覆盖：

1. `legacy_time_offset` 与当前实现输出一致。
2. `source_id_time_offset` 只改变参考帧 segment：

```text
target segment: allclose(new, old)
ref segment:    new = old * source_phase
motion segment: allclose(new, old)
```

3. `source_id_local` 只改变参考帧 segment 的 grid 坐标和 source phase，目标/motion 不变。
4. source phase 支持 float：

```text
s2v_ref_source_id = 1.0
s2v_ref_source_id = 1.5
```

5. shape 不变：

```text
pre_compute_freqs: [1, total_seq_len, num_heads, head_dim/2]
```

### 9.2 训练 smoke test

用极小数据跑 1 到 5 step：

```bash
S2V_REF_ROPE_MODE=legacy_time_offset \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh

S2V_REF_ROPE_MODE=source_id_time_offset \
bash examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh
```

检查：

- loss 正常反传。
- 没有 complex dtype/device mismatch。
- `use_gradient_checkpointing_offload` 下正常。
- cache 训练 `sft:train` 和 raw 训练 `sft` 都能跑通。

### 9.3 推理一致性

对同一 checkpoint 做对照：

```text
训练 mode = legacy_time_offset       推理必须 legacy_time_offset
训练 mode = source_id_time_offset    推理必须 source_id_time_offset
训练 mode = source_id_local          推理必须 source_id_local
```

特别注意 LoRA：LoRA 只保存增量权重，不保存本设计的 RoPE 配置。验证脚本和推理脚本必须显式设置。

## 10. 推荐实施顺序

1. 先新增参数和 setter，默认 `legacy_time_offset`。
2. 把 S2V RoPE 构造收敛到 `WanS2VModel.build_s2v_rope()`。
3. 在 `legacy_time_offset` 下做等价性测试，确认没有行为变化。
4. 实现 `source_id_time_offset`，先跑 LoRA 小规模训练。
5. 再实现/打开 `source_id_local` 做 Bernini-style ablation。
6. 把训练参数同步到 full、LoRA、cache、validate 脚本。

推荐首个实验配置：

```bash
S2V_REF_ROPE_MODE=source_id_time_offset
S2V_REF_SOURCE_ID=1.0
S2V_REF_ROPE_THETA=10000.0
S2V_REF_TIME_BASE=30
S2V_REF_TIME_MARGIN=9
```

理由：它保留当前参考帧的时间偏移位置，同时只给参考帧增加 Bernini 的 source_id 相位，迁移风险最低。

## 11. 结论

Bernini 的关键增量不是重新发明 3D RoPE，而是在原有三轴 RoPE 上乘一个连续的 `source_id` 旋转相位，用于区分 target latent 和不同条件源。

迁移到 Wan2.2-S2V 时，不建议全局替换位置编码。更稳妥的方案是只对参考帧 token 增加可配置 source_id RoPE：

```text
默认:      legacy_time_offset，完全保留现有行为
推荐迁移:  source_id_time_offset，当前参考帧时间偏移 + source_id
消融实验:  source_id_local，Bernini-style 局部坐标 + source_id
```

这样既保留 Wan2.2-S2V 的原始 target/motion 位置编码方案，又可以把 Bernini 的 source_id 思路引入参考帧分支，并通过参数控制训练和推理使用同一种编码方式。
