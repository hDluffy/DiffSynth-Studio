# FaceID S2V LoRA 能力增强：分阶段对照实验

适用范围：Wan2.2-S2V-14B、每条视频单一身份、81 帧、16 fps、ArcFace iResNet100、ZeRO-3 LoRA。目标是提高**完整推理视频**的人脸身份保持，同时保留嘴形、表情、动作和画质。

本文件只安排实验，不改变训练实现。操作入口见 [README.md](README.md) 和 [run.sh](run.sh)。

## 一、当前事实与问题定位

此前核查得到：

- 本地 ArcFace 与官方 iResNet100 在相同权重、测试输入下输出一致。
- 抽查 128 条缓存，123 条参考有效；参考 embedding 归一化正常，没有非有限值。
- 8 条原视频重新提取参考 embedding，与缓存 cosine 约为 0.9999999。
- 同一个真实视频不同帧与参考图的 cosine 也可能从约 0.85 降到 0.45～0.50。因此训练损失的变化同时包含人物、姿态、画质、对齐和噪声难度，不完全是模型能力变化。
- 全区间 timestep 索引采样下，实际 sigma 落入 [0.1, 0.5] 的比例曾约为 14%，并非 40%。`every=2` 后每个 rank 的身份监督更加稀疏。
- 当前每次监督同一解码块的两帧，相关性高；8 卡训练也经常只有一个 rank 激活身份分支。
- 当前质量权重是检测置信度，尚无专门的模糊度、姿态、遮挡质量估计。
- 当前身份梯度只回传选定 latent chunk；历史解码上下文的梯度被截断。

这些结果排除了部分明显实现错误，但不证明所有缓存正确，也不能证明训练指标可以直接代表推理质量。

## 二、所有实验统一的条件

### 2.1 固定数据、初始化和训练预算

1. 固定训练 CSV。按身份划分验证/测试集；若只有一个身份，按原视频来源/拍摄场次划分。不能将同一视频的相邻裁剪片段分散到训练和测试集。
2. 建议先选 32～64 个固定验证案例，覆盖正脸、侧脸、转头、大表情、不同人脸大小与光照。该数量只是启动规模，不是统计充分性的保证。
3. 固定 base checkpoint、VAE、ArcFace、RoPE、LoRA rank/target modules、学习率、GPU 数、梯度累积、分辨率和采样配置。
4. 每组从**同一个 base 和同一个初始 LoRA**开始。不能把 A 从头训练、B 从 step-450 续训当成单变量对照。
5. 比较共同 checkpoint 步数，例如新增训练 step 50/100/200/400；同时记录耗时和实际身份监督样本数。不能只比较 epoch。
6. 初筛每组观察到 step 200～400；入选方案再延长，避免未经验证直接跑长训。实际轮数需保证产生目标 checkpoint。
7. 入选结果至少重复两次独立运行。现有入口未暴露固定训练 seed/严格确定性开关，独立运行存在随机差异；精确控制数据顺序、Python/NumPy/PyTorch seed 属于后续开发项。推理评估必须固定 seed。
8. 不同时改权重、学习率、sigma 采样和筛选阈值。每个实验记录“相对于谁，只变了什么”。

当前入口只恢复 base/LoRA 权重，不恢复 optimizer、global step、scheduler 和 FaceID warmup。所有续训分支都应从同一个权重文件以相同方式重新启动。`warmup=50` 表示从 0 升到目标权重，不是从上次权重接着升。

### 2.2 固定推理评估

- 固定参考图、音频、prompt、negative prompt、seed、分辨率、81 帧、采样器、scheduler、步数、CFG。
- LoRA strength 首轮固定为 1.0。选择最佳训练方案后，再单独扫描 strength，如 0.5/0.75/1.0，不与训练参数实验混在一起。
- 必须使用训练对应的 base。若用四步蒸馏底座，先记录实际推理 sigma 序列；确认模型求值点是否覆盖拟训练的 sigma 区间。文件名里的“4step”不足以确认。
- 若 ComfyUI 需要转换，使用仓库根目录的 `script/convert_lora_to_comfyui.py`；训练继续加载原始 LoRA 文件。

### 2.3 评价指标与选优规则

| 类型 | 指标 | 用途 |
|---|---|---|
| 主指标 | 固定完整生成视频与参考图的身份 cosine：每视频均值、低分位数 | 关注整体提升及身份漂移最严重部分 |
| 独立验证 | 不同于训练 ArcFace 的识别模型分数 | 避免仅迎合训练编码器 |
| 稳定性 | 每视频跨帧身份分数波动、人工身份漂移检查 | 区分平均变好与局部崩坏 |
| 约束指标 | 嘴形同步、表情幅度、转头动作、画质、伪影 | 防止身份增强以牺牲 S2V 功能为代价 |
| 覆盖率 | 生成帧检测成功率、可评估人脸数量 | 检测失败不能被悄悄过滤后报高分 |
| 效率 | optimizer step、墙钟时间、FaceID 有效样本数、显存峰值 | 不同采样策略的预算解释 |

先逐视频计算，再跨视频汇总，避免长视频或可检测帧更多的视频支配指标。对同一验证案例使用成对差值；有条件时按视频 bootstrap 估计置信区间。不得从多个 checkpoint 中挑训练 loss 最低的一步直接宣称最佳，应在固定验证集选型，最后用未参与选型的测试集确认。

没有通用的 cosine 提升门槛。先固定一组基线重复推理，确定随机波动，再预先定义可接受的嘴形/画质退化范围。出现明显脸僵硬、嘴部动作缩小、身份混淆或画质崩坏时停止加权。

## 三、阶段 0：建立可信诊断基线（先做）

### D0：三路身份对照与裁剪可视化【需要补充评估代码】

在固定案例的固定帧（前/中/后）、固定 sigma（例如 0.15/0.30/0.45）和固定噪声上比较：

1. 真实视频帧 → ArcFace → 与参考图比较。
2. 缓存干净 latent → VAE 解码 → 同样对齐 → 与参考图比较。
3. 模型预测干净 latent → VAE 解码 → 同样对齐 → 与参考图比较。

保存参考图与三路人脸裁剪，并记录姿态、检测大小、目标帧索引及对齐矩阵。统计 decoder 输出超出 [-1,1] 的比例。检查 tiled 编码/身份分支解码是否引入可见偏移。

解释：

- 第一路就低：先查数据、参考图、姿态、对齐，不能完全归咎于模型。
- 第二路明显劣化：先查 VAE 重建/解码分布与坐标匹配。
- 只有第三路低：更支持模型训练不足。

真实视频分数是诊断基准，不是严格的数学上限；模型可能生成更正脸、更静态的脸而获得更高分，因此必须一起看动作与画质。

### D1：无 LoRA、现有 LoRA、FM-only 基线

- B0：base 无 LoRA，仅推理，不训练。
- B1：选定共同初始 LoRA，仅推理，不训练。
- F0：从共同初始化继续训练，`FACEID_WEIGHT=0`，其余条件一致。

F0 用于区分普通 SFT 带来的提升与身份损失的增益。当前入口即使 weight=0 也会加载教师，所以它是算法对照，不是最省资源的 FM 性能基准。

## 四、阶段 1：增加有效监督频率【现有脚本支持】

所有组使用同一份 sim0.2 缓存（或明确选定的其他阈值缓存），不能在此阶段换数据。

| ID | every | FaceID weight | warmup | 学习率 | 对比目标 |
|---|---:|---:|---:|---:|---|
| F0 | 1 | 0 | 50 | 1e-5 | FM-only 对照 |
| F1 | 2 | 0.05 | 50 | 1e-5 | 原始稀疏身份监督 |
| F2 | 1 | 0.05 | 50 | 1e-5 | 只提高调度频率 |

保持 sigma gate=[0.1,0.5]、原始全范围采样、每次两帧。F2 对 F1 检验增加身份更新机会是否有效；不能仅因 F2 的当步 loss 更平滑就判胜。

检查：`faceid/active`、`faceid/eligible`、有效监督累计数量、同训练 step 及相近墙钟预算的完整推理结果。`active` 是各 rank 激活比例；单样本/卡且梯度累积为 1 时，累加 `active × world_size` 得到 rank 级身份监督次数。

## 五、阶段 2：提高身份损失权重【现有脚本支持】

从同一初始化并行比较，不用上一组最终模型作为下一组起点：

| ID | every | FaceID weight | 其他设置 |
|---|---:|---:|---|
| W0 | 1 | 0.05 | 复用 F2 结果 |
| W1 | 1 | 0.1 | 相同 warmup、学习率、缓存 |
| W2 | 1 | 0.5 | 仅 W1 未明显损伤嘴形/表情时尝试 |
| W3 | 1 | 1.0 | 可选，高风险加强组，先短程验证 |

默认保持 warmup=50、lr=1e-5，不同时加学习率。优先比较 W0/W1，随后再决定是否需要 W2/W3。你的目标确实是身份增强，但“loss 数值占主导”不等于“LoRA 梯度占主导”。

后续建议增加 FM/FaceID 对 LoRA 的分项梯度范数及方向夹角诊断【需要开发】。ZeRO-3 下不能简单读取 `.grad` 当作完整全局梯度，需要正确聚合分片，并控制额外 backward 开销。

## 六、阶段 3：改变 sigma 采样【已实现】

`FACEID_SIGMA_MIN/MAX` 目前只是计算身份损失的门控。它们**不会限制 timestep 的采样范围**。`min/max_timestep_boundary` 是索引比例，不等于 sigma，不用它冒充区间采样。

现已支持 `SIGMA_SAMPLING_MODE=original|mixed|interval` 和 `SIGMA_SAMPLING_PROB`；完整设置见 [SIGMA_SAMPLING.md](SIGMA_SAMPLING.md)。区间按 BF16 舍入后的实际 sigma 筛选。

以阶段 2 的较保守优胜配置为参照；首次改变采样时建议 weight=0.05 或 0.1，并在原采样组设置相同权重重新对照。

| ID | 有效区间采样 | 原始全范围采样 | 目的 |
|---|---:|---:|---|
| S0 | 0% | 100% | 对照 |
| S1 | 50% | 50% | 温和增加有效监督 |
| S2 | 75% | 25% | 身份为主要训练内容，保留全范围约束 |
| S3 | 100% | 0% | 专项短程对照，不先设为默认方案 |

“75% 区间采样”指 75% 概率从真实 scheduler.sigmas 符合范围的索引中抽样；其余全范围分支也可能落入有效区间，所以实际命中率会高于 75%。记录真实命中率。

必须说明 FM 的优化目标：不做重要性加权时，混合采样会有意提高该区间的 FM 权重。若要保持原始 FM 期望，应按采样概率加权并单独作为实验。纯区间采样无法覆盖区间外目标，不能仅靠加权恢复全区间 FM。

风险：LoRA 作用于所有推理噪声阶段，区间外缺少约束；低训练 loss 不保证完整生成改善。对四步底座尤需检查实际推理 sigma。

sigma 区间宽度（如 [0.1,0.35] 对 [0.1,0.5]）留到采样策略选定后再单独试验；缩窄区间也会减少全范围采样时的激活率，不能把更低原始 loss 直接当改善。

## 七、阶段 4：数据筛选与身份特征方案

### Q0/Q1/Q2：候选帧阈值【现有脚本支持，需要重建缓存】

对完全相同 CSV、VAE、ArcFace、分辨率及采样配置，分别建：

| ID | min_similarity | 目的 |
|---|---:|---|
| Q0 | 0.2 | 宽松保留侧脸、大表情 |
| Q1 | 0.4 | 中等筛选 |
| Q2 | 0.5 | 更严格的参考相似度筛选 |

固定同一训练配方对比。阈值高意味着任务可能更容易，并不等于高质量；严重侧脸可能被系统性排除。必须保留困难姿态的独立验证集，检查是否“正脸变好、侧脸变差”。

记录 audit 的总样本、参考有效率、可监督样本数、每视频有效候选帧数量；这些缓存仍保留无有效身份监督的样本用于 FM，不是把整个训练集简单删小。

### T0/T1/T2：身份目标【需要开发和升级缓存】

| ID | 身份目标 | 目的 |
|---|---|---|
| T0 | 只对参考图 embedding | 当前基线 |
| T1 | 参考图与对应真实帧 embedding 各占 0.5 | 降低姿态/表情差异干扰 |
| T2 | 只对对应真实帧 embedding | 消融，判断是否偏离参考条件 |

使用 `L_ID = alpha*L_reference + (1-alpha)*L_paired`，身份项外部总权重保持不变。逐帧目标 embedding 必须与真实采样帧一致且 detach。不要把两项都乘 1 后，与原单项直接比较，否则同时改变目标和梯度强度。

后续可单独比较“首帧参考”与“同身份高质量多帧归一化均值参考”；但多帧参考改变了与推理参考图的一致性，不能直接认定必然更好。避免为提高分数而把同身份姿态多样性全部过滤。

## 八、阶段 5：加强身份主导、稳定 batch【需要开发】

只在前述效果证实后逐项进行：

1. 新增 `FM_WEIGHT`：对照 1.0 / 0.5 / 0.2，固定身份权重与采样。不能现阶段只 export FM_WEIGHT 就期待生效。
2. 全局有效 FaceID 样本归一化：明确是按有效样本、有效帧还是有效 rank 求平均。无有效样本时安全返回零；所有 rank 同步参与必要 collective。
3. 更稳定的有效身份 batch：增加独立样本覆盖或身份分支有效更新次数。不能把两张相邻帧当成两个独立身份样本。
4. 更多时间窗口：对照每样本一个窗口/两个间隔较远窗口，评估额外解码、CPU offload 和梯度图的开销。当前 FACEID_FRAMES 只支持同一窗口 1 或 2 帧，不支持跨窗口。
5. 学习率最后再比较，如 5e-6 / 1e-5 / 2e-5，其他条件固定；不是通过加大学习率掩盖目标问题。

归一化改变身份项强度，必须重新校准权重，不能与“加大 FACEID_WEIGHT”同时进行。仅 rank0 有效时按全局有效数归一化可能明显放大身份梯度，不是无代价的日志优化。

## 九、当前可执行命令模板

以下示例从仓库根目录执行。先根据自己的文件替换路径；不要直接使用不存在的占位符。

```bash
cd /data-training/hjq/DiffSynth-Studio
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export FACEID_WEIGHTS="$PWD/faceid_model/backbone.pth"
export FACEID_DETECTOR_PATH="$PWD/faceid_model/det_10g.onnx"
export FACEID_VAE_PATH=/data-training/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors
export MODEL_BASE_PATH=/data-training/models
export NUM_PROCESSES=8
export MAX_PIXELS=589824
export LEARNING_RATE=1e-5
export LORA_RANK=32
export FACEID_FRAMES=2
export FACEID_SIGMA_MIN=0.1
export FACEID_SIGMA_MAX=0.5
export FACEID_WARMUP_STEPS=50
export SAVE_STEPS=50
export NUM_EPOCHS=2  # 按实际数据量确认足以生成 step-400；不是 max_steps
export CACHE_PATH=/data-training/multi_face/cache_faceid_f81

# 方案 A：所有组都从原始 base 新建 LoRA
unset LORA_CHECKPOINT
# 若使用官方 base：
unset BASE_CHECKPOINT
# 若使用自定义 base，则所有组显式固定同一路径：
# export BASE_CHECKPOINT=/absolute/path/base.safetensors

# 或方案 B：所有组从同一训练 checkpoint 启动；不要与方案 A 混用
# export LORA_CHECKPOINT=/absolute/path/common_initial_lora.safetensors
```

先核对当前缓存的阈值：以 `_cache_manifest.json` 为准。当前 run.sh 的默认 `FACEID_MIN_SIMILARITY` 已是 0.5，旧缓存可能仍是 0.2，所以重建时必须显式设置，不能依赖历史默认值。

依次执行 F0/F1/F2，分别完成后评估相同步数：

```bash
FACEID_EVERY=1 FACEID_WEIGHT=0 \
OUTPUT_PATH=./models/train/faceid_experiments/F0_fm_only \
bash examples/wanvideo/model_training/faceid/run.sh train

FACEID_EVERY=2 FACEID_WEIGHT=0.05 \
OUTPUT_PATH=./models/train/faceid_experiments/F1_every2_w005 \
bash examples/wanvideo/model_training/faceid/run.sh train

FACEID_EVERY=1 FACEID_WEIGHT=0.05 \
OUTPUT_PATH=./models/train/faceid_experiments/F2_every1_w005 \
bash examples/wanvideo/model_training/faceid/run.sh train
```

上述命令是完整训练命令，不会自动在 step-400 停止；现有脚本没有 max_train_steps。可统一使用相同 epoch 数并比较共同 checkpoint，或开发精确步数停止机制后再严格等预算运行。不要在同一组输出目录混入多次不同配置的事件文件。

权重组 W1/W2（W0 复用 F2）：

```bash
FACEID_EVERY=1 FACEID_WEIGHT=0.1 \
OUTPUT_PATH=./models/train/faceid_experiments/W1_every1_w01 \
bash examples/wanvideo/model_training/faceid/run.sh train

# 仅评估 W1 后有必要时执行
FACEID_EVERY=1 FACEID_WEIGHT=0.5 \
OUTPUT_PATH=./models/train/faceid_experiments/W2_every1_w05 \
bash examples/wanvideo/model_training/faceid/run.sh train
```

阈值组缓存（同一训练 CSV，不包含验证集；每个配置使用新目录）：

```bash
export DATASET_BASE_PATH=/data-training/multi_face
export DATASET_METADATA_PATH=/absolute/path/train_only.csv

FACEID_MIN_SIMILARITY=0.4 \
CACHE_PATH=/data-training/multi_face/cache_faceid_f81_sim04 \
bash examples/wanvideo/model_training/faceid/run.sh cache

FACEID_MIN_SIMILARITY=0.5 \
CACHE_PATH=/data-training/multi_face/cache_faceid_f81_sim05 \
bash examples/wanvideo/model_training/faceid/run.sh cache

python examples/wanvideo/model_training/faceid/audit_cache.py \
  /data-training/multi_face/cache_faceid_f81_sim04
python examples/wanvideo/model_training/faceid/audit_cache.py \
  /data-training/multi_face/cache_faceid_f81_sim05
```

Q 组训练时只更换 CACHE_PATH 和 OUTPUT_PATH，其他参数使用选定的同一配方。S/T 阶段不提供伪命令：需要先实现对应功能再运行。

## 十、TensorBoard 如何比较

```bash
tensorboard --logdir ./models/train/faceid_experiments --host 0.0.0.0 --port 6006
```

优先查看：

- `loss/faceid_when_active`：只对当步有效 rank 的身份 loss 求均值。
- `faceid/cosine_when_active`：只对有效 rank 的身份 cosine 求均值。
- `faceid/active`、`faceid/eligible`：解释该点由多少有效样本构成。
- `loss/fm`、`loss/faceid_weighted`、`loss/total`：监督强度与原任务是否异常。
- `faceid/weight`：实际 warmup 权重。
- `memory/cuda_peak_allocated_gib`：累计峰值，不能当成当前单步峰值。

相同平滑系数可辅助观察，但平滑不是训练改善。某步只有 1 个 rank 激活和 4 个 rank 激活时，单步 when_active 曲线的统计可靠性不同。跨 100 步汇总时应同时报告有效次数；推荐用 `sum(loss/faceid) / sum(faceid/active)` 求有效 rank 加权的窗口平均，而不是直接平均所有 when_active 点。全窗口无有效监督时记为缺失，不记作 0。

`loss/faceid_weighted` 会受激活率和 warmup 影响，不能单独用于判断身份提升。不同阈值/噪声区间的 raw loss 不可直接横向比较，必须看统一验证集。

## 十一、实验记录表（每组填写）

| 字段 | 记录 |
|---|---|
| 实验 ID / 对照 ID / 唯一改动 | |
| 代码版本 / 启动命令 / 日期 | |
| base 路径与 SHA256 / 初始 LoRA 路径与 SHA256 | |
| train CSV 指纹 / cache manifest 指纹 | |
| ArcFace、VAE 权重指纹 | |
| 缓存阈值 / 有效参考样本 / 有效监督帧 | |
| LR / rank / target modules / GPUs / grad accumulation | |
| FM 权重 / FaceID 权重 / warmup / every | |
| sigma gate / 采样策略 / 实际有效率 | |
| 训练新增 step / 身份有效样本次数 / 总耗时 | |
| 评估 checkpoint / 推理完整参数 / seed | |
| 验证身份均值 / 低分位数 / 检测失败率 | |
| 嘴形、表情、转头、画质人工对照结论 | |
| 是否优于对照 / 是否复现实验 / 下一步 | |

CSV 汇总模板见同目录 `experiment_results_template.csv`。新增 sigma 策略、验证脚本等功能时应同时记录版本，避免把实现变化当成参数变化。

## 十二、最小推荐执行顺序

1. 固定数据划分、共同初始化与推理协议，完成 D0 三路诊断。
2. 评估 B0/B1，训练 F0/F1/F2。
3. F2 有明确收益后试 W1；有必要才试 W2/W3。
4. 使用现有采样参数对比 S0/S1/S2，S3 仅作短程边界对照。
5. 固定选定配方，对比 Q0/Q1/Q2，避免严格筛选掩盖困难姿态退化。
6. 再开发 T1/T2 的逐帧目标；最后才调 FM 权重、全局归一化及学习率。
7. 对前两名重复训练、验证，最终在保留测试集上确认完整生成效果。

不需要把所有组合做笛卡尔积。每阶段只保留有证据的候选方案；若 D0 暴露明显对齐/重建问题，优先修复，再重新建立后续基线。
