# MiniMax-H3 Ref2VA 数据采集、制作与交付规范

> 文档用途：本文件可直接作为数据外包公司的采集、制作、标注、质检和交付依据。
> 适用工程：DiffSynth-Studio MiniMax-H3 Ref2VA 全参训练与 LoRA 训练。
> 目标画布：横屏 1280×736 或竖屏 736×1280，恒定 24 FPS。
> 目标时长：名义 5～15 秒；训练成品必须满足 `17n+5` 帧，实际合法范围为 124～345 帧（约 5.167～14.375 秒）。
> 规范版本：v2.0。

## 1. 项目目标

每条 Ref2VA 训练样本由以下内容组成：

1. 一段作为训练目标的带声音视频；
2. 一项或多项参考条件，包括参考图像、参考视频、参考音频或带声音的参考视频；
3. 一条准确描述“如何利用参考条件生成目标音视频”的 Prompt；
4. 可追溯的样本编号、授权信息和质检记录。

Ref2VA 的训练目标不是简单的视频描述，而是学习以下映射：

```text
Prompt + References -> Target Video + Target Audio
```

因此，参考素材、Prompt 和目标音视频之间必须存在明确、可验证的关系。例如：

- 参考图像提供人物、商品、服装、场景或视觉风格；
- 参考视频提供主体身份、动作、镜头、构图或待编辑的源视频；
- 参考音频提供音色、说话风格、音乐、环境声或待复用声轨；
- Prompt 说明哪些内容应保留、哪些内容应改变，以及目标画面和目标声音最终是什么。

## 2. 术语和样本边界

| 术语 | 含义 |
|---|---|
| Target Video | 模型应学习生成的目标视频，对应元数据字段 `video` |
| Target Audio | 与目标视频同步的目标音频，对应元数据字段 `input_audio` |
| Reference | 模型生成目标时使用的参考条件，对应 `references` |
| Picture Reference | 参考图像，类型为 `image` |
| Video Reference | 无声参考视频，类型为 `video` |
| Audio Reference | 纯音频参考，类型为 `audio` |
| Video-Audio Reference | 同时提供画面和声轨的参考视频，类型为 `video_audio` |
| Prompt | 对目标内容、参考关系、动作、镜头和声音的文字说明 |
| Sample | 一条完整的“目标音视频 + 参考条件 + Prompt”训练记录 |

一个样本应当是语义完整的最小训练单元。每个样本只包含一段 5～15 秒范围内、帧数合法的目标片段，不允许在一条记录中串联多个互不相关的视频。横屏和竖屏是两种独立的交付规格，不得通过 rotation metadata 互相代替。

## 3. 最终交付物

每个交付批次必须包含以下内容：

```text
delivery_YYYYMMDD_batchNN/
├── metadata.json
├── media/
│   ├── targets/
│   │   ├── S00000001_target.mp4
│   │   └── ...
│   └── references/
│       ├── images/
│       ├── videos/
│       └── audios/
├── qa/
│   ├── qa_manifest.csv
│   ├── rights_manifest.csv
│   └── rejected_samples.csv
└── README_DELIVERY.md
```

交付要求：

- `metadata.json`：训练程序直接读取的 UTF-8 JSON 文件；
- `media/`：所有目标和参考素材，不接受依赖公网 URL 的正式交付；
- `qa_manifest.csv`：逐样本技术检测和人工审核结果；
- `rights_manifest.csv`：来源、授权、肖像权、音乐版权等记录；
- `rejected_samples.csv`：本批次被剔除的样本及原因，防止后续重复交付；
- `README_DELIVERY.md`：批次数量、类型分布、已知问题、制作工具版本和联系人。

除非甲方书面要求，不要交付缓存、中间工程文件、代理视频、缩略图或重复转码版本。

## 4. 文件命名规范

### 4.1 样本编号

样本编号采用：

```text
S + 8 位十进制数字
```

例如：

```text
S00000001
S00000002
```

编号在整个项目中必须全局唯一，返工时不得更换编号。

### 4.2 文件名

```text
S00000001_target.mp4
S00000001_ref_img_01.png
S00000001_ref_video_01.mp4
S00000001_ref_audio_01.wav
```

要求：

- 只使用英文字母、数字、下划线和扩展名；
- 不使用空格、中文、括号、`#`、`?`、`&` 等特殊字符；
- 文件名大小写固定为小写，样本编号 `S` 除外；
- 元数据中使用相对于交付根目录的 POSIX 路径，即统一使用 `/`；
- 不允许两个路径指向内容相同但文件名不同的重复文件，复用参考素材时应复用同一路径。

## 5. 训练元数据结构

### 5.1 顶层结构

`metadata.json` 必须是 JSON 数组，每个元素是一条样本：

```json
[
  {
    "sample_id": "S00000001",
    "domain": "ecommerce",
    "task_type": "product_showcase",
    "orientation": "landscape",
    "target_width": 1280,
    "target_height": 736,
    "target_fps": 24,
    "target_frames": 124,
    "video": "media/targets/S00000001_target.mp4",
    "input_audio": "media/targets/S00000001_target.mp4",
    "prompt": "……",
    "references": [
      {
        "type": "image",
        "image": "media/references/images/S00000001_ref_img_01.png"
      }
    ]
  }
]
```

### 5.2 字段定义

| 字段 | 必填 | 类型 | 说明 |
|---|---:|---|---|
| `sample_id` | 是 | string | 全局唯一样本编号，用于追踪；训练代码会忽略此辅助字段 |
| `task_type` | 是 | string | 任务分类，见第 10 节；训练代码会忽略此辅助字段 |
| `video` | 是 | string | 目标视频相对路径 |
| `domain` | 是 | string | 垂域分类，例如 `ecommerce`、`tourism`、`digital_human`；训练代码会忽略此辅助字段 |
| `orientation` | 是 | string | 只允许 `landscape` 或 `portrait` |
| `target_width` | 是 | integer | 横屏为 1280，竖屏为 736 |
| `target_height` | 是 | integer | 横屏为 736，竖屏为 1280 |
| `target_fps` | 是 | integer | 固定为 24 |
| `target_frames` | 是 | integer | 目标视频实际帧数，必须满足 `17n+5`，范围见第 7 节 |
| `input_audio` | 是 | string | 当前训练加载器要求填写带目标声轨的目标 MP4，与 `video` 使用同一路径 |
| `prompt` | 是 | string | 非空 Prompt，规则见第 9 节 |
| `references` | 是 | array | 一个或多个参考块，顺序具有语义，不得随意调整 |
| `source_group_id` | 建议 | string | 同一原始长视频、人物、商品或录音来源的分组编号，用于去重和数据划分 |
| `prompt_language` | 建议 | string | `zh`、`en` 或 `mixed` |
| `rights_id` | 建议 | string | 对应 `rights_manifest.csv` 的授权记录编号 |

训练代码的最小必需字段是 `video`、`input_audio`、`prompt` 和 `references`。本项目交付验收还要求填写 `sample_id`、`domain`、`task_type`、`orientation`、`target_width`、`target_height`、`target_fps` 和 `target_frames`；这些辅助字段用于分桶、统计和质检，当前训练代码会安全忽略。

### 5.3 四种参考块

#### 参考图像

```json
{
  "type": "image",
  "image": "media/references/images/S00000001_ref_img_01.png"
}
```

#### 无声参考视频

```json
{
  "type": "video",
  "video": "media/references/videos/S00000002_ref_video_01.mp4"
}
```

`video` 类型只使用画面。即使文件中存在声轨，训练 Pipeline 也不会把它当作音频参考。需要声轨时必须使用 `video_audio`。

#### 参考音频

```json
{
  "type": "audio",
  "audio": "media/references/audios/S00000003_ref_audio_01.wav"
}
```

#### 带声音参考视频

```json
{
  "type": "video_audio",
  "video": "media/references/videos/S00000004_ref_video_01.mp4",
  "audio": "media/references/videos/S00000004_ref_video_01.mp4"
}
```

`video` 和 `audio` 可以指向同一个 MP4，也可以分别指向已严格同步的视频和 WAV。若采用分离文件，两者的起止时间必须一致。

### 5.4 多参考样本

```json
{
  "sample_id": "S00000005",
  "domain": "digital_human",
  "task_type": "video_edit_voice_reference",
  "orientation": "portrait",
  "target_width": 736,
  "target_height": 1280,
  "target_fps": 24,
  "target_frames": 192,
  "video": "media/targets/S00000005_target.mp4",
  "input_audio": "media/targets/S00000005_target.mp4",
  "prompt": "……",
  "references": [
    {
      "type": "video_audio",
      "video": "media/references/videos/S00000005_ref_video_01.mp4",
      "audio": "media/references/videos/S00000005_ref_video_01.mp4"
    },
    {
      "type": "audio",
      "audio": "media/references/audios/S00000005_ref_audio_02.wav"
    },
    {
      "type": "image",
      "image": "media/references/images/S00000005_ref_img_01.png"
    }
  ]
}
```

单条样本建议使用 1～3 个参考块。超过 4 个参考块会显著增加标注复杂度和模型输入长度，除非任务方案明确要求，否则不予验收。

## 6. Reference 编号规则

这是外包标注中最容易出错的部分，必须严格执行。

### 6.1 按模态分别编号

模型按以下三个序列分别从 1 编号：

- 图像：`<Picture 1>`、`<Picture 2>`……
- 视频：`<Video 1>`、`<Video 2>`……
- 音频：`<Audio 1>`、`<Audio 2>`……

编号不是 `references` 数组的统一序号。

### 6.2 `video_audio` 同时占用两种编号

例如：

```json
"references": [
  {"type": "video_audio", "video": "a.mp4", "audio": "a.mp4"},
  {"type": "audio", "audio": "voice.wav"},
  {"type": "image", "image": "product.png"},
  {"type": "video", "video": "motion.mp4"}
]
```

对应关系为：

| 参考块 | Prompt 中的名称 |
|---|---|
| `a.mp4` 的声轨 | `<Audio 1>` |
| `a.mp4` 的画面 | `<Video 1>` |
| `voice.wav` | `<Audio 2>` |
| `product.png` | `<Picture 1>` |
| `motion.mp4` | `<Video 2>` |

### 6.3 Prompt 引用要求

- 结构化 Prompt 必须使用精确形式 `<Picture N>`、`<Video N>`、`<Audio N>`；
- 简短中文 Prompt 可以使用“图片1、视频1、音频1”，但同一项目必须统一写法；
- 禁止写“上图”“前面的音频”“这个视频”“附件”等依赖界面位置的表达；
- Prompt 中出现的每个编号必须有对应参考素材；
- 每个参考素材都必须在 Prompt 中说明用途，不允许存在未被解释的无效参考。

## 7. 目标视频与音频技术规范

### 7.1 目标视频硬性要求

| 项目 | 验收标准 |
|---|---|
| 容器 | MP4 |
| 视频编码 | H.264/AVC，建议 High Profile |
| 像素格式 | `yuv420p` |
| 横屏分辨率 | `1280×736`，即宽 1280、高 736 |
| 竖屏分辨率 | `736×1280`，即宽 736、高 1280 |
| 帧率 | 恒定 24 FPS，不接受可变帧率 |
| 帧数 | 124～345 帧范围内的合法 `17n+5` 帧数 |
| 名义时长 | 5～15 秒 |
| 实际训练时长 | 约 5.167～14.375 秒 |
| 扫描方式 | 逐行扫描，不接受隔行扫描 |
| 旋转 | 像素已正向，不依赖 rotation metadata |
| 黑边 | 不允许 letterbox/pillarbox 黑边 |
| 水印/字幕 | 默认不允许，除非任务本身是 UI、字幕或文字生成 |

横屏和竖屏是两个明确的数据桶：

```text
landscape: width=1280, height=736
portrait:  width=736,  height=1280
```

禁止以下做法：

- 把竖屏内容放进 1280×736 横屏画布并添加左右黑边；
- 把横屏内容放进 736×1280 竖屏画布并添加上下黑边；
- 使用 rotation metadata 让播放器旋转，实际像素宽高与元数据不一致；
- 为凑分辨率而非等比拉伸人物、商品或场景；
- 用横屏训练参数直接处理竖屏素材，或反向操作。

两个尺寸都能被 32 整除，符合当前训练预处理要求。画面必须在交付前完成构图重排、裁剪和缩放，确保主体、字幕安全区和关键动作在对应方向中完整可见。

MiniMax-H3 的目标帧数必须满足：

```text
num_frames % 17 == 5
```

24 FPS、名义 5～15 秒范围内的合法训练帧数如下：

| 帧数 | 时长 | 帧数 | 时长 |
|---:|---:|---:|---:|
| 124 | 5.167 秒 | 243 | 10.125 秒 |
| 141 | 5.875 秒 | 260 | 10.833 秒 |
| 158 | 6.583 秒 | 277 | 11.542 秒 |
| 175 | 7.292 秒 | 294 | 12.250 秒 |
| 192 | 8.000 秒 | 311 | 12.958 秒 |
| 209 | 8.708 秒 | 328 | 13.667 秒 |
| 226 | 9.417 秒 | 345 | 14.375 秒 |

精确 5.000 秒是 120 帧，精确 15.000 秒是 360 帧，两者都不满足 `17n+5`。本项目所说的“支持 5～15 秒”按以下方式执行：

- 原始采集素材可以覆盖名义 5～15 秒，但必须为规范化剪辑保留足够余量；
- 训练成品最短交付 124 帧，约 5.167 秒；
- 训练成品最长交付 345 帧，约 14.375 秒；
- 推荐首期使用 124、192、260、345 四个时长桶；
- 如果合同要求不短于 15 秒，必须单独审批 362 帧（约 15.083 秒）规格，本规范默认不包含。

当前加载器从视频起点按 24 FPS 读取，不会随机选择中间片段。外包方必须提前完成精确剪辑，关键动作、对白和声音必须全部位于交付片段内。不得提交长视频并依赖训练程序自动寻找有效内容。

同一阶段一缓存任务只配置一个 `height/width/num_frames` 组合。横竖屏和不同帧数应按桶分别处理，不能依靠单个命令自动识别并切换规格。

### 7.2 采集与规范化裁剪要求

外包采集必须同时支持以下两种独立构图：

| 采集桶 | 成品画布 | 主体安全区要求 |
|---|---:|---|
| 横屏 | 1280×736 | 关键主体、动作和文字不得依赖竖屏外扩区域 |
| 竖屏 | 736×1280 | 人脸、手部、商品和关键文字不得位于横向裁切边缘 |

采集和剪辑要求：

- 原始拍摄分辨率不得低于对应成品画布，推荐使用更高分辨率采集后等比缩放和裁剪；
- 横屏与竖屏必须按各自构图拍摄或逐条重新构图，不能只给同一文件增加 rotation metadata；
- 同一原片同时制作横屏和竖屏版本时，两版都必须单独通过主体完整性、动作完整性和文字安全区审核；
- 原始素材建议在目标有效片段前后各保留至少 0.5 秒余量，方便剪成合法 `17n+5` 帧；
- “约 5 秒”样本的原始拍摄时长应至少覆盖 5.7 秒，确保可稳定交付 124 帧且保留剪辑余量；
- “约 15 秒”样本最终交付 345 帧、约 14.375 秒，原始拍摄可以达到 15 秒，但有效动作和对白必须在 345 帧内完整结束；
- 成品必须从第 0 帧开始就是有效内容，不保留场记板、对焦过程、拍摄准备、无意义静止或尾部黑场；
- 不允许通过重复帧、改变播放速度、插帧或补黑场凑齐帧数，除非项目任务本身明确要求该效果；
- 横屏、竖屏和各时长桶的数量比例由任务单规定；未规定时，试制集横竖屏按 1:1，四个推荐时长桶均需覆盖；
- 同一原始拍摄制作出的多个方向或时长版本必须使用相同 `source_group_id`，防止训练集与验证集同源泄漏。

横屏和竖屏的参考素材可以使用不同原始宽高，但参考视频在进入训练前会被裁剪到当前目标画布。因此参考视频也必须按目标方向检查主体安全区，不能假设模型会保留画布边缘内容。


### 7.3 目标音频硬性要求

| 项目 | 验收标准 |
|---|---|
| 声道 | 立体声；单声道可接受但会在训练时复制为双声道 |
| 采样率 | 优先 32000 Hz；允许 44100/48000 Hz，训练时会重采样 |
| 时长 | 与目标视频实际帧数严格对齐，即 `target_frames / 24` 秒 |
| 编码 | 必须封装在目标 MP4 中，建议 AAC-LC |
| 削波 | 不允许数字削波 |
| 峰值 | 建议不高于 -1 dBTP |
| 响度 | 对话/通用内容建议约 -16 LUFS，批次内保持一致 |
| 同步 | 口型、动作声误差建议不超过 2 帧，最大不超过 80 ms |

当前 `LoadAudioWithTorchaudio` 会先把 `input_audio` 当作视频读取帧数和时长，因此本项目交付必须把目标声轨封装进目标 MP4，并让两个字段填写同一路径：

```json
{
  "video": "media/targets/S00000001_target.mp4",
  "input_audio": "media/targets/S00000001_target.mp4"
}
```

不接受把独立 WAV 直接填入 `input_audio`。如后续训练代码改造为显式接收独立目标音频，甲方会另行发布新版规范；参考音频不受此限制。

不得用与画面不相关的随机音乐填充目标音频。没有自然声音的任务，应按项目要求采集合理环境底噪，或明确采用静音样本；不得因制作疏忽产生意外静音。

### 7.4 目标内容质量

必须满足：

- 主体清晰，目标动作或变化在片段中完整发生；
- 无损坏帧、黑帧、绿帧、卡帧、花屏和明显重复帧；
- 无严重抖动、失焦、过曝、欠曝、压缩块和摩尔纹；
- 除专门任务外，5～8 秒片段建议不超过 2 个镜头，8～15 秒片段建议不超过 4 个镜头；
- 对话内容与嘴部运动一致，动作声与事件一致；
- Prompt 中描述的主体、动作、镜头、对白和声音都能在目标中被观察或听到；
- 不在 Prompt 中描述目标中不存在的内容。

## 8. 参考素材技术规范

### 8.1 参考图像

| 项目 | 要求 |
|---|---|
| 格式 | PNG、JPEG、WebP；正式交付优先 PNG/JPEG |
| 色彩 | RGB、sRGB；不交付 CMYK、索引色或异常 ICC 文件 |
| 分辨率 | 短边建议不低于 1024；主体身份精细任务建议短边 1600～2048 |
| 清晰度 | 主体关键细节清晰，无严重运动模糊和压缩损伤 |
| 构图 | 主体无遮挡或遮挡符合目标任务，重要部位不得出画 |
| 透明通道 | 不依赖透明背景；有 alpha 时必须确认转为 RGB 后视觉正确 |

Pipeline 默认会把参考图像短边缩放到 2048。低分辨率素材会被放大，不能因为模型会缩放而降低采集质量。

人物或商品一致性任务建议同时覆盖：

- 正脸或主要识别角度；
- 关键服饰、发型、配件、Logo、纹理和材质；
- 与目标中主体一致但不必逐像素相同的内容。

### 8.2 参考视频

| 项目 | 要求 |
|---|---|
| 容器/编码 | MP4、H.264、`yuv420p` |
| 帧率 | 恒定 24 FPS |
| 分辨率 | 与对应目标桶一致：横屏 1280×736，竖屏 736×1280；至少保证中心裁剪后主体完整 |
| 推荐帧数 | 与目标时长桶一致，且满足 `17n+5` |
| 最低帧数 | 22 帧；短于 22 帧当前加载器无法形成有效参考分组 |
| 时长 | 不超过目标时长；超出部分会从尾部截断 |
| 内容 | 能清楚提供 Prompt 声称要参考的身份、动作、镜头或场景 |

参考视频会从起点取帧并做中心裁剪。不要依赖尾部才出现的关键信息，也不要把主体放在会被中心裁剪掉的画面边缘。

使用 `type: video` 时，声轨不会成为条件；需要复用背景音乐、对白或环境声时必须标为 `video_audio`。

### 8.3 参考音频

| 项目 | 要求 |
|---|---|
| 格式 | WAV PCM 优先；也支持常见 MP3、MP4 等可解码格式 |
| 采样率 | 优先 32000 Hz，其他采样率会重采样 |
| 声道 | 单声道或立体声；超过 2 声道时只使用前 2 声道 |
| 时长 | 不超过对应目标样本时长；超出会从尾部截断 |
| 语音音色参考 | 建议 2～5 秒清晰单人语音，无背景人声和混响 |
| 音乐/环境参考 | 建议覆盖最具辨识度且能在目标中验证的片段 |
| 质量 | 无削波、爆音、静电噪声、编码错误和异常长静音 |

纯音色参考不得包含第二个人声。若目标需要多人说话，应分别提供可区分的参考并在 Prompt 中明确角色与音频编号。

### 8.4 参考与目标之间的关系

每个参考至少满足一项可验收关系：

- `identity`：人物、动物、商品或角色身份一致；
- `appearance`：服装、材质、配色、发型、Logo 等一致；
- `style`：视觉风格、UI 设计、光影、摄影或动画风格一致；
- `motion`：动作、节奏、运镜或交互方式一致；
- `layout`：构图、空间关系、页面布局一致；
- `voice`：说话人音色或说话方式一致；
- `music`：音乐主题、配器或指定声轨复用；
- `ambience`：环境声或声场一致；
- `edit_source`：目标是对参考视频的明确编辑版本。

禁止以下配对：

- 参考与目标完全无关；
- Prompt 声称“保持身份”，但人物明显不是同一身份；
- Prompt 声称“复用音频”，但目标音频并未复用；
- 为凑数量，把目标视频截图直接作为参考图，却没有标明这是首帧/源内容保持任务；
- 目标与参考逐帧完全相同，却标成“风格参考”或“动作参考”；
- 同一目标被轻微改名后重复交付。

## 9. Prompt 标注规范

### 9.1 总原则

合格 Prompt 必须同时回答：

1. 目标视频中有什么；
2. 主体在做什么，镜头如何变化；
3. 每个参考分别提供什么；
4. 哪些参考内容保留、复用、借鉴或改变；
5. 目标音频包含什么，与画面如何同步。

Prompt 描述的是目标结果，不是制作指令清单。应使用确定、可观察的语言，不写“效果高级一些”“看起来不错”“自由发挥”等无法验收的表达。

### 9.2 Prompt 语言

- 中文数据使用完整、自然的简体中文；
- 英文数据使用自然英文，不使用逐词机翻；
- 一条 Prompt 内可以为对白保留原语言，但叙述部分尽量保持单一语言；
- 专有名词、UI 文案和对白必须与目标素材一致；
- 不允许用关键词堆砌代替完整描述。

建议长度：

- 简短自然语言 Prompt：80～300 个中文字符，或 50～180 个英文单词；
- 结构化 Prompt：300～1200 个中文字符，或 180～700 个英文单词；
- 单图、单主体、简单运动可使用短 Prompt；
- 多参考、视频编辑、对白和音频复用必须使用结构化 Prompt。

以上长度是生产质检建议，不是解析器的硬限制。不得为了达到字数重复描述同一信息。

### 9.3 简短自然语言模板

适用于一张参考图、关系简单的样本：

```text
以<Picture 1>中的【主体/商品/角色】为核心，生成一段【视觉风格】视频。
画面中【主体外观和必须保持的特征】，在【场景】中执行【动作及时间顺序】。
镜头采用【景别、角度、运镜】，光线为【光线】，整体色彩为【色彩】。
声音包含【对白/音乐/环境声/动作声】，并与【对应动作】同步。
```

填写示例：

```text
以<Picture 1>中的黑色运动鞋为核心，生成一段动感产品官网展示视频。保持鞋面的银色标志、网眼纹理和白色中底结构。运动鞋先在暗色碳纤维背景中央缓慢旋转，随后页面向下滚动，鞋身快速放大并出现红黑色速度光带。镜头以近景环绕和快速推进为主，画面采用高对比商业广告风格。声音包含低沉电子节拍、页面切换的短促呼啸声和鞋底落地声，声音与画面切换同步。
```

### 9.4 结构化 Prompt 模板

适用于多参考、视频编辑、声音参考或复杂动作：

```text
subject_definitions:
<Subject 1> 是【主体定义】，来自<Picture 1>/<Video 1>。
<Picture 1> 提供【身份/商品外观/视觉风格】参考。
<Video 1> 是【源视频/动作/镜头】参考。
<Audio 1> 提供【背景音乐/环境声/源声轨】。
<Audio 2> 提供【角色】的音色参考。

summary:
【一句话说明目标任务，以及每个参考如何作用于目标。】

retention_analysis:
<Subject 1>: fully_preserved/partially_preserved/reference - 【保留内容】。
<Picture 1>: reference - 【参考内容】。
<Video 1>: fully_preserved/partially_copy/reference - 【保留、复制或修改内容】。
<Audio 1>: fully_copy/partially_copy/reference - 【目标中如何使用】。
<Audio 2>: reference - 【音色、说话方式如何使用】。

detailed_description:
目标视频采用【写实/动画/UI/广告等】风格。
[Shot 1] 【起始画面、主体、动作、场景、光线、镜头运动、声音事件】。
[Shot 2] 【如有第二镜头，说明转场、后续动作和结尾状态】。

overall_soundscape:
【对白、音乐、环境声、动作声、前后层次和同步关系】。

non_diegetic_music:
【没有配乐则写 None；有配乐则说明来源、风格、强弱和起止方式】。
```

字段名推荐保持上述英文形式，与仓库官方复杂 Prompt 示例一致；字段正文可以使用中文。

### 9.5 保留关系词汇

| 标记 | 使用场景 |
|---|---|
| `fully_preserved` | 身份、构图、镜头或源视频内容基本完整保留 |
| `partially_preserved` | 只保留部分外观、场景或镜头内容 |
| `fully_copy` | 目标完整复制参考音频或指定片段 |
| `partially_copy` | 只复用部分音乐、环境声或源视频元素 |
| `reference` | 只参考身份、风格、音色、动作或氛围，不直接复制 |
| `replace` | 明确把参考源中的某元素替换为新元素 |
| `remove` | 明确移除参考源中的某元素 |

保留关系必须与目标素材一致，不能机械地全部填写 `fully_preserved`。

### 9.6 对白写法

有明确台词时，建议使用：

```text
<Subject 1> 用平静的男声说：<d>[Chinese] 风会带我们去新的地方。</d>
```

英文对白：

```text
<Subject 1> says softly: <d>[English] Follow the wind, and leave your worries behind.</d>
```

要求：

- `<d>...</d>` 内填写目标音频中实际出现的逐字对白；
- 标明语言，混合语言按实际顺序分别标注；
- 不得用同音替代、概括或漏写明显台词；
- 说明说话人、语气、音量、情绪和口型同步；
- 没有对白时不要虚构对白。

`<d>` 和语言标签属于推荐标注约定，当前代码不会单独解析它们，而是作为 Prompt 文本交给文本编码器。

### 9.7 音频描述要求

至少描述以下信息：

- 是否有人声，谁在说话；
- 人声语言、台词、情绪、音色来源；
- 是否有音乐，音乐来自参考还是新配乐；
- 环境声和关键动作声；
- 声音何时出现、持续多久、与什么动作同步；
- 多音轨的前后层次，例如“对白清晰居前，背景音乐较轻”。

不合格写法：

```text
配上合适的声音。
```

合格写法：

```text
<Audio 1>中的轻柔钢琴旋律作为低音量背景音乐贯穿全片。第 2 秒商品落到桌面时出现一次短促、清晰的撞击声，随后加入轻微室内环境声；没有人声。
```

### 9.8 Prompt 禁止事项

- 不写素材文件名和磁盘路径；
- 不使用错误或不存在的参考编号；
- 不描述目标中不存在的主体、动作、声音或文字；
- 不把主观质量词作为主要内容，如只写“高清、电影感、大师作品”；
- 不大量复制同义句凑长度；
- 不遗漏对白、关键音效和参考关系；
- 不包含外包制作备注、审核意见、客户沟通记录；
- 不出现“AI 生成”“模型应该”“请生成”等与目标画面无关的元叙述；
- 不用含糊代词指代多个主体；
- 不写无法从目标或参考素材验证的身份、品牌、地点和事实。

## 10. 推荐任务类型和 Prompt 形式

`task_type` 建议使用以下枚举。项目合同可从中选择所需类型并约定数量比例。`task_type` 表示参考与目标之间的学习关系，`domain` 表示业务行业；例如电商商品广告应填写 `domain=ecommerce`、`task_type=product_showcase`。当前训练代码不会根据 `task_type` 自动切换模型分支。

业务选型、能力边界和验收指标详见 [`ref2va-lora-business-solution-selection.md`](./ref2va-lora-business-solution-selection.md)。

| `task_type` | 参考组合 | 适用 Prompt |
|---|---|---|
| `image_reference` | 1～2 张图像 | 简短或结构化 |
| `identity_animation` | 人物/角色图像，可加动作视频 | 结构化 |
| `product_showcase` | 商品图像，可加风格图像 | 简短或结构化 |
| `style_reference` | 风格图像/视频 | 简短或结构化 |
| `video_motion_reference` | 无声参考视频 | 结构化 |
| `video_edit` | 源参考视频 | 结构化，必须写 retention |
| `video_edit_audio_reuse` | `video_audio` | 结构化，必须写画面与声轨保留关系 |
| `video_edit_voice_reference` | `video_audio` + 独立音色音频 | 结构化，必须写源声轨和新音色的用途 |
| `voice_reference` | 音色参考音频 | 结构化，必须有逐字对白 |
| `music_reference` | 音乐参考音频 | 结构化，说明复用或风格参考 |
| `audio_driven_visual` | 参考音频 | 结构化，说明声音与画面关系 |
| `multi_reference_composition` | 两种以上模态 | 结构化 |
| `scene_ambience_reference` | 场景图像/视频 + 环境音频 | 结构化，说明场景与声场关系 |

如果甲方未指定比例，外包方必须先提交不少于 50 条的试制集，由甲方确认任务分布、Prompt 粒度和质量阈值后再批量生产。未经试制验收直接批量制作产生的返工由外包方承担。

## 11. 完整元数据示例

### 11.1 单图商品参考

```json
{
  "sample_id": "S00000011",
  "domain": "ecommerce",
  "task_type": "product_showcase",
  "orientation": "landscape",
  "target_width": 1280,
  "target_height": 736,
  "target_fps": 24,
  "target_frames": 124,
  "source_group_id": "G_PRODUCT_0007",
  "prompt_language": "zh",
  "rights_id": "R0000011",
  "video": "media/targets/S00000011_target.mp4",
  "input_audio": "media/targets/S00000011_target.mp4",
  "prompt": "以<Picture 1>中的黑色运动鞋为核心，生成一段动感产品官网展示视频。保持鞋面的银色标志、网眼纹理和白色中底结构。运动鞋先在暗色碳纤维背景中央缓慢旋转，随后页面向下滚动，鞋身快速放大并出现红黑色速度光带。镜头以近景环绕和快速推进为主。声音包含低沉电子节拍、页面切换的短促呼啸声和鞋底落地声，声音与画面切换同步。",
  "references": [
    {
      "type": "image",
      "image": "media/references/images/S00000011_ref_img_01.png"
    }
  ]
}
```

### 11.2 视频编辑并参考另一段人声音色

JSON 中的多行 Prompt 必须使用 `\n` 转义。下面为便于阅读进行了格式化展示：

```json
{
  "sample_id": "S00000012",
  "domain": "digital_human",
  "task_type": "video_edit_voice_reference",
  "orientation": "landscape",
  "target_width": 1280,
  "target_height": 736,
  "target_fps": 24,
  "target_frames": 124,
  "source_group_id": "G_PERSON_0031",
  "prompt_language": "zh",
  "rights_id": "R0000012",
  "video": "media/targets/S00000012_target.mp4",
  "input_audio": "media/targets/S00000012_target.mp4",
  "prompt": "subject_definitions:\n<Subject 1>是<Video 1>中身穿粉色西装、怀抱黑色小羊的年轻男子。\n<Video 1>是待编辑的源视频。\n<Audio 1>是<Video 1>的同步背景音乐。\n<Audio 2>是<Subject 1>的男声音色参考。\n\nsummary:\n目标基本保留<Video 1>的主体身份、服装、构图和场景，为<Subject 1>增加新的说话口型和对白。<Audio 1>的背景音乐低音量复用，角色声音参考<Audio 2>。\n\nretention_analysis:\n<Subject 1>: fully_preserved - 保持人物身份、金色卷发、粉色西装和怀中的黑色小羊。\n<Video 1>: fully_preserved - 保持原始镜头、草地、光线和背景羊群，只编辑人物口型。\n<Audio 1>: partially_copy - 作为目标的连续背景音乐，音量降低。\n<Audio 2>: reference - 用作角色平静男声音色参考。\n\ndetailed_description:\n目标视频采用写实摄影风格。\n[Shot 1] <Subject 1>站在夕阳下的草坡上怀抱黑色小羊，镜头缓慢推进。他看向镜头，用平静的声音说：<d>[Chinese] 跟着风走，把烦恼留在身后。</d> 说话结束时嘴唇自然闭合，随后低头轻抚小羊。\n\noverall_soundscape:\n清晰男声位于前景，<Audio 1>的轻柔背景音乐低音量持续播放，并保留轻微草地环境声。\n\nnon_diegetic_music:\n低音量复用<Audio 1>中的背景音乐，贯穿全片并在结尾自然淡出。",
  "references": [
    {
      "type": "video_audio",
      "video": "media/references/videos/S00000012_ref_video_01.mp4",
      "audio": "media/references/videos/S00000012_ref_video_01.mp4"
    },
    {
      "type": "audio",
      "audio": "media/references/audios/S00000012_ref_audio_02.wav"
    }
  ]
}
```

在这个例子中，`video_audio` 对应 `<Video 1>` 和 `<Audio 1>`，随后单独的音色参考对应 `<Audio 2>`。

## 12. 内容安全、版权与隐私

每条样本必须具备可用于模型训练的合法授权。外包方必须保存原始授权证据，并在 `rights_manifest.csv` 中提供：

- `rights_id`；
- `sample_id` 或 `source_group_id`；
- 素材来源；
- 著作权/许可类型；
- 肖像权和声音授权状态；
- 音乐、字体、Logo、商标的授权状态；
- 授权地域、期限和用途；
- 证据文件路径或合同编号；
- 经办人与复核人。

未经书面许可，不得交付：

- 从短视频平台、影视作品、广告、直播、监控中抓取的未授权内容；
- 未取得肖像权和声音权授权的可识别真人素材；
- 未成年人敏感内容；
- 受版权保护但无训练授权的音乐、配音、角色和商品素材；
- 包含身份证、住址、电话号码、车牌、病历等个人敏感信息的素材；
- 色情、血腥、违法、仇恨、歧视、危险行为或其他合同禁止内容；
- 带有第三方水印、平台 Logo、时间戳或不可清除版权标记的素材。

如任务必须包含品牌、文字、真实人物或特殊敏感类别，必须由甲方以书面任务单另行确认。

## 13. 去重和数据隔离

### 13.1 去重要求

- 对目标视频、参考视频、图像和音频分别计算 SHA-256；
- 同一文件内容不得以改名方式重复交付；
- 对视频进行感知去重，禁止仅改变封装、码率、色温或裁剪少量边缘后重复交付；
- 对音频进行指纹或波形相似度去重，禁止只调整音量后重复交付；
- 同一长视频切片之间不得高度重叠，默认重叠时长不超过片段长度的 20%；
- 同一 Prompt 不应机械复制到大量内容不同的样本。

### 13.2 分组要求

来自同一人物、商品、原始长视频、同场拍摄或同一次录音的样本必须使用相同 `source_group_id`。后续划分训练集、验证集和测试集时以该字段整体分组，避免同源泄漏。

## 14. 质检流程

每条样本必须经过以下四层质检：

1. 文件和 JSON 自动检查；
2. 音视频技术指标检查；
3. 参考关系和 Prompt 人工检查；
4. 版权、安全和最终复核。

### 14.1 自动检查

至少检查：

- `metadata.json` 可被标准 JSON 解析器读取；
- `sample_id` 唯一；
- 所有路径存在且位于交付目录内；
- `prompt` 非空；
- `references` 是非空数组；
- 每个参考块只有合法 `type` 和对应必填字段；
- 目标视频为横屏 1280×736 或竖屏 736×1280，恒定 24 FPS；
- `orientation`、`target_width`、`target_height` 与视频实际方向和尺寸一致；
- `target_frames` 与视频实际帧数一致，范围为 124～345 且满足 `target_frames % 17 == 5`；
- 目标音频时长与 `target_frames / 24` 一致；
- 音视频可完整解码且时长匹配；
- 参考视频不少于 22 帧；
- 文件 hash 无重复；
- Prompt 中引用编号与实际 references 一致。

JSON 基础检查示例：

```bash
python -m json.tool metadata.json >/dev/null
```

视频指标可使用 `ffprobe` 检查：

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames \
  -of json media/targets/S00000001_target.mp4
```

### 14.2 人工审核

审核员必须同时观看/收听目标和所有参考，不得只看缩略图。逐项确认：

- 参考与目标的身份、外观、动作、风格、镜头或声音关系真实存在；
- Prompt 准确描述目标，不遗漏关键事件；
- Prompt 中每个参考编号正确；
- `video` 与 `video_audio` 类型使用正确；
- 对白逐字稿、语言、说话人和目标声轨一致；
- retention 描述与实际保留程度一致；
- 音画同步、声轨质量和结尾完整；
- 不存在版权、隐私和内容安全问题。

### 14.3 质检状态

`qa_manifest.csv` 的 `status` 只允许：

- `pass`：可直接训练；
- `rework`：可修复，写明问题和截止时间；
- `reject`：不可修复或不允许进入数据集。

不得用“基本通过”“待定”等模糊状态。

### 14.4 建议的 QA 表字段

```text
sample_id,status,domain,task_type,orientation,reviewer_1,reviewer_2,
target_width,target_height,target_fps,target_frames,target_duration,
target_audio_sr,target_audio_channels,reference_count,prompt_language,
reference_relation_pass,prompt_accuracy_pass,av_sync_pass,rights_pass,
duplicate_pass,sha256_target,issue_code,issue_description,review_time
```

## 15. 验收标准

### 15.1 一票否决项

出现以下任一情况，该样本直接拒收：

- 文件缺失、损坏或无法解码；
- JSON 无法解析或必填字段缺失；
- 目标视频不是约定的横屏 1280×736 或竖屏 736×1280、恒定 24 FPS；
- 目标帧数不在 124～345 范围内，或不满足 `17n+5`；
- 元数据中的方向、宽高、帧率、帧数与文件实际属性不一致；
- Prompt 为空或引用编号错误；
- 参考与目标无可验证关系；
- 音画严重不同步或目标声轨错误；
- 样本重复或近重复；
- 无法提供合法授权；
- 存在合同禁止的隐私、安全或版权内容。

### 15.2 批次建议验收方式

推荐按以下方式写入外包合同：

- 自动规则对全量数据检查；
- 甲方对每批随机抽检不少于 10%，且不少于 100 条；
- 对高风险类别，如真人、音色、音乐、品牌、视频编辑任务，提高到 100% 人工审核；
- 抽检发现严重错误时，扩大抽检或整批退回；
- 外包方完成返工后必须重新执行全量自动检查和双人复核；
- 同一问题连续两批出现时，暂停批量生产并重新试制。

具体合格率、抽检比例和违约条款以合同为准。本规范定义“什么是合格样本”，不替代商务合同中的数量和结算条款。

## 16. 常见错误示例

### 错误 1：把数组序号当作统一编号

```json
"references": [
  {"type": "image", "image": "a.png"},
  {"type": "audio", "audio": "b.wav"}
]
```

错误 Prompt：

```text
使用<Picture 1>的主体和<Audio 2>的声音。
```

正确 Prompt：

```text
使用<Picture 1>的主体和<Audio 1>的声音。
```

### 错误 2：希望使用参考视频声轨却标记为 `video`

错误：

```json
{"type": "video", "video": "source.mp4"}
```

正确：

```json
{"type": "video_audio", "video": "source.mp4", "audio": "source.mp4"}
```

### 错误 3：Prompt 只描述目标，没有说明参考用途

不合格：

```text
一个男人在草地上说话，电影感，高清。
```

合格：

```text
保持<Video 1>中男子的身份、粉色西装、怀中的黑色小羊以及草地构图，只为男子增加新的说话口型。角色使用<Audio 2>中的平静男声音色说：<d>[Chinese] 跟着风走，把烦恼留在身后。</d> <Audio 1>中的轻柔音乐作为低音量背景音乐保留。
```

### 错误 4：长视频未预切

训练加载器只读取从起点开始、由本次 `--num_frames` 指定的帧数，不会自动搜索精彩片段。把完整长视频直接交付，会导致 Prompt 描述的动作可能不在实际训练片段中。必须先按合法时长桶剪出精确片段，再编写 Prompt。

### 错误 5：目标音频路径缺失

即使目标没有对白，也必须正确填写 `input_audio`。如果项目规定静音，应交付长度正确的静音声轨或由训练方明确启用静音兜底，不允许遗漏字段。

## 17. 外包生产流程

1. 接收垂域、任务类型、横竖屏比例、时长桶比例、数量、语言和授权要求；
2. 提交 50 条以上试制集；
3. 甲方确认参考配对和 Prompt 粒度；
4. 采集原始素材并登记授权和 `source_group_id`；
5. 按横屏/竖屏和合法帧数桶剪辑目标与参考片段，统一转码；
6. 编写 Prompt 和 `metadata.json`；
7. 执行文件、JSON、方向、宽高、合法帧数、时长、帧率和 hash 自动检查；
8. 第一名审核员检查内容、参考关系和 Prompt；
9. 第二名审核员复核高风险与抽检样本；
10. 生成 QA、rights、reject 清单；
11. 按固定目录结构打包交付；
12. 根据甲方验收结果返工，保持原 `sample_id` 不变。

## 18. 与当前训练代码的对应关系

阶段一必须按方向和帧数桶分别运行。横屏 124 帧示例：

```bash
--dataset_metadata_path /path/to/landscape_frames_124/metadata.json \
--data_file_keys "video,input_audio,references" \
--extra_inputs "input_audio,references" \
--height 736 \
--width 1280 \
--num_frames 124
```

竖屏 124 帧示例：

```bash
--dataset_metadata_path /path/to/portrait_frames_124/metadata.json \
--data_file_keys "video,input_audio,references" \
--extra_inputs "input_audio,references" \
--height 1280 \
--width 736 \
--num_frames 124
```

8 秒、约 10.83 秒和约 14.38 秒桶分别把 `--num_frames` 改为 192、260 和 345。不得让一个阶段一任务同时读取横屏和竖屏，也不得让源视频长度决定随机输出帧数。

当前工程行为：

- 目标视频从起点按 24 FPS 采样并中心裁剪到训练画布；
- 目标音频按目标视频时长截断或补齐，并重采样到 32000 Hz；
- 参考图像按原始图像读取，Pipeline 内部默认把短边缩放至 2048；
- 参考视频按 24 FPS 采样、中心裁剪，并截断到不超过目标时长的 `17n+5` 帧；
- 参考音频最多读取目标时长并重采样到 32000 Hz；
- 单声道音频会转换为立体声，超过两声道时只使用前两声道；
- `references` 的数组顺序决定条件呈现顺序，但 Picture/Video/Audio 编号各自独立；
- 两阶段训练会把这些预处理结果写入缓存，任何素材、Prompt、顺序、分辨率或帧数变化后都必须重建缓存。

代码依据：

- [`examples/minimax_h3/model_training/train.py`](../../examples/minimax_h3/model_training/train.py)
- [`diffsynth/utils/data/minimax_h3.py`](../../diffsynth/utils/data/minimax_h3.py)
- [`diffsynth/pipelines/minimax_h3_audio_video.py`](../../diffsynth/pipelines/minimax_h3_audio_video.py)
- [`diffsynth/models/minimax_h3_text_encoder.py`](../../diffsynth/models/minimax_h3_text_encoder.py)

## 19. 交付前最终检查清单

- [ ] 每条记录都有唯一 `sample_id`。
- [ ] 每条记录都填写 `domain`、`task_type`、`orientation`、`target_width`、`target_height`、`target_fps` 和 `target_frames`。
- [ ] `orientation` 与实际宽高一致：横屏 1280×736，竖屏 736×1280。
- [ ] `target_frames` 在 124～345 范围内且满足 `target_frames % 17 == 5`。
- [ ] `video`、`input_audio`、`prompt`、`references` 全部存在。
- [ ] `input_audio` 与 `video` 指向同一个带声轨目标 MP4。
- [ ] 所有文件路径均为相对路径且可访问。
- [ ] 目标视频为横屏 1280×736 或竖屏 736×1280、恒定 24 FPS。
- [ ] 目标音频时长等于 `target_frames / 24`，并与画面同步。
- [ ] 每个参考块的 `type` 与字段匹配。
- [ ] 参考视频不少于 22 帧。
- [ ] `video_audio` 的画面和音频起止时间一致。
- [ ] Picture、Video、Audio 分别从 1 编号。
- [ ] Prompt 中每个参考都被正确解释。
- [ ] Prompt 准确描述目标画面、动作、镜头和声音。
- [ ] 有对白时提供逐字稿、说话人、语言、语气和音色来源。
- [ ] 所有样本完成 hash 和感知去重。
- [ ] 同源数据填写相同 `source_group_id`。
- [ ] 所有授权、肖像权和声音权记录完整。
- [ ] QA 状态为 `pass`，无未关闭的返工项。
- [ ] `metadata.json` 已通过标准 JSON 解析检查。
