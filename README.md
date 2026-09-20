# ComfyUI-H3-upgrade-kit

**把 MiniMax H3 的潜空间放大、分块采样和多段续接，放进一套更方便搭建的节点里。**

这个插件基于下面三个开源项目，结合 H3 工作流的实际使用，做了优化、升级和整合：

- [NikoDemon80 / ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)：动作续接与音画衔接。
- [LBH-123-AI / Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)：H3 潜空间放大与高清分块采样。
- [facok / comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift)：先低清采样、再放大续采的 SelfLift 思路。

感谢三位作者把代码和经验分享出来，让这个整合版有了基础。我们在此基础上整理了节点接口，加入 VAE 分块解码、可串联的 SelfLift 采样和音画拼接，希望大家搭建 H3 工作流时能少绕一些弯路。

[安装](#安装) · [工作流示例](#工作流示例) · [SelfLift 多段续接](#selflift-多段续接) · [其他节点怎么用](#其他节点怎么用) · [常见问题](#常见问题)

## 能做什么

在 ComfyUI 中搜索 `H3Kit`，或打开 `H3 Upgrade Kit` 分类就能找到以下节点。

| 你想做的事 | 使用的节点 |
| --- | --- |
| 先低清生成，再放大到高清继续采样 | **H3Kit SelfLift K采样器** |
| 沿用参考素材，只替换下一段的提示词和长度 | **H3Kit 提示词与长度替换** |
| 裁掉重复开头、对齐音频，或把前后两段音画拼起来 | **H3Kit 音画裁剪与拼接** |
| 单独放大 H3 视频 latent | **H3Kit 3D 潜空间放大** |
| 对高清画面分块采样 | **H3Kit 高清分块采样** |
| 让下一段接着上一段的动作生成 | **H3Kit 动作续接** |
| 分批处理 H3 视频 VAE 解码的小块 | **H3Kit VAE 分块解码** |
| 将 latent、图片、音频分别循环回传和收集 | **H3Kit Start Loop 多路循环开始** / **H3Kit End Loop 多路循环结束** |
| 将循环输出的声音列表拼成完整音轨 | **H3Kit 声音列表到声音批次** |

使用本插件不需要另外安装上面三个原插件，也不会修改 ComfyUI 核心文件。

## 安装

### 通过 ComfyUI Manager

搜索 **`ComfyUI-H3-upgrade-kit`**，安装后重启 ComfyUI。如果列表还没有同步，可以使用仓库地址安装：

```text
https://github.com/flywhale-666/ComfyUI-H3-upgrade-kit
```

### 手动安装

在 `ComfyUI/custom_nodes/` 目录打开终端，执行：

```bash
git clone https://github.com/flywhale-666/ComfyUI-H3-upgrade-kit.git
```

也可以下载仓库 ZIP，将解压后的插件目录放进 `ComfyUI/custom_nodes/`，然后重启。

**没有额外的 Python 依赖需要安装。** 插件使用 ComfyUI 已提供的依赖，`requirements.txt` 留空。

请使用已支持 MiniMax H3 的新版 ComfyUI。动作续接需要支持任意关键帧锚点的 H3 布局，首次使用时会检查；若提示布局不支持，请先更新 ComfyUI。

## 工作流示例

### MINIMAX 数字人 v2 加速版

使用参考图片和音频生成数字人视频。为了测试两段 SelfLift 采样、续接和音画拼接，示例将同一条音频切成两段，分别生成后再衔接。

H3 视频长度按 **17 帧周期**对齐（有效帧数为 `17k+5`），音频按秒裁切，需要结合视频的 **24fps** 换算，因此切分点的计算稍微麻烦一些。工作流保留了相关计算与裁切节点，调整段长时请一起检查音频切分点。下载 JSON 后拖入 ComfyUI，即可查看完整工作流。

请自行选择参考图片、音频和本机对应的 H3 模型、文本编码器、音视频 VAE、加速 LoRA 及 latent 放大模型，素材和模型不随示例提供。示例还使用 ComfyUI-UniversalToolkit、ComfyUI-VideoHelperSuite、ComfyUI-KJNodes、ComfyUI_LayerStyle、ComfyUI_Comfyroll_CustomNodes、ComfyUI-ReservedVRAM 和 rgthree-comfy，缺失时需另行安装。

**[下载工作流 JSON](https://raw.githubusercontent.com/flywhale-666/ComfyUI-H3-upgrade-kit/main/example_workflows/minimax_digital_human_v2_accelerated.json)** · [查看工作流文件](example_workflows/minimax_digital_human_v2_accelerated.json)

## SelfLift 多段续接

### 先理解它怎么工作

SelfLift 把一次采样分成低清和高清两个阶段：**先在较低分辨率下采样，再用潜空间放大模型放大，最后完成高清部分。**

例如：`steps = 8`、`high_resolution_steps = 2`、`lowres_scale = 0.5`，就是先以一半宽高采样 6 步，放大后再采样 2 步。

> **8 步需要搭配适用的加速模型或 LoRA。** 使用普通模型时，请沿用适合该模型的步数，不要直接照搬 8 步。

### 怎么连接

每一段的 `latent_image` 都接这一段的**目标高清音视频 latent**，不需要提前缩小。前一个 K采的 `sampled_latent` 直接接后一个的 `previous_latent`。

两段保持相同的目标分辨率和低清比例，连接中间不要拆分或重新合并 latent，以免丢失续接状态。

每段采样后分别解码，再接 **H3Kit 音画裁剪与拼接** 节点：

| 输入 | 连接内容 |
| --- | --- |
| `sampled_latent` | 本段 SelfLift K采的原始输出 |
| `decoded_frames` / `decoded_audio` | 本段完整解码的画面和音频 |
| `previous_frames` / `previous_audio` | 前段拼接后的画面和音频；第一段留空 |

最后将拼接节点的 `images`、`audio` 接到视频保存节点，帧率使用 **24fps**。

拼接节点的 `latent` 原样输出本段采样结果，可接下一段 K采的 `previous_latent`，也可通过多路循环回传。

### 常用参数

| 参数 | 含义与用法 |
| --- | --- |
| `steps` | 总采样步数，按模型与 LoRA 选择 |
| `high_resolution_steps` | 留给高清阶段的步数 |
| `lowres_scale` | 低清阶段的宽高比例，例如 `0.5` |
| `upscale_weights` | H3 潜空间放大权重 |
| `overlap_frames` | 续接上下文长度，默认 **17**，使用 **17、34、51……** |
| `context_vae` | 接 H3 视频 VAE，让两阶段使用同一段真实尾部画面作为续接依据 |
| `previous_frames` | 配合 `context_vae` 接前段解码画面，可省去一次重复解码 |
| `continue_audio` | 开启时对接缝尾音做平滑过渡 |
| `spatial_tiles` | 是否在高清采样阶段启用空间分块，默认关闭 |
| `minimum_tiles` | 高清采样最少分块数，默认 4；按显存规划最多 8 块，小画面可能更少 |

**段长按新增帧数填写。** 比如两段各 124 帧，拼接后就是 248 帧。第二段会自动增加上下文前缀，拼接节点再裁掉重复画面，不需要手动补帧。

<details>
<summary><strong>展开查看：续接与采样的补充说明</strong></summary>

- 当前仅支持 `euler`。常用参数顺序为 `steps → cfg → sampler_name → scheduler → denoise`。
- `denoise` 默认 1.0；设为 0 时直接返回输入，不执行采样或续接，也不创建新的拼接信息。
- 接入外部 `sigmas` 后，它会覆盖 `steps`、`scheduler`、`denoise` 的调度设置，`high_resolution_steps` 仍然生效。
- 不连接 `context_vae` 时，使用前段保存的低清与高清 latent 尾部。连接后，两个阶段都从前段最终高清画面取得上下文；只处理尾部，整段放大仍使用潜空间放大模型。
- 输入包含按时间分段的提示词时，需要考虑自动增加的上下文前缀。17 帧上下文对应 5 个视频 latent 时间步。
- 单张首帧条件在续接时转为外观参考，中间帧、尾帧、多帧 Guide 及音频锚点随前缀后移。续接会增加运动条件的计算量，效果仍取决于模型、提示词和素材。
- 高清阶段保留真实上下文，并对放大后的新画面进行衔接补偿；不增加采样步数，也不改动原音轨。
- `continue_audio` 开启时，前段最后约 0.2 秒的尾音平滑释放，拼接时使用后段过渡音频替换对应区间。关闭时保留前段音频并裁掉后段重复音频。已锁定的输入音轨不会被覆盖。
- 支持音视频噪波遮罩。音频遮罩会按 latent 尺寸对齐，全零表示锁定；输入音轨较短时只补齐缺少的部分，各段解码音频按最终画面帧数对齐。
- 拼接节点不接前段画面时，只输出当前新增片段；音频未连接时可以输出无声画面。
- SelfLift 直连续接流程不需要额外叠加“动作续接”或重复裁剪。普通 K采输出不包含这里需要的双分辨率状态，不能直接替代前段 SelfLift K采的 previous_latent。

</details>

## 其他节点怎么用

### 提示词与长度替换

放在 **MiniMax H3 参考转视频 → 第二段 SelfLift K采样器** 之间：

| 连接来源 | 连接目标 |
| --- | --- |
| 参考转视频的正向 | 本节点 `positive` |
| 同一个参考转视频的 Latent | 本节点 `latent_image` |
| 本节点 `positive` | 第二段 K采的 `positive` |
| 本节点 `latent` | 第二段 K采的 `latent_image` |

在本节点填写新的 `prompt` 和 `length` 即可。长度单位为帧，24fps，沿用 H3 的 `17k+5` 向上对齐规则（124 帧约 5.17 秒）。第一段仍接原参考节点；第二段的 `previous_latent` 仍接前段 K采输出，`negative` 保持原来的连接。

节点沿用原来的 CLIP、视频/音频 VAE、宽高、参考图尺寸设置，以及全部参考图、视频和音频输入。它使用新提示词重新执行原生参考编码，并生成新长度的空白音视频 latent；参考视频仍按原生规则根据新长度裁切。无需额外接 CLIP 或重复连素材，不会修改第一段。

输入必须来自同一个原生参考转视频节点，也可来自同一个本替换节点以继续串联。不要在这两条输入线上插入条件合并、采样或循环回传节点；循环内使用时，从循环外的原参考节点直接接入，提示词和长度可转换为输入由循环提供。

### 多路循环

需要当前 ComfyUI 已提供原生 `Start Loop` / `End Loop` 与循环边界执行支持。搜索 `H3Kit Start Loop`、`H3Kit End Loop`，成对使用。新节点加入后需要重启 ComfyUI 并刷新页面加载端口扩展脚本。

连接一个端口后会自动增加下一个空位，最多 100 路（ComfyUI Autogrow 上限）。各路按编号对应，断开中间一路不会把后面的数据移到前一号端口。

End Loop 的输入按 `output_value`、`next_iteration_value`、`termination` 分组排列，组内按编号排序；新增端口与重新载入工作流时都会保持此顺序。

| 通道用途 | Start Loop 输入 | Start Loop 输出 | End Loop 回传输入 |
| --- | --- | --- | --- |
| 上一段 latent | `initial_iteration_value1` | `current_iteration_value1` | `next_iteration_value1` |
| 上一段图片 | `initial_iteration_value2` | `current_iteration_value2` | `next_iteration_value2` |
| 上一段音频 | `initial_iteration_value3` | `current_iteration_value3` | `next_iteration_value3` |

将本轮结果分别接入 End Loop 的回传输入。未连接回传的路保留原值。`output_value1`、`output_value2`……是独立的结果收集输入，分别对应 `outputs1`、`outputs2`……，数量不必与回传路数一致，无需创建列表或取列表项来打包、拆包。

- `accumulate=false`：每路输出最后一轮结果。若循环内已用 SelfLift 音画拼接累积完整视频，使用此设置。
- `accumulate=true`：每路按执行顺序输出自己的 ComfyUI 列表，例如图片列表、音频列表，互不交错。此选项不会自动把帧批次或音频拼成一个对象。
- 保留 `simple`、`For`、`List` 模式、`parent_iteration` 嵌套、`cache_iterations` 缓存和 `termination` 每轮必执行分支。`termination` 是执行依赖，不是提前退出条件。
- 次数为 0 时各路输出空列表；不要在同一对循环边界混用 H3Kit 和原生节点。

### 声音列表到声音批次

连接：**End Loop 的音频 `outputsN` → 本节点 `audio` → Video Combine 的音频输入**。节点一次接收整个列表，沿时间维按轮次顺序拼成一条完整音轨，与“图像列表到图像批次”后的画面配合使用。

不同采样率统一到最高采样率，单声道与双声道混用时自动复制单声道；每段原有时长保留，不混音、不自动去重。End Loop 应收集每轮新增片段，已累积成完整音轨的结果直接使用最后一轮即可。

### 3D 潜空间放大

将 H3 视频 latent 接到 `source_latent`，选择放大权重与目标尺寸。支持按倍率、目标宽高或百万像素设置大小。

- 权重放在 `ComfyUI/models/latent_upscale_models/`，支持子目录和 ComfyUI 额外模型路径；不自动下载。
- 接收单路图像或视频 latent。音视频复合 latent 需要先分离视频，放大后再与音频合并。
- `temporal_chunks` 控制时间分块，`spatial_tiles` 控制空间分块，`pixel_alignment` 默认 32。
- `compute_backend`、`compute_precision` 控制设备和精度。开启 `release_weights` 时，CUDA 推理结束后会将放大权重移回 CPU。

### 高清分块采样

在独立采样流程中，将 H3 音视频复合 latent 接入 **H3Kit 高清分块采样**。节点沿长边重叠分块，再融合视频结果；音频保留完整，并采用第一块的预测。

默认至少 4 块，按显存规划最多 8 块。**不支持 ControlNet。** 分块只作用于采样，不代替 VAE 分块解码。

### 动作续接与音画裁剪拼接

用于在普通 H3 采样流程里接续前一段：

```text
H3 条件与目标 latent → 动作续接 → 采样 → 解码 → 音画裁剪与拼接 → 保存
```

`video_vae` 必接。前段可以通过 `previous_frames` 或 `previous_latent` 提供：传画面时直接编码尾部；只传 latent 时先解码再编码。没有前段上下文时直接透传，`prefix_frames` 输出 0。

`overlap_frames` 默认 17，可选 0、17、34、51……。0 时只用前段最后一帧引导本段，不增加重叠区；大于 0 时增加并锁定对应长度的重叠区，`prefix_frames` 输出实际重叠帧数。音频参考长度自动取 `max(24, overlap_frames)`，不再单独暴露音频参考长度设置；素材不足时使用现有尾音。

| 重叠帧数 | 音频参考长度 | 裁剪行为 |
| --- | --- | --- |
| 0 | 24 帧时长 | 图像和音频都不裁开头，只自动对齐音频尾部 |
| 17 | 24 帧时长 | 裁掉 17 帧画面及同等时长的音频 |
| 34 | 34 帧时长 | 裁掉 34 帧画面及同等时长的音频 |
| 51 | 51 帧时长 | 裁掉 51 帧画面及同等时长的音频 |

0 模式的尾音参考位于本段时间轴起点之前，不会占用本段音频长度。重叠模式裁剪的是实际增加的重复区，不是整段音频参考长度。

音频优先取 `previous_latent` 的尾部；只提供 `previous_audio` 时，需要连接 `sound_vae`。将续接节点的 `prefix_frames` 和 `delivery_frames` 接到 **H3Kit 音画裁剪与拼接** 的同名输入，裁掉重复开头并同步音频长度。普通采样内部按 H3 原生 24fps 对齐，SelfLift 自动读取采样记录的帧率，无需手动设置；保存 H3 原生视频使用 24fps。

统一节点前两个输入为 `decoded_frames` / `decoded_audio`，接本段完整解码结果；第三个输入 `sampled_latent` 为可选。普通高级采样器可以不接 latent，按动作续接输出的 `prefix_frames` / `delivery_frames` 裁剪；SelfLift 流程请接本段原始 latent，以自动读取实际重叠帧数、新增帧数和音频过渡信息。不要提前裁剪解码结果。

不接 `previous_frames` / `previous_audio` 时只裁剪本段；接入前段音画时自动拼接。三个输出固定为 `images`、`audio`、`latent`，其中 latent 原样保留本段采样结果；未连接 `sampled_latent` 时该输出为空。音频尾部始终自动对齐画面时长，无需开关。原“SelfLift 续接音画拼接”和“音画同步裁剪”合并为这一个节点。

**升级到 0.0.9：** 旧工作流中的 `H3KitSelfLiftAVJoin` / `H3KitAVTrim` 需要替换为 `H3KitAVJoin` 并按输入名称重新连接；动作续接改用 `overlap_frames`，请检查重叠帧数。随包示例已改用新节点。

### VAE 分块解码

用 **H3Kit VAE 分块解码** 替换画面分支的普通 VAE 解码：采样结果接 `video_latent`，H3 视频 VAE 接 `video_vae`，从 `decoded_frames` 取出画面。

音视频复合 latent 会自动提取视频，音频分支仍使用音频解码器。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `tile_edge` | 256 | 跨行调度的纵向跨度，单位为像素 |
| `tile_blend` | 64 | 原生小块的最小空间重叠 |
| `tiles_per_batch` | 8 | 每批最多处理的小块数 |

内部始终使用原生 256 像素窗口；`tile_edge` 调大改变的是调度范围。合批越大，显存占用越高；发生显存不足时会自动减半重试，降到 1 块仍不足则报错。

可以先用 **512 / 64 / 2** 做对照，再按显存情况尝试每批 4 或 8 块。对比画质时保持重叠为 64。节点保留 H3 原生时间分块，不提供任意时间块长。

## 常见问题

**安装后找不到节点？**

先重启 ComfyUI，再搜索 `H3Kit`。如果仍找不到，检查启动日志中的导入错误，并确认 ComfyUI 版本支持 MiniMax H3。

**用了分块就一定更快、更省显存吗？**

不保证。分块和合批的效果取决于分辨率、模型、显存和参数。分块采样需要逐块计算，缺少整幅画面的全局注意力，结果也可能与不分块时不同。

**能保证每次续接都完全无缝吗？**

不能保证。插件提供上下文续接、高清衔接补偿和音频过渡，但动作、人物、背景的一致性仍受模型与输入内容影响。

## 来源与许可

完整的代码来源、移植关系和修改说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本插件按 [GPL-3.0](LICENSE) 提供，MIT 许可部分保留原作者的版权与[许可文本](licenses/latent-upscaler-MIT.txt)。
