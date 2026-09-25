# ComfyUI-H3-upgrade-kit

**把 MiniMax H3 的潜空间放大、分块采样和多段续接，放进一套更方便搭建的节点里。**

**当前续接规则：SelfLift 和动作续接均使用 22 帧上下文时段，由音画拼接节点自动裁掉这 22 帧。** 动作续接优先继承原生 latent，也支持图片帧、图片加音频、纯音频输入；首段没有续接前缀，无需裁掉开头。

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
| 修改视频播放帧率后，同步缩短或拉长音频 | **H3Kit 音频帧率同步** |
| 使用 BUNNY V2/V1 或原版语义桥改善复杂条件关系 | **H3Kit 语义桥（Semantic Bridge）** |

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

使用参考图片和音频生成数字人视频。示例将同一条音频切成两段，通过两个 SelfLift 采样器直接续接，分别保存首段和去除重复上下文后的续段。两段均采用8步采样、其中2步高清续采，需搭配示例中的加速LoRA。

示例采用 **22 帧续接上下文**，音画裁剪节点自动去掉重复的22帧。音频时长为 `a` 秒时，首段长度使用 `max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17`；续段使用 `max(17, ceil(a * 24 / 17) * 17)`，按17帧周期向上覆盖音频。上下文由采样器自动添加，不要再手动加22。例如续段音频3.83秒时新增102帧，内部生成124帧。

音频按 **24fps** 换算切分位置；第二段从首段结束处开始。需要合成完整视频时，将首段解码画面和音频分别接入音画裁剪与拼接的 `previous_frames`、`previous_audio`。段长对齐可能产生多余尾部，严格匹配音频时长时需裁齐。若要保留原歌，最终合成使用原始完整音频；`continue_audio=true` 会重新生成前段最后约0.2秒的过渡尾音。下载JSON后拖入ComfyUI即可查看完整工作流。

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
| `context_vae` | 直连时用于边界检查及低清校准；外部视频续接时用于建立时间对齐的高清、低清上下文 |
| `boundary_check` | 是否检查并校准上下文；开启时需要连接 `context_vae`，不接 VAE 时请关闭 |
| `previous_frames` | 配合 `context_vae` 接前段解码画面，可省去一次重复解码 |
| `continue_audio` | 开启时对接缝尾音做平滑过渡 |
| `spatial_tiles` | 是否在高清采样阶段启用空间分块，默认关闭 |
| `minimum_tiles` | 高清采样最少分块数，默认 4；按显存规划最多 8 块，小画面可能更少 |

**段长按目标帧数填写。** 续段新增长度按模型周期向下对齐。两段目标各124帧，交付124＋119＝243帧；内部增加22帧上下文，由拼接节点自动裁除。

**SelfLift 统一续接：** 所有段长均按实际窗口采样，与动作续接一样固定继承 22 帧上下文。不额外生成未来部分，也不将继承区额外添加为历史视频条件。每段目标都填 124 帧时，两段合计 243 帧，三段合计 362 帧。

<details>
<summary><strong>展开查看：续接与采样的补充说明</strong></summary>

- 当前仅支持 `euler`。常用参数顺序为 `steps → cfg → sampler_name → scheduler → denoise`。
- `denoise` 默认 1.0；设为 0 时直接返回输入，不执行采样或续接，也不创建新的拼接信息。
- 接入外部 `sigmas` 后，它会覆盖 `steps`、`scheduler`、`denoise` 的调度设置，`high_resolution_steps` 仍然生效。
- SelfLift直连时，高清阶段使用原始latent尾部；低清阶段开启 `boundary_check` 后，用最终高清画面校准继承区的空间差异。需要 `context_vae`；关闭检查时沿用前段保存的双分辨率状态。外部视频续接的VAE要求见下文。
- 输入包含按时间分段的提示词时，需要考虑自动增加的上下文前缀。22帧上下文对应7个视频latent时间步。
- 单张首帧条件在续接时转为外观参考，中间帧、尾帧、多帧 Guide 及音频锚点随前缀后移。
- 高清阶段恢复原生22帧上下文，不增加采样步数。
- `continue_audio` 开启时，前段最后约 0.2 秒的尾音平滑释放，拼接时使用后段过渡音频替换对应区间。关闭时保留前段音频并裁掉后段重复音频。已锁定的输入音轨不会被覆盖。
- 支持音视频噪波遮罩。音频遮罩会按 latent 尺寸对齐，全零表示锁定；输入音轨较短时只补齐缺少的部分，各段解码音频按最终画面帧数对齐。
- 拼接节点不接前段画面时，只输出当前新增片段；音频未连接时可以输出无声画面。
- SelfLift 直连续接流程不需要额外叠加“动作续接”或重复裁剪。普通 K采输出不包含这里需要的双分辨率状态，不能直接替代前段 SelfLift K采的 previous_latent。

**第一段来自外部视频或普通K采，第二段才使用SelfLift：**

1. 前段H3音视频latent接“动作续接”的 `previous_latent`；也可以直接接图片帧/音频及对应VAE。当前段条件和目标latent接动作续接的两个必填输入。
2. 动作续接的 `positive_conditioning` → SelfLift的 `positive`；动作续接的 `target_latent` → SelfLift的 `latent_image`。
3. SelfLift的 `previous_latent` **留空**，视频续接时 `context_vae` **必须接H3视频VAE**，遗漏会在采样前报错停止，不再退回latent插值。它会读取已准备的22帧前缀，保持音频条件与关键帧时间，不再次添加前缀。若SelfLift接了 `previous_frames`，以这些图片的末尾22帧分别编码高清、低清上下文，统一时间终点；例如81帧外部视频编码后只对应73帧，不能把原视频末尾与截尾latent混用。没接图片时保留高清latent，并解码它来构造低清上下文。该要求不受 `boundary_check` 控制；纯音频续接不构造视频上下文，无需视频VAE。原来的SelfLift直连规则不变。
4. SelfLift输出分别解码，再接音画裁剪与拼接；需要完整成片时，将前段画面/音频接到拼接节点。

新路径的低清上下文始终通过VAE从图片编码，`lowres_scale=1` 时也不绕过；接入 `previous_frames` 时两个分辨率都使用相同尾帧，不再恢复时间终点可能不同的高清前缀。上下文不使用latent插值回退，也不向新生成帧叠加上下文的通道均值偏差。

外部视频续接的高清阶段以低清放大结果保留第一组新帧，在首个17帧周期内逐渐放开高清续采，避免第23帧开始突然改写导致颜色闪动。这些帧已在低清阶段生成动作，不是重复前段尾帧；已有的锁定遮罩保持不变。

这条路径无需补跑第一段SelfLift，也不伪造缺失的前段低清状态；从本段生成后，输出已包含SelfLift低清状态，下一段可以直接串联。直接串联前段SelfLift的原有双分辨率续接路径保持不变。

动作续接走latent路径且 `boundary_check=true` 时，H3视频VAE必须接到**动作续接自己的 `video_vae`**；接在编码/解码节点或SelfLift的 `context_vae` 上不能替代它。也可关闭动作续接的 `boundary_check`。图片帧输入路径始终需要 `video_vae`。

</details>

## 其他节点怎么用

### 语义桥（Semantic Bridge）

搜索 **H3Kit 语义桥**，连接：**H3 编码后的正向 CONDITIONING → 语义桥 → 采样器的 positive / Guider 的正向条件**。SelfLift 同样接入其 `positive`；负向条件保持原连接。每段若重新编码提示词，在重新编码之后应用语义桥。

默认选择 **BUNNY V2**。保持 `enabled` 开启，点击运行；模型不存在时，节点会从作者的 Hugging Face 仓库自动下载所选权重，下载完成后继续执行。V2 约 22 MB，保存到：

```text
ComfyUI/models/semantic_bridge/BUNNY_H3_ActionLogic_Bridge_V2.safetensors
```

目录自动创建；也可选择自动下载 BUNNY V1 或原版 `MiniMaxH3_SemanticBridge_v1.safetensors`，每次只下载所选的一份。已有本地模型时直接使用，不联网。支持手动放入子目录和 ComfyUI 额外模型路径；自定义文件名缺失时提示手动安装。

下载显示节点进度，并校验文件大小和 SHA-256；失败或取消会清理临时文件，下次运行重新下载。无法访问 Hugging Face 时，可从 [BUNNY 模型仓库](https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge/tree/main) 手动下载到上述目录。打开页面、校验工作流和检查执行缓存都不会下载。更新节点代码后重启 ComfyUI 并刷新页面。

- `enabled`：点击切换 **开启 / 关闭**。关闭时原样传递条件，不下载、不读取模型，即使没有权重也能运行。
- `alpha`：默认 `0.10`；`0` 等同关闭。固定种子对比后再调整，强度并非越高越好。
- `magnitude_match`：默认 `per_token`，逐 token 匹配原始幅度；另有整体匹配 `global` 和不匹配 `none`。

语义桥作用于条件向量，不是 LoRA；不要在同一条条件上串联多个语义桥。Ref2VA、参考音频、歌唱和口型属于需要独立验证的用法，可能退化。模型不随插件打包；训练来源和许可见 [来源说明](THIRD_PARTY_NOTICES.md)。

### 提示词与长度替换

放在 **MiniMax H3 参考转视频 → 第二段 SelfLift K采样器** 之间：

| 连接来源 | 连接目标 |
| --- | --- |
| 参考转视频的正向 | 本节点 `positive` |
| 同一个参考转视频的 Latent | 本节点 `latent_image` |
| 本节点 `positive` | 第二段 K采的 `positive` |
| 本节点 `latent` | 第二段 K采的 `latent_image` |

在本节点填写新的 `prompt` 和 `length` 即可。`length` 是目标帧数，视频帧率为 24fps。用于续段时，SelfLift 会按固定 **22 帧上下文**准备采样窗口；例如目标填 124 帧，拼接后实际新增 119 帧（约 4.96 秒）。第一段仍接原参考节点；第二段的 `previous_latent` 仍接前段 K采输出，`negative` 保持原来的连接。

长度对齐分两步：本节点先按 H3 原生的 `17k+5` 规则向上对齐目标 latent 长度，续接再按模型周期计算实际新增帧数。这里的 17 是模型长度对齐周期，**续接上下文始终是 22 帧**。

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

### 音频帧率同步

连接：**VAE 解码（音频）／最终拼接音频 → H3Kit 音频帧率同步 → Video Combine 的音频输入**。图像仍直接接 Video Combine。

`source_fps` 是音频原本对应的视频帧率，H3 默认 **24**；`target_fps` 填 Video Combine 当前设置的帧率。它们不是音频的采样率。

| 原始帧率 | 目标帧率 | 原来 6 秒的音频 |
| --- | --- | --- |
| 24 | 36 | 缩短到 4 秒 |
| 24 | 12 | 拉长到 12 秒 |
| 24 | 24 | 原样输出，仍为 6 秒 |

按 `新时长 = 原时长 × source_fps / target_fps` 做保音高变速，保留原采样率、声道和批次，时长按采样点取整。适用于**视频帧数不变，只修改播放帧率**；如果是插帧后帧率提高、总时长不变，则无需使用。多段视频应先完成原帧率下的音画裁剪与拼接，再对最终音轨应用一次。变速会重新处理波形，大幅变速可能影响音质。

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

续接固定使用 **22帧上下文（7个视频latent时间步）**，按 **24fps** 解释输入。根据已连接的输入自动选择：

| 前段输入 | 需要的 VAE | 行为 |
| --- | --- | --- |
| `previous_latent` | 边界检查或原片尾帧重新编码需要 `video_vae` | 优先沿用前段同分辨率H3音视频latent；忽略外部 `previous_audio`，图片用于边界检查或外部视频截尾对齐 |
| 仅 `previous_frames` | `video_vae` | 编码末尾22张图片，锁定视频上下文；不引入前段音频 |
| `previous_frames` + `previous_audio` | `video_vae` + `audio_vae` | 编码末尾22帧及音频末尾22/24秒，建立音画上下文 |
| 仅 `previous_audio` | `audio_vae` | 使用末尾22/24秒音频作为续接条件，不锁定视频画面 |
| 三种前段输入都不接 | 无 | 首段透传，`prefix_frames=0` |

图片帧路径使用H3视频VAE，按目标latent尺寸缩放。超过22张只取末尾22张；不足22张在开头重复首帧补齐，单张图片作为静止上下文。音频路径使用H3音频VAE，自动重采样；不足约0.917秒在开头补静音。音画同时传入时，两者应对应同一结束时刻；外部视频应先按24fps加载。输入图片来自视频时，连续尾帧比单图提供更多运动信息。

外部视频先编码再接入 `previous_latent` 时，若完整 `previous_frames` 比视频latent多不足一个17帧周期，并连接了 `video_vae`，动作续接会重新编码真实末尾22帧，并将音频参考同步取到原片终点。例如81帧输入编码为73帧时，不再截取早8帧的尾音，避免拼接后重复尾句。

音频取尾不依赖 `previous_frames`。没有完整原片帧数时，若音频比视频latent长不足一个17帧周期，则按40Hz音频长度估计24fps的末帧，取实际音频尾部并保留取整偏差。正常H3生成的音视频长度及带有分段信息的latent保持原取尾规则。这项时长估计不能恢复视频latent已丢失的图片；准确恢复画面尾部仍需原始图片。

节点输出的正向条件与 `target_latent` 都连接当前普通K采，或连接SelfLift的 `positive` 与 `latent_image`（SelfLift的 `previous_latent` 留空）。四种续接路径均输出 `prefix_frames=22`；纯音频路径的前22帧画面自由生成，随后也由音画裁剪与拼接去除。裁剪节点接完整解码画面和音频；纯音频续接只输出本段新音画时，不要把前段音频单独接到裁剪节点的 `previous_audio`（该接口用于与前段画面一起拼接）。

低清续接接前段低清采样器的去噪输出，高清续接接前段高清采样结果。循环内两路latent必须分别回传，不能只建立执行依赖。单张首帧条件转为身份参考；上下文直接复制，不反复编码图片。

原生latent路径中，`boundary_check` 开启时连接 `video_vae`；低清、高清两处的 `previous_frames` 都接上一段最终高清画面，不要预先缩小。节点通过图片和latent的尺寸识别低清阶段：只有低清继承区参与校准，同分辨率高清上下文保持原始值。关闭时原生latent路径不调用VAE，也不执行校准；图片或音频路径仍须进行VAE编码。普通动作续接没有接图片时只能检查自身latent，无法据此恢复最终高清背景。

低清校准处理尾部22帧：分别编码缩小后的最终画面与原低清尾部的解码结果，取二者差值，去除每帧每通道的空间均值变化，再以0.5强度加到原低清上下文。候选解码的整体误差至少改善2%、末尾5帧误差不增加时才采用。原输入latent、高清上下文、音频和后续新画面初始化不被这项校准修改，不额外追加历史视频Guide。

存在明显差异时，低清检查最多额外增加两次22帧编码和一次候选尾部解码；不重复解码完整视频。SelfLift 同样以最终高清画面为依据，没有接 `previous_frames` 时仅解码必要的高清尾部取得参照。低清校准的实际视频效果尚未验证，局部误差改善不保证背景、人物和色彩连续性均改善。

**固定 22 帧续接的长度示例：** 目标填 124 帧，首段交付 124 帧；续段内部为 141 帧，裁掉 22 帧后新增 119 帧。内部片段长度遵循 H3 的 `17k+5` 对齐规则，续接上下文固定为 22 帧。后续音频起点必须按实际交付长度推进。

采样器的 MODEL 直接接原模型/LoRA链路。SelfLift 的 `model` 也接原模型/LoRA链路，并可选用外部 `sigmas`。

### 分段解码、裁剪与拼接

**H3Kit 音画裁剪与拼接** 必须接本段 `sampled_latent` 和完整 `decoded_frames`，可接 `decoded_audio`。它自动读取续接信息，去除22帧重复开头并对齐音频，不再提供手动裁剪输入框。接入 `previous_frames` / `previous_audio` 时拼合前后段，否则只交付本段新画面。latent 输出保留本段状态。

每段完整解码一次，画面复用为下一段的图片参考与检查依据；裁掉重复22帧后直接拼接已有画面。下一段接收本段原始采样latent，不累计整条latent，也不在结尾重新解码全部视频。

请使用随包示例工作流，并按当前节点的输入名称连接。

**已知限制：** 接缝处仍可能改变背景物体或纹理，不能保证长视频无漂移。

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

不能保证。插件提供原生上下文续接和音频过渡，但动作、人物、背景的一致性仍受模型与输入内容影响。

## 来源与许可

完整的代码来源、移植关系和修改说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本插件按 [GPL-3.0](LICENSE) 提供，MIT 许可部分保留原作者的版权与[许可文本](licenses/latent-upscaler-MIT.txt)。
