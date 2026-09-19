# ComfyUI-H3-upgrade-kit

**把 MiniMax H3 的潜空间放大、分块采样和多段续接，放进一套更方便搭建的节点里。**

这个插件基于下面三个开源项目，结合 H3 工作流的实际使用，做了优化、升级和整合：

- [NikoDemon80 / ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)：动作续接与音画衔接。
- [LBH-123-AI / Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)：H3 潜空间放大与高清分块采样。
- [facok / comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift)：先低清采样、再放大续采的 SelfLift 思路。

感谢三位作者把代码和经验分享出来，让这个整合版有了基础。我们在此基础上整理了节点接口，加入 VAE 分块解码、可串联的 SelfLift 采样和音画拼接，希望大家搭建 H3 工作流时能少绕一些弯路。

[安装](#安装) · [工作流示例](#工作流示例) · [SelfLift 多段续接](#selflift-多段续接) · [其他节点怎么用](#其他节点怎么用) · [常见问题](#常见问题)

## 能做什么

插件共 **7 个节点**。在 ComfyUI 中搜索 `H3Kit`，或打开 `H3 Upgrade Kit` 分类就能找到。

| 你想做的事 | 使用的节点 |
| --- | --- |
| 先低清生成，再放大到高清继续采样 | **H3Kit SelfLift K采样器** |
| 把前后两段视频和音频拼起来 | **H3Kit SelfLift 续接音画拼接** |
| 单独放大 H3 视频 latent | **H3Kit 3D 潜空间放大** |
| 对高清画面分块采样 | **H3Kit 高清分块采样** |
| 让下一段接着上一段的动作生成 | **H3Kit 动作续接** |
| 裁掉续接时重复的开头，同步音频长度 | **H3Kit 音画同步裁剪** |
| 分批处理 H3 视频 VAE 解码的小块 | **H3Kit VAE 分块解码** |

使用本插件不需要另外安装上面三个原插件，也不会修改 ComfyUI 核心文件。示例工作流用到的其他辅助节点包，见下面的准备说明。

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

**[下载工作流 JSON](https://raw.githubusercontent.com/flywhale-666/ComfyUI-H3-upgrade-kit/main/example_workflows/minimax_digital_human_v2_accelerated.json)** · [查看工作流文件](example_workflows/minimax_digital_human_v2_accelerated.json)

这是一条完整的数字人工作流，包含参考图、歌曲加载与裁剪、模型加载、两段 SelfLift 采样、音视频解码和视频保存。

1. 下载 JSON，拖入 ComfyUI。
2. 安装缺失的辅助节点，并在加载器中选好本机模型。
3. 在 `LoadImage` 和 `LoadAudio` 中换成自己的参考图片、音频。
4. 检查提示词、分辨率和采样参数，再开始生成。

示例不附带模型、图片、歌曲或生成结果，需要自行准备。模型文件不在同一子目录时，在对应加载器里重新选择即可。

<details>
<summary><strong>展开查看：示例使用的模型</strong></summary>

下面的目录均相对于 `ComfyUI/models/`。

| 用途 | 示例选择的文件 | 放置目录 |
| --- | --- | --- |
| H3 主模型 | `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors` | `diffusion_models/` |
| 文本编码器 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders/` |
| 视频 VAE | `minimax_h3_video_vae_fp16.safetensors` | `vae/` |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` | `vae/` |
| 潜空间放大模型 | `minimax_h3_latent_upscaler_3d_bf16.safetensors` | `latent_upscale_models/` |
| 加速 LoRA | `minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors` | `loras/minimaxH3/` |
| 另一处 LoRA 加载器 | `MysticXXX_MMH3-V4.safetensors` | `loras/minimaxH3/` |

</details>

<details>
<summary><strong>展开查看：示例使用的其他节点包</strong></summary>

- `ComfyUI-ReservedVRAM`
- `ComfyUI_LayerStyle`
- `ComfyUI-UniversalToolkit`
- `ComfyUI-VideoHelperSuite`
- `ComfyUI-KJNodes`
- `rgthree-comfy`

可通过 Manager 的缺失节点安装功能补齐。这些是这条示例工作流使用的辅助节点包，不是本插件需要额外安装的 Python 依赖。

</details>

## SelfLift 多段续接

### 先理解它怎么工作

SelfLift 把一次采样分成低清和高清两个阶段：**先在较低分辨率下采样，再用潜空间放大模型放大，最后完成高清部分。**

例如：`steps = 8`、`high_resolution_steps = 2`、`lowres_scale = 0.5`，就是先以一半宽高采样 6 步，放大后再采样 2 步。

> **8 步需要搭配适用的加速模型或 LoRA。** 使用普通模型时，请沿用适合该模型的步数，不要直接照搬 8 步。

### 怎么连接

每一段的 `latent_image` 都接这一段的**目标高清音视频 latent**，不需要提前缩小。前一个 K采的 `sampled_latent` 直接接后一个的 `previous_latent`。

两段保持相同的目标分辨率和低清比例，连接中间不要拆分或重新合并 latent，以免丢失续接状态。

每段采样后分别解码，再接对应的 **SelfLift 续接音画拼接** 节点：

| 输入 | 连接内容 |
| --- | --- |
| `sampled_latent` | 本段 SelfLift K采的原始输出 |
| `decoded_frames` / `decoded_audio` | 本段完整解码的画面和音频 |
| `previous_frames` / `previous_audio` | 前段拼接后的画面和音频；第一段留空 |

最后将拼接节点的 `images`、`audio` 接到视频保存节点，帧率使用 **24fps**。

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
- SelfLift 直连续接流程不需要额外叠加“动作续接”或“音画同步裁剪”节点。普通 K采输出不包含这里需要的双分辨率状态，不能直接替代前段 SelfLift 输出。

</details>

## 其他节点怎么用

### 3D 潜空间放大

将 H3 视频 latent 接到 `source_latent`，选择放大权重与目标尺寸。支持按倍率、目标宽高或百万像素设置大小。

- 权重放在 `ComfyUI/models/latent_upscale_models/`，支持子目录和 ComfyUI 额外模型路径；不自动下载。
- 接收单路图像或视频 latent。音视频复合 latent 需要先分离视频，放大后再与音频合并。
- `temporal_chunks` 控制时间分块，`spatial_tiles` 控制空间分块，`pixel_alignment` 默认 32。
- `compute_backend`、`compute_precision` 控制设备和精度。开启 `release_weights` 时，CUDA 推理结束后会将放大权重移回 CPU。

### 高清分块采样

在独立采样流程中，将 H3 音视频复合 latent 接入 **H3Kit 高清分块采样**。节点沿长边重叠分块，再融合视频结果；音频保留完整，并采用第一块的预测。

默认至少 4 块，按显存规划最多 8 块。**不支持 ControlNet。** 分块只作用于采样，不代替 VAE 分块解码。

### 动作续接与音画同步裁剪

用于在普通 H3 采样流程里接续前一段：

```text
H3 条件与目标 latent → 动作续接 → 采样 → 解码 → 音画同步裁剪 → 保存
```

`video_vae` 必接。前段可以通过 `previous_frames` 或 `previous_latent` 提供：传画面时直接编码末尾 17 帧；只传 latent 时先解码再编码。没有前段上下文时直接透传。

音频优先取 `previous_latent` 的尾部；只提供 `previous_audio` 时，需要连接 `sound_vae`。将续接节点的 `prefix_frames` 和 `delivery_frames` 接到裁剪节点同名输入，裁掉重复开头并同步音频长度。`frame_rate` 要与保存视频的帧率一致，H3 默认 24。

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

**导入示例后出现红色缺失节点？**

示例还使用了一些辅助节点包，按“工作流示例”中的清单补齐；同时检查模型加载器中的文件选择。

**用了分块就一定更快、更省显存吗？**

不保证。分块和合批的效果取决于分辨率、模型、显存和参数。分块采样需要逐块计算，缺少整幅画面的全局注意力，结果也可能与不分块时不同。

**能保证每次续接都完全无缝吗？**

不能保证。插件提供上下文续接、高清衔接补偿和音频过渡，但动作、人物、背景的一致性仍受模型与输入内容影响。

## 来源与许可

完整的代码来源、移植关系和修改说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本插件按 [GPL-3.0](LICENSE) 提供，MIT 许可部分保留原作者的版权与[许可文本](licenses/latent-upscaler-MIT.txt)。
