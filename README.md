# ComfyUI-H3-upgrade-kit

提供 H3 的 3D 潜空间放大、高清分块采样、VAE 分块解码、动作续接、音画裁剪，以及可直接串联的 SelfLift K采和音画拼接，共七个独立节点。无需安装原节点包，也不修改 ComfyUI 核心。

## 安装与查找

将本目录放入 `ComfyUI/custom_nodes/`，使用 ComfyUI 的 Python 安装 `requirements.txt` 后重启。当前整合包已提供这些依赖时无需重复安装。在节点搜索中输入 `H3Kit`，或打开 `H3 Upgrade Kit` 分类。

也可在 ComfyUI Manager 中搜索 `ComfyUI-H3-upgrade-kit`，或使用 `comfy node install comfyui-h3-upgrade-kit` 安装 Registry 版本。

Python 依赖已列入 `requirements.txt`：`torch`、`torchaudio`、`einops`、`safetensors`。其中 `torch` 与 `torchaudio` 应沿用当前 ComfyUI 环境相互兼容的版本；`safetensors` 用于经 ComfyUI 读取放大权重。`comfy`、`comfy_api`、`comfy_extras` 等由 ComfyUI 自带，无需另外 pip 安装。前端扩展不需要 npm 安装。

动作续接要求具有新版 MiniMax H3 `PackedLayout` 的 ComfyUI（上游节点注明 0.34.0 起）；首次续接会检查真实布局行为。放大与采样使用 ComfyUI V3 节点接口，由标准入口映射注册。

| 原节点 | 新显示名称 | 新注册 ID / 类名 |
| --- | --- | --- |
| Minimax H3 Latent Upscaler (3D) | H3Kit 3D 潜空间放大 | `H3KitLatentUpscale3D` |
| Minimax H3 高清分块采样 | H3Kit 高清分块采样 | `H3KitTiledSampler` |
| H3 Motion Context | H3Kit 动作续接 | `H3KitMotionBridge` |
| H3 Motion Context Trim | H3Kit 音画同步裁剪 | `H3KitAVTrim` |
| 新增专用解码节点 | H3Kit VAE 分块解码 | `H3KitVAEDecodeTiled` |
| 新增双分辨率续接采样 | H3Kit SelfLift K采样器 | `H3KitSelfLiftSampler` |
| 新增重叠裁剪与音频过渡 | H3Kit SelfLift 续接音画拼接 | `H3KitSelfLiftAVJoin` |

新旧节点 ID 不同，可以同时安装。现有工作流继续使用原节点；使用新节点时需替换节点并重新连线。

## 示例工作流

[MINIMAX数字人v2加速版.json](example_workflows/MINIMAX数字人v2加速版.json) 是完整的数字人工作流，包含参考图、歌曲加载与裁剪、H3 模型和条件、两段 SelfLift 采样、视频/音频解码及视频保存。下载 JSON 后拖入 ComfyUI 即可导入。

示例保留原工作流的节点、连线、参数与布局，仅清理本机输出预览记录和工作区标识，并补充 H3Kit 节点的 Registry 标识。

运行前需准备：

- 在 `LoadImage`、`LoadAudio` 中选择自己的参考图片和音频。示例不附带图片、歌曲、生成结果或模型权重。
- 主模型：`minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors`，放入 `models/diffusion_models/`。
- 文本编码器：`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`，放入 `models/text_encoders/`。
- 视频及音频 VAE：`minimax_h3_video_vae_fp16.safetensors`、`minimax_h3_audio_vae_fp32.safetensors`，放入 `models/vae/`。
- 放大权重：`minimax_h3_latent_upscaler_3d_bf16.safetensors`，放入 `models/latent_upscale_models/`。
- 检查两个 LoRA 加载器中的 `minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors` 与 `MysticXXX_MMH3-V4.safetensors`，按本机文件位置重新选择；原工作流选择的是 `models/loras/minimaxH3/` 子目录。
- 示例还使用 `ComfyUI-ReservedVRAM`、`ComfyUI_LayerStyle`、`ComfyUI-UniversalToolkit`、`ComfyUI-VideoHelperSuite`、`ComfyUI-KJNodes`、`rgthree-comfy`。可用 Manager 的缺失节点安装功能补齐。这些是示例工作流的节点包依赖，不是本插件的 pip 依赖。

## 使用

- 放大模型仍从 `ComfyUI/models/latent_upscale_models/` 读取，支持子目录及 ComfyUI 额外模型路径。沿用已有 `.safetensors` / `.pth` 权重，不自动下载。放大节点接单路图像或视频 latent；H3 音视频复合 latent 请先分离视频、放大后再与音频合并。
- `resize_settings` 提供倍率、目标尺寸和百万像素三种模式，`pixel_alignment` 默认 32。`temporal_chunks` 控制时间分块，`spatial_tiles` 控制空间分块。权重按次加载，不保留全局模型缓存；`release_weights` 开启时在 CUDA 推理结束后先移回 CPU。
- 高清采样接 H3 音视频复合 latent，默认至少 4 块，显存规划最多 8 块。它复制 guider 和 model patcher，不改变其他节点共享的 guider。分块采样不支持 ControlNet。
- 动作续接已恢复视频 VAE 对齐路径，`video_vae` 必接。只传 `previous_latent` 时先解码画面，再重新编码实际末尾 17 帧；传 `previous_frames` 时优先复用画面，跳过解码但仍执行编码。音频优先直接取 `previous_latent` 尾部；仅传 `previous_audio` 时才需要 `sound_vae` 编码。没有上下文时直接透传。
- 续接节点的 `target_latent` 输出接采样器；`prefix_frames` 和 `delivery_frames` 接裁剪节点同名输入。裁剪节点接解码后的画面和音频，`frame_rate` 应与最终视频一致（H3 默认 24）。保留原定帧数并裁掉新增的 17 帧头部，同时对齐音频长度。
- 高清二采 context：独立放大节点保留续接信息、缩放视频遮罩并标记放大结果。再次经过动作续接时，按上一段真实高清 context 的末端残差衔接生成区域；以新段首个时间周期为基准比较同相位内容，画面变化后衰减末端补差，退回扣除时间波动后的中值残差，避免运镜时固定位置残影。然后锁定真实高清前缀。校正只消费一次放大标记，不增加采样步数、不改音频，也不改变普通直出续接。现有连线无需调整。

## SelfLift K采与多段续接

这是普通节点连接流程，不需要时间线导演台，也不需要安装 `comfyui-SelfLift` 或 TimelineDirector。

1. 本段的 **目标高清** H3 音视频复合 latent 接 `latent_image`；模型、正负条件按普通 K采连接。不要预先把该 latent 缩小。
2. 选择 `upscale_weights`。例如总步数 `8`、`high_resolution_steps=2`、`lowres_scale=0.5` 表示半宽半高采样 6 步，学习型 latent 放大后再采样 2 步。固定使用 Euler；没有像素/VAE 往返校正，也不启用 TST。8 步应配合适用的加速模型/LoRA；普通模型沿用原工作流的总步数。
3. 前一个 SelfLift K采的 `sampled_latent` **直接**连接后一个的 `previous_latent`。后一节点的 `latent_image` 仍接后一段自己的目标 latent。保持目标分辨率和低清比例一致；不要经过会丢失附加状态的拆分、合并、裁剪或独立放大节点。
4. 每段 `sampled_latent` 分别接视频/音频解码器；也连接本段拼接节点的 `sampled_latent`。完整解码结果接 `decoded_frames` / `decoded_audio`。前段拼接输出接后段拼接的 `previous_frames` / `previous_audio`。最后的 `images` / `audio` 接 Create Video（24fps）和保存节点。

```text
本段目标 latent A → SelfLift K采 A ──sampled_latent──→ SelfLift K采 B.previous_latent
本段目标 latent B ─────────────────────────────────→ SelfLift K采 B.latent_image

K采 A → 视频/音频解码 → 拼接 A ──images/audio──→ 拼接 B.previous_frames/previous_audio
K采 B → 视频/音频解码 ────────────────────────→ 拼接 B.decoded_frames/decoded_audio
K采 B.sampled_latent ─────────────────────────→ 拼接 B.sampled_latent
拼接 B → Create Video（24fps）→ 保存
```

完整示例见上方的「MINIMAX数字人v2加速版」。以下说明用于理解各节点及按需搭建自己的续接流程。

- `latent_image` 的长度代表**本段新增帧数**。`previous_latent` 不连接时生成首段；接入后只在头部增加上下文，**不再补视频尾帧**。拼接节点裁掉重复前缀，保留原定新增帧数。两段各 124 帧，最终为 **248 帧**；三段为 **372 帧**。不需要手动加帧，也不要再叠加旧动作续接节点。
- `overlap_frames` 默认 **17**，以 **17** 为步长，使用 **17、34、51、68……**。17 帧对应 5 个 latent 时间步；将前段最后 5/10/15……个时间步分别放到下一段新增的头部，低清与高清状态使用同一区间。比如第二段目标新增 124 帧、上下文 17 帧，内部生成 **141 帧**，裁掉前 17 帧后仍保留 124 帧；上下文 34 帧则生成 158 帧、裁头 34 帧，同样无需裁视频尾部。
- 旧工作流的 22/39 等非整周期值在载入时迁移为 17/34；接口直接传入时也向下对齐，最少 17 帧。若使用旧版保存的 latent 且有效终点落在一个时间步内部，需要重新运行前段 K采。新流程的有效终点就是完整输出尾部。
- 不接 `context_vae` 时采用纯 latent 的尾部搬移，不经 VAE 或时间插值；这不同于把实际解码画面重新编码，因此不宣称逐像素重建旧尾帧。新段已有的初始化视频与遮罩一起前移 5/10/15……个时间步，音轨和音频遮罩按 40Hz 后移，中间帧、尾帧及多帧Guide条件后移相同帧数，单张首帧按下述规则转为外观参考。输入有明确分段时间的提示词，也需要考虑这段自动预留的上下文。
- 第二段连接 `context_vae`（H3视频VAE）时，两个阶段统一以前段最终高清画面为上下文来源：原尺寸重新编码用于高清阶段，先把尾部画面缩小到低清尺寸、再编码用于低清阶段。避免低清采样先继承高清修复前的背景、高清阶段再强行拉回。这里缩放的是实际画面，不是直接缩放高清latent。`previous_frames` 可复用第一段实际解码画面；未连接时只解码一次前段高清结果。不再单独解码旧低清结果。仅处理上下文，整段仍使用学习型latent放大。
- 输出 latent 同时保存原生低清 x0 与最终高清结果。下一段在两个分辨率阶段分别接入对应尾部。放大后、高清采样前，复用动作续接的空间残差补偿：按高清上下文末端衔接新画面，随同相位内容变化减弱末端补差，退回扣除时间波动后的稳定残差；尊重原视频遮罩，再恢复真实高清前缀。校正作用于保留的新画面，不再只修正将被裁掉的前缀。不增加采样步数，不改音频、原生低清状态或动作续接节点；现有连线无需调整。
- 续接固定使用一种方案：两个阶段分别锁定对应分辨率的重叠区，并像动作续接一样把连续上下文作为一整组运动Guide放在第0帧；默认17帧对应5个连续latent时间步，较长重叠区使用完整前缀。接入 `context_vae` 后，两个Guide由同一段真实尾部画面按各自分辨率编码；未接VAE时仍分别使用保存的原生低清和高清尾部。继续保留放大后的高清尾部补偿，不增加采样步数，但运动条件会增加注意力计算和显存开销。
- 单张首帧条件转为外观参考，保留人物/服装信息，避免在接缝处重置姿势；中间帧、尾帧、多帧Guide和音频锚点仍按新增前缀后移，已有参考保留。运动Guide只加入正条件，输入条件不被原地修改。这里使用真实连续尾部，不重复静态末帧。Guide参与整段注意力，实际动作与背景连续性仍需完整模型实测。节点没有模式选项，旧工作流载入时自动删除旧模式控件值。
- 学习型放大使用真实连续时间序列，首段与续接段沿用同一时间分块路径。已移除把接缝两侧分别填充大量重复帧的处理：放大网络的 GroupNorm 包含时间维，这种填充会改变统计分布，影响可以扩散到保留区间。
- `continue_audio` 开启时，尾音在最后 8 个音频 token（约 0.2 秒）平滑释放；拼接用后段的过渡音频替换前段对应尾音。关闭时保留前段音频、裁掉后段重复音频。输入音轨已锁定时不覆盖它。各段解码音频按画面帧数补齐/裁剪，避免长度误差累积。
- 支持“设置Latent噪波遮罩”提供的64×64等任意尺寸音频遮罩，按标准采样器规则缩放到音频latent尺寸；全零仍表示锁定。分离后合并的原音轨若短于视频，只补齐缺少的音频latent并将该部分设为可生成，已有锁定音轨不变；长于视频则截取对应时段。
- 拼接节点不接前段画面时，只输出裁掉头部上下文的当前新增片段；音频未连接时可以输出无声画面。要保留完整 Soft AV 过渡，应连接前段完整的音画输出。解码节点仍输出含上下文的完整片段，必须经过本拼接节点才能得到准确的交付长度。音频的40Hz网格仍可能有取整差异，拼接节点按24fps精确对齐波形长度。
- `denoise` 为可调降噪强度，默认 1.0；未连接外部 `sigmas` 时，与 `steps/scheduler` 一起生成内置调度。低于 1 时保留更多输入 latent 内容；0 时原样返回输入，不执行二采或续接，也不创建新的低清状态和拼接信息。
- 常用参数顺序为 `steps → cfg → sampler_name → scheduler → denoise`，之后再显示高清步数等 SelfLift 参数。`sampler_name` 当前仅支持 `euler`：分辨率交接复用了最后一次低清预测，按 Euler 公式重建交接状态以维持总预测次数；其他算法需要单独适配。前端扩展会在载入旧工作流时迁移参数顺序，保留原降噪值。
- 可选 `sigmas` 接外部调度，连接后覆盖 `steps/scheduler/denoise`，降噪强度改由外部调度器控制。`high_resolution_steps` 仍生效。不提供高级 K采的任意起止步：低、高阶段属于同一条 Euler 调度。
- `spatial_tiles` 开启高清阶段的空间分块，默认关闭。`minimum_tiles` 默认 4，范围 2–8，按显存预算增加到最多 8 块；小画面受网格限制可能更少。复用本插件“高清分块采样”的实现：沿长边重叠分块、裁剪视频遮罩和关键帧、保留全局位置、融合视频预测；音频及其遮罩保持完整，音频采用第一块的预测。支持 SelfLift 的原生音视频遮罩和双分辨率续接，不支持 ControlNet。
- 分块只影响高清去噪，不改变低清采样、latent 放大或 VAE 解码。每一步高清采样会分别计算各块，不保证提速；模型权重、完整采样状态和参考数据仍占用内存。由于块间没有完整的全局注意力，画质可能与整幅采样不同。
- “动作续接”节点采用 17 帧前缀，支持直接 latent 或图片编码，继续服务原工作流；SelfLift 直连流程无需它，也不要再叠加旧的裁剪节点。只有新采样器输出包含所需低清状态，普通 K采输出不能直接代替。

## H3 VAE 分块解码

替换工作流中画面分支的普通“VAE 解码”：采样器输出接 `video_latent`，H3 视频 VAE 接 `video_vae`，`decoded_frames` 接原来的图像/视频处理节点。音视频复合 latent 会自动提取视频，音频分支保持原接法。

本机 H3 VAE 已内置 256 像素空间分块和时间分块；普通分块解码节点的参数会被该实现忽略。本节点默认保持原生 256 / 64 空间布局，通过独立 model patcher 调节每次解码的块数，继续使用原生时间解码、重叠融合及输出缓冲，完成或异常退出后撤销临时补丁。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `tile_edge` | 256 | 跨行调度的纵向跨度，单位为像素；不再表示模型窗口大小 |
| `tile_blend` | 64 | 原生 256 小块的最小空间重叠 |
| `tiles_per_batch` | 8 | 每次合并解码的原生小块数上限 |

三个分块参数设置后直接生效，无需额外开启跨行调度。可以用 512 / 64 / 2 做画质对照，再比较每批 4 和 8 的耗时。合批越大越占显存；OOM 时自动减半并重试，降至 1 块仍不足则报告错误。调度跨度至少为 256，按 16 像素网格对齐；重叠限制在 16 到 240 之间，避免原生融合的零重叠错误。画质对照请保持重叠 64。

开发验证中已撤掉直接放大模型窗口的路径。真实 FP16 权重对照中，原生解码器直接使用 512 会产生网格，单独修正空间位置编码也未解决。现在所有调度尺寸都使用同一组原生 256 窗口；调度器按原生坐标截取窗口、合批计算，并按原生顺序融合，因此不会因设置 512 而改变模型的空间位置编码或注意力窗口。

分块不保证提速。窄画面可以把不同行的小块放进同一批，减少 decoder 调用；实际效果取决于分辨率、合批上限和显存。384 与 512 等跨度在部分画面上可能容纳相同行数，因而调度相同。`tile_edge` 表示调度跨度。继续保持 H3 的原生时间周期，不提供任意时间块长。

## 来源

见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。移植保留来源及许可；重新命名不改变原许可要求。
