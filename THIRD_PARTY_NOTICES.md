# 来源及修改

## 固定22帧原生续接（2026-09-22）

当前 `native_context.py` 负责H3时间布局、原生尾部复制和边界检查，并用最终高清画面对低清上下文做局部空间校准。高清原始latent保持不变，不包含旧版全局颜色补偿或动态Drift算法。续接接口固定22帧；成片复用各段已解码画面进行拼接。

两次同布局编码差值的思路参考 [ethanfel/ComfyUI-MiniMaxH3-Context-Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop) 的 `latent_color_carry.py`（GPL-3.0）。本地低清适配去掉差值的逐帧通道空间均值，以半强度构造候选，并在解码误差改善后采用；这不代表上游已验证本地效果。

## 多路循环（2026-09-20）

`nodes_loop.py` 的循环展开与逐轮执行逻辑改编自 ComfyUI `comfy_extras/nodes_loop.py`，沿用 GPL-3.0 许可。复用原生循环边界、模式、缓存和进度机制，新增独立编号的数据回传与输出；`web/loops.js` 配合原生 Autogrow 整理输入分组并同步显示输出端口。无需 Easy Use 等外部循环插件。

## SelfLift 续接采样（2026-09-18）

`selflift_sampling.py` 的渐进分辨率交接、双分辨率上下文、Soft AV和升采样时间边界改编自 [Songssx/ComfyUI-MiniMaxH3-TimelineDirector](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector)，提交 `03915aae320d186f1498d12a847689af0902785d` 的 `selflift_runtime/`、`drift_control_av.py` 和 `experimental_latent_guide.py`，遵循其 GPL-3.0 许可，完整许可见本目录 `LICENSE`。

该项目的渐进采样来源为 [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift)，对照版本 `19ec540505dcbc261aaecf450d83c7536be2826d`；Drift-Control AV 来源为 [ethanfel/ComfyUI-MiniMaxH3-Contex-Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop)。保留这些来源说明。本次仅移植所需 H3 推理算法，复用本插件的 LBH-123-AI 放大网络，不引入时间线、像素/VAE 校正、TST 或运行时外部插件依赖。

`nodes_selflift.py` 提供独立采样节点接口；`nodes_av_join.py` 整合按 SelfLift 采样元数据进行的音画拼接与普通采样的裁剪逻辑。节点改名、接口调整和裁减功能不改变所移植代码的许可义务。

本插件最初根据本机版本移植（2026-09-18）。保留时间分块、空间分块和身份参考处理；当前上下文固定22帧。上文保留历史移植来源，不表示已移除的算法仍在运行。

- `nodes_latent_upscale.py`：来自 `Comfyui_Minimax_h3_latent_Upscaler/nodes/minimax_h3_latent_upscaler_3d.py`，Copyright (c) 2026 LBH-123-AI，MIT，完整许可见 `licenses/latent-upscaler-MIT.txt`。
- `nodes_tiled_sampler.py`、`sampling_tiles.py`：来自同一插件的 `minimax_h3_tiled_sampler.py`、`h3_tiling.py`；原分块文件注明 adapted from the local SelfLift implementation，该来源说明保留。
- `nodes_motion.py`、`motion_layout.py`：来自 [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) 的 `nodes.py` 和 `layout_contract.py`，Copyright (C) 2026 NikoDemon80，GPL-3.0，完整许可见 `LICENSE`。仅移植续接、裁剪及其必要依赖。
- `ComfyUI-relayapi` 仅作为插件入口布局与 Registry 元数据格式参考，未复制 API 请求、密钥设置或联网功能。
- `nodes_vae_decode.py` 的窗口融合顺序改编自本机 ComfyUI `comfy/ldm/minimax/vae.py` 的 `MiniMaxH3VideoVAE.tiled_decode`；新增原生窗口跨行合批调度，不复制模型权重。沿用 ComfyUI 的 GPL-3.0 许可。

修改包括独立包结构、节点和参数重命名、独立补丁标识、裁剪无关节点/路由、使用 ComfyUI 安全权重读取器和 meta 初始化、移除跨执行模型缓存。权重层名及 ComfyUI 数据协议保留。组合插件按 `LICENSE` 所列 GPL-3.0 许可提供，MIT 部分同时保留原版权和许可。
