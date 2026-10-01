"""可串联的 SelfLift K采、解码后的音画拼接与续接状态存取。"""

import logging
import os

import torch
import comfy.samplers
import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import BasicScheduler

from .nodes_latent_upscale import list_upscale_weights
from .selflift_sampling import LOW_CARRY, SEGMENT, progressive_sample


class H3KitSelfLiftSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT", {"tooltip": "本段目标高清H3音视频latent；支持动作续接输出的target_latent，此时positive也接动作续接输出，previous_latent留空。自动继承22帧前缀；124帧目标交付119帧新画面。"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 8, "min": 2, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0, "max": 100, "step": 0.1}),
                "sampler_name": (["euler"], {"default": "euler",
                    "tooltip": "当前 SelfLift 交接使用 Euler 更新公式，只支持标准 euler；其他算法需要单独适配交接与历史状态。"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "降噪强度，仅在未连接外部 sigmas 时生效。低于 1 时保留更多输入 latent 内容；0 时直接返回输入，不执行二采或续接。"}),
                "high_resolution_steps": ("INT", {"default": 2, "min": 1, "max": 9999,
                    "tooltip": "总步数中的高清部分。8 步、高清 2 步表示低清 6 步＋高清 2 步；固定使用 Euler。"}),
                "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05,
                    "tooltip": "低清宽高比例。直接串联时前后段须使用相同设置，保留原生高清上下文；动作续接输入时两阶段以相同尾帧建立上下文。"}),
                "upscale_weights": (list_upscale_weights(),),
                "continue_audio": ("BOOLEAN", {"default": True,
                    "tooltip": "直接串联SelfLift时续接前段尾音，最后8个音频token平滑释放。动作续接输入的音频条件和遮罩由上游决定。"}),
            },
            "optional": {
                "previous_latent": ("LATENT", {"tooltip": "仅用于直接串联前一个SelfLift的sampled_latent，保留低清和高清状态。外部视频或普通K采经动作续接接入latent_image时，此处留空。"}),
                "sigmas": ("SIGMAS", {"tooltip": "可选外部调度；连接后取代 steps/scheduler/denoise，总步数和降噪强度由外部调度决定。"}),
                "spatial_tiles": ("BOOLEAN", {"default": False,
                    "label_on": "高清采样分块：开启", "label_off": "高清采样分块：关闭",
                    "tooltip": "仅对 SelfLift 高清采样阶段按自动网格重叠分块：优先沿长边分割，实际单块长宽比不超过2:1；支持视频/音频遮罩及固定上下文续接。不影响 latent 放大或 VAE 解码。"}),
                "minimum_tiles": ("INT", {"default": 2, "min": 2, "max": 8,
                    "tooltip": "自动网格的最少块数，按块形状和显存预算增加到最多8块；小画面可能更少，极端长宽比无法满足2:1时提示并选最接近的布局。音频取第一块预测，不支持ControlNet。"}),
                "upscaler_unload": ("BOOLEAN", {"default": True,
                    "label_on": "放大后卸载：开启", "label_off": "放大后卸载：关闭",
                    "tooltip": "放大完成后、进入高清阶段前，立即把 latent 放大模型从显存卸载，为高清采样腾出显存；放大出错时同样卸载。权重文件有缓存，下次运行无需重新读盘，但需重新上传显存。显存充足且连续运行时可关闭，省去重复加载。"}),
                "context_vae": ("VAE", {"tooltip": "H3视频VAE。动作续接→SelfLift的视频续接必须连接，否则采样前报错；已接previous_frames或关闭boundary_check也不能省略。用于建立两阶段时间对齐的上下文。纯音频路径不需要；原来的SelfLift直连仍按boundary_check决定是否需要。"}),
                "previous_frames": ("IMAGE", {"tooltip": "连接前段完整画面或末尾22帧。动作续接输入时，以这些图片的末尾22帧分别编码高清和低清上下文，避免外部视频编码截尾造成时间错位；不连接则解码原高清上下文。SelfLift直连时仍用于原来的边界检查。"}),
                "boundary_check": ("BOOLEAN", {"default": True,
                    "tooltip": "检查22帧上下文；以最终高清画面校准低清空间差异，仅接受局部误差改善的候选。关闭则不检查、不校准。需要context_vae。"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("sampled_latent",)
    FUNCTION = "sample"
    CATEGORY = "H3 Upgrade Kit/采样"
    DESCRIPTION = "低清采样→学习型latent放大→高清续采。支持直接串联SelfLift，或接收动作续接已准备的22帧上下文；后者不需要前段SelfLift，previous_latent留空。解码后接音画裁剪与拼接去除重复前缀。"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, scheduler,
               high_resolution_steps, lowres_scale, upscale_weights,
               continue_audio, previous_latent=None, sigmas=None, denoise=1.0, sampler_name="euler",
               spatial_tiles=False, minimum_tiles=2, upscaler_unload=True, context_vae=None,
               previous_frames=None, boundary_check=True):
        if sigmas is None:
            if denoise == 0:
                return (latent_image,)
            sigmas = BasicScheduler.execute(model, scheduler, steps, denoise)[0]
        return (progressive_sample(model, positive, negative, latent_image, sigmas, seed, cfg,
                                   high_resolution_steps, lowres_scale, upscale_weights,
                                   previous_latent, continue_audio, sampler_name=sampler_name,
                                   upscaler_unload=upscaler_unload,
                                   spatial_tiles=spatial_tiles, minimum_tiles=minimum_tiles,
                                   context_vae=context_vae, previous_frames=previous_frames,
                                   boundary_check=boundary_check),)


STATE_FORMAT = "h3kit_selflift_state"
STATE_VERSION = 1


def _resolve_state_path(path):
    path = (path or "").strip().strip('"').strip("'")
    if not path:
        raise ValueError("请填写续接状态文件路径；相对路径基于 ComfyUI 的 output 目录。")
    if not os.path.isabs(path):
        path = os.path.join(folder_paths.get_output_directory(), path)
    return path


def _extract_state(latent):
    """提取 previous_latent 消费的数据：音视频复合 samples、低清状态与段信息。"""
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not getattr(samples, "is_nested", False):
        raise ValueError("sampled_latent 需要连接 H3Kit SelfLift K采样器的 sampled_latent 输出（音视频复合 latent）。")
    video, audio = samples.unbind()
    low = latent.get(LOW_CARRY)
    if low is None:
        raise ValueError("sampled_latent 缺少低清续接状态（h3kit_selflift_low），只能连接 SelfLift 采样器的输出。")
    state = {
        "samples": NestedTensor([video.detach().to("cpu"), audio.detach().to("cpu")]),
        LOW_CARRY: low.detach().to("cpu"),
    }
    if latent.get(SEGMENT) is not None:
        state[SEGMENT] = latent[SEGMENT]
    return state


def _save_state(path, state):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    video, audio = state["samples"].unbind()
    torch.save({
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "video": video,
        "audio": audio,
        LOW_CARRY: state[LOW_CARRY],
        SEGMENT: state.get(SEGMENT),
    }, path)
    logging.info("[H3Kit SelfLift] 续接状态已保存：%s", path)


def _load_state(path):
    if not os.path.isfile(path):
        raise ValueError(f"找不到续接状态文件：{path}。请先连接 sampled_latent 运行一次以保存状态，或检查路径。")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"读取续接状态文件失败：{path}（{exc}）") from exc
    if not isinstance(payload, dict) or payload.get("format") != STATE_FORMAT:
        raise ValueError(f"{path} 不是 H3Kit SelfLift 续接状态文件。")
    video, audio, low = payload.get("video"), payload.get("audio"), payload.get(LOW_CARRY)
    if video is None or audio is None or low is None:
        raise ValueError(f"状态文件 {path} 缺少音视频或低清数据，可能已损坏，请重新保存。")
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4:
        raise ValueError(f"状态文件 {path} 中的 latent 形状不符合 H3 音视频格式。")
    state = {"samples": NestedTensor([video, audio]), LOW_CARRY: low}
    if payload.get(SEGMENT) is not None:
        state[SEGMENT] = payload[SEGMENT]
    logging.info("[H3Kit SelfLift] 已从文件载入续接状态：%s", path)
    return state


class H3KitSelfLiftLatentSave:
    """SelfLift 续接状态存储：保存 sampled_latent 中 previous_latent 所需的数据；输入留空时从文件读取并输出，可作为工作流启动节点。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "h3kit_selflift/continuation.pt", "multiline": False,
                    "tooltip": "续接状态文件的保存/读取路径。相对路径基于 ComfyUI 的 output 目录；多段串联时请为每一段使用不同文件。"}),
            },
            "optional": {
                "sampled_latent": ("LATENT", {"tooltip": "连接上一个 H3Kit SelfLift K采样器的 sampled_latent：提取 previous_latent 所需的音视频 latent 与低清状态，保存到 file_path 并输出。留空时改为从 file_path 读取并输出——此时可断开或旁路上一段采样器，让本节点作为启动节点，避免缓存不足或重启 ComfyUI 后重新采样上一段。"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("previous_latent",)
    FUNCTION = "save"
    CATEGORY = "H3 Upgrade Kit/采样"
    OUTPUT_NODE = True
    DESCRIPTION = "SelfLift 续接状态存储：接入 sampled_latent 时提取并保存 previous_latent 所需数据到文件；输入留空时从文件读取并输出。本节点是输出节点，无任何连线也能单独运行，可作为工作流启动节点。"

    def save(self, file_path, sampled_latent=None):
        path = _resolve_state_path(file_path)
        if sampled_latent is None:
            return (_load_state(path),)
        state = _extract_state(sampled_latent)
        _save_state(path, state)
        return (state,)


class H3KitSelfLiftLatentLoad:
    """SelfLift 续接状态输出：从存储节点保存的状态文件读取 previous_latent 所需的数据。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "h3kit_selflift/continuation.pt", "multiline": False,
                    "tooltip": "由 H3Kit SelfLift Latent 存储节点保存的状态文件路径。相对路径基于 ComfyUI 的 output 目录；多段串联时选择对应段落的文件。"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("previous_latent",)
    FUNCTION = "load"
    CATEGORY = "H3 Upgrade Kit/采样"
    DESCRIPTION = "SelfLift 续接状态输出：读取存储节点保存的状态文件，输出可接入 SelfLift K采样器 previous_latent 的 latent，实现跨会话持久化续接。"

    def load(self, file_path):
        return (_load_state(_resolve_state_path(file_path)),)
