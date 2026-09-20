"""可串联的 SelfLift K采与解码后的音画拼接。"""

import torch
import torch.nn.functional as F
import torchaudio.functional

import comfy.samplers
from comfy_extras.nodes_custom_sampler import BasicScheduler

from .nodes_latent_upscale import list_upscale_weights
from .selflift_sampling import SEGMENT, progressive_sample


class H3KitSelfLiftSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT", {"tooltip": "本段目标高清 H3 音视频 latent；长度代表本段新增帧数。续接时只在头部增加17帧的整数倍上下文，拼接后保留原定新增长度，不补视频尾帧。"}),
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
                    "tooltip": "低清宽高比例。前后段须使用相同设置；整段仍使用latent放大，接context_vae时仅把实际尾部画面缩小后编码为低清上下文。"}),
                "upscale_weights": (list_upscale_weights(),),
                "overlap_frames": ("INT", {"default": 17, "min": 17, "max": 3587, "step": 17,
                    "tooltip": "额外增加的头部上下文：17、34、51……帧，对应前段尾部的5、10、15……个latent时间步；不占用本段新增帧数。"}),
                "continue_audio": ("BOOLEAN", {"default": True,
                    "tooltip": "续接前段尾音，最后 8 个音频 token 平滑释放。已锁定的输入音轨优先。"}),
            },
            "optional": {
                "previous_latent": ("LATENT", {"tooltip": "直接接前一个 H3Kit SelfLift K采的 sampled_latent；同时携带低清和高清状态。"}),
                "sigmas": ("SIGMAS", {"tooltip": "可选外部调度；连接后取代 steps/scheduler/denoise，总步数和降噪强度由外部调度决定。"}),
                "spatial_tiles": ("BOOLEAN", {"default": False,
                    "label_on": "高清采样分块：开启", "label_off": "高清采样分块：关闭",
                    "tooltip": "仅对 SelfLift 高清采样阶段沿长边重叠分块，支持视频/音频遮罩及固定上下文续接。不影响 latent 放大或 VAE 解码。"}),
                "minimum_tiles": ("INT", {"default": 4, "min": 2, "max": 8,
                    "tooltip": "高清采样的最少分块数，按显存预算增加到最多 8 块；小画面受网格限制可能更少。音频取第一块预测，不支持 ControlNet。"}),
                "context_vae": ("VAE", {"tooltip": "连接H3视频VAE，以前段最终高清画面作为统一来源：原尺寸编码高清上下文，缩小画面后编码低清上下文，供锁定前缀和连续运动引导使用。不连接则沿用原生latent尾部；只处理上下文，不做整段像素校正。"}),
                "previous_frames": ("IMAGE", {"tooltip": "可选：前段实际解码画面，配合context_vae避免再次解码高清前段；不连接则自动解码。"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("sampled_latent",)
    FUNCTION = "sample"
    CATEGORY = "H3 Upgrade Kit/采样"
    DESCRIPTION = "低清采样→学习型latent放大→高清续采。前段输出接previous_latent；两个阶段各自锁定连续重叠区并加入整段运动引导，单张首帧转为外观参考；放大后按高清尾部校正新画面。解码后接‘SelfLift续接音画拼接’。"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, scheduler,
               high_resolution_steps, lowres_scale, upscale_weights, overlap_frames,
               continue_audio, previous_latent=None, sigmas=None, denoise=1.0, sampler_name="euler",
               spatial_tiles=False, minimum_tiles=4, context_vae=None, previous_frames=None):
        if sigmas is None:
            if denoise == 0:
                return (latent_image,)
            sigmas = BasicScheduler.execute(model, scheduler, steps, denoise)[0]
        return (progressive_sample(model, positive, negative, latent_image, sigmas, seed, cfg,
                                   high_resolution_steps, lowres_scale, upscale_weights,
                                   previous_latent, overlap_frames, continue_audio, sampler_name=sampler_name,
                                   spatial_tiles=spatial_tiles, minimum_tiles=minimum_tiles,
                                   context_vae=context_vae, previous_frames=previous_frames),)


def fit_audio(waveform, length):
    if waveform.shape[-1] < length:
        return F.pad(waveform, (0, length - waveform.shape[-1]))
    return waveform[..., :length]


class H3KitSelfLiftAVJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampled_latent": ("LATENT", {"tooltip": "本段 SelfLift K采输出，用于读取实际重叠和音频续接信息。"}),
                "decoded_frames": ("IMAGE", {"tooltip": "本段完整解码画面，不要预先裁掉重叠。"}),
            },
            "optional": {
                "decoded_audio": ("AUDIO",),
                "previous_frames": ("IMAGE", {"tooltip": "前一拼接节点的 images，或第一段完整解码画面。"}),
                "previous_audio": ("AUDIO", {"tooltip": "与 previous_frames 同步的累积音频；多段视频可连续串联。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "LATENT")
    RETURN_NAMES = ("images", "audio", "latent")
    FUNCTION = "join"
    CATEGORY = "H3 Upgrade Kit/视频续接"
    DESCRIPTION = "裁掉17帧整数倍的重复前缀，保留本段原定新增帧数；启用 Soft AV 时用后段过渡音频替换前段对应尾音。按24fps对齐音频长度。latent 原样输出本段 sampled_latent，可接下一段 K采的 previous_latent。"

    def join(self, sampled_latent, decoded_frames, decoded_audio=None, previous_frames=None, previous_audio=None):
        info = sampled_latent.get(SEGMENT)
        if info is None:
            raise ValueError("请连接 H3Kit SelfLift K采的原始输出，以读取续接信息。")
        if decoded_frames.shape[0] != info["frames"]:
            raise ValueError("解码帧数与采样 latent 不匹配，请连接本段未经裁剪的完整画面。")
        overlap = info["overlap_frames"]
        joined = previous_frames is not None
        if joined and not overlap:
            raise ValueError("本段未接入前段 latent；拼接前请先连接 K采的 previous_latent。")
        if previous_audio is not None and not joined:
            raise ValueError("previous_audio 需要匹配的 previous_frames。")
        delivery = info["delivery_frames"]
        end = overlap + delivery
        if delivery <= 0 or end > info["frames"]:
            raise ValueError("本段有效长度与采样帧数不匹配。")
        frames = decoded_frames[overlap:end].clone()
        if joined:
            if previous_frames.shape[0] < overlap:
                raise ValueError("前段画面短于重叠区。")
            frames = torch.cat((previous_frames, frames.to(previous_frames)), dim=0)
        if decoded_audio is None and previous_audio is None:
            return frames, None, sampled_latent
        source = decoded_audio if decoded_audio is not None else previous_audio
        sr = source["sample_rate"]
        template = source["waveform"]
        current_length = round(end * sr / 24)
        current = (decoded_audio["waveform"] if decoded_audio is not None
                   else template.new_zeros((*template.shape[:-1], current_length)))
        current = fit_audio(current, current_length)
        cut = round(overlap * sr / 24)
        if joined:
            prior_length = round(previous_frames.shape[0] * sr / 24)
            if previous_audio is None:
                prior = template.new_zeros((*template.shape[:-1], prior_length))
            else:
                prior = previous_audio["waveform"].to(template)
                if previous_audio["sample_rate"] != sr:
                    prior = torchaudio.functional.resample(prior, previous_audio["sample_rate"], sr)
                if prior.shape[:-1] != current.shape[:-1]:
                    raise ValueError("前后段音频的批次和声道数必须一致。")
                prior = fit_audio(prior, prior_length)
            if info["soft_audio"] and decoded_audio is not None and previous_audio is not None:
                waveform = torch.cat((prior[..., :prior_length - cut], current), dim=-1)
            else:
                waveform = torch.cat((prior, current[..., cut:]), dim=-1)
        else:
            waveform = current[..., cut:]
        waveform = fit_audio(waveform, round(frames.shape[0] * sr / 24)).clone()
        return frames, {"waveform": waveform, "sample_rate": sr}, sampled_latent
