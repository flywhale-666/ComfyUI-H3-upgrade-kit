"""可串联的 SelfLift K采与解码后的音画拼接。"""

import comfy.samplers
from comfy_extras.nodes_custom_sampler import BasicScheduler

from .nodes_latent_upscale import list_upscale_weights
from .selflift_sampling import progressive_sample


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
                    "tooltip": "低清宽高比例。前后段须使用相同设置；高清直接传原始latent，开启边界检查时低清按最终高清画面做局部空间校准。"}),
                "upscale_weights": (list_upscale_weights(),),
                "continue_audio": ("BOOLEAN", {"default": True,
                    "tooltip": "直接串联SelfLift时续接前段尾音，最后8个音频token平滑释放。动作续接输入的音频条件和遮罩由上游决定。"}),
            },
            "optional": {
                "previous_latent": ("LATENT", {"tooltip": "仅用于直接串联前一个SelfLift的sampled_latent，保留低清和高清状态。外部视频或普通K采经动作续接接入latent_image时，此处留空。"}),
                "sigmas": ("SIGMAS", {"tooltip": "可选外部调度；连接后取代 steps/scheduler/denoise，总步数和降噪强度由外部调度决定。"}),
                "spatial_tiles": ("BOOLEAN", {"default": False,
                    "label_on": "高清采样分块：开启", "label_off": "高清采样分块：关闭",
                    "tooltip": "仅对 SelfLift 高清采样阶段沿长边重叠分块，支持视频/音频遮罩及固定上下文续接。不影响 latent 放大或 VAE 解码。"}),
                "minimum_tiles": ("INT", {"default": 4, "min": 2, "max": 8,
                    "tooltip": "高清采样的最少分块数，按显存预算增加到最多 8 块；小画面受网格限制可能更少。音频取第一块预测，不支持 ControlNet。"}),
                "context_vae": ("VAE", {"tooltip": "H3视频VAE。动作续接→SelfLift的视频续接必须连接，否则采样前报错；已接previous_frames或关闭boundary_check也不能省略。用于缩小图片后编码低清上下文。纯音频路径不需要；原来的SelfLift直连仍按boundary_check决定是否需要。"}),
                "previous_frames": ("IMAGE", {"tooltip": "连接与前段latent对应的完整画面或末尾22帧。动作续接输入时，直接缩小这些图片并用context_vae编码低清上下文，省去高清解码；不连接则自动解码。SelfLift直连时仍用于原来的边界检查。"}),
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
               spatial_tiles=False, minimum_tiles=4, context_vae=None, previous_frames=None,
               boundary_check=True):
        if sigmas is None:
            if denoise == 0:
                return (latent_image,)
            sigmas = BasicScheduler.execute(model, scheduler, steps, denoise)[0]
        return (progressive_sample(model, positive, negative, latent_image, sigmas, seed, cfg,
                                   high_resolution_steps, lowres_scale, upscale_weights,
                                   previous_latent, continue_audio, sampler_name=sampler_name,
                                   spatial_tiles=spatial_tiles, minimum_tiles=minimum_tiles,
                                   context_vae=context_vae, previous_frames=previous_frames,
                                   boundary_check=boundary_check),)
