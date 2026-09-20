"""普通采样和 SelfLift 共用的音画裁剪与拼接。"""

import torch
import torch.nn.functional as F
import torchaudio.functional


def fit_audio(waveform, length):
    if waveform.shape[-1] < length:
        return F.pad(waveform, (0, length - waveform.shape[-1]))
    return waveform[..., :length]


class H3KitAVJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoded_frames": ("IMAGE", {"tooltip": "本段未经裁剪的完整解码画面。"}),
            },
            "optional": {
                "decoded_audio": ("AUDIO",),
                "sampled_latent": ("LATENT", {"tooltip": "普通采样可不接，按 prefix_frames / delivery_frames 裁剪；SelfLift 请接原始输出以自动读取续接信息。连接时原样输出 latent，不连接时 latent 输出为空。"}),
                "previous_frames": ("IMAGE", {"tooltip": "可选：前段完整或累积画面。连接后自动拼接；不接则只裁剪本段。"}),
                "previous_audio": ("AUDIO", {"tooltip": "与 previous_frames 对应的音频。"}),
                "prefix_frames": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 17,
                    "tooltip": "普通采样接动作续接的 prefix_frames / trim_frames，或按实际上下文填写 17、34、51……；首段为 0。SelfLift 自动读取。"}),
                "delivery_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "forceInput": True,
                    "tooltip": "可接动作续接的 delivery_frames，去除多余尾帧；0 保留裁掉前缀后的全部画面。SelfLift 自动读取。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "LATENT")
    RETURN_NAMES = ("images", "audio", "latent")
    FUNCTION = "join"
    CATEGORY = "H3 Upgrade Kit/视频续接"
    DESCRIPTION = "统一裁剪重复前缀并自动对齐音频尾部；接入前段音画时自动拼接。兼容普通高级采样器和 SelfLift，保留 SelfLift 音频过渡。latent 原样输出本段采样结果。"

    def join(self, sampled_latent=None, decoded_frames=None, decoded_audio=None, previous_frames=None,
             previous_audio=None, prefix_frames=0, delivery_frames=0):
        frame_rate = 24.0
        total = int(decoded_frames.shape[0])
        info = sampled_latent.get("h3kit_selflift_segment") if sampled_latent is not None else None
        overlap = max(0, int(prefix_frames))
        delivery = int(delivery_frames) or total - overlap
        soft_audio = False
        if info is not None:
            if total != info["frames"]:
                raise ValueError("解码帧数与采样 latent 不匹配，请连接本段未经裁剪的完整画面。")
            overlap = info["overlap_frames"]
            delivery = info["delivery_frames"]
            frame_rate = info["fps"]
            soft_audio = info["soft_audio"]
            if previous_frames is not None and not overlap:
                raise ValueError("本段未接入前段 latent；SelfLift 续接请先连接 K采的 previous_latent。")
        if frame_rate <= 0:
            raise ValueError("采样 latent 中记录的帧率必须大于 0。")
        end = overlap + delivery
        if delivery <= 0 or end > total:
            raise ValueError("裁剪帧数超出本段画面长度，请检查 prefix_frames 和 delivery_frames。")
        joined = previous_frames is not None
        if previous_audio is not None and not joined:
            raise ValueError("previous_audio 需要匹配的 previous_frames。")
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
        current_length = round(end * sr / frame_rate)
        current = (decoded_audio["waveform"] if decoded_audio is not None
                   else template.new_zeros((*template.shape[:-1], current_length)))
        current = fit_audio(current, current_length)
        cut = round(overlap * sr / frame_rate)
        if joined:
            prior_length = round(previous_frames.shape[0] * sr / frame_rate)
            if previous_audio is None:
                prior = template.new_zeros((*template.shape[:-1], prior_length))
            else:
                prior = previous_audio["waveform"].to(template)
                if previous_audio["sample_rate"] != sr:
                    prior = torchaudio.functional.resample(prior, previous_audio["sample_rate"], sr)
                if prior.shape[:-1] != current.shape[:-1]:
                    raise ValueError("前后段音频的批次和声道数必须一致。")
                prior = fit_audio(prior, prior_length)
            if soft_audio and decoded_audio is not None and previous_audio is not None:
                waveform = torch.cat((prior[..., :prior_length - cut], current), dim=-1)
            else:
                waveform = torch.cat((prior, current[..., cut:]), dim=-1)
        else:
            waveform = current[..., cut:]
        waveform = fit_audio(waveform, round(frames.shape[0] * sr / frame_rate)).clone()
        return frames, {"waveform": waveform, "sample_rate": sr}, sampled_latent
