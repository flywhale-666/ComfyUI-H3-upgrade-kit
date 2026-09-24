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
                "sampled_latent": ("LATENT", {"tooltip": "本段采样输出，用于自动读取22帧续接裁剪信息。"}),
                "decoded_frames": ("IMAGE", {"tooltip": "本段未经裁剪的完整解码画面。"}),
            },
            "optional": {
                "decoded_audio": ("AUDIO",),
                "previous_frames": ("IMAGE", {"tooltip": "可选：前段完整或累积画面。连接后自动拼接；不接则只裁剪本段。"}),
                "previous_audio": ("AUDIO", {"tooltip": "与 previous_frames 对应的音频。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "LATENT")
    RETURN_NAMES = ("images", "audio", "latent")
    FUNCTION = "join"
    CATEGORY = "H3 Upgrade Kit/视频续接"
    DESCRIPTION = "统一裁剪重复前缀并自动对齐音频尾部；接入前段音画时自动拼接。兼容普通高级采样器和 SelfLift，保留 SelfLift 音频过渡。latent 原样输出本段采样结果。"

    def join(self, sampled_latent, decoded_frames, decoded_audio=None, previous_frames=None,
             previous_audio=None):
        frame_rate = 24.0
        total = int(decoded_frames.shape[0])
        info = sampled_latent.get("h3kit_selflift_segment")
        delivery, overlap = sampled_latent.get("h3kit_motion_length", (total, 0))
        soft_audio = False
        if info is not None:
            if total != info["frames"]:
                raise ValueError("解码帧数与采样 latent 不匹配，请连接本段未经裁剪的完整画面。")
            overlap = info["overlap_frames"]
            delivery = info["delivery_frames"]
            frame_rate = info["fps"]
            soft_audio = info["soft_audio"]
            if previous_frames is not None and not overlap:
                raise ValueError("本段没有续接前缀；请将动作续接的条件和target_latent接入SelfLift，或将前一个SelfLift输出接入previous_latent。")
        if overlap not in (0, 22) or overlap + delivery != total:
            raise ValueError("续接信息与完整解码帧数不符；请用固定22帧节点重新生成。")
        if frame_rate <= 0:
            raise ValueError("采样 latent 中记录的帧率必须大于 0。")
        end = overlap + delivery
        if delivery <= 0 or end > total:
            raise ValueError("裁剪帧数超出本段画面长度，请检查本段采样输出与解码画面是否对应。")
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
