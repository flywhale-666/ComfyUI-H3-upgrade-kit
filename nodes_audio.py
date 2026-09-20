"""将循环收集的音频列表按时间顺序合并。"""

import torch
import torchaudio.functional


class H3KitAudioListToBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "连接 End Loop 的音频列表输出；按列表顺序拼接所有片段。"}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "combine"
    CATEGORY = "H3 Upgrade Kit/音频"
    DESCRIPTION = "将声音列表沿时间维依次拼成一条完整音轨，可直接连接 Video Combine。不同采样率统一到最高采样率；单声道按需要扩展为相同声道数。不会混音或自动裁剪重复片段。"

    def combine(self, audio):
        segments = audio
        if not segments:
            return (None,)
        sample_rate = max(segment["sample_rate"] for segment in segments)
        channels = max(segment["waveform"].shape[1] for segment in segments)
        template = segments[0]["waveform"]
        waveforms = []
        for segment in segments:
            waveform = segment["waveform"].to(template)
            if segment["sample_rate"] != sample_rate:
                waveform = torchaudio.functional.resample(waveform, segment["sample_rate"], sample_rate)
            if waveform.shape[1] == 1 and channels > 1:
                waveform = waveform.expand(-1, channels, -1)
            waveforms.append(waveform)
        return ({**segments[0], "waveform": torch.cat(waveforms, dim=-1), "sample_rate": sample_rate},)
