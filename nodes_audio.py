"""音频列表拼接及按视频播放帧率调整音频时长。"""

import math

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


class H3KitAudioFrameSync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "连接音频 VAE 解码输出，或已经拼接完成的音频。"}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                                         "tooltip": "填 Video Combine 的目标帧率。原始 24 时：36 缩短到 2/3，12 拉长到 2 倍。"}),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                                         "tooltip": "音频原本对应的视频播放帧率，H3 默认 24；不是音频采样率。"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "sync"
    CATEGORY = "H3 Upgrade Kit/音频"
    DESCRIPTION = "在视频帧数不变、修改播放帧率时同步音频时长。新时长 = 原时长 × 原始帧率 ÷ 目标帧率；变速时保留音高及采样率。输出接 Video Combine 的音频输入，目标帧率填同一数值。插帧后总时长不变的情况无需使用。"

    def sync(self, audio, target_fps, source_fps=24.0):
        if not all(math.isfinite(fps) and fps > 0 for fps in (source_fps, target_fps)):
            raise ValueError("原始帧率和目标帧率必须是大于 0 的有限数值。")
        if audio is None or source_fps == target_fps:
            return (audio,)
        waveform = audio["waveform"]
        if waveform.ndim != 3 or not waveform.is_floating_point():
            raise ValueError("音频需要 [批次, 声道, 采样点] 格式的浮点 waveform。")
        samples = waveform.shape[-1]
        if samples == 0:
            return (audio,)
        target_samples = max(1, round(samples * source_fps / target_fps))
        if target_samples == samples:
            return (audio,)

        # 使用已有 torchaudio 的保音高变速；显式指定长度，避免时频窗口造成尾部时长误差。
        n_fft, hop_length = 2048, 512
        working = waveform.reshape(-1, samples).float()
        window = torch.hann_window(n_fft, device=working.device)
        spectrum = torch.stft(working, n_fft=n_fft, hop_length=hop_length, window=window,
                              center=True, pad_mode="constant", return_complex=True)
        phase_advance = torch.linspace(0, math.pi * hop_length, n_fft // 2 + 1,
                                       device=working.device).unsqueeze(-1)
        stretched = torchaudio.functional.phase_vocoder(spectrum, samples / target_samples, phase_advance)
        output = torch.istft(stretched, n_fft=n_fft, hop_length=hop_length, window=window,
                             center=True, length=target_samples)
        output = output.reshape(*waveform.shape[:-1], target_samples).to(dtype=waveform.dtype)
        return ({**audio, "waveform": output},)
