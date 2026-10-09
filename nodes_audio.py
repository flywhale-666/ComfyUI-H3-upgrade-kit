"""音频列表拼接及按视频播放帧率调整音频时长。"""

import math

import torch
import torchaudio.functional


def audio_segment_plan(samples, sample_rate, segment_seconds):
    frames = math.ceil(segment_seconds * 24)
    frames += (5 - frames) % 17

    # 用累计帧数换算采样点，避免逐段取整导致丢失或重复采样点。
    def boundary(index):
        return (index * frames * sample_rate + 12) // 24

    full_segments = samples * 24 // (frames * sample_rate)
    if boundary(full_segments + 1) <= samples:
        full_segments += 1
    count = full_segments + (boundary(full_segments) < samples)
    plan = []
    for index in range(count):
        start = boundary(index)
        if index == full_segments:
            seconds = max(3, math.ceil((samples - start) / sample_rate))
            actual_frames = seconds * 24
            actual_frames += (5 - actual_frames) % 17
            end = samples
        else:
            seconds = segment_seconds
            actual_frames = frames
            end = boundary(index + 1)
        target_samples = ((actual_frames * sample_rate + 12) // 24
                          if index == full_segments else end - start)
        plan.append((start, end, actual_frames, target_samples, seconds))
    return plan


def audio_input_segment_plan(audio, segment_seconds):
    if not isinstance(segment_seconds, int) or segment_seconds < 3:
        raise ValueError("每段目标时长必须是至少 3 秒的整数。")
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim != 3 or not waveform.is_floating_point():
        raise ValueError("音频需要 [批次, 声道, 采样点] 格式的浮点 waveform。")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("音频采样率必须是大于 0 的整数。")
    if waveform.shape[-1] == 0:
        raise ValueError("输入音频为空，无法分段。")
    return audio_segment_plan(waveform.shape[-1], sample_rate, segment_seconds)


def audio_segment_clip(audio, entry):
    start, end, _, target_samples, _ = entry
    segment = audio["waveform"][..., start:end].clone()
    if target_samples > segment.shape[-1]:
        segment = torch.nn.functional.pad(segment, (0, target_samples - segment.shape[-1]))
    return {**audio, "waveform": segment}


class H3KitAudioSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "连接完整音频；保持原采样率、声道和音高。"}),
                "segment_seconds": ("INT", {"default": 9, "min": 3, "max": 150, "step": 1,
                    "tooltip": "每段的目标整数秒数，最低 3 秒。按 24fps 向上对齐 17N+5 帧后，以实际时长连续切分音频。"}),
                "segment_index_from_second": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "从第二段开始的索引：0 是第二段，1 是第三段。首段始终从 first_* 独立输出。"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "FLOAT", "AUDIO", "INT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("first_audio", "first_frame_count", "first_duration",
                    "audio", "frame_count", "duration", "segment_count", "segment_info")
    FUNCTION = "split"
    CATEGORY = "H3 Upgrade Kit/音频"
    DESCRIPTION = "按 H3 的 24fps、17N+5 帧规则连续分段。first_* 始终输出首段；audio、frame_count、duration 从第二段起按 segment_index_from_second 选择，0 对应第二段。segment_count 只统计后续片段。末段向上取整数秒、最低 3 秒，再对齐合法帧数并补静音。"

    def split(self, audio, segment_seconds=9, segment_index_from_second=0):
        plan = audio_input_segment_plan(audio, segment_seconds)
        sample_rate = audio["sample_rate"]
        samples = audio["waveform"].shape[-1]
        remaining_count = len(plan) - 1
        if not isinstance(segment_index_from_second, int) or not 0 <= segment_index_from_second < max(1, remaining_count):
            raise ValueError(f"后续片段序号超出范围：共 {remaining_count} 段，请选择 0 至 {max(0, remaining_count - 1)}。")

        first_audio = audio_segment_clip(audio, plan[0])
        first_frames = plan[0][2]
        selected_audio, frames = None, 0
        if remaining_count:
            selected_audio = audio_segment_clip(audio, plan[segment_index_from_second + 1])
            frames = plan[segment_index_from_second + 1][2]
        selected = str(segment_index_from_second) if remaining_count else "无"
        lines = [f"原音频：{samples / sample_rate:.6f} 秒；共 {len(plan)} 段（首段独立，后续 {remaining_count} 段）；当前后续索引：{selected}",
                 "片段 / 后续索引 | 原音频范围（秒） | 整数目标（秒） | 实际帧数 | 实际时长（秒） | 补静音（秒）"]
        for index, (begin, stop, frame_count, length, seconds) in enumerate(plan):
            label = "首段（独立）" if index == 0 else f"第 {index + 1} 段 / {index - 1}"
            lines.append(f"{label} | {begin / sample_rate:.6f}–{stop / sample_rate:.6f} | {seconds} | "
                         f"{frame_count} | {frame_count / 24:.6f} | {(length - (stop - begin)) / sample_rate:.6f}")
        return (first_audio, first_frames, first_frames / 24,
                selected_audio, frames, frames / 24, remaining_count, "\n".join(lines))


class AudioSegmentOutputTypes(tuple):
    """每新增一段，重复 AUDIO / INT / FLOAT 的端口类型。"""

    def __getitem__(self, index):
        if isinstance(index, int) and index >= 0:
            index %= 3
        return super().__getitem__(index)


class H3KitAudioSegments:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = H3KitAudioSegment.INPUT_TYPES()["required"]
        return {"required": {name: inputs[name] for name in ("audio", "segment_seconds")}}

    RETURN_TYPES = AudioSegmentOutputTypes(("AUDIO", "INT", "FLOAT"))
    RETURN_NAMES = ("1_audio", "1_frame_count", "1_duration")
    FUNCTION = "split"
    CATEGORY = "H3 Upgrade Kit/音频"
    OUTPUT_NODE = True
    DESCRIPTION = "直接输出全部音频片段，不需要循环或索引。首次运行后，按实际段数展开 N_audio、N_frame_count、N_duration 三个端口。按 24fps、17N+5 帧规则连续切分；末段向上取整数秒、最低 3 秒，再对齐帧数并补静音。"

    def split(self, audio, segment_seconds=9):
        plan = audio_input_segment_plan(audio, segment_seconds)
        outputs = []
        for entry in plan:
            frames = entry[2]
            outputs.extend((audio_segment_clip(audio, entry), frames, frames / 24))
        return {"ui": {"segment_count": [len(plan)]}, "result": tuple(outputs)}


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
