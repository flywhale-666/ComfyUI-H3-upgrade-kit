"""H3 渐进分辨率采样；改编自 TimelineDirector / SelfLift，见 THIRD_PARTY_NOTICES。"""

import math
import logging
from bisect import bisect_left

import torch
import torch.nn.functional as F

import comfy.model_base
import comfy.model_management as mm
import comfy.model_patcher
import comfy.model_sampling
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils
from comfy.nested_tensor import NestedTensor
from comfy.patcher_extension import WrappersMP
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
import latent_preview

from .nodes_latent_upscale import H3KitTemporalConv, load_upscale_network, make_latent_statistics
from .nodes_motion import match_upscaled_motion_context
from .sampling_tiles import create_tiled_patcher


LOW_CARRY = "h3kit_selflift_low"
SEGMENT = "h3kit_selflift_segment"


def av_streams(latent):
    samples = latent["samples"]
    if not samples.is_nested:
        raise ValueError("SelfLift K采需要 H3 音视频复合 latent，请勿只连接视频分支。")
    video, audio = samples.unbind()
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4:
        raise ValueError("SelfLift K采需要 H3 的 24 通道视频和音频 latent。")
    return video, audio


def pixel_frames(tokens):
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(tokens))


def aligned_overlap(frames):
    # 新增上下文按完整周期增加：17 帧 = 5 个 latent 时间步。
    # 旧工作流的 22/39 等值迁移为 17/34，最少一个周期。
    return max(17, int(frames) // 17 * 17)


def frame_boundaries(tokens):
    boundaries = [0]
    for i in range(tokens):
        boundaries.append(boundaries[-1] + FRAME_PER_TOKEN[i % 5])
    return boundaries


def encode_context_prefix(vae, samples, frames, previous_frames=None):
    if previous_frames is None:
        previous_frames = vae.decode(samples)
        previous_frames = previous_frames.reshape(-1, *previous_frames.shape[-3:])
    if previous_frames.shape[0] < frames:
        raise ValueError("前段实际画面不足以编码指定的上下文。")
    if tuple(previous_frames.shape[1:3]) != (samples.shape[-2] * 16, samples.shape[-1] * 16):
        raise ValueError("previous_frames 请连接前段完整尺寸的VAE解码画面。")
    tail = previous_frames[-frames:, ..., :3]
    # 补5帧只为取得完整17n帧的VAE编码，编码后丢弃额外的两个时间步。
    window = torch.cat((tail, tail[-1:].repeat(5, 1, 1, 1)), dim=0)
    encoded = vae.encode(window)
    tokens = frames // 17 * 5
    if encoded.shape[2] != tokens + 2:
        raise ValueError("上下文VAE的时间压缩不符合H3布局，请连接H3视频VAE。")
    return encoded[:, :, :tokens].clone()


def continuation_window(previous, requested, context_vae=None, previous_frames=None):
    video, audio = av_streams(previous)
    bounds = frame_boundaries(video.shape[2])
    info = previous.get(SEGMENT, {})
    end = info.get("overlap_frames", 0) + info.get(
        "delivery_frames", bounds[-1] - info.get("overlap_frames", 0))
    if not requested <= end <= bounds[-1]:
        raise ValueError("上一段有效画面不足以提供所选重叠区，或其长度信息与 latent 不匹配。")
    stop_token = bisect_left(bounds, end)
    if bounds[stop_token] != end:
        raise ValueError("上一段有效终点落在 latent 时间步内部，请重新运行前段 SelfLift K采以更新旧的补尾结果。")
    start_token = stop_token - requested // 17 * 5
    overlap = requested
    low = previous.get(LOW_CARRY)
    if low is None:
        raise ValueError("previous_latent 请直接连接前一个 SelfLift K采的输出；普通采样或拆分重组会丢失低清状态。")
    if low.shape[:3] != video.shape[:3]:
        raise ValueError("上一段低清状态与高清 latent 不匹配，请直接连接采样器输出。")
    ticks = round(overlap * 40 / 24)
    audio_end = round(end * 40 / 24)
    if audio_end > audio.shape[-1] or audio_end < ticks:
        raise ValueError("上一段音频 latent 无法覆盖有效画面的续接区间。")
    high_tail = video[:, :, start_token:stop_token].clone()
    low_tail = low[:, :, start_token:stop_token].clone()
    if context_vae is not None:
        if previous_frames is None:
            previous_frames = context_vae.decode(video[:, :, :stop_token])
            previous_frames = previous_frames.reshape(-1, *previous_frames.shape[-3:])
        high_tail = encode_context_prefix(context_vae, video[:, :, :stop_token], requested, previous_frames)
        # 两个阶段都以最终交付画面为准，避免低清阶段先延续高清修复前的旧背景。
        low_frames = F.interpolate(previous_frames[-requested:, ..., :3].movedim(-1, 1),
                                   size=(low.shape[-2] * 16, low.shape[-1] * 16), mode="area").movedim(1, -1)
        low_tail = encode_context_prefix(context_vae, low[:, :, :stop_token], requested, low_frames)
    elif previous_frames is not None:
        raise ValueError("使用previous_frames进行上下文对齐时，请同时连接context_vae。")
    return (high_tail, low_tail,
            audio[..., audio_end - ticks:audio_end].clone(), overlap)


def shift_conditioning(conditioning, frames):
    if not frames:
        return conditioning
    result = []
    for embedding, data in conditioning:
        data = data.copy()
        if "minimax_keyframes" in data:
            data["minimax_keyframes"] = [
                {**kf, "resolved_frame_index": kf.get("resolved_frame_index", 0) + frames}
                for kf in data["minimax_keyframes"]]
        result.append((embedding, data))
    return result


def add_motion_context(conditioning, reference, overlap_frames):
    if not overlap_frames:
        return conditioning
    # 一整段连续条件保留运动时序，不能拆成独立首帧或重复最后一帧。
    guide = {"resolved_frame_index": 0, "latent": reference.clone(), "h3kit_motion": True}
    result = []
    for embedding, data in conditioning:
        data = data.copy()
        keyframes = []
        identity_refs = []
        for keyframe in data.get("minimax_keyframes") or []:
            if keyframe.get("h3kit_motion"):
                continue
            latent = keyframe.get("latent")
            # 首帧已随新增前缀后移；沿用动作续接的外观参考处理，避免在接缝处重置姿势。
            if (keyframe.get("resolved_frame_index", 0) == overlap_frames
                    and latent is not None and latent.ndim == 5 and latent.shape[2] == 1):
                identity_refs.append({"kind": "image", "latent_h": latent.shape[-2],
                                      "latent_w": latent.shape[-1], "latent": latent})
                if keyframe.get("audio_latent") is not None:
                    keyframe = keyframe.copy()
                    keyframe.pop("latent")
                    keyframes.append(keyframe)
            else:
                keyframes.append(keyframe)
        data["minimax_keyframes"] = keyframes + [guide]
        if identity_refs:
            data["minimax_refs"] = list(data.get("minimax_refs") or []) + identity_refs
        result.append((embedding, data))
    return result


def resize_video(video, size):
    return F.interpolate(video.float(), size=(video.shape[2], *size), mode="trilinear", align_corners=False).to(video)


def resize_conditioning(conditioning, size):
    result = []
    for embedding, data in conditioning:
        data = data.copy()
        if "minimax_keyframes" in data:
            keyframes = []
            for source in data["minimax_keyframes"]:
                keyframe = source.copy()
                latent = keyframe.get("latent")
                if latent is not None and latent.shape[-2:] != size:
                    if latent.ndim == 5:
                        resized = resize_video(latent, size)
                    else:
                        resized = F.interpolate(latent.float(), size=size, mode="bilinear", align_corners=False).to(latent)
                    keyframe["latent"] = resized + latent.mean((-2, -1), keepdim=True) - resized.mean((-2, -1), keepdim=True)
                keyframes.append(keyframe)
            data["minimax_keyframes"] = keyframes
        result.append((embedding, data))
    return result


def prepare_segment(latent, previous, overlap_frames, continue_audio, context_vae=None, previous_frames=None):
    video, audio = av_streams(latent)
    delivery_frames = pixel_frames(video.shape[2])
    raw = latent.get("noise_mask")
    if raw is None:
        masks = [torch.ones((video.shape[0], 1, 1, 1, 1)), torch.ones((audio.shape[0], 1, 1, 1))]
    elif raw.is_nested:
        masks = list(raw.unbind())
    else:
        masks = [raw, torch.ones((audio.shape[0], 1, 1, 1))]
    vm = masks[0]
    if vm.ndim == 3:
        vm = vm[:, None, None]
    elif vm.ndim == 4:
        vm = vm[:, :, None]
    # Concat AV 会用 ones_like(video) 补出24通道遮罩；H3视频token使用第一通道。
    vm = comfy.sampler_helpers.prepare_mask(vm[:, :1].float(),
                                            (video.shape[0], 1, *video.shape[2:]), video.device)
    # SetLatentNoiseMask 可为音频提供任意尺寸的二维遮罩，沿用标准采样器的缩放规则。
    am = comfy.sampler_helpers.prepare_mask(masks[1].float(), audio.shape, audio.device).clone()
    video, audio = video.clone(), audio.clone()
    # 分离后再合并的音轨可能短于视频；与标准AV替换相同，缺少的部分留给模型生成。
    audio_padding = round(delivery_frames * 40 / 24) - audio.shape[-1]
    if audio_padding:
        audio = F.pad(audio, (0, audio_padding))
        am = F.pad(am, (0, audio_padding), value=1)
    overlap = tokens = 0
    low = None
    soft_audio = False
    if previous is not None:
        pv, low, pa, overlap = continuation_window(previous, aligned_overlap(overlap_frames), context_vae, previous_frames)
        tokens = pv.shape[2]
        if pv.shape[:2] != video.shape[:2] or pv.shape[-2:] != video.shape[-2:]:
            raise ValueError("前后段 latent 的批次、通道和目标分辨率必须相同。")
        generated_frames = overlap + delivery_frames
        # 按整周期前移，原初始化内容及遮罩的时间分组不变，无需插值或补视频尾帧。
        video = F.pad(video, (0, 0, 0, 0, tokens, 0))
        vm = F.pad(vm, (0, 0, 0, 0, tokens, 0), value=1)
        audio_head = round(overlap * 40 / 24)
        audio_total = round(generated_frames * 40 / 24)
        audio_padding = audio_total - audio.shape[-1] - audio_head
        audio = F.pad(audio, (audio_head, audio_padding))
        am = F.pad(am, (audio_head, audio_padding), value=1)
        logging.info("[H3Kit SelfLift] new=%d context=%d internal=%d frames (no video tail padding)",
                     delivery_frames, overlap, generated_frames)
        video[:, :, :tokens] = pv.to(video)
        vm[:, :, :tokens] = 0
        if continue_audio:
            ticks = round(overlap * 40 / 24)
            if ticks > min(pa.shape[-1], audio.shape[-1]) or pa.shape[:-1] != audio.shape[:-1]:
                raise ValueError("前后段音频 latent 无法容纳所选重叠区。")
            # 尾音只填新增前缀，原音轨及其锁定遮罩已一起后移。
            soft_audio = True
            audio[..., :ticks] = pa.to(audio)
            am[..., :ticks] = 0
            release = min(8, ticks)
            ramp = 0.5 - 0.5 * torch.cos(torch.pi * torch.arange(1, release + 1, device=am.device) / release)
            am[..., ticks - release:ticks] = ramp
    info = {"overlap_frames": overlap, "soft_audio": soft_audio, "fps": 24,
            "frames": pixel_frames(video.shape[2]), "delivery_frames": delivery_frames}
    return video, audio, vm, am, low, tokens, info


class ContinuationMask:
    """每次采样独立持有遮罩，同时更新原生 inpaint 输入与 H3 token 标签。"""

    def __init__(self, video, audio, video_mask, audio_mask, pool_masks):
        self.shapes = [video.shape, audio.shape]
        self.video_mask = video_mask.expand(video.shape[0], 1, *video.shape[2:]).clone()
        self.audio_mask = audio_mask
        self.pool_masks = pool_masks

    def denoise_mask(self, sigma, denoise_mask, extra_options=None):
        video = self.video_mask.to(denoise_mask)
        audio = self.audio_mask.to(denoise_mask)
        return comfy.utils.pack_latents([video.expand(self.shapes[0]), audio.expand(self.shapes[1])])[0]

    def apply_model(self, executor, *args, **kwargs):
        video = self.video_mask.to(args[0])
        video, audio = self.pool_masks([video, self.audio_mask.to(args[0])])
        kwargs["denoise_mask"] = torch.ceil(video.float() * 256) / 256
        kwargs["audio_denoise_mask"] = torch.ceil(audio.amax(dim=1, keepdim=True).float() * 256) / 256
        return executor(*args, **kwargs)


def stage_model(model, video, audio, vm, am, prefix):
    patched = model.clone()
    if prefix:
        if "denoise_mask_function" in patched.model_options:
            raise ValueError("SelfLift 续接不能与另一个动态 denoise mask 补丁同时使用。")
        state = ContinuationMask(video, audio, vm, am, model.model._pool_masks_to_token_grid)
        patched.set_model_denoise_mask_function(state.denoise_mask)
        patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "h3kit_selflift_context", state.apply_model)
    return patched


def learned_lift(low, size, weights):
    device = mm.get_torch_device()
    precision = "fp32" if device.type == "cpu" else "fp16"
    network = load_upscale_network(weights, torch.device("cpu"), precision)
    patcher = comfy.model_patcher.CoreModelPatcher(network, load_device=device, offload_device=mm.unet_offload_device())
    overlap = next((block.dwconv.kernel_size[0] for block in network.in_blocks if isinstance(block, H3KitTemporalConv)), 0)
    window = low.shape[2] if low.shape[2] <= 32 else min(low.shape[2] + 2 * overlap, 32 + 4 * overlap)
    budget = low.shape[0] * network.conv_in.out_channels * window * math.prod(size) * network.conv_in.weight.element_size() * 8
    mm.load_models_gpu([patcher], memory_required=budget)
    mean, std = make_latent_statistics(device, network.conv_in.weight.dtype)
    x = (low.to(device=device, dtype=mean.dtype) - mean) / std
    # GroupNorm统计包含时间维。按真实连续序列放大，不能在接缝两侧大量复制帧后分别归一化。
    scale = (size[0] / x.shape[-2] + size[1] / x.shape[-1]) / 2
    output = network(x, scale=scale, target_size=(x.shape[2], *size))
    return (output * std + mean).float().to(mm.intermediate_device())


def match_seam(lifted, previous, prefix, video_mask):
    if not prefix:
        return lifted
    lifted = lifted.clone()
    reference = previous[:, :, :prefix]
    # 把真实高清尾部与本次放大的空间差异传到新画面；仅修正待生成区域。
    match_upscaled_motion_context(lifted, reference, video_mask)
    lifted[:, :, :prefix].copy_(reference.to(lifted))
    return lifted


def euler_step(state, x0, sigma, sigma_next):
    return state + (state - x0.to(state)) * ((sigma_next - sigma) / sigma).to(state)


def progressive_sample(model, positive, negative, latent, sigmas, seed, cfg,
                       high_steps, lowres_scale, weights, previous=None, overlap_frames=17,
                       continue_audio=True, lifter=learned_lift, sampler_name="euler",
                       spatial_tiles=False, minimum_tiles=4, context_vae=None, previous_frames=None):
    if sampler_name != "euler":
        raise ValueError("当前 SelfLift 分辨率交接仅支持标准 euler。")
    if not isinstance(model.model, comfy.model_base.MiniMaxH3):
        raise ValueError("此采样器仅支持 MiniMax H3 音视频模型。")
    sampling = model.get_model_object("model_sampling")
    if not isinstance(sampling, comfy.model_sampling.CONST):
        raise ValueError("SelfLift 需要 H3 rectified-flow 采样模型。")
    steps = sigmas.numel() - 1
    split = steps - high_steps
    if not 1 <= split < steps:
        raise ValueError("高清步数必须大于 0 且小于总步数。")
    if sigmas.ndim != 1 or not torch.isfinite(sigmas).all() or (sigmas[:-1] <= 0).any() or (sigmas[1:] > sigmas[:-1]).any() or sigmas[-1] != 0 or sigmas[split] >= 1:
        raise ValueError("SelfLift 需要递减且以 0 结束的 sigma，高清起始 sigma 必须小于 1。")
    video, audio, vm, am, low_carry, prefix, info = prepare_segment(
        latent, previous, overlap_frames, continue_audio, context_vae, previous_frames)
    del previous_frames
    positive = shift_conditioning(positive, info["overlap_frames"])
    negative = shift_conditioning(negative, info["overlap_frames"])
    size = tuple(video.shape[-2:])
    low_size = tuple(max(2, round(s * lowres_scale / 2) * 2) for s in size)
    low = resize_video(video, low_size)
    if low_carry is not None:
        if low_carry.shape[-2:] != low_size:
            raise ValueError("前后段低清尺寸不一致，请保持目标分辨率和 lowres_scale 相同。")
        low[:, :, :prefix] = low_carry[:, :, -prefix:].to(low)
    low_mask = resize_video(vm, low_size)
    low_model = stage_model(model, low, audio, low_mask, am, prefix)
    low_positive = resize_conditioning(positive, low_size)
    low_positive = add_motion_context(low_positive, low[:, :, :prefix], info["overlap_frames"])
    low_negative = resize_conditioning(negative, low_size)
    positive = add_motion_context(positive, video[:, :, :prefix], info["overlap_frames"])
    sampler = comfy.samplers.sampler_object(sampler_name)
    preview = latent_preview.prepare_callback(model, steps)
    captured = {}
    counts = [0, 0]

    def low_callback(step, x0, x, total):
        counts[0] += 1
        if counts[0] == split:
            captured["x0"], captured["state"] = x0, x
        streams = list(x0.unbind())
        display = NestedTensor([resize_video(streams[0], size), streams[1]])
        preview(step, display, display, steps)

    low_latent = NestedTensor([low, audio])
    comfy.samplers.sample(low_model, comfy.sample.prepare_noise(low_latent, seed, latent.get("batch_index")),
                          low_positive, low_negative,
                          cfg, low_model.load_device, sampler, sigmas[:split + 1], low_model.model_options,
                          latent_image=low_latent, denoise_mask=NestedTensor([low_mask, am]),
                          callback=low_callback, seed=seed, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED)
    if counts[0] != split:
        raise RuntimeError("低清采样步数与计划不符，请检查外部采样补丁。")
    x0_video, x0_audio = captured.pop("x0").unbind()
    _, state_audio = captured.pop("state").unbind()
    sigma, next_sigma = sigmas[split - 1], sigmas[split]
    audio_next = euler_step(state_audio, x0_audio, sigma, next_sigma).to(mm.intermediate_device())
    latent_format = model.get_model_object("latent_format")
    native_low = latent_format.process_out(x0_video.float()).to(mm.intermediate_device()).clone()
    del low_model, low_latent, low, x0_video, x0_audio, state_audio, low_positive, low_negative
    lifted = lifter(native_low, size, weights)
    lifted = match_seam(lifted, video, prefix, vm)
    high_x0 = latent_format.process_in(lifted)
    noise = comfy.sample.prepare_noise(high_x0, (seed + 1) % (1 << 64), latent.get("batch_index")).to(high_x0)
    video_next = euler_step(sampling.noise_scaling(sigma, noise, high_x0), high_x0, sigma, next_sigma)
    high_anchor = NestedTensor([video, audio])
    anchors = model.model.process_latent_in(high_anchor).unbind()
    noise_scale = float(getattr(sampling, "noise_scale", 1.0))
    resume = []
    for i, (state, anchor) in enumerate(zip((video_next, audio_next), anchors)):
        clean = anchor.to(state)
        s = next_sigma.to(state)
        residual = (state - (1 - s) * clean) / (s * noise_scale)
        if i == 0:
            residual = torch.where(vm.to(state) == 0, noise.to(state), residual)
        resume.append(residual)
    high_model = stage_model(model, video, audio, vm, am, prefix)
    if spatial_tiles:
        high_model = create_tiled_patcher(high_model, [video.shape, audio.shape], min_tiles=minimum_tiles)
    del lifted, high_x0, video_next, audio_next, noise, anchors

    def high_callback(step, x0, x, total):
        counts[1] += 1
        preview(step + split, x0, x, steps)

    output = comfy.samplers.sample(high_model, NestedTensor(resume), positive, negative, cfg,
                                   high_model.load_device, sampler, sigmas[split:], high_model.model_options,
                                   latent_image=high_anchor, denoise_mask=NestedTensor([vm, am]),
                                   callback=high_callback, seed=seed, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED)
    if counts[1] != high_steps:
        raise RuntimeError("高清采样步数与计划不符，请检查外部采样补丁。")
    result = latent.copy()
    result.pop("noise_mask", None)
    result["samples"] = output.to(device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
    result[LOW_CARRY] = native_low.to(dtype=mm.intermediate_dtype())
    result[SEGMENT] = info
    return result
