"""H3 渐进分辨率采样；改编自 TimelineDirector / SelfLift，见 THIRD_PARTY_NOTICES。"""

import math
import logging

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
import latent_preview

from .nodes_latent_upscale import H3KitTemporalConv, load_upscale_network, make_latent_statistics
from .native_context import native_video_tail, decoded_video_tail, pixel_frames, continuation_size, CONTEXT_FRAMES, CONTEXT_TOKENS, TOKEN_PERIOD
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


def continuation_window(previous, context_vae=None, previous_frames=None, boundary_check=True):
    video, audio = av_streams(previous)
    low = previous.get(LOW_CARRY)
    if low is None or low.shape[:2] != video.shape[:2]:
        raise ValueError("previous_latent 需要前一个 SelfLift 的低清状态。外部视频或普通K采请先接动作续接，再将其条件和target_latent接到本节点positive和latent_image，previous_latent留空。")
    if boundary_check and previous_frames is None:
        previous_frames = decoded_video_tail(video, context_vae)
    high_tail, _ = native_video_tail(video, context_vae, previous_frames, boundary_check)
    low_tail, _ = native_video_tail(low, context_vae, previous_frames, boundary_check,
                                    align_to_frames=low.shape[-2:] != video.shape[-2:])
    ticks = round(CONTEXT_FRAMES * 40 / 24)
    audio_end = round(pixel_frames(video.shape[2]) * 40 / 24)
    if audio_end > audio.shape[-1] or audio_end < ticks:
        raise ValueError("上一段音频未覆盖22帧续接区间。")
    return high_tail, low_tail, audio[..., audio_end - ticks:audio_end].clone(), CONTEXT_FRAMES


def shift_conditioning(conditioning, frames, delivery):
    if not frames:
        return conditioning
    result = []
    for embedding, data in conditioning:
        if data.get("h3kit_motion_prefix") == frames:
            result.append((embedding, data))
            continue
        data = data.copy()
        if "minimax_keyframes" in data:
            data["minimax_keyframes"] = [
                {**kf, "resolved_frame_index": min(kf.get("resolved_frame_index", 0), delivery - 1) + frames}
                for kf in data["minimax_keyframes"]]
        result.append((embedding, data))
    return result


def add_motion_context(conditioning, overlap_frames):
    if not overlap_frames:
        return conditioning
    result = []
    for embedding, data in conditioning:
        if data.get("h3kit_motion_prefix") == overlap_frames:
            result.append((embedding, data))
            continue
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
        data["minimax_keyframes"] = keyframes
        if identity_refs:
            data["minimax_refs"] = list(data.get("minimax_refs") or []) + identity_refs
        result.append((embedding, data))
    return result


def resize_video(video, size):
    return F.interpolate(video.float(), size=(video.shape[2], *size), mode="trilinear", align_corners=False).to(video)


def encode_low_prefix(video, vae, size, previous_frames=None):
    if previous_frames is None:
        decoded = vae.decode(video[:, :, :CONTEXT_TOKENS])
        batches = decoded.reshape(video.shape[0], CONTEXT_FRAMES, *decoded.shape[-3:])
    else:
        frames = previous_frames[-CONTEXT_FRAMES:]
        if len(frames) == 0:
            raise ValueError("previous_frames没有图片帧。")
        if len(frames) < CONTEXT_FRAMES:
            frames = torch.cat((frames[:1].expand(CONTEXT_FRAMES-len(frames), -1, -1, -1), frames))
        batches = frames.unsqueeze(0).expand(video.shape[0], -1, -1, -1, -1)
    prefixes = []
    for frames in batches:
        pixels = F.interpolate(frames[..., :3].movedim(-1, 1), size=(size[0]*16, size[1]*16), mode="area").movedim(1, -1)
        prefixes.append(vae.encode(pixels))
    return torch.cat(prefixes, dim=0)


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


def prepare_segment(latent, previous, continue_audio, context_vae=None, previous_frames=None,
                    boundary_check=True):
    video, audio = av_streams(latent)
    delivery_frames = pixel_frames(video.shape[2])
    prepared = latent.get("h3kit_motion_length")
    if prepared is not None and previous is not None:
        raise ValueError("动作续接已准备22帧上下文，请将SelfLift的previous_latent留空；直接串联SelfLift时使用未加前缀的目标latent。")
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
        pv, low, pa, overlap = continuation_window(previous, context_vae, previous_frames, boundary_check)
        tokens = pv.shape[2]
        if pv.shape[:2] != video.shape[:2] or pv.shape[-2:] != video.shape[-2:]:
            raise ValueError("前后段 latent 的批次、通道和目标分辨率必须相同。")
        delivery_frames, target_tokens = continuation_size(video.shape[2])
        padding_tokens = target_tokens - video.shape[2]
        generated_frames = overlap + delivery_frames
        # 按整周期前移，原初始化内容及遮罩的时间分组不变，无需插值或补视频尾帧。
        video = F.pad(video, (0, 0, 0, 0, padding_tokens, 0))
        vm = F.pad(vm, (0, 0, 0, 0, padding_tokens, 0), value=1)
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
    elif prepared is not None:
        delivery_frames, overlap = prepared
        if overlap != CONTEXT_FRAMES or delivery_frames <= 0 or delivery_frames + overlap != pixel_frames(video.shape[2]):
            raise ValueError("动作续接的22帧信息与输入latent长度不符，请连接动作续接输出的target_latent。")
        # 已锁定的图片/原生视频前缀在两个分辨率阶段复用；纯音频续接不锁画面。
        tokens = CONTEXT_TOKENS if torch.all(vm[:, :, :CONTEXT_TOKENS] == 0) else 0
        if tokens and context_vae is None:
            raise ValueError("动作续接 → SelfLift 的视频续接需要连接 context_vae（H3视频VAE）来编码低清上下文；即使已接 previous_frames 或关闭 boundary_check，也必须连接。")
        logging.info("[H3Kit SelfLift] prepared context=%d, new=%d, locked video tokens=%d", overlap, delivery_frames, tokens)
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


def euler_step(state, x0, sigma, sigma_next):
    return state + (state - x0.to(state)) * ((sigma_next - sigma) / sigma).to(state)


def align_external_lift(lifted, reference, prefix):
    """以已知上下文估计放大偏差，在第一个新生成周期内平滑释放。"""
    count = min(TOKEN_PERIOD, lifted.shape[2]-prefix)
    if count < 1:
        return
    delta = (reference[:, :, :prefix].to(lifted)-lifted[:, :, :prefix]).mean((2, 3, 4), keepdim=True)
    weight = 0.5+0.5*torch.cos(torch.linspace(0, torch.pi, count, device=lifted.device, dtype=lifted.dtype))
    lifted[:, :, prefix:prefix+count].add_(delta*weight[None, None, :, None, None])


def progressive_sample(model, positive, negative, latent, sigmas, seed, cfg,
                       high_steps, lowres_scale, weights, previous=None,
                       continue_audio=True, lifter=learned_lift, sampler_name="euler",
                       spatial_tiles=False, minimum_tiles=4, context_vae=None, previous_frames=None,
                       boundary_check=True):
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
        latent, previous, continue_audio, context_vae, previous_frames, boundary_check)
    positive = shift_conditioning(positive, info["overlap_frames"], info["delivery_frames"])
    negative = shift_conditioning(negative, info["overlap_frames"], info["delivery_frames"])
    size = tuple(video.shape[-2:])
    low_size = tuple(max(2, round(s * lowres_scale / 2) * 2) for s in size)
    if low_carry is None and prefix and latent.get("h3kit_motion_length") is not None:
        # 新路径的上下文只从图片编码；latent缩放仅用于后面待生成的部分。
        low_prefix = encode_low_prefix(video, context_vae, low_size, previous_frames).to(video)
        low = torch.cat((low_prefix, resize_video(video[:, :, prefix:], low_size)), dim=2)
        del low_prefix
        logging.info("[H3Kit SelfLift] encoded external 22-frame context at low resolution")
    else:
        low = resize_video(video, low_size)
        if low_carry is not None:
            if low_carry.shape[-2:] != low_size:
                raise ValueError("前后段低清尺寸不一致，请保持目标分辨率和 lowres_scale 相同。")
            low[:, :, :prefix] = low_carry[:, :, -prefix:].to(low)
    del previous_frames
    low_mask = resize_video(vm, low_size)
    low_model = stage_model(model, low, audio, low_mask, am, prefix)
    low_positive = resize_conditioning(positive, low_size)
    low_positive = add_motion_context(low_positive, info["overlap_frames"])
    low_negative = resize_conditioning(negative, low_size)
    positive = add_motion_context(positive, info["overlap_frames"])
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
    if low_carry is None and prefix and latent.get("h3kit_motion_length") is not None:
        align_external_lift(lifted, video, prefix)
    lifted[:, :, :prefix].copy_(video[:, :, :prefix].to(lifted))
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
    result[LOW_CARRY] = native_low[:, :, -CONTEXT_TOKENS:].clone().to(dtype=mm.intermediate_dtype())
    result[SEGMENT] = info
    return result
