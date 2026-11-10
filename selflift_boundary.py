"""续接首12帧使用像素锚点；原采样历史只在解码副本中统一表征。"""
import copy
import hashlib
import math

import torch
import torch.nn.functional as F
import comfy.model_management as mm
from .selflift_geometry import to_high_grid

DECODE_PREFIX = 'h3kit_selflift_decode_prefix_v1'


def video_samples(latent):
    samples = latent['samples']
    return samples.unbind()[0] if samples.is_nested else samples


def video_digest(video):
    raw = video.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def attach_decode_prefix(latent, prefix):
    latent.pop(DECODE_PREFIX, None)
    if prefix is None:
        return
    video = video_samples(latent)
    if ((latent.get('h3kit_selflift_segment') or {}).get('overlap_frames') != 22
            or tuple(prefix.shape) != (video.shape[0], 24, 7, *video.shape[-2:]) or video.shape[2] <= 7):
        raise ValueError('SelfLift解码前缀必须对应可裁剪的22帧历史区。')
    latent[DECODE_PREFIX] = {'version': 1, 'shape': tuple(video.shape),
                             'source_digest': video_digest(video),
                             'samples': prefix.detach().cpu().clone()}


def prepare_video_decode(latent):
    profile = latent.get(DECODE_PREFIX)
    if not isinstance(profile, dict) or profile.get('version') != 1:
        return latent
    video = video_samples(latent)
    if ((latent.get('h3kit_selflift_segment') or {}).get('overlap_frames') != 22
            or tuple(profile.get('shape', ())) != tuple(video.shape)
            or profile.get('source_digest') != video_digest(video)):
        return latent
    prefix = profile['samples']
    if tuple(prefix.shape) != (video.shape[0], 24, 7, *video.shape[-2:]):
        raise ValueError('SelfLift解码前缀尺寸与当前视频不一致。')
    decoded_video = video.clone()
    decoded_video[:, :, :7] = prefix.to(decoded_video)
    result = latent.copy()
    samples = latent['samples']
    result['samples'] = type(samples)([decoded_video, *samples.unbind()[1:]]) if samples.is_nested else decoded_video
    result.pop(DECODE_PREFIX, None)
    return result


def pixel_boundary(low, vae, size, count, *, lowres_scale=None):
    # 22帧历史＋12帧过渡对应前10个token；原生解码需多读2个重叠token。
    tokens = min(low.shape[2], math.ceil(count/5)*5+2)
    frames_needed = math.ceil(count/5)*17
    anchors = []
    for sample in low.split(1):
        frames = vae.decode(sample[:, :, :tokens])
        frames = frames.reshape(-1, *frames.shape[-3:])[:frames_needed, ..., :3]
        dtype = torch.float32 if vae.device.type == 'cpu' else vae.vae_dtype
        pixels = torch.empty((len(frames), size[0]*16, size[1]*16, 3), dtype=dtype, device='cpu')
        for index in range(0, len(frames), 32):
            chunk = frames[index:index+32].movedim(-1, 1).to(device=vae.device, dtype=dtype)
            resized = (to_high_grid(chunk, size, lowres_scale) if lowres_scale is not None else
                       F.interpolate(chunk, size=(size[0]*16, size[1]*16), mode='bicubic',
                                     align_corners=False, antialias=True))
            pixels[index:index+32] = resized.movedim(1, -1).cpu()
        # 这里是完整视频中间的窗口，不是视频终点。通过原生patcher保留
        # 两个完整编码块的尾部token，避免额外编码一整组后续画面。
        encoder = copy.copy(vae)
        encoder.patcher = vae.patcher.clone()
        encoder.patcher.add_object_patch('token_drop', 0)
        try:
            encoded = encoder.encode(pixels)
        finally:
            encoder.patcher.unpatch_model(unpatch_weights=False)
        expected = (1, 24, math.ceil(len(pixels)/17)*5, *size)
        if tuple(encoded.shape) != expected or encoded.shape[2] < count:
            raise ValueError('SelfLift边界重编码没有保持H3时间网格。')
        anchors.append(encoded[:, :, :count].detach().float().cpu().clone())
    return torch.cat(anchors, dim=0)


def anchor_prefix(video, mask, lifted, low, vae, prefix, *, lowres_scale=None):
    count = min(3, video.shape[2]-prefix)
    if prefix != 7 or vae is None or count <= 0:
        return None
    end = prefix+count
    free = mask[:, :, prefix:end] == 1
    if not bool(free.any()):
        return None
    try:
        pixels = pixel_boundary(low, vae, tuple(lifted.shape[-2:]), end, lowres_scale=lowres_scale)
    finally:
        mm.unload_model_and_clones(vae.patcher)
    phase = torch.linspace(0, torch.pi, count, device=lifted.device, dtype=lifted.dtype)
    weight = (.5+.5*torch.cos(phase))[None, None, :, None, None]
    body = lifted[:, :, prefix:end]
    candidate = body+(pixels[:, :, prefix:end].to(body)-body)*weight
    body.copy_(torch.where(free.to(body.device), candidate, body))
    active = free & (weight.to(mask) > 0)
    video[:, :, prefix:end] = torch.where(active, body.to(video), video[:, :, prefix:end])
    mask[:, :, prefix:end] = torch.where(free, (1-weight).to(mask), mask[:, :, prefix:end])
    return pixels[:, :, :prefix].clone()
