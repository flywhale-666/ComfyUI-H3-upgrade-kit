# 移植自ComfyUI-H3-Motion-Context；原始GPL-3.0许可见LICENSE。
"""固定22帧的原生H3音视频续接。"""

import logging

import torch
import torch.nn.functional as F

import comfy.nested_tensor
import comfy.utils

from .motion_layout import ensure_motion_layout
from .native_context import CONTEXT_FRAMES, CONTEXT_TOKENS, continuation_size, native_video_tail, pixel_frames


H3KIT_NATIVE_FPS = 24
H3KIT_AUDIO_RATE = 40
H3KIT_AUDIO_FRAME_SCALE = H3KIT_AUDIO_RATE / H3KIT_NATIVE_FPS


def split_av_streams(latent):
    samples = latent['samples']
    if isinstance(samples, (tuple, list)):
        parts = samples
    elif samples.is_nested:
        parts = samples.unbind()
    else:
        raise ValueError('请连接H3音视频复合latent，不要只接视频分支。')
    if len(parts) != 2:
        raise ValueError('H3续接需要视频、音频两路latent。')
    return tuple(parts)


def video_stream(latent):
    video = split_av_streams(latent)[0]
    if video.ndim != 5:
        raise ValueError('H3视频latent须为[B,C,T,H,W]。')
    return video


def reserve_motion_prefix(target_latent):
    previous = target_latent.get('h3kit_motion_length')
    if previous is not None:
        if previous[1] != CONTEXT_FRAMES:
            raise ValueError('该latent含旧的续接信息，请从未添加前缀的输入重新生成。')
        return target_latent, previous[0]
    video, audio = split_av_streams(target_latent)
    delivery, tokens = continuation_size(video.shape[2])
    audio_head = round(CONTEXT_FRAMES * H3KIT_AUDIO_FRAME_SCALE)
    audio_total = round((delivery + CONTEXT_FRAMES) * H3KIT_AUDIO_FRAME_SCALE)
    video_pad = tokens - video.shape[2]
    audio_pad = audio_total - audio.shape[-1] - audio_head
    out = target_latent.copy()
    out['samples'] = comfy.nested_tensor.NestedTensor((
        F.pad(video, (0,0,0,0,video_pad,0)), F.pad(audio, (audio_head,audio_pad))))
    if target_latent.get('noise_mask') is not None:
        vm, am = target_latent['noise_mask'].unbind()
        vm = comfy.utils.reshape_mask(vm, video.shape)
        am = comfy.utils.reshape_mask(am, audio.shape)
        out['noise_mask'] = comfy.nested_tensor.NestedTensor((
            F.pad(vm, (0,0,0,0,video_pad,0), value=1),
            F.pad(am, (audio_head,audio_pad), value=1)))
    out['h3kit_motion_length'] = (delivery, CONTEXT_FRAMES)
    return out, delivery


def lock_motion_prefix(target_latent, reference):
    video, audio = split_av_streams(target_latent)
    video = video.clone()
    masks = target_latent.get('noise_mask')
    if masks is None:
        vm, am = torch.ones_like(video), torch.ones_like(audio)
    else:
        vm, am = masks.unbind()
        vm = comfy.utils.reshape_mask(vm, video.shape).clone()
    video[:, :, :CONTEXT_TOKENS].copy_(reference.to(video))
    vm[:, :, :CONTEXT_TOKENS] = 0
    out = target_latent.copy()
    out['samples'] = comfy.nested_tensor.NestedTensor((video, audio))
    out['noise_mask'] = comfy.nested_tensor.NestedTensor((vm, am))
    return out


def previous_audio_tail(latent, frames=24):
    video, audio = split_av_streams(latent)
    if audio.ndim != 4:
        raise ValueError('H3音频latent须为[B,C,2,T]。')
    count = pixel_frames(video.shape[2])
    length = latent.get('h3kit_motion_length')
    if length is not None:
        count = sum(length)
    total = min(audio.shape[-1], round(count * H3KIT_AUDIO_FRAME_SCALE))
    overhang = total - count * H3KIT_AUDIO_FRAME_SCALE
    if not -0.5 < overhang < 0.5:
        logging.warning('[H3Kit context] audio grid differs from video; ignoring fractional overhang')
        overhang = 0.0
    ticks = min(total, round(frames * H3KIT_AUDIO_FRAME_SCALE))
    if ticks < 1:
        raise ValueError('前段音频上下文为空。')
    return audio[:1, ..., total-ticks:total].clone(), ticks, overhang


class H3KitMotionBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
                    'positive_conditioning': ('CONDITIONING',),
                    'target_latent': ('LATENT',),
                },
                'optional': {
                    'previous_latent': ('LATENT', {'tooltip':'前段同分辨率H3音视频latent；固定继承末尾22帧。首段不连接。'}),
                    'video_vae': ('VAE', {'tooltip':'边界检查使用；低清阶段另用局部编码差值校准上下文，高清原始latent不变。'}),
                    'previous_frames': ('IMAGE', {'tooltip':'连接上一段最终高清解码画面或连续22张尾帧；低清和高清两处都接同一画面，不要提前缩小。'}),
                    'boundary_check': ('BOOLEAN', {'default':True,'tooltip':'检查22帧边界；图片大于当前latent分辨率时校准低清空间差异，仅接受局部误差改善的候选。关闭则不检查、不校准。'}),
                }}

    RETURN_TYPES = ('CONDITIONING','INT','LATENT','INT')
    RETURN_NAMES = ('positive_conditioning','prefix_frames','target_latent','delivery_frames')
    FUNCTION = 'bridge_motion'
    CATEGORY = 'H3 Upgrade Kit/视频续接'
    DESCRIPTION = '统一22帧原生续接，继承7个latent时间步。请求124帧时续段交付119帧；音频分段须按delivery_frames推进。背景结构连续性仍有已知限制。'

    def bridge_motion(self, positive_conditioning, target_latent, previous_latent=None,
                      video_vae=None, previous_frames=None, boundary_check=True):
        if previous_latent is None:
            return positive_conditioning, 0, target_latent, pixel_frames(video_stream(target_latent).shape[2])
        ensure_motion_layout('22-frame native continuation')
        video = video_stream(target_latent)
        source = video_stream(previous_latent)
        if source.shape[:2] != video.shape[:2] or source.shape[-2:] != video.shape[-2:]:
            raise ValueError('前后段视频latent的批次、通道和分辨率须一致。')
        # 普通两阶段链路在两处都接最终高清画面，以实际尺寸识别低清阶段。
        # 不根据模型连接或已删除的放大标记猜测，也不修改同分辨率的高清上下文。
        low_stage = (previous_frames is not None
                     and previous_frames.shape[1] >= source.shape[-2] * 16
                     and previous_frames.shape[2] >= source.shape[-1] * 16
                     and tuple(previous_frames.shape[1:3]) != (source.shape[-2] * 16, source.shape[-1] * 16))
        reference, _ = native_video_tail(source, video_vae, previous_frames, boundary_check,
                                        align_to_frames=low_stage)
        target_latent, delivery = reserve_motion_prefix(target_latent)
        target_latent = lock_motion_prefix(target_latent, reference)
        total = pixel_frames(video_stream(target_latent).shape[2])
        audio, ticks, overhang = previous_audio_tail(previous_latent)
        end_coord = round(CONTEXT_FRAMES * H3KIT_AUDIO_FRAME_SCALE + overhang)
        audio_guide = {'resolved_frame_index':end_coord/H3KIT_AUDIO_FRAME_SCALE-ticks/H3KIT_AUDIO_FRAME_SCALE,
                       'audio_latent':audio, 'h3kit_motion':True}
        out = []
        for embedding, extra in positive_conditioning:
            data = extra.copy()
            kept, identities = [], []
            for original in data.get('minimax_keyframes') or []:
                if original.get('h3kit_motion'):
                    continue
                keyframe = original.copy()
                position = keyframe.get('resolved_frame_index', 0)
                if position > 0 and not data.get('h3kit_motion_prefix'):
                    position = min(position, delivery-1) + CONTEXT_FRAMES
                    keyframe['resolved_frame_index'] = position
                if position >= total:
                    raise ValueError('关键帧超出当前视频，请连接同一条件节点的正向和latent。')
                if position < CONTEXT_FRAMES:
                    image = keyframe.get('latent')
                    if position == 0 and image is not None and image.ndim == 5 and image.shape[2] == 1:
                        identities.append({'kind':'image','latent_h':image.shape[-2],
                                           'latent_w':image.shape[-1],'latent':image})
                    continue
                kept.append(keyframe)
            data['minimax_keyframes'] = kept + [audio_guide]
            data['h3kit_motion_prefix'] = CONTEXT_FRAMES
            if identities:
                data['minimax_refs'] = list(data.get('minimax_refs') or []) + identities
            out.append([embedding,data])
        logging.info('[H3Kit context] 22-frame native prefix, %d new frames, %d internal frames',delivery,total)
        return out, CONTEXT_FRAMES, target_latent, delivery
