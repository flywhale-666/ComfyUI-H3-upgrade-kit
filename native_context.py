"""固定22帧原生上下文；用最终画面校准低清空间差异。"""

import logging
import time

import torch
import torch.nn.functional as F

from comfy.ldm.minimax.model import FRAME_PER_TOKEN


CONTEXT_FRAMES = 22
CONTEXT_TOKENS = 7
FRAME_PERIOD = 17
TOKEN_PERIOD = 5


def pixel_frames(tokens):
    return sum(FRAME_PER_TOKEN[i % TOKEN_PERIOD] for i in range(tokens))


def continuation_size(tokens):
    """净新增长度按模型网格对齐；124帧请求对应119新帧＋22帧上下文。"""
    delivery = max(FRAME_PERIOD, pixel_frames(tokens) // FRAME_PERIOD * FRAME_PERIOD)
    total = delivery + CONTEXT_FRAMES
    return delivery, (total - 5) // FRAME_PERIOD * TOKEN_PERIOD + 2


def image_frames(decoded):
    return decoded.reshape(-1, *decoded.shape[-3:])


def comparison_error(actual, expected):
    a = F.interpolate(actual[..., :3].float().movedim(-1, 1), size=(64, 64), mode='area')
    b = F.interpolate(expected[..., :3].float().movedim(-1, 1), size=(64, 64), mode='area').to(a)
    error = (a-b).square().mean((1, 2, 3)).sqrt()
    return float(error.mean()), float(error[-5:].mean())


def decoded_video_tail(video, vae):
    if vae is None:
        raise ValueError('检查22帧上下文需要连接H3视频VAE，或关闭boundary_check。')
    start = max(0, video.shape[2] - CONTEXT_TOKENS - TOKEN_PERIOD)
    return image_frames(vae.decode(video[:, :, start:]))[-CONTEXT_FRAMES:]


def native_video_tail(video, vae=None, previous_frames=None, check=False, align_to_frames=False):
    if video.ndim != 5 or video.shape[2] < CONTEXT_TOKENS or video.shape[2] % TOKEN_PERIOD != 2:
        raise ValueError('22帧续接需要至少7个时间步、长度为5k+2的完整H3视频latent。')
    tail = video[:, :, -CONTEXT_TOKENS:].clone()
    info = {'frames': CONTEXT_FRAMES, 'checked': False, 'corrected': False}
    if not check:
        return tail, info
    if vae is None:
        raise ValueError('检查22帧上下文需要连接H3视频VAE，或关闭boundary_check。')
    started = time.perf_counter()
    if previous_frames is None:
        previous_frames = decoded_video_tail(video, vae)
    if len(previous_frames) < CONTEXT_FRAMES:
        raise ValueError('previous_frames须包含前段完整画面或至少22张连续尾帧。')
    expected = previous_frames[-CONTEXT_FRAMES:, ..., :3]
    size = (video.shape[-2]*16, video.shape[-1]*16)
    if tuple(expected.shape[1:3]) != size:
        expected = F.interpolate(expected.movedim(-1, 1), size=size, mode='area').movedim(1, -1)
    decoded = image_frames(vae.decode(tail))
    if len(decoded) != CONTEXT_FRAMES:
        raise ValueError('原生上下文未解码为22帧，请检查H3视频VAE。')
    error, seam = comparison_error(decoded, expected)
    info.update(checked=True, error=error, seam_error=seam)
    if align_to_frames and error > 1 / 255:
        # 两次同布局编码的差值抵消共同的VAE重建偏差，避免用重编码结果替换原latent。
        # 只处理低清继承区；高清原生状态和下一段尚未生成的区域不参与校准。
        target = vae.encode(expected).to(tail)
        baseline = vae.encode(decoded[..., :3]).to(tail)
        if target.shape != tail.shape or baseline.shape != tail.shape:
            raise ValueError('低清校准需要22帧对应7个时间步的H3视频VAE。')
        delta = target.float() - baseline.float()
        # 保留每帧每通道的空间均值，不把全局曝光/色彩偏移作为结构补偿传下去。
        delta -= delta.mean(dim=(-2, -1), keepdim=True)
        if torch.isfinite(delta).all() and delta.abs().max() > 1e-6:
            candidate = (tail.float() + 0.5 * delta).to(tail)
            candidate_frames = image_frames(vae.decode(candidate))
            after, seam_after = comparison_error(candidate_frames, expected)
            accepted = after < error * 0.98 and seam_after <= seam
            info.update(candidate_error=after, candidate_seam_error=seam_after, corrected=accepted)
            if accepted:
                tail = candidate
            logging.info('[H3Kit low/high sync] tail RMSE %.4f -> %.4f, seam %.4f -> %.4f, accepted=%s',
                         error, after, seam, seam_after, accepted)
    info['check_seconds'] = time.perf_counter() - started
    logging.info('[H3Kit context] 22 frames, RMSE %.4f, seam %.4f, %.2fs; low/high correction=%s',
                 error, seam, info['check_seconds'], info['corrected'])
    return tail, info
