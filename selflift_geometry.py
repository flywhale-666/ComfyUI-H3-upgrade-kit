"""SelfLift 的高低清空间网格，不改变时间轴。"""

import math

import torch
import torch.nn.functional as F


def low_grid_size(size, scale):
    return tuple(max(2, round(side * scale / 2) * 2) for side in size)


def lift_geometry(source, target, scale):
    integer = round(scale)
    if (integer >= 2 and math.isclose(scale, integer, rel_tol=0, abs_tol=1e-9)
            and all(abs(dst - src * integer) <= integer for src, dst in zip(source, target))):
        return tuple(src * integer for src in source)
    return tuple(target)


def fit_lifted_video(video, size):
    if video.shape[-2:] == tuple(size):
        return video
    top, left = (max(0, (actual - wanted) // 2) for actual, wanted in zip(video.shape[-2:], size))
    video = video[..., top:top + size[0], left:left + size[1]]
    dh, dw = size[0] - video.shape[-2], size[1] - video.shape[-1]
    if dh or dw:
        shape = video.shape
        video = F.pad(video.reshape(-1, 1, *shape[-2:]),
                      (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), mode="replicate")
        video = video.reshape(*shape[:-2], *size)
    return video


def resize_spatial(samples, size, scale=None, mode="bilinear"):
    source = samples.shape[-2:]
    if scale is None or all(math.isclose(dst / src, scale, abs_tol=1e-9, rel_tol=0)
                            for src, dst in zip(source, size)):
        options = {} if mode == "area" else {"align_corners": False}
        if mode == "bicubic":
            options["antialias"] = True
        target = (samples.shape[2], *size) if samples.ndim == 5 else size
        method = "trilinear" if samples.ndim == 5 and mode == "bilinear" else mode
        return F.interpolate(samples.float(), size=target, mode=method, **options).to(samples)

    theta = torch.tensor([[size[1] / (source[1] * scale), 0., 0.],
                          [0., size[0] / (source[0] * scale), 0.]],
                         dtype=torch.float32, device=samples.device)
    grid = F.affine_grid(theta[None], (1, 1, *size), align_corners=False)
    if samples.ndim == 5:
        batch, channels, frames = samples.shape[:3]
        flat = samples.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, *source)
    else:
        flat = samples
    result = F.grid_sample(flat.float(), grid.expand(flat.shape[0], -1, -1, -1),
                           mode="bicubic" if mode == "bicubic" else "bilinear",
                           padding_mode="border", align_corners=False)
    if samples.ndim == 5:
        result = result.reshape(batch, frames, channels, *size).permute(0, 2, 1, 3, 4)
    return result.to(samples)


def to_low_grid(samples, low_size, scale, *, high_size=None, pixels=False):
    unit = 16 if pixels else 1
    high_size = tuple(high_size or (side // unit for side in samples.shape[-2:]))
    high_pixels = tuple(side * unit for side in high_size)
    if samples.shape[-2:] != high_pixels:
        samples = resize_spatial(samples, high_pixels, mode="area" if pixels else "bilinear")
    canvas = lift_geometry(low_size, high_size, 1 / scale)
    samples = fit_lifted_video(samples, tuple(side * unit for side in canvas))
    return resize_spatial(samples, tuple(side * unit for side in low_size), scale,
                          mode="area" if pixels else "bilinear")


def to_high_grid(samples, high_size, scale):
    low_size = tuple(side // 16 for side in samples.shape[-2:])
    canvas = lift_geometry(low_size, high_size, 1 / scale)
    resized = resize_spatial(samples, tuple(side * 16 for side in canvas), 1 / scale, mode="bicubic")
    return fit_lifted_video(resized, tuple(side * 16 for side in high_size))
