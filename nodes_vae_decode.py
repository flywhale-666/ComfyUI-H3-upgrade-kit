"""复用 H3 原生时间解码和融合，通过 model patcher 调节空间块与合批。"""

import copy
from functools import partial
from itertools import islice
import logging
import time

import torch

from comfy import model_management
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE


def decode_h3_tile_batches(decoder, tile_iterator, batch_state):
    tile_iterator = iter(tile_iterator)
    pending = []
    while True:
        model_management.throw_exception_if_processing_interrupted()
        if not pending:
            pending = list(islice(tile_iterator, batch_state["limit"]))
        if not pending:
            return
        group_size = min(batch_state["limit"], len(pending))
        try:
            decoded_tiles = decoder._decode_pixels(torch.cat(pending[:group_size], dim=0))
        except torch.OutOfMemoryError:
            if group_size == 1:
                raise
            batch_state["limit"] = max(1, group_size // 2)
        else:
            yield from decoded_tiles.chunk(group_size, dim=0)
            del decoded_tiles
            del pending[:group_size]
            continue
        # 离开异常作用域后再释放临时张量与 allocator 缓存。
        model_management.soft_empty_cache()
        logging.warning("[H3Kit VAE] 显存不足，合批降为 %d 块后重试。", batch_state["limit"])


def decode_h3_scheduled_tiles(decoder, samples, *, schedule_height, batch_state):
    """调度跨度可变，模型始终读取原生 256 窗口；拼接顺序与原生相同。"""
    height, width = (size * decoder.vae_ratio for size in samples.shape[-2:])
    row_starts, row_sizes, row_overlaps = decoder.split_tiles(height)
    col_starts, col_sizes, col_overlaps = decoder.split_tiles(width)

    def predictions():
        first_row = 0
        while first_row < len(row_starts):
            last_row = first_row + 1
            band_end = row_starts[first_row] + max(256, schedule_height)
            while last_row < len(row_starts) and row_starts[last_row] + row_sizes[last_row] <= band_end:
                last_row += 1
            tiles = (
                samples[..., row_starts[row] // decoder.vae_ratio:(row_starts[row] + row_sizes[row]) // decoder.vae_ratio,
                        col // decoder.vae_ratio:(col + size) // decoder.vae_ratio]
                for row in range(first_row, last_row)
                for col, size in zip(col_starts, col_sizes)
            )
            yield from decode_h3_tile_batches(decoder, tiles, batch_state)
            first_row = last_row

    # 沿用 MiniMaxH3VideoVAE.tiled_decode 的原始边缘保存和融合顺序。
    tile_stream = predictions()
    canvas = None
    row_tails = []
    output_y = 0
    for row in range(len(row_starts)):
        next_row_tails = []
        left_tail = None
        output_x = 0
        for column in range(len(col_starts)):
            tile = next(tile_stream)
            if row < len(row_starts) - 1:
                next_row_tails.append(tile[..., -row_overlaps[row]:, :].clone())
            next_left = tile[..., :, -col_overlaps[column]:].clone() if column < len(col_starts) - 1 else None
            if row > 0:
                tile = decoder.blend(row_tails[column], tile, row_overlaps[row - 1], dim=-2)
            if column > 0:
                tile = decoder.blend(left_tail, tile, col_overlaps[column - 1], dim=-1)
            left_tail = next_left
            if row < len(row_starts) - 1:
                tile = tile[..., :-row_overlaps[row], :]
            if column < len(col_starts) - 1:
                tile = tile[..., :, :-col_overlaps[column]]
            if canvas is None:
                canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
            canvas[..., output_y:output_y + tile.shape[-2], output_x:output_x + tile.shape[-1]].copy_(tile)
            tile_height = tile.shape[-2]
            output_x += tile.shape[-1]
            del tile
        row_tails = next_row_tails
        output_y += tile_height
    return canvas


def decode_memory_budget(original_estimate, tile_batch, sample_shape, dtype):
    # 沿用本机 H3 每个 256 小块约 128 MiB 的合批估算。
    temporal_scale = min(sample_shape[2], 7) / 7
    extra_batch = 128 * 2**20 * max(0, tile_batch - 1) * sample_shape[0]
    return int(original_estimate(sample_shape, dtype) + extra_batch * temporal_scale)


class H3KitVAEDecodeTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_latent": ("LATENT", {"tooltip": "H3 视频 latent 或音视频复合 latent；复合输入自动取视频。"}),
            "video_vae": ("VAE", {"tooltip": "连接 MiniMax H3 视频 VAE，音频仍使用音频解码节点。"}),
            "tile_edge": ("INT", {"default": 256, "min": 256, "max": 2048, "step": 16,
                                  "tooltip": "分块调度跨度（像素），设置后直接生效，可设 384、512、768 等。内部保持原生解码窗口。"}),
            "tile_blend": ("INT", {"default": 64, "min": 16, "max": 512, "step": 16,
                                   "tooltip": "分块重叠（像素），建议保持 64；改动重叠会改变融合结果。"}),
            "tiles_per_batch": ("INT", {"default": 8, "min": 1, "max": 16, "step": 1,
                                        "tooltip": "每次解码的 256 小块数量上限。跨行调度可跨多行合批；越大越占显存，OOM 自动减半。"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("decoded_frames",)
    FUNCTION = "decode_video"
    CATEGORY = "H3 Upgrade Kit/视频解码"
    DESCRIPTION = ("固定 H3 原生 256 模型窗口，保留时间解码与融合；支持可调跨度的跨行合批。"
                   "512 等设置仅影响调度，不改变模型的位置编码与注意力窗口。合批不保证提速。")

    def decode_video(self, video_latent, video_vae, tile_edge=256, tile_blend=64, tiles_per_batch=8):
        decoder = video_vae.first_stage_model
        if not isinstance(decoder, MiniMaxH3VideoVAE):
            raise ValueError("H3Kit VAE：请连接 MiniMax H3 视频 VAE，不能使用音频 VAE 或其他视频模型的 VAE。")
        video_samples = video_latent["samples"]
        if video_samples.is_nested:
            video_samples = video_samples.unbind()[0]
        if video_samples.ndim != 5 or video_samples.shape[1] != 24:
            raise ValueError("H3Kit VAE：需要形状为 [B, 24, T, H, W] 的 H3 视频 latent。")

        # 原生融合不支持零重叠，空间坐标必须落在 VAE 的 16 像素网格。
        grid = decoder.vae_ratio
        schedule_height = max(256, (int(tile_edge) + grid - 1) // grid * grid)
        tile_blend = max(grid, min(int(tile_blend) // grid * grid, 256 - grid))
        batch_state = {"limit": max(1, int(tiles_per_batch))}
        scoped_vae = copy.copy(video_vae)
        scoped_vae.patcher = video_vae.patcher.clone()
        scoped_vae.patcher.add_object_patch("tiling", True)
        scoped_vae.patcher.add_object_patch("tile_size", 256)
        scoped_vae.patcher.add_object_patch("tile_overlap_min", tile_blend)
        scoped_vae.patcher.add_object_patch("tiled_decode", partial(
            decode_h3_scheduled_tiles, decoder, schedule_height=schedule_height, batch_state=batch_state))
        scoped_vae.memory_used_decode = partial(
            decode_memory_budget, video_vae.memory_used_decode, batch_state["limit"])

        started = time.perf_counter()
        try:
            decoded_frames = scoped_vae.decode(video_samples)
        finally:
            # 保留已加载权重，只撤销本次对象补丁，包括异常或取消的情况。
            scoped_vae.patcher.unpatch_model(unpatch_weights=False)
        if decoded_frames.ndim == 5:
            decoded_frames = decoded_frames.reshape(-1, *decoded_frames.shape[-3:])
        logging.info("[H3Kit VAE] %d 帧，模型窗口 256 / 重叠 %d，调度跨度 %d，合批上限 %d，耗时 %.2fs。",
                     decoded_frames.shape[0], tile_blend, schedule_height, batch_state["limit"],
                     time.perf_counter() - started)
        return (decoded_frames,)
