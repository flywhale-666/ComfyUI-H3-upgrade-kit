"""Spatial H3 sampling tiling, adapted from the local SelfLift implementation."""

from functools import partial
import logging
import math

import torch

import comfy.ldm.common_dit
from comfy.ldm.minimax.model import PackedLayout
import comfy.model_base
import comfy.patcher_extension
import comfy.sampler_helpers


def sampling_axis_regions(length, tile_count):
    patches = (length + 1) // 2
    tile_count = max(1, min(tile_count, max(1, patches // 2)))
    if patches < 4 or tile_count <= 1:
        return [(0, length)]
    overlap = min(4, max(1, patches // 8), patches // tile_count // 2)
    regions = []
    for index in range(tile_count):
        start = max(0, index * patches // tile_count - overlap) * 2
        end = min(patches, (index + 1) * patches // tile_count + overlap) * 2
        end = min(end, length)
        regions.append((start, end))
    regions[-1] = (regions[-1][0], length)
    return regions


def sampling_grid_plans(height, width, minimum_tiles):
    plans = []
    max_rows = min(8, max(1, ((height + 1) // 2) // 2))
    max_cols = min(8, max(1, ((width + 1) // 2) // 2))
    minimum_tiles = min(minimum_tiles, max_rows * max_cols)
    for rows in range(1, max_rows + 1):
        for cols in range(1, max_cols + 1):
            count = rows * cols
            if not minimum_tiles <= count <= 8:
                continue
            ys = sampling_axis_regions(height, rows)
            xs = sampling_axis_regions(width, cols)
            regions = [(y0, y1, x0, x1) for y0, y1 in ys for x0, x1 in xs]
            aspect = max(max(y1 - y0, x1 - x0) / min(y1 - y0, x1 - x0)
                         for y0, y1, x0, x1 in regions)
            ph, pw = (height + 1) // 2, (width + 1) // 2
            core_heights = [min(height, (i + 1) * ph // rows * 2) - i * ph // rows * 2 for i in range(rows)]
            core_widths = [min(width, (i + 1) * pw // cols * 2) - i * pw // cols * 2 for i in range(cols)]
            # 重叠不能把狭长的核心区伪装成合理块形状。
            aspect = max(aspect, max(max(h, w) / min(h, w) for h in core_heights for w in core_widths))
            area = sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in regions)
            plans.append(dict(rows=rows, cols=cols, tiles=count, regions=regions,
                              aspect=aspect, area=area))
    # 优先满足实际块形状约束，再按最少块数、紧凑程度及重叠开销选择。
    return sorted(plans, key=lambda p: (p['aspect'] > 2,
                  p['tiles'] if p['aspect'] <= 2 else p['aspect'], p['aspect'], p['area'],
                  p['rows'] if width >= height else p['cols']))


def build_tile_layout(signature, payload):
    return PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=payload.get("refs"))


def crop_tile_payload(payload, context, video, audio, region):
    y0, y1, x0, x1 = region
    height, width = video.shape[-2:]
    padded_height, padded_width = (height + 1) // 2 * 2, (width + 1) // 2 * 2
    full_layout = payload.get("layout")
    signature = (context.shape[1], video.shape[2], padded_height, padded_width, audio.shape[-1])
    if full_layout is None or full_layout.signature != signature:
        full_layout = build_tile_layout(signature, payload)
    tiled = payload.copy()
    if payload.get("keyframes"):
        keyframes = []
        for keyframe in payload["keyframes"]:
            latent = keyframe.get("latent")
            if latent is None:
                keyframes.append(keyframe.copy())
                continue
            if latent.shape[-2:] != (height, width):
                raise ValueError("H3Kit: tiled H3 keyframes must match the target latent height and width")
            cropped = latent[..., y0:y1, x0:x1]
            keyframes.append({**keyframe, "latent": comfy.ldm.common_dit.pad_to_patch_size(
                cropped, (1, 2, 2)).contiguous()})
        tiled["keyframes"] = keyframes
        tiled["cond_video_latents"] = [keyframe["latent"] for keyframe in keyframes
                                       if keyframe.get("latent") is not None]
        tiled["cond_video_latents"] += [reference["latent"] for reference in payload.get("refs") or []
                                        if reference.get("latent") is not None]
    tile_height, tile_width = y1 - y0, x1 - x0
    layout = build_tile_layout((context.shape[1], video.shape[2], (tile_height + 1) // 2 * 2,
                             (tile_width + 1) // 2 * 2, audio.shape[-1]), tiled)
    for (source_start, source_end, kind), (target_start, target_end, _) in zip(full_layout.segments, layout.segments):
        positions = full_layout.position_ids[source_start:source_end]
        if kind in ("cond", "video"):
            positions = positions.reshape(-1, padded_height // 2, padded_width // 2, 3)
            positions = positions[:, y0 // 2:(y1 + 1) // 2, x0 // 2:(x1 + 1) // 2].reshape(-1, 3)
        layout.position_ids[target_start:target_end].copy_(positions)
    tiled["layout"] = layout
    return tiled


def tile_axis_window(regions, index):
    start, end = regions[index]
    window = torch.ones(end - start, dtype=torch.float32, device="cpu")
    if index > 0:
        overlap = min(end, regions[index - 1][1]) - start
        window[:overlap] *= (torch.arange(overlap, dtype=torch.float32, device="cpu") + 0.5) / overlap
    if index + 1 < len(regions):
        overlap = end - regions[index + 1][0]
        window[-overlap:] *= 1.0 - (torch.arange(overlap, dtype=torch.float32, device="cpu") + 0.5) / overlap
    return window


def run_tiled_diffusion(executor, streams, timestep, context, transformer_options, minimax_payload=None,
                   n_tiles=2, plan=None, **kwargs):
    video, audio = streams
    grid = plan if plan is not None and 'regions' in plan else sampling_grid_plans(
        video.shape[3], video.shape[4], plan.get('min_tiles', n_tiles) if plan is not None else n_tiles)[0]
    regions = grid['regions']
    if len(regions) == 1:
        return executor(streams, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    if kwargs.get("control") is not None:
        raise ValueError("H3Kit: high-resolution H3 tiling does not support ControlNet")
    # Keep the stitched accumulator on CPU so previous tiles do not remain on
    # the accelerator while the next tile is evaluated.
    video_output = torch.zeros(video.shape, dtype=torch.float32, device="cpu")
    audio_output = None
    weights = torch.zeros(video.shape[-2:], dtype=torch.float32, device="cpu")
    ys = sampling_axis_regions(video.shape[3], grid['rows'])
    xs = sampling_axis_regions(video.shape[4], grid['cols'])
    for index, (y0, y1, x0, x1) in enumerate(regions):
        tile = video[..., y0:y1, x0:x1].contiguous()
        payload = crop_tile_payload(minimax_payload or {}, context, video, audio, (y0, y1, x0, x1))
        tile_kwargs = kwargs.copy()
        mask = tile_kwargs.get("denoise_mask")
        if mask is not None:
            tile_kwargs["denoise_mask"] = mask[..., y0:y1, x0:x1]
        predicted_video, predicted_audio = executor(
            [tile, audio], timestep, context, transformer_options.copy(), minimax_payload=payload, **tile_kwargs)
        row, col = divmod(index, grid['cols'])
        window = tile_axis_window(ys, row)[:, None] * tile_axis_window(xs, col)[None, :]
        predicted_video = predicted_video.float().cpu()
        weights[y0:y1, x0:x1].add_(window)
        video_output[..., y0:y1, x0:x1].addcmul_(predicted_video, window)
        if audio_output is None:
            audio_output = predicted_audio.float().clone()
        del tile, payload, tile_kwargs, predicted_video, predicted_audio, window
    video_output.div_(weights)
    return [video_output.to(device=video.device, dtype=video.dtype), audio_output.to(audio.dtype)]


def count_condition_elements(condition, tile_height, tile_width, channels):
    text = condition.get("cross_attn")
    elements = text.shape[-2] * channels * 4 if text is not None else 0
    for keyframe in condition.get("minimax_keyframes") or []:
        latent = keyframe.get("latent")
        if latent is not None:
            shape = latent.shape
            elements += shape[1] * shape[2] * tile_height * tile_width
        audio = keyframe.get("audio_latent")
        if audio is not None:
            elements += math.prod(audio.shape[1:])
    for reference in condition.get("minimax_refs") or []:
        latent = reference.get("latent")
        if latent is not None:
            shape = latent.shape
            elements += shape[1] * shape[2] * ((shape[3] + 1) // 2 * 2) * ((shape[4] + 1) // 2 * 2)
        audio = reference.get("audio_latent")
        if audio is not None:
            elements += math.prod(audio.shape[1:])
    return elements


def estimate_tile_budget(model, noise_shape, conds, latent_shapes, regions):
    video_shape, audio_shape = latent_shapes
    full_elements = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    tile_shape = list(video_shape)
    tile_shape[3] = max(y1 - y0 for y0, y1, x0, x1 in regions)
    tile_shape[4] = max(x1 - x0 for y0, y1, x0, x1 in regions)
    tile_shape[3] = (tile_shape[3] + 1) // 2 * 2
    tile_shape[4] = (tile_shape[4] + 1) // 2 * 2
    tile_elements = math.prod(tile_shape[1:]) + math.prod(audio_shape[1:])
    condition_elements = max((count_condition_elements(condition, tile_shape[3], tile_shape[4], video_shape[1])
                              for group in conds.values() for condition in (group or [])), default=0)
    buffer_bytes = full_elements * 4 * 8
    bytes_per_element = model.model.memory_required((1, 1, 1))
    budget_elements = tile_elements + condition_elements + math.ceil(buffer_bytes / bytes_per_element)
    budget_shape = (noise_shape[0], 1, budget_elements)
    preferred, minimum = comfy.sampler_helpers.estimate_memory(model, budget_shape, conds)
    return budget_shape, tuple(tile_shape), buffer_bytes, preferred, minimum


def prepare_tile_sampling(executor, model, noise_shape, conds, model_options=None,
                            force_full_load=False, force_offload=False, *, latent_shapes, plan=None):
    video_shape, audio_shape = latent_shapes
    full_elements = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    if tuple(noise_shape) != (video_shape[0], 1, full_elements):
        raise ValueError("H3Kit: tiled memory planning received a different latent shape than the sampling input")
    available = available_tile_workspace(model)
    candidates = sampling_grid_plans(video_shape[3], video_shape[4], plan.get('min_tiles', 1) if plan is not None else 2)
    valid = [grid for grid in candidates if grid['aspect'] <= 2]
    if valid:
        candidates = valid
    selected = None
    for grid in candidates:
        budget = estimate_tile_budget(model, noise_shape, conds, latent_shapes, grid['regions'])
        if selected is None or budget[-1] < selected[1][-1]:
            selected = (grid, budget)
        if budget[-1] <= available:
            selected = (grid, budget)
            break
    grid, budget = selected
    if plan is not None:
        plan.update(grid)
    budget_shape, tile_shape, buffer_bytes, preferred, minimum = budget
    logging.info("[H3Kit tiling plan] grid=%dx%d tiles=%d max_aspect=%.3f target=%.2f MiB estimate_fits=%s",
                 grid['rows'], grid['cols'], grid['tiles'], grid['aspect'], available / 2**20, minimum <= available)
    if grid['aspect'] > 2:
        logging.warning("[H3Kit tiling] 8-tile/patch-grid limit prevents a 2:1 tile aspect ratio; using the closest layout")
    if minimum > available:
        logging.warning("[H3Kit tiling] estimated workspace exceeds budget; reduce resolution or video length if out of memory")
    if grid['tiles'] == 1 or force_offload:
        return executor(model, noise_shape, conds, model_options=model_options,
                        force_full_load=force_full_load, force_offload=force_offload)
    logging.info("[H3Kit tiling memory] largest_tile=%s full_audio=%s full_state_buffers=%.2f MiB "
                 "minimum=%.2f MiB preferred=%.2f MiB (ComfyUI estimates; additional models and reserves excluded)",
                 tuple(tile_shape), audio_shape, buffer_bytes * noise_shape[0] / 2**20,
                 minimum / 2**20, preferred / 2**20)
    return executor(model, budget_shape, conds, model_options=model_options,
                    force_full_load=force_full_load, force_offload=force_offload)


def available_tile_workspace(model):
    manager = comfy.model_management
    free = manager.get_free_memory(model.load_device)
    reclaimable = 0
    seen = set()
    for patcher in manager.loaded_models():
        identity = id(patcher.model)
        if patcher.load_device == model.load_device and identity not in seen:
            seen.add(identity)
            reclaimable += patcher.loaded_size()
    pool = min(manager.get_total_memory(model.load_device), free + reclaimable)
    weights = min(model.model_size(), pool * manager.MIN_WEIGHT_MEMORY_RATIO)
    available = max(0, pool - weights - manager.minimum_inference_memory())
    logging.info("[H3Kit tiling capacity] free=%.2f MiB reclaimable_weights=%.2f MiB "
                 "weight_allowance=%.2f MiB workspace=%.2f MiB",
                 free / 2**20, reclaimable / 2**20, weights / 2**20, available / 2**20)
    return available


def create_tiled_patcher(model, latent_shapes, min_tiles=1):
    if not isinstance(model.model, comfy.model_base.MiniMaxH3):
        raise ValueError("H3Kit: high-resolution tiling requires a MiniMax H3 model")
    if len(latent_shapes) != 2 or len(latent_shapes[0]) != 5 or len(latent_shapes[1]) != 4:
        raise ValueError("H3Kit: high-resolution tiling requires H3 video and audio latent streams")
    if not 1 <= min_tiles <= 8:
        raise ValueError("H3Kit: min_tiles must be between 1 and 8")
    plan = {'min_tiles': min_tiles}
    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 "h3kit_spatial_sampling", partial(run_tiled_diffusion, plan=plan))
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                                 "h3kit_spatial_sampling",
                                 partial(prepare_tile_sampling, latent_shapes=tuple(tuple(shape) for shape in latent_shapes), plan=plan))
    return patched
