# 移植自 Comfyui_Minimax_h3_latent_Upscaler；原始 MIT 许可见 licenses/。
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
import math
import folder_paths
import re
from einops import rearrange
from enum import Enum
from typing import TypedDict

import comfy.model_management as mm
import comfy.utils
from comfy_api.latest import io

# ==========================================
# Register model folder
# ==========================================
H3KIT_WEIGHTS_FOLDER = "latent_upscale_models"
if H3KIT_WEIGHTS_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        H3KIT_WEIGHTS_FOLDER,
        os.path.join(folder_paths.models_dir, H3KIT_WEIGHTS_FOLDER)
    )

H3KIT_SPATIAL_FACTOR = 16

# ==========================================
# Minimax H3 latent normalization stats (24 channels)
# ==========================================
H3KIT_LATENT_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
H3KIT_LATENT_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

def make_latent_statistics(device, dtype):
    mean = torch.tensor(H3KIT_LATENT_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(H3KIT_LATENT_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std

# ==========================================
# ROCm / Device Helper Functions
# ==========================================
def is_rocm_backend():
    return getattr(torch.version, "hip", None) is not None

def resolve_compute_backend(backend):
    if backend == "cpu":
        return torch.device("cpu")
    if backend == "rocm":
        if not is_rocm_backend():
            raise RuntimeError("ROCm was selected, but this PyTorch build has no HIP/ROCm support.")
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm was selected, but PyTorch cannot access an AMD GPU.")
        return torch.device("cuda")
    if backend == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported device backend: {backend}")

def compute_backend_label(device):
    if device.type == "cuda" and is_rocm_backend():
        return f"ROCm/HIP {torch.version.hip}"
    if device.type == "cuda":
        return f"CUDA {getattr(torch.version, 'cuda', None) or 'unknown'}"
    return "CPU"

# ==========================================
# 3D network components
# ==========================================
def upscale_group_norm(channels):
    return nn.GroupNorm(32, channels)

class H3KitAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = upscale_group_norm(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.q(h), "b c t h w -> b 1 (t h w) c")
        k = rearrange(self.k(h), "b c t h w -> b 1 (t h w) c")
        v = rearrange(self.v(h), "b c t h w -> b 1 (t h w) c")
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (t h w) c -> b c t h w", t=x.shape[2], h=x.shape[3], w=x.shape[4])
        return x + self.proj_out(h)

class H3KitResidual3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            upscale_group_norm(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = upscale_group_norm(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Identity(),
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h

class H3KitTemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = upscale_group_norm(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h

# ==========================================
# Pure-3D backbone with Temporal Chunking
# ==========================================
class H3KitResizeNetwork(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(H3KitAttention3D(channels))
            self.in_blocks.append(H3KitResidual3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(H3KitTemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(H3KitAttention3D(channels))
            self.out_blocks.append(H3KitResidual3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(H3KitTemporalConv(channels, temporal_kernel))

        self.norm_out = upscale_group_norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        B, C, T, H, W = x.shape

        tk = 0
        for b in self.in_blocks:
            if isinstance(b, H3KitTemporalConv):
                tk = b.dwconv.weight.shape[2]
                break

        overlap = tk
        chunk = 32

        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, size)

        print(f"[H3Kit Upscale] temporal chunking: T={T} chunks={(T + chunk - 1) // chunk} overlap={overlap}")

        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode='replicate')

        out_full = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        start = 0
        while start < T:
            seg_start = start
            seg_end = min(T, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(T, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(T + 2 * overlap, out_end + overlap)

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)

            start += chunk
            del seg, seg_out, valid_out
            if start % (chunk * 4) == 0:
                gc.collect()

        out_full = out_full / weight_full.clamp(min=1e-8)
        return out_full

    def _forward_seg(self, x, scale, size):
        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, H3KitResidual3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, H3KitResidual3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

# ==========================================
# Model loading
# ==========================================

def upscale_weights_directory():
    return folder_paths.get_folder_paths(H3KIT_WEIGHTS_FOLDER)[0]

def list_upscale_weights():
    names = [
        name for name in folder_paths.get_filename_list(H3KIT_WEIGHTS_FOLDER)
        if os.path.splitext(name)[1].lower() in (".pth", ".safetensors")
    ]
    return names if names else [f"(place models in: {upscale_weights_directory()})"]

def read_upscale_weights(path):
    state = comfy.utils.load_torch_file(path, safe_load=True)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    return {key: value.to(torch.float16) if value.dtype == torch.float8_e4m3fn else value
            for key, value in state.items()}

def extract_upscale_weights(sd):
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd

def detect_upscale_architecture(sd):
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "attn": False, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_in_indices, temporal_out_indices = set(), set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m: out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_out_indices.add(int(m.group(1)))

    if in_ids: cfg["in_blocks"] = len(in_ids)
    if out_ids: cfg["out_blocks"] = len(out_ids)

    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    if any('attn' in k for k in sd): cfg["attn"] = True
    cfg["attn"] = False
    return cfg

def load_upscale_network(name, device, precision, memory_required=0):
    backend_lbl = compute_backend_label(device)
    try:
        path = folder_paths.get_full_path_or_raise(H3KIT_WEIGHTS_FOLDER, name)
    except Exception as e:
        raise FileNotFoundError(f"Model file not found: {name}") from e

    raw_sd = read_upscale_weights(path)
    up_sd = extract_upscale_weights(raw_sd)
    cfg = detect_upscale_architecture(up_sd)

    with torch.device("meta"):
        model = H3KitResizeNetwork(
            in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
            channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
            temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
        )
    model.load_state_dict(up_sd, strict=True, assign=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    if memory_required and device.type != "cpu":
        weight_bytes = sum(p.numel() for p in model.parameters()) * dtype.itemsize
        mm.free_memory(weight_bytes + memory_required, device)
    model = model.to(device=device, dtype=dtype).eval()

    print(f"[H3Kit Upscale] Loaded upscale model: {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
          f"Attn: forced off | Temporal: {'on' if cfg['temporal_every'] > 0 else 'off'} "
          f"(every={cfg['temporal_every']}, kernel={cfg['temporal_kernel']}) | "
          f"Backend: {backend_lbl} | Precision: {precision}")
    return model

def upscale_tile_regions(length, target, tile_size, overlap):
    starts = range(0, max(1, length - overlap), tile_size - overlap) if length > tile_size else [0]
    # Map both boundaries so fractional scaling covers the exact target size.
    regions = [(start, min(start + tile_size, length), round(start * target / length),
                round(min(start + tile_size, length) * target / length)) for start in starts]
    return [region for region in regions if region[3] > region[2]]


def upscale_tiled_volume(model, samples, scale, target_size, enable_chunking, tile_size):
    device, dtype = model.conv_in.weight.device, model.conv_in.weight.dtype
    batch, channels, frames, height, width = samples.shape
    _, out_h, out_w = target_size
    overlap = max(1, tile_size // 4)
    rows = upscale_tile_regions(height, out_h, tile_size, overlap)
    cols = upscale_tile_regions(width, out_w, tile_size, overlap)
    mean, std = make_latent_statistics(device, dtype)
    # Full output and blending buffers stay on CPU; only one tile lives on GPU.
    output = torch.zeros((batch, channels, frames, out_h, out_w), dtype=torch.float32, device="cpu")
    weights = torch.zeros((out_h, out_w), dtype=torch.float32, device="cpu")
    print(f"[H3Kit Upscale] spatial tiling: {len(rows)}x{len(cols)} per batch, tile={tile_size}, overlap={overlap}, output=CPU")
    for y0, y1, oy0, oy1 in rows:
        for x0, x1, ox0, ox1 in cols:
            window = torch.ones((oy1 - oy0, ox1 - ox0), dtype=torch.float32, device="cpu")
            for axis, start, end, total, feather in (
                (0, oy0, oy1, out_h, round(overlap * out_h / height)),
                (1, ox0, ox1, out_w, round(overlap * out_w / width)),
            ):
                feather = min(feather, window.shape[axis])
                ramp = torch.arange(1, feather + 1, dtype=torch.float32, device="cpu") / (feather + 1)
                shape = (-1, 1) if axis == 0 else (1, -1)
                if start > 0:
                    window.narrow(axis, 0, feather).mul_(ramp.reshape(shape))
                if end < total:
                    window.narrow(axis, window.shape[axis] - feather, feather).mul_(ramp.flip(0).reshape(shape))
            weights[oy0:oy1, ox0:ox1].add_(window)
            for index in range(batch):
                mm.throw_exception_if_processing_interrupted()
                tile = samples[index:index + 1, :, :, y0:y1, x0:x1].to(device=device, dtype=dtype).contiguous()
                tile = (tile - mean) / std
                result = model(tile, scale=scale, target_size=(frames, oy1 - oy0, ox1 - ox0),
                               enable_chunking=enable_chunking)
                result = (result * std + mean).to(device="cpu", dtype=torch.float32)
                output[index:index + 1, :, :, oy0:oy1, ox0:ox1].add_(result * window)
                del tile, result
    output.div_(weights)
    return output

# ==========================================
# ComfyUI node (new API)
# ==========================================
class H3KitResizeMode(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"
    MEGAPIXELS = "megapixels"

class H3KitResizeSettings(TypedDict):
    resize_settings: H3KitResizeMode
    scale_factor: float
    target_width: int
    target_height: int
    target_megapixels: float

class H3KitLatentUpscale3D(io.ComfyNode):
    """Minimax H3 latent upscaler with Temporal Chunking and pixel-space alignment."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3KitLatentUpscale3D",
            display_name="H3Kit 3D 潜空间放大",
            category="H3 Upgrade Kit/放大与采样",
            search_aliases=["minimax", "h3", "source_latent", "upscale", "3d"],
            inputs=[
                io.Latent.Input("source_latent", tooltip="Input latent (image or video)."),
                io.Combo.Input("upscale_weights", options=list_upscale_weights(), tooltip="Minimax H3 upscale model."),

                io.DynamicCombo.Input(
                    "resize_settings",
                    tooltip="How the target size is computed.",
                    options=[
                        io.DynamicCombo.Option(H3KitResizeMode.SCALE_BY, [
                            io.Float.Input("scale_factor", default=2.0, min=1.0, max=4.0, step=0.05, tooltip="Upscale factor."),
                        ]),
                        io.DynamicCombo.Option(H3KitResizeMode.TARGET_DIMENSIONS, [
                            io.Int.Input("target_width", default=1280, min=64, max=8192, step=8, tooltip="Target pixel width."),
                            io.Int.Input("target_height", default=704, min=64, max=8192, step=8, tooltip="Target pixel height.")
                        ]),
                        io.DynamicCombo.Option(H3KitResizeMode.MEGAPIXELS, [
                            io.Float.Input("target_megapixels", default=1.0, min=0.1, max=16.0, step=0.1, tooltip="Target megapixels.")
                        ])
                    ],
                ),

                io.Int.Input("pixel_alignment", default=32, min=1, max=512, step=1,
                             tooltip="Pixel-space alignment. 32 is strictly recommended."),

                # ---- 推理行为开关组 ----
                io.Boolean.Input("temporal_chunks", default=True,
                                 tooltip="Enable temporal chunking to save VRAM for long videos and fix end-frame flickering."),
                io.Boolean.Input("release_weights", default=True,
                                 tooltip="推理结束后先将权重移回 CPU，释放显存给后续节点；模型不跨执行缓存。"),

                # ---- 硬件 / 精度选项组 ----
                io.Combo.Input("compute_backend", options=["cuda", "rocm", "cpu"], default="cuda"),
                io.Combo.Input("compute_precision", options=["fp32", "fp16", "bf16"], default="fp16"),
                io.Boolean.Input("spatial_tiles", default=False, optional=True,
                                 label_on="高分辨率分块：开启", label_off="高分辨率分块：关闭",
                                 tooltip="按空间分块放大，在 CPU 内存中融合结果，降低显存峰值；显存不足时自动缩小分块。"
                                         "可与时间分块同时开启。速度会降低，画质可能有差异；仅影响本放大节点。"),
            ],
            outputs=[
                io.Latent.Output("upscaled_latent", tooltip="Upscaled latent."),
            ],
        )

    @classmethod
    def execute(cls, source_latent: dict, upscale_weights: str, resize_settings: H3KitResizeSettings,
                pixel_alignment: int, temporal_chunks: bool, release_weights: bool,
                compute_backend: str, compute_precision: str, spatial_tiles: bool = False) -> io.NodeOutput:

        if upscale_weights.startswith('('):
            raise ValueError("Please place model files into the latent_upscale_models directory")

        selected_mode = resize_settings["resize_settings"]
        source_samples = source_latent["samples"]
        orig_dtype = source_samples.dtype
        was_4d = (source_samples.dim() == 4)

        compute_device = resolve_compute_backend(compute_backend)
        compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[compute_precision]

        working_samples = source_samples.to(device="cpu") if spatial_tiles else source_samples.to(device=compute_device, dtype=compute_dtype, copy=True)
        if was_4d:
            working_samples = working_samples.unsqueeze(2)

        b, c, t, h_in, w_in = working_samples.shape
        downsample = H3KIT_SPATIAL_FACTOR

        # 1. Calculate Target Size
        if selected_mode == H3KitResizeMode.SCALE_BY:
            scale_val = resize_settings["scale_factor"]
            w_pixel_target = w_in * downsample * scale_val
            h_pixel_target = h_in * downsample * scale_val
            effective_scale = scale_val
        elif selected_mode == H3KitResizeMode.TARGET_DIMENSIONS:
            w_pixel_target = float(resize_settings["target_width"])
            h_pixel_target = float(resize_settings["target_height"])
            effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
        elif selected_mode == H3KitResizeMode.MEGAPIXELS:
            mp = resize_settings["target_megapixels"]
            target_pixels = mp * 1024 * 1024
            aspect_ratio = w_in / h_in
            h_pixel_target = (target_pixels / aspect_ratio) ** 0.5
            w_pixel_target = h_pixel_target * aspect_ratio
            effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
        else:
            raise ValueError(f"Unsupported mode: {selected_mode}")

        # 2. Pixel-space alignment (双向对齐)
        alignment = max(1, pixel_alignment)
        w_pixel_aligned = round(w_pixel_target / alignment) * alignment
        h_pixel_aligned = round(h_pixel_target / alignment) * alignment

        w_pixel_final = round(w_pixel_aligned / downsample) * downsample
        h_pixel_final = round(h_pixel_aligned / downsample) * downsample

        w_out = max(1, int(w_pixel_final // downsample))
        h_out = max(1, int(h_pixel_final // downsample))

        if effective_scale < 1.0 and (w_out < w_in or h_out < h_in):
            raise ValueError("This model only supports upscaling (effective scale >= 1.0).")

        if w_out == w_in and h_out == h_in:
            return io.NodeOutput(source_latent)

        print(f"[H3Kit Upscale] Latent {w_in}x{h_in} -> {w_out}x{h_out} | "
              f"Pixels {w_out * downsample}x{h_out * downsample} | scale={effective_scale:.3f}")

        # 3. Inference
        model = load_upscale_network(upscale_weights, compute_device, compute_precision, memory_required=2 * 1024**3 if spatial_tiles else 0)
        try:
            if spatial_tiles:
                tile_size = max(8, min(32, math.floor(64 / max(h_out / h_in, w_out / w_in))))
                while True:
                    try:
                        upscaled_samples = upscale_tiled_volume(model, working_samples, effective_scale, (t, h_out, w_out),
                                                    temporal_chunks, tile_size)
                        break
                    except torch.OutOfMemoryError:
                        if compute_device.type != "cuda" or tile_size <= 8:
                            raise
                    tile_size = max(8, tile_size // 2)
                    print(f"[H3Kit Upscale] VRAM insufficient; retrying with tile={tile_size}")
                    gc.collect()
                    mm.soft_empty_cache()
            else:
                norm_mean, norm_std = make_latent_statistics(compute_device, compute_dtype)
                s_norm = (working_samples - norm_mean) / norm_std
                del working_samples
                upscaled_samples = model(s_norm, scale=effective_scale, target_size=(t, h_out, w_out),
                            enable_chunking=temporal_chunks)
                del s_norm
                upscaled_samples = upscaled_samples * norm_std + norm_mean
        finally:
            if release_weights and compute_device.type == "cuda":
                model.to("cpu")
                print("[H3Kit Upscale] Model offloaded to CPU.")

        if was_4d:
            upscaled_samples = upscaled_samples.squeeze(2)

        upscaled_samples = upscaled_samples.to(device="cpu", dtype=orig_dtype, non_blocking=True)

        # 4. VRAM Management
        if compute_device.type == "cuda":
            mm.soft_empty_cache()
            gc.collect()

        output = source_latent.copy()
        output["samples"] = upscaled_samples
        if source_latent.get("noise_mask") is not None:
            mask = comfy.utils.reshape_mask(source_latent["noise_mask"], source_samples.shape)
            output["noise_mask"] = F.interpolate(
                mask.to(device="cpu"), size=upscaled_samples.shape[2:], mode="nearest")
        return io.NodeOutput(output)
