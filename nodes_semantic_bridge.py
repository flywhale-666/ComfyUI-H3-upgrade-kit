"""H3 条件语义适配器，支持原版及 BUNNY V1/V2 的六张量权重。"""

import hashlib
import logging
import os
import tempfile
import urllib.request

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.rmsnorm
import comfy.utils
import folder_paths


MODEL_FOLDER = "semantic_bridge"
# 固定作者仓库版本及 SHA-256，只在执行节点且缺少所选权重时下载。
DOWNLOADS = {
    "BUNNY_H3_ActionLogic_Bridge_V2.safetensors": (
        "https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge/resolve/1c46814a7034030920e1d4a6d4e587bf876b7f9c/BUNNY_H3_ActionLogic_Bridge_V2.safetensors",
        "aeb6dc9bcf5fd8e25290a44ac295ed83239e6b3054ebb86cf1227b1d95bd0952", 22045472,
    ),
    "BUNNY_H3_ActionLogic_Bridge_V1.safetensors": (
        "https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge/resolve/1c46814a7034030920e1d4a6d4e587bf876b7f9c/BUNNY_H3_ActionLogic_Bridge_V1.safetensors",
        "983380be6bf790544dbfa9be1bbe42e60ea841c7b6f7c5aac668de9380ab277a", 22045536,
    ),
    "MiniMaxH3_SemanticBridge_v1.safetensors": (
        "https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge/resolve/b9fe58ba6f428d990a59f20f09f719c8fbc67f7d/MiniMaxH3_SemanticBridge_v1.safetensors",
        "ac0dc8ac05f545ebdee12e2fcebe4515b049f9cfd9558eb4887a9bf3fd6d562e", 11023032,
    ),
}
folder_paths.add_model_folder_path(MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER))
folder_paths.folder_names_and_paths[MODEL_FOLDER][1].add(".safetensors")


def adapter_path(adapter, download=False):
    if not adapter.lower().endswith(".safetensors"):
        raise ValueError("请选择语义桥 .safetensors 权重，放入 ComfyUI/models/semantic_bridge 后刷新模型列表。")
    path = folder_paths.get_full_path(MODEL_FOLDER, adapter)
    if path is not None:
        return path
    if adapter in DOWNLOADS:
        return download_adapter(adapter) if download else None
    raise FileNotFoundError(f"找不到语义桥模型 {adapter}，请放入 ComfyUI/models/semantic_bridge。")


def download_adapter(adapter):
    url, expected_hash, expected_size = DOWNLOADS[adapter]
    directory = os.path.join(folder_paths.models_dir, MODEL_FOLDER)
    target = os.path.join(directory, adapter)
    temporary = None
    try:
        mm.throw_exception_if_processing_interrupted()
        os.makedirs(directory, exist_ok=True)
        logging.info("[H3Kit 语义桥] 正在下载 %s（%.1f MB）", adapter, expected_size / 1_000_000)
        progress = comfy.utils.ProgressBar(expected_size)
        progress.update_absolute(0)
        digest = hashlib.sha256()
        received = 0
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{adapter}.", suffix=".part", delete=False) as output:
            temporary = output.name
            with urllib.request.urlopen(url, timeout=20) as response:
                while True:
                    mm.throw_exception_if_processing_interrupted()
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > expected_size:
                        raise ValueError("下载文件大小与作者发布的模型不符")
                    output.write(chunk)
                    digest.update(chunk)
                    progress.update_absolute(received)
        mm.throw_exception_if_processing_interrupted()
        if received != expected_size or digest.hexdigest() != expected_hash:
            raise ValueError("下载不完整或 SHA-256 校验失败")
        if not os.path.exists(target):
            os.replace(temporary, target)
        folder_paths.filename_list_cache.pop(MODEL_FOLDER, None)
        logging.info("[H3Kit 语义桥] 模型已就绪：%s", target)
        return target
    except (OSError, ValueError) as error:
        raise RuntimeError(f"语义桥自动下载失败：{error}。检查网络后重新运行，或手动下载到 {target}。") from error
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.remove(temporary)


def load_weights(path, device):
    weights = comfy.utils.load_torch_file(path, safe_load=True)
    keys = {f"fc{i}.{kind}" for i in (1, 2, 3) for kind in ("weight", "bias")}
    if set(weights) != keys or any(not isinstance(weights[k], torch.Tensor) for k in keys):
        raise ValueError("语义桥权重必须包含 fc1、fc2、fc3 的 weight 和 bias；不能选择 LoRA 或 H3 主模型。")
    first = weights["fc1.weight"]
    if first.ndim != 2:
        raise ValueError("语义桥 fc1.weight 必须是二维矩阵。")
    hidden = first.shape[0]
    shapes = {
        "fc1.weight": (hidden, 5120), "fc1.bias": (hidden,),
        "fc2.weight": (hidden, hidden), "fc2.bias": (hidden,),
        "fc3.weight": (5120, hidden), "fc3.bias": (5120,),
    }
    if hidden == 0 or any(tuple(weights[k].shape) != shape for k, shape in shapes.items()):
        raise ValueError("语义桥权重维度不匹配，需要 5120 → hidden → hidden → 5120 的适配器。")
    return {k: value.to(device=device, dtype=torch.float32) for k, value in weights.items()}


class H3KitSemanticBridge:
    @classmethod
    def INPUT_TYPES(cls):
        models = [name for name in folder_paths.get_filename_list(MODEL_FOLDER) if name.lower().endswith(".safetensors")]
        models = list(DOWNLOADS) + sorted(set(models) - DOWNLOADS.keys(), key=str.lower)
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "连接 H3 编码完成后的正向条件，输出接原来的采样器或 Guider。"}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "tooltip": "关闭时原样传递条件，不加载模型；没有下载模型也可以关闭运行。"}),
                "adapter": (models, {"tooltip": "默认 BUNNY V2。开启后点击运行，缺少所选的 V2/V1/原版模型时自动下载到 models/semantic_bridge；本地已有模型直接使用。"}),
                "alpha": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "语义混合强度，建议从 0.10 开始做同种子对比；0 等同关闭。"}),
                "magnitude_match": (["per_token", "global", "none"], {"default": "per_token", "tooltip": "建议 per_token：逐 token 匹配原始幅度；global：整体匹配；none：不匹配。"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "H3 Upgrade Kit/条件"
    DESCRIPTION = "在 H3 条件编码后应用语义桥。开启后运行时自动下载缺失的内置模型；关闭或强度为 0 时不下载、不加载，原样传递。参考音频、唱歌及 Ref2VA 效果需要单独对比，不要串联叠加多个语义桥。"

    @classmethod
    def VALIDATE_INPUTS(cls, adapter, enabled, alpha):
        if not enabled or alpha == 0:
            return True
        if not 0 <= alpha <= 1:
            return "alpha 必须在 0 到 1 之间。"
        try:
            adapter_path(adapter)
        except (ValueError, FileNotFoundError) as error:
            return str(error)
        return True

    @classmethod
    def IS_CHANGED(cls, adapter, enabled, alpha):
        if not enabled or alpha == 0:
            return "disabled"
        path = adapter_path(adapter)
        if path is None:
            return ("not_downloaded", adapter)
        stat = os.stat(path)
        return (path, stat.st_mtime_ns, stat.st_size)

    def apply(self, conditioning, enabled, adapter, alpha, magnitude_match):
        if not enabled or alpha == 0:
            return (conditioning,)
        if not 0 <= alpha <= 1:
            raise ValueError("alpha 必须在 0 到 1 之间。")
        if magnitude_match not in ("per_token", "global", "none"):
            raise ValueError("未知的幅度匹配模式。")
        for native, _metadata in conditioning:
            if native.ndim != 3 or native.shape[-1] != 5120 or not native.is_floating_point():
                raise ValueError("语义桥需要 H3 的 [B, T, 5120] 浮点条件，请接在 H3 条件编码节点之后。")

        device = mm.get_torch_device()
        weights = load_weights(adapter_path(adapter, download=True), device)
        result = []
        for native, metadata in conditioning:
            h = native.to(device=device, dtype=torch.float32)
            projected = torch.empty_like(h)
            for start in range(0, h.shape[1], 256):
                mm.throw_exception_if_processing_interrupted()
                tokens = h[:, start:start + 256]
                x = comfy.rmsnorm.rms_norm(tokens, eps=1e-6)
                for layer in (1, 2):
                    x = F.silu(F.linear(x, weights[f"fc{layer}.weight"], weights[f"fc{layer}.bias"]))
                projected[:, start:start + 256] = F.linear(x, weights["fc3.weight"], weights["fc3.bias"])
            if magnitude_match != "none":
                dims = (-1,) if magnitude_match == "per_token" else (0, 1, 2)
                target_rms = (h.square().mean(dim=dims, keepdim=True) + 1e-8).sqrt()
                source_rms = (projected.square().mean(dim=dims, keepdim=True) + 1e-8).sqrt()
                projected.mul_(target_rms / source_rms)
            mixed = h + alpha * (projected - h)
            result.append([mixed.to(device=native.device, dtype=native.dtype), metadata.copy()])
        return (result,)
