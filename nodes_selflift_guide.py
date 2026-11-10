"""给 SelfLift 提供低清 VAE 重编码的单帧引导。"""

import math

from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, _resize

from .selflift_geometry import low_grid_size, to_low_grid


LOW_GUIDE = "h3kit_selflift_low_guide"


class H3KitSelfLiftGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "latent": ("LATENT",),
            "vae": ("VAE",),
            "image": ("IMAGE",),
            "frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05,
                                      "tooltip": "与 SelfLift 的 lowres_scale 保持一致。"}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "encode"
    CATEGORY = "H3 Upgrade Kit/引导"
    DESCRIPTION = "为同一帧分别编码高清和低清图像。正向输出接 SelfLift；多张图可逐个串联，引导节点的 latent 都接同一个目标。"

    def encode(self, positive, latent, vae, image, frame_idx, lowres_scale):
        if len(image) != 1:
            raise ValueError("SelfLift 双尺度引导每个节点只接收一张图；多帧引导请串联多个节点。")
        guided = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=frame_idx,
                                          vae=vae, image=image).result[0]
        video = latent["samples"].unbind()[0]
        size = tuple(video.shape[-2:])
        low_size = low_grid_size(size, lowres_scale)
        pixels = _resize(image, size[1] * 16, size[0] * 16, "center")
        if all(math.isclose(dst / src, lowres_scale, abs_tol=1e-9, rel_tol=0)
               for src, dst in zip(size, low_size)):
            pixels = _resize(pixels, low_size[1] * 16, low_size[0] * 16, "disabled")
        else:
            pixels = to_low_grid(pixels.movedim(-1, 1), low_size, lowres_scale,
                                 high_size=size, pixels=True).movedim(1, -1)
        low = vae.encode(pixels)
        result = []
        for embedding, metadata in guided:
            frames = metadata["minimax_keyframes"]
            guide = {**frames[-1], LOW_GUIDE: (size, lowres_scale, low)}
            result.append((embedding, {**metadata, "minimax_keyframes": [*frames[:-1], guide]}))
        return (result,)


def take_low_guide(frame, size, scale, target):
    profile = frame.pop(LOW_GUIDE, None)
    if profile is not None and target is not None:
        source_size, source_scale, encoded = profile
        if (source_size == target.shape[-2:]
                and math.isclose(source_scale, scale, abs_tol=1e-9, rel_tol=0)
                and encoded.shape == (*target.shape[:-2], *size)):
            return encoded.to(target)
    return None


def strip_low_guides(conditioning):
    result = []
    for embedding, metadata in conditioning:
        frames = metadata.get("minimax_keyframes", [])
        if any(LOW_GUIDE in frame for frame in frames):
            metadata = {**metadata, "minimax_keyframes": [
                {key: value for key, value in frame.items() if key != LOW_GUIDE} for frame in frames
            ]}
        result.append((embedding, metadata))
    return result
