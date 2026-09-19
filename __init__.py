from .nodes_latent_upscale import H3KitLatentUpscale3D
from .nodes_tiled_sampler import H3KitTiledSampler
from .nodes_motion import H3KitMotionBridge, H3KitAVTrim
from .nodes_vae_decode import H3KitVAEDecodeTiled
from .nodes_selflift import H3KitSelfLiftSampler, H3KitSelfLiftAVJoin

NODE_CLASS_MAPPINGS = {
    "H3KitLatentUpscale3D": H3KitLatentUpscale3D,
    "H3KitTiledSampler": H3KitTiledSampler,
    "H3KitMotionBridge": H3KitMotionBridge,
    "H3KitAVTrim": H3KitAVTrim,
    "H3KitVAEDecodeTiled": H3KitVAEDecodeTiled,
    "H3KitSelfLiftSampler": H3KitSelfLiftSampler,
    "H3KitSelfLiftAVJoin": H3KitSelfLiftAVJoin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3KitLatentUpscale3D": "H3Kit 3D 潜空间放大",
    "H3KitTiledSampler": "H3Kit 高清分块采样",
    "H3KitMotionBridge": "H3Kit 动作续接",
    "H3KitAVTrim": "H3Kit 音画同步裁剪",
    "H3KitVAEDecodeTiled": "H3Kit VAE 分块解码",
    "H3KitSelfLiftSampler": "H3Kit SelfLift K采样器",
    "H3KitSelfLiftAVJoin": "H3Kit SelfLift 续接音画拼接",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
