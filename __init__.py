from .nodes_latent_upscale import H3KitLatentUpscale3D
from .nodes_tiled_sampler import H3KitTiledSampler
from .nodes_motion import H3KitMotionBridge, H3KitAVTrim
from .nodes_vae_decode import H3KitVAEDecodeTiled
from .nodes_selflift import H3KitSelfLiftSampler, H3KitSelfLiftAVJoin
from .nodes_loop import H3KitStartLoop, H3KitEndLoop, H3KitLoopIteration, H3KitLoopProgress, H3KitLoopResult
from .nodes_audio import H3KitAudioListToBatch
from .nodes_reference import H3KitReferencePromptLength

NODE_CLASS_MAPPINGS = {
    "H3KitLatentUpscale3D": H3KitLatentUpscale3D,
    "H3KitTiledSampler": H3KitTiledSampler,
    "H3KitMotionBridge": H3KitMotionBridge,
    "H3KitAVTrim": H3KitAVTrim,
    "H3KitVAEDecodeTiled": H3KitVAEDecodeTiled,
    "H3KitSelfLiftSampler": H3KitSelfLiftSampler,
    "H3KitSelfLiftAVJoin": H3KitSelfLiftAVJoin,
    "H3KitStartLoop": H3KitStartLoop,
    "H3KitEndLoop": H3KitEndLoop,
    "H3KitLoopIteration": H3KitLoopIteration,
    "H3KitLoopProgress": H3KitLoopProgress,
    "H3KitLoopResult": H3KitLoopResult,
    "H3KitAudioListToBatch": H3KitAudioListToBatch,
    "H3KitReferencePromptLength": H3KitReferencePromptLength,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3KitLatentUpscale3D": "H3Kit 3D 潜空间放大",
    "H3KitTiledSampler": "H3Kit 高清分块采样",
    "H3KitMotionBridge": "H3Kit 动作续接",
    "H3KitAVTrim": "H3Kit 音画同步裁剪",
    "H3KitVAEDecodeTiled": "H3Kit VAE 分块解码",
    "H3KitSelfLiftSampler": "H3Kit SelfLift K采样器",
    "H3KitSelfLiftAVJoin": "H3Kit SelfLift 续接音画拼接",
    "H3KitStartLoop": "H3Kit Start Loop 多路循环开始",
    "H3KitEndLoop": "H3Kit End Loop 多路循环结束",
    "H3KitAudioListToBatch": "H3Kit 声音列表到声音批次",
    "H3KitReferencePromptLength": "H3Kit 提示词与长度替换",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
