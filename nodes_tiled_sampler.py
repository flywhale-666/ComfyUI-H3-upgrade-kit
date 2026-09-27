"""通过独立 model patcher 执行 H3 高清分块采样。"""

import copy
import logging
import math

from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from .sampling_tiles import create_tiled_patcher
from .sampling_global import create_global_patcher


class H3KitTiledSampler(SamplerCustomAdvanced):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3KitTiledSampler",
            display_name="H3Kit 高清分块采样",
            category="H3 Upgrade Kit/放大与采样",
            inputs=[
                io.Noise.Input("noise_source"),
                io.Guider.Input("sampling_guider"),
                io.Sampler.Input("sampling_algorithm"),
                io.Sigmas.Input("sigma_schedule"),
                io.Latent.Input("initial_latent"),
                io.Boolean.Input("spatial_tiles", default=True,
                                 label_on="采样分块：开启", label_off="采样分块：关闭",
                                 tooltip="自动判断：完整噪声起采或空视频 latent 保留全局画面并分批计算；已有画面低噪声续采使用空间分块。关闭时使用原生高级自定义采样。"),
                io.Int.Input("minimum_tiles", default=2, min=2, max=8,
                             tooltip="放大后续采：自动网格最少块数，优先沿长边分割，实际单块长宽比不超过2:1，按形状和显存增加到最多8块；极端比例无法满足时提示并选最接近布局。独立完整采样：内部计算分批数，保留全局画面信息。"),
            ],
            outputs=[
                io.Latent.Output(display_name="sampled_latent"),
                io.Latent.Output(display_name="denoised_latent"),
            ],
        )

    @classmethod
    def execute(cls, noise_source, sampling_guider, sampling_algorithm,
                sigma_schedule, initial_latent, spatial_tiles=True, minimum_tiles=2):
        if spatial_tiles and len(sigma_schedule) > 1:
            av_samples = initial_latent["samples"]
            streams = av_samples.unbind() if av_samples.is_nested else (av_samples,)
            sigma = float(sigma_schedule[0])
            max_sigma = float(sampling_guider.model_patcher.get_model_object("model_sampling").sigma_max)
            # 与原生采样器的完整去噪判定一致；空视频也不能独立裁块生成。
            full_context = (math.isclose(max_sigma, sigma, rel_tol=1e-5)
                            or sigma > max_sigma or not streams[0].any().item())
            sampling_guider = copy.copy(sampling_guider)
            if full_context:
                sampling_guider.model_patcher = create_global_patcher(
                    sampling_guider.model_patcher, chunks=minimum_tiles)
                logging.info("[H3Kit Sampler] auto: full-context sampling chunks=%d", minimum_tiles)
            else:
                stream_shapes = [tuple(part.shape) for part in streams]
                sampling_guider.model_patcher = create_tiled_patcher(
                    sampling_guider.model_patcher, stream_shapes, min_tiles=minimum_tiles)
                logging.info("[H3Kit Sampler] auto: spatial continuation minimum_tiles=%d video=%s audio=%s",
                             minimum_tiles, stream_shapes[0], stream_shapes[1])
            sampling_guider.model_options = sampling_guider.model_patcher.model_options
        return super().execute(noise_source, sampling_guider, sampling_algorithm,
                               sigma_schedule, initial_latent)
