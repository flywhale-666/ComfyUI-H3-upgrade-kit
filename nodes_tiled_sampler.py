"""通过独立 model patcher 执行 H3 高清分块采样。"""

import copy
import logging

from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from .sampling_tiles import create_tiled_patcher


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
                                 label_on="采样分块：开启", label_off="采样分块：关闭"),
                io.Int.Input("minimum_tiles", default=4, min=2, max=8,
                             tooltip="沿长边重叠分块；根据显存预算增加到最多 8 块。音频和参考图保持完整。"),
            ],
            outputs=[
                io.Latent.Output(display_name="sampled_latent"),
                io.Latent.Output(display_name="denoised_latent"),
            ],
        )

    @classmethod
    def execute(cls, noise_source, sampling_guider, sampling_algorithm,
                sigma_schedule, initial_latent, spatial_tiles=True, minimum_tiles=4):
        if spatial_tiles:
            av_samples = initial_latent["samples"]
            stream_shapes = ([tuple(part.shape) for part in av_samples.unbind()]
                             if av_samples.is_nested else [tuple(av_samples.shape)])
            sampling_guider = copy.copy(sampling_guider)
            sampling_guider.model_patcher = create_tiled_patcher(
                sampling_guider.model_patcher, stream_shapes, min_tiles=minimum_tiles)
            sampling_guider.model_options = sampling_guider.model_patcher.model_options
            logging.info("[H3Kit Sampler] minimum_tiles=%d video=%s audio=%s",
                         minimum_tiles, stream_shapes[0], stream_shapes[1])
        return super().execute(noise_source, sampling_guider, sampling_algorithm,
                               sigma_schedule, initial_latent)
