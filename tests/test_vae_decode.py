"""真实 H3 拼接/时间逻辑与 ComfyUI VAE 包装器，像素网络使用轻量替身。"""

from functools import partial
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nodes import PACKAGE, NestedTensor
import torch
import comfy.model_patcher
import comfy.sd
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from h3kit_tests import nodes_vae_decode as decode


class SmallH3Decoder(MiniMaxH3VideoVAE):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.vae_ratio = 16
        self.vae_ratio_t = 4
        self.clip_length = 17
        self.token_drop = 3
        self.frame_pre_padding = 3
        self.tokens_chunk_size = 5
        self.token_overlap = 2
        self.frame_overlap = 5
        self.tiling = True
        self.tile_size = 64
        self.tile_overlap_min = 16
        self.decoder = SimpleNamespace(out_channels=3)
        self.register_buffer("latents_mean", torch.zeros(24))
        self.register_buffer("latents_std", torch.ones(24))
        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1, 1))
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1, 1))
        self.call_batches = []
        self.fail_above = None

    def _decode_pixels(self, samples):
        self.call_batches.append(samples.shape[0])
        if self.fail_above is not None and samples.shape[0] > self.fail_above:
            raise torch.OutOfMemoryError("simulated OOM")
        return torch.nn.functional.interpolate(samples[:, :3], scale_factor=(4, 16, 16), mode="nearest")


def make_vae():
    vae = comfy.sd.VAE.__new__(comfy.sd.VAE)
    vae.first_stage_model = SmallH3Decoder()
    vae.latent_dim = 3
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.disable_offload = False
    vae.memory_used_decode = lambda shape, dtype: 2**20
    vae.process_output = lambda output: output
    vae.patcher = comfy.model_patcher.ModelPatcher(
        vae.first_stage_model, load_device=vae.device, offload_device=vae.device)
    return vae


class VAEDecodeTests(unittest.TestCase):
    def test_node_registered(self):
        self.assertIs(PACKAGE.NODE_CLASS_MAPPINGS["H3KitVAEDecodeTiled"], decode.H3KitVAEDecodeTiled)

    def test_real_wrapper_nested_input_and_restored_settings(self):
        vae = make_vae()
        samples = torch.rand(2, 24, 7, 5, 10)
        audio = torch.rand(2, 32, 2, 37)
        expected = vae.decode(samples).reshape(-1, 80, 160, 3)
        before = dict(vae.patcher.object_patches)
        output = decode.H3KitVAEDecodeTiled().decode_video(
            {"samples": NestedTensor([samples, audio])}, vae, 64, 16, 8)[0]
        torch.testing.assert_close(output, expected)
        self.assertEqual(tuple(output.shape), (44, 80, 160, 3))
        self.assertEqual(vae.patcher.object_patches, before)
        self.assertEqual(vae.first_stage_model.tile_size, 64)
        self.assertEqual(vae.first_stage_model.tile_overlap_min, 16)
        self.assertNotIsInstance(vae.first_stage_model._decode_tile_row, partial)
        torch.testing.assert_close(vae.decode(samples).reshape_as(expected), expected)

    def test_actual_tile_parameters_and_cleanup_after_failure(self):
        vae = make_vae()
        samples = torch.rand(1, 24, 7, 5, 10)
        observed = []

        def failing_pixels(tensor):
            observed.append((vae.first_stage_model.tile_size, vae.first_stage_model.tile_overlap_min))
            raise RuntimeError("decode failure")

        with patch.object(vae.first_stage_model, "_decode_pixels", side_effect=failing_pixels):
            with self.assertRaisesRegex(RuntimeError, "decode failure"):
                decode.H3KitVAEDecodeTiled().decode_video({"samples": samples}, vae, 96, 32, 2)
        self.assertEqual(observed, [(256, 32)])
        self.assertEqual((vae.first_stage_model.tile_size, vae.first_stage_model.tile_overlap_min), (64, 16))
        self.assertNotIsInstance(vae.first_stage_model._decode_tile_row, partial)

    def test_native_temporal_frames_and_batching_equivalence(self):
        model = SmallH3Decoder()
        patcher = comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        for steps in (1, 2, 5, 7, 12):
            with self.subTest(steps=steps):
                samples = torch.rand(1, 24, steps, 5, 10)
                expected = model.decode(samples)
                patcher.add_object_patch("tiled_decode", partial(decode.decode_h3_scheduled_tiles, model, schedule_height=256, batch_state={"limit": 2}))
                patcher.patch_model(load_weights=False)
                try:
                    output = model.decode(samples)
                finally:
                    patcher.unpatch_model(unpatch_weights=False)
                self.assertEqual(tuple(output.shape), model.decode_output_shape(samples.shape))
                torch.testing.assert_close(output, expected)

    def test_512_uses_native_model_windows_without_a_switch(self):
        vae = make_vae()
        vae.first_stage_model.tile_size = 256
        vae.first_stage_model.tile_overlap_min = 64
        samples = torch.rand(1, 24, 2, 17, 35)
        expected = vae.decode(samples).reshape(-1, 272, 560, 3)
        with patch.object(vae.first_stage_model, "_decode_pixels", wraps=vae.first_stage_model._decode_pixels) as pixels:
            # 512 调度直接生效，模型内部仍读取原生 256 窗口。
            output = decode.H3KitVAEDecodeTiled().decode_video({"samples": samples}, vae, 512, 64, 2)[0]
        self.assertTrue(pixels.called)
        for call in pixels.call_args_list:
            self.assertEqual(tuple(call.args[0].shape[-2:]), (16, 16))
        torch.testing.assert_close(output, expected)
        self.assertEqual(vae.first_stage_model.tile_size, 256)
        self.assertEqual(vae.first_stage_model.tile_overlap_min, 64)

    def test_schedule_sizes_keep_native_windows_and_image_order(self):
        vae = make_vae()
        vae.first_stage_model.tile_size = 256
        vae.first_stage_model.tile_overlap_min = 64
        samples = torch.rand(2, 24, 2, 17, 35)
        expected = vae.decode(samples).reshape(-1, 272, 560, 3)
        for span in (256, 384, 512, 768, 1024):
            with self.subTest(span=span), patch.object(vae.first_stage_model, "_decode_pixels", wraps=vae.first_stage_model._decode_pixels) as pixels:
                actual = decode.H3KitVAEDecodeTiled().decode_video(
                    {"samples": samples}, vae, span, 64, 8)[0]
            torch.testing.assert_close(actual, expected)
            for call in pixels.call_args_list:
                self.assertEqual(tuple(call.args[0].shape[-2:]), (16, 16))
            self.assertNotIsInstance(vae.first_stage_model.tiled_decode, partial)

    def test_scheduling_can_batch_across_rows(self):
        vae = make_vae()
        samples = torch.rand(1, 24, 1, 48, 8)
        counts = []
        for span in (256, 512, 1024):
            vae.first_stage_model.call_batches.clear()
            decode.H3KitVAEDecodeTiled().decode_video({"samples": samples}, vae, span, 64, 8)
            counts.append(len(vae.first_stage_model.call_batches))
        self.assertEqual(counts, [4, 2, 1])

    def test_oom_halves_batch_without_missing_or_reordering_tiles(self):
        model = SmallH3Decoder()
        model.fail_above = 2
        samples = torch.rand(1, 24, 2, 2, 8)
        starts, sizes = [0, 32, 64, 96], [32] * 4
        state = {"limit": 4}
        with patch.object(decode.model_management, "soft_empty_cache"):
            tiles = (samples[..., start // 16:(start + size) // 16] for start, size in zip(starts, sizes))
            results = list(decode.decode_h3_tile_batches(model, tiles, state))
        self.assertEqual(state["limit"], 2)
        self.assertEqual(model.call_batches, [4, 2, 2])
        for index, pixels in enumerate(results):
            expected = torch.nn.functional.interpolate(samples[:, :3, :, :, index * 2:(index + 1) * 2],
                                                       scale_factor=(4, 16, 16), mode="nearest")
            torch.testing.assert_close(pixels, expected)

    def test_single_tile_oom_propagates(self):
        model = SmallH3Decoder()
        model.fail_above = 0
        with self.assertRaises(torch.OutOfMemoryError):
            list(decode.decode_h3_tile_batches(model, [torch.rand(1, 24, 1, 2, 2)], {"limit": 1}))

    def test_reject_wrong_vae(self):
        with self.assertRaisesRegex(ValueError, "H3 视频 VAE"):
            decode.H3KitVAEDecodeTiled().decode_video({}, SimpleNamespace(first_stage_model=object()))


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
