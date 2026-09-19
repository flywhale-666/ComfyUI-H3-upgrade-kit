"""使用真实 ComfyUI/PyTorch 的 CPU 回归检查，不加载大型生成模型。"""

import asyncio
import copy
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
import comfy.options

comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]

import torch
import comfy.model_base
import comfy.nested_tensor
import nodes as comfy_nodes
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
from safetensors.torch import save_file

SPEC = importlib.util.spec_from_file_location(
    "h3kit_tests", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
from h3kit_tests import nodes_latent_upscale as upscale
from h3kit_tests import nodes_motion as motion
from h3kit_tests import nodes_tiled_sampler as sampling
from h3kit_tests import sampling_tiles as tiles
from h3kit_tests.motion_layout import ensure_motion_layout

NestedTensor = comfy.nested_tensor.NestedTensor


def av_latent(video_steps=7, audio_steps=37):
    return {"samples": NestedTensor([
        torch.zeros(1, 24, video_steps, 2, 2),
        torch.arange(audio_steps).float().reshape(1, 1, 1, -1).repeat(1, 32, 2, 1),
    ])}


class VideoVAE:
    def encode(self, pixels):
        assert pixels.shape[0] == 22
        return torch.ones(1, 24, 7, 2, 2)

    def decode(self, samples):
        return torch.ones(22, 32, 32, 3)


class ModelPatcher:
    def __init__(self):
        self.model = comfy.model_base.MiniMaxH3.__new__(comfy.model_base.MiniMaxH3)
        torch.nn.Module.__init__(self.model)
        self.model_options = {}
        self.wrappers = {}

    def clone(self):
        result = copy.copy(self)
        result.model_options = copy.deepcopy(self.model_options)
        result.wrappers = copy.deepcopy(self.wrappers)
        return result

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers[kind] = (key, wrapper)


class RegistrationTests(unittest.TestCase):
    def test_official_loader_and_renamed_inputs(self):
        before = dict(comfy_nodes.NODE_CLASS_MAPPINGS)
        self.assertTrue(asyncio.run(comfy_nodes.load_custom_node(str(ROOT))))
        expected = {"H3KitLatentUpscale3D", "H3KitTiledSampler", "H3KitMotionBridge", "H3KitAVTrim", "H3KitVAEDecodeTiled",
                    "H3KitSelfLiftSampler", "H3KitSelfLiftAVJoin"}
        self.assertEqual(set(PACKAGE.NODE_CLASS_MAPPINGS), expected)
        self.assertEqual(set(comfy_nodes.NODE_CLASS_MAPPINGS) - set(before), expected)
        for node_id in expected:
            node_type = comfy_nodes.NODE_CLASS_MAPPINGS[node_id]
            self.assertEqual(node_type.__name__, node_id)
            entrypoint = node_type.execute if issubclass(node_type, io.ComfyNode) else getattr(node_type, node_type.FUNCTION)
            signature = inspect.signature(entrypoint)
            for group in node_type.INPUT_TYPES().values():
                for field in group:
                    self.assertIn(field, signature.parameters)
            self.assertTrue(node_type.CATEGORY.startswith("H3 Upgrade Kit/"))


class MotionTests(unittest.TestCase):
    def test_endpoint_context_follows_join_and_releases_moving_content(self):
        video = torch.ones(1, 24, 17, 2, 2)
        video[:, :, 10:] = 100
        audio = torch.zeros(1, 32, 2, 93)
        residual = torch.tensor([-4., -2., 0., 2., 4.]).reshape(1, 1, 5, 1, 1)
        reference = video[:, :, :5] + residual
        latent = {"samples": NestedTensor([video, audio]), "h3kit_upscaled_motion": True}
        corrected = motion.lock_motion_prefix(latent, reference)["samples"].unbind()[0]
        torch.testing.assert_close(corrected[:, :, :5], reference)
        # 时间中值为0，但接缝末端是+4；新段开头应承接末端差异。
        torch.testing.assert_close(corrected[:, :, 5:10], video[:, :, 5:10] + 4)
        # 内容已离开原位置，不把旧物品残差固定在后续画面上。
        torch.testing.assert_close(corrected[:, :, 10:], video[:, :, 10:])

    def test_upscaled_context_corrects_static_structure_without_freezing_motion(self):
        # 柜子区域有固定的空间偏差，右半区域保留持续运动。
        video = torch.zeros(1, 24, 12, 4, 4)
        video[..., 2:] = torch.arange(12).view(1, 1, 12, 1, 1) * 0.1
        cabinet = torch.zeros(1, 24, 1, 4, 4)
        cabinet[..., :2] = torch.tensor([0.2, -0.3, 0.4, -0.1]).view(1, 1, 1, 4, 1)
        reference = video[:, :, :5] + cabinet
        audio = torch.randn(1, 32, 2, 65)
        latent = {"samples": NestedTensor([video, audio]),
                  "h3kit_motion_length": (22, 17), "h3kit_upscaled_motion": True}
        result = motion.lock_motion_prefix(latent, reference)
        corrected, result_audio = result["samples"].unbind()
        torch.testing.assert_close(corrected, video + cabinet)
        torch.testing.assert_close(corrected[:, :, 5:] - corrected[:, :, 4:-1],
                                   video[:, :, 5:] - video[:, :, 4:-1])
        self.assertIs(result_audio, audio)
        self.assertNotIn("h3kit_upscaled_motion", result)
        torch.testing.assert_close(motion.lock_motion_prefix(result, reference)["samples"].unbind()[0], corrected)
        self.assertTrue(latent["h3kit_upscaled_motion"])
        torch.testing.assert_close(latent["samples"].unbind()[0], video)

    def test_context_transfer_rejects_unstable_residual_and_respects_locked_content(self):
        latent = av_latent(12, 65)
        video, audio = latent["samples"].unbind()
        reference = torch.tensor([-2., 2., -2., 2., 0.]).view(1, 1, 5, 1, 1).expand(1, 24, 5, 2, 2)
        latent["h3kit_upscaled_motion"] = True
        output = motion.lock_motion_prefix(latent, reference)
        torch.testing.assert_close(output["samples"].unbind()[0][:, :, 5:], video[:, :, 5:])
        mask = torch.ones_like(video)
        mask[:, :, -1] = 0
        audio_mask = torch.zeros_like(audio)
        latent["noise_mask"] = NestedTensor([mask, audio_mask])
        output = motion.lock_motion_prefix(latent, torch.ones_like(reference))
        self.assertEqual(output["samples"].unbind()[0][:, :, -1].count_nonzero().item(), 0)
        self.assertIs(output["noise_mask"].unbind()[1], audio_mask)

    def test_plain_motion_and_repeated_bridge_do_not_transfer_context(self):
        original = av_latent(12, 65)
        original["h3kit_motion_length"] = (22, 17)
        before_video, before_audio = original["samples"].unbind()
        conditions, head, result, length = motion.H3KitMotionBridge().bridge_motion(
            [[torch.zeros(1), {}]], VideoVAE(), original,
            previous_frames=torch.ones(17, 32, 32, 3))
        torch.testing.assert_close(result["samples"].unbind()[0][:, :, 5:], before_video[:, :, 5:])
        self.assertIs(result["samples"].unbind()[1], before_audio)
        self.assertEqual((head, length), (17, 22))

    def test_motion_vae_path_restored_for_latent_and_picture_inputs(self):
        schema = motion.H3KitMotionBridge.INPUT_TYPES()
        self.assertIn("video_vae", schema["required"])
        self.assertNotIn("video_vae", schema["optional"])
        for images in (None, torch.ones(17, 32, 32, 3)):
            with self.subTest(images_supplied=images is not None):
                vae = VideoVAE()
                with patch.object(vae, "decode", wraps=vae.decode) as decode, \
                     patch.object(vae, "encode", wraps=vae.encode) as encode:
                    out, trim, target, delivery = motion.H3KitMotionBridge().bridge_motion(
                        [[torch.zeros(1), {}]], vae, av_latent(),
                        previous_latent=av_latent(), previous_frames=images)
                self.assertEqual(decode.call_count, 1 if images is None else 0)
                self.assertEqual(encode.call_count, 1)
                self.assertEqual(encode.call_args.args[0].shape[0], 22)
                self.assertEqual((trim, delivery), (17, 22))
                actual = target["samples"].unbind()[0][:, :, :5]
                torch.testing.assert_close(actual, torch.ones_like(actual))

    def test_real_h3_layout_contract(self):
        ensure_motion_layout("CPU regression")

    def test_no_context_is_identity(self):
        original = av_latent()
        conditions = [[torch.zeros(1), {}]]
        result = motion.H3KitMotionBridge().bridge_motion(conditions, VideoVAE(), original)
        self.assertIs(result[0], conditions)
        self.assertIs(result[2], original)
        self.assertEqual((result[1], result[3]), (0, 0))

    def test_prefix_mask_audio_shift_and_original_unchanged(self):
        original = av_latent()
        video, audio = original["samples"].unbind()
        original["noise_mask"] = NestedTensor([torch.ones_like(video), torch.zeros_like(audio)])
        identity = torch.zeros(1, 24, 1, 2, 2)
        conditions = [[torch.zeros(1), {"minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": identity},
            {"resolved_frame_index": 21, "latent": identity},
        ]}]]
        result = motion.H3KitMotionBridge().bridge_motion(
            conditions, VideoVAE(), original, previous_frames=torch.ones(17, 32, 32, 3))
        out_conditions, head, extended, delivered = result
        self.assertEqual((head, delivered), (17, 22))
        output_video, output_audio = extended["samples"].unbind()
        self.assertEqual(motion.count_pixel_frames(output_video.shape[2]), 39)
        torch.testing.assert_close(output_video[:, :, :5], torch.ones_like(output_video[:, :, :5]))
        masks = extended["noise_mask"].unbind()
        self.assertEqual(masks[0][:, :, :5].count_nonzero().item(), 0)
        torch.testing.assert_close(output_audio[..., 28:65], audio)
        self.assertEqual(masks[1][..., 28:65].count_nonzero().item(), 0)
        self.assertEqual(video.count_nonzero().item(), 0)
        self.assertNotIn("h3kit_motion_length", original)
        self.assertEqual(out_conditions[0][1]["minimax_keyframes"][0]["resolved_frame_index"], 38)
        self.assertIs(out_conditions[0][1]["minimax_refs"][0]["latent"], identity)
        self.assertEqual(len(conditions[0][1]["minimax_keyframes"]), 2)

    def test_audio_tail_and_repeat_extension(self):
        previous = av_latent()
        result = motion.H3KitMotionBridge().bridge_motion(
            [[torch.zeros(1), {}]], VideoVAE(), av_latent(), previous_latent=previous)
        audio_keyframe = result[0][0][1]["minimax_keyframes"][-1]
        self.assertIn("audio_latent", audio_keyframe)
        self.assertLess(audio_keyframe["resolved_frame_index"], 0)
        repeated, delivered = motion.reserve_motion_prefix(result[2])
        self.assertIs(repeated, result[2])
        self.assertEqual(delivered, 22)

    def test_trim_audio_exactly_and_pad_short_tail(self):
        frames = torch.arange(39).reshape(39, 1, 1, 1).float()
        for audio_length in (4000, 3800):
            with self.subTest(audio_length=audio_length):
                audio = {"waveform": torch.ones(1, 2, audio_length), "sample_rate": 2400}
                trimmed, sound = motion.H3KitAVTrim().trim_av(
                    frames, 17, decoded_audio=audio, frame_rate=24, delivery_frames=22)
                self.assertEqual(trimmed.shape[0], 22)
                self.assertEqual(trimmed[0].item(), 17)
                self.assertEqual(sound["waveform"].shape[-1], 2200)
                if audio_length == 3800:
                    self.assertEqual(sound["waveform"][..., -100:].count_nonzero().item(), 0)


class SamplingTests(unittest.TestCase):
    def test_independent_guider_and_patch_namespace(self):
        model = ModelPatcher()
        guider = SimpleNamespace(model_patcher=model, model_options=model.model_options)
        latent = av_latent()
        with patch.object(SamplerCustomAdvanced, "execute", return_value="ok") as execute:
            result = sampling.H3KitTiledSampler.execute(None, guider, None, None, latent)
        copied = execute.call_args.args[1]
        self.assertEqual(result, "ok")
        self.assertIsNot(copied, guider)
        self.assertIsNot(copied.model_patcher, model)
        self.assertEqual(model.wrappers, {})
        self.assertEqual(len(copied.model_patcher.wrappers), 2)
        for key, wrapper in copied.model_patcher.wrappers.values():
            self.assertEqual(key, "h3kit_spatial_sampling")
            self.assertEqual(wrapper.keywords["plan"]["min_tiles"], 4)

    def test_disabled_tiling_passes_original(self):
        guider = object()
        with patch.object(SamplerCustomAdvanced, "execute", return_value="ok") as execute:
            sampling.H3KitTiledSampler.execute(None, guider, None, None, {}, spatial_tiles=False)
        self.assertIs(execute.call_args.args[1], guider)

    def test_tile_blending_and_audio_preservation(self):
        video, audio = torch.randn(1, 24, 3, 40, 20), torch.randn(1, 32, 2, 8)
        mask = torch.rand(1, 1, 3, 40, 20)
        regions, seen = tiles.sampling_tile_regions(40, 4), []

        def predict(streams, timestep, context, options, **kwargs):
            start, end = regions[len(seen)]
            seen.append(start)
            self.assertIs(streams[1], audio)
            torch.testing.assert_close(kwargs["denoise_mask"], mask[:, :, :, start:end])
            return streams[0] * 2, streams[1] * 3

        with patch.object(tiles, "crop_tile_payload", return_value={}):
            output = tiles.run_tiled_diffusion(predict, [video, audio], None, None, {}, n_tiles=4, denoise_mask=mask)
        self.assertEqual(len(seen), 4)
        torch.testing.assert_close(output[0], video * 2)
        torch.testing.assert_close(output[1], audio * 3)


class UpscaleTests(unittest.TestCase):
    def test_safe_weight_loading_and_network_execution(self):
        network = upscale.H3KitResizeNetwork(in_channels=24, in_blocks=1, out_blocks=1,
                                             channels=32, temporal_every=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.safetensors"
            save_file(network.state_dict(), str(path))
            with patch.object(upscale.folder_paths, "get_full_path_or_raise", return_value=str(path)):
                loaded = upscale.load_upscale_network("tiny.safetensors", torch.device("cpu"), "fp32")
            sample = torch.randn(1, 24, 2, 2, 2)
            with torch.inference_mode():
                torch.testing.assert_close(loaded(sample, scale=2, target_size=(2, 4, 4)),
                                           network.eval()(sample, scale=2, target_size=(2, 4, 4)))

    def test_renamed_resize_modes_and_tiled_path(self):
        class NearestVolume:
            conv_in = SimpleNamespace(weight=torch.empty(1))

            def __call__(self, samples, scale, target_size, enable_chunking):
                return torch.nn.functional.interpolate(samples, size=target_size, mode="nearest")

        original = {"samples": torch.randn(1, 24, 12, 8, 8), "h3kit_motion_length": (22, 17)}
        original["noise_mask"] = torch.ones(1, 1, 12, 8, 8)
        original["noise_mask"][:, :, :5] = 0
        modes = [
            {"resize_settings": "scale by multiplier", "scale_factor": 2.0},
            {"resize_settings": "target dimensions", "target_width": 256, "target_height": 256},
            {"resize_settings": "megapixels", "target_megapixels": 0.0625},
        ]
        for mode in modes:
            for tiled in (False, True):
                with self.subTest(mode=mode, tiled=tiled), patch.object(upscale, "load_upscale_network", return_value=NearestVolume()):
                    result = upscale.H3KitLatentUpscale3D.execute(
                        source_latent=original, upscale_weights="tiny.safetensors", resize_settings=mode,
                        pixel_alignment=32, temporal_chunks=True, release_weights=True,
                        compute_backend="cpu", compute_precision="fp32", spatial_tiles=tiled)[0]
                self.assertEqual(tuple(result["samples"].shape), (1, 24, 12, 16, 16))
                expected = torch.nn.functional.interpolate(original["samples"], size=(12, 16, 16), mode="nearest")
                torch.testing.assert_close(result["samples"], expected)
                self.assertEqual(result["h3kit_motion_length"], (22, 17))
                self.assertTrue(result["h3kit_upscaled_motion"])
                self.assertEqual(result["noise_mask"][:, :, :5].count_nonzero().item(), 0)
                self.assertTrue(torch.all(result["noise_mask"][:, :, 5:] == 1))
                self.assertNotIn("h3kit_upscaled_motion", original)

                audio = av_latent(12, 65)
                audio["h3kit_motion_length"] = (22, 17)
                _, audio_stream = LTXVSeparateAVLatent.execute(audio).result
                merged = LTXVConcatAVLatent.execute(result, audio_stream)[0]
                self.assertTrue(merged["h3kit_upscaled_motion"])
                video = merged["samples"].unbind()[0]
                reference = video[:, :, :5] + 0.25
                with patch.object(motion, "encode_motion_prefix", return_value=reference):
                    _, head, bridged, length = motion.H3KitMotionBridge().bridge_motion(
                        [[torch.zeros(1), {}]], VideoVAE(), merged,
                        previous_frames=torch.ones(17, 256, 256, 3))
                self.assertEqual((head, length), (17, 22))
                torch.testing.assert_close(bridged["samples"].unbind()[0], video + 0.25)
                self.assertIs(bridged["samples"].unbind()[1], audio_stream["samples"])
                self.assertNotIn("h3kit_upscaled_motion", bridged)


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
