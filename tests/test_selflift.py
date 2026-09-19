"""使用真实 ComfyUI Euler、H3 AV 包装器；仅大型像素网络替换为轻量算子。"""

import unittest
from unittest.mock import patch

import test_nodes  # 初始化 ComfyUI CPU 环境并加载插件。
import torch
import comfy.model_base
import comfy.model_patcher
import comfy.supported_models
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_lt import LTXVConcatAVLatent
from h3kit_tests import selflift_sampling as sl
from h3kit_tests import nodes_selflift as selflift_nodes
from h3kit_tests.nodes_selflift import H3KitSelfLiftAVJoin


def latent(tokens=12, size=4):
    frames = sl.pixel_frames(tokens)
    return {"samples": NestedTensor([
        torch.zeros(1, 24, tokens, size, size),
        torch.zeros(1, 32, 2, round(frames * 40 / 24)),
    ])}


class TinyH3(MiniMaxH3Model):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.dtype = torch.float32
        self.hidden_size = 8
        self.patch_size = (1, 2, 2)
        self.sigma_shift_video, self.sigma_shift_audio = 12.0, 3.0
        self.calls = []

    def _forward(self, x, timestep, context, transformer_options, **kwargs):
        self.calls.append(tuple(x[0].shape))
        return [torch.zeros_like(x[0]), torch.zeros_like(x[1])]


def make_model():
    config = comfy.supported_models.MiniMaxH3({"disable_unet_model_creation": True})
    model = comfy.model_base.MiniMaxH3(config, device=torch.device("cpu"))
    model.diffusion_model = TinyH3()
    return comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def nearest_lift(low, size, weights):
    return torch.nn.functional.interpolate(low, size=(low.shape[2], *size), mode="nearest")


class SelfLiftTests(unittest.TestCase):
    def test_context_keeps_prefix_with_continuous_motion_guide(self):
        class ContextVAE:
            def __init__(self):
                self.decoded_sizes = []

            def decode(self, samples):
                self.decoded_sizes.append(samples.shape[-1])
                return torch.zeros(sl.pixel_frames(samples.shape[2]), samples.shape[-2]*16, samples.shape[-1]*16, 3)

            def encode(self, frames):
                tokens = 1 if len(frames) == 1 else (len(frames)-5)//17*5+2
                return torch.full((1, 24, tokens, frames.shape[1]//16, frames.shape[2]//16), float(frames.mean()))

        class InspectBoundary(TinyH3):
            def _forward(self, x, timestep, context, transformer_options, **kwargs):
                guides = kwargs["minimax_payload"]["keyframes"]
                assert len(guides) == 1
                assert guides[0]["resolved_frame_index"] == 0
                assert guides[0]["latent"].shape[2] == 5
                assert torch.all(kwargs["denoise_mask"][:, :, :5] == 0)
                return super()._forward(x, timestep, context, transformer_options, **kwargs)

        prior = latent()
        prior[sl.LOW_CARRY] = torch.zeros(1, 24, 12, 2, 2)
        frames = torch.zeros(39, 64, 64, 3)
        frames[-1] = .9
        condition = [[torch.zeros(1, 2, 8), {}]]
        model = make_model()
        model.model.diffusion_model = InspectBoundary()
        vae = ContextVAE()
        with patch.object(sl.latent_preview, "prepare_callback", return_value=lambda *args: None):
            output = sl.progressive_sample(model, condition, condition, latent(), torch.linspace(1, 0, 5),
                                           42, 1, 2, .5, "test", prior, lifter=nearest_lift,
                                           context_vae=vae, previous_frames=frames)
        self.assertEqual(vae.decoded_sizes, [])
        self.assertEqual(output[sl.SEGMENT]["delivery_frames"], 39)
        self.assertNotIn("minimax_keyframes", condition[0][1])

    def test_aligned_motion_guide_converts_first_frame_to_identity(self):
        class ContextVAE:
            def __init__(self):
                self.encoded_lengths = []

            def decode(self, samples):
                return torch.zeros(sl.pixel_frames(samples.shape[2]), samples.shape[-2] * 16, samples.shape[-1] * 16, 3)

            def encode(self, frames):
                self.encoded_lengths.append(frames.shape[0])
                tokens = 1 if frames.shape[0] == 1 else (frames.shape[0] - 5) // 17 * 5 + 2
                return torch.full((1, 24, tokens, frames.shape[1] // 16, frames.shape[2] // 16), float(frames.mean()))

        class InspectBoundary(TinyH3):
            def __init__(self):
                super().__init__()
                self.guides = []
                self.refs = []

            def _forward(self, x, timestep, context, transformer_options, **kwargs):
                self.guides.append(kwargs["minimax_payload"].get("keyframes", []))
                self.refs.append(kwargs["minimax_payload"].get("refs", []))
                return super()._forward(x, timestep, context, transformer_options, **kwargs)

        prior = latent()
        prior[sl.LOW_CARRY] = torch.zeros(1, 24, 12, 2, 2)
        frames = torch.zeros(39, 64, 64, 3)
        frames[-1] = .9
        for explicit in (False, True):
            vae = ContextVAE()
            model = make_model()
            network = InspectBoundary()
            model.model.diffusion_model = network
            data = {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": torch.full((1, 24, 1, 4, 4), .4)}]} if explicit else {}
            condition = [[torch.zeros(1, 2, 8), data]]
            with patch.object(sl.latent_preview, "prepare_callback", return_value=lambda *args: None):
                sl.progressive_sample(model, condition, condition, latent(), torch.linspace(1, 0, 5),
                                      42, 1, 2, .5, "test", prior, lifter=nearest_lift,
                                      context_vae=vae, previous_frames=frames)
            self.assertEqual(vae.encoded_lengths, [22, 22])
            for guides, refs in zip(network.guides, network.refs):
                self.assertEqual(len(guides), 1)
                self.assertEqual(guides[0]["resolved_frame_index"], 0)
                self.assertEqual(guides[0]["latent"].shape[2], 5)
                expected = .9 * 6 / 22
                torch.testing.assert_close(guides[0]["latent"], torch.full_like(guides[0]["latent"], expected))
                self.assertEqual(len(refs), int(explicit))
                if explicit:
                    self.assertEqual(refs[0]["kind"], "image")
                    torch.testing.assert_close(refs[0]["latent"], torch.full_like(refs[0]["latent"], .4))

    def test_locked_motion_guides_use_each_stage_native_tail(self):
        class InspectContext(TinyH3):
            def __init__(self):
                super().__init__()
                self.contexts = []

            def _forward(self, x, timestep, context, transformer_options, **kwargs):
                guide = kwargs["minimax_payload"].get("keyframes", [])
                self.contexts.append((kwargs["denoise_mask"][:, :, :5].clone(), guide))
                return super()._forward(x, timestep, context, transformer_options, **kwargs)

        previous = latent()
        sl.av_streams(previous)[0][:] = torch.arange(12).view(1, 1, 12, 1, 1)
        previous[sl.LOW_CARRY] = torch.arange(12).view(1, 1, 12, 1, 1).expand(1, 24, 12, 2, 2).float() + 100
        condition = [[torch.zeros(1, 2, 8), {}]]
        for overlap in (17, 34):
            for tiled in (False, True):
                model = make_model()
                network = InspectContext()
                model.model.diffusion_model = network
                with patch.object(sl.latent_preview, "prepare_callback", return_value=lambda *args: None):
                    sl.progressive_sample(model, condition, condition, latent(), torch.linspace(1, 0, 5),
                                          42, 1, 2, .5, "test", previous, overlap,
                                          lifter=nearest_lift, spatial_tiles=tiled, minimum_tiles=2)
                self.assertTrue(all(torch.all(mask == 0) for mask, _ in network.contexts))
                prefix = overlap // 17 * 5
                for index, (_, guides) in enumerate(network.contexts):
                    self.assertEqual(len(guides), 1)
                    guide = guides[0]
                    self.assertEqual(guide["resolved_frame_index"], 0)
                    self.assertEqual(guide["latent"].shape[2], prefix)
                    expected = torch.arange(12 - prefix, 12).float() + (100 if index < 2 else 0)
                    torch.testing.assert_close(guide["latent"][0, 0, :, 0, 0], expected)
        self.assertNotIn("minimax_keyframes", condition[0][1])

    def test_motion_context_preserves_explicit_guides_audio_and_existing_refs(self):
        identity = torch.ones(1, 24, 1, 4, 4)
        audio = torch.ones(1, 32, 2, 5)
        multi = torch.ones(1, 24, 5, 4, 4)
        existing_ref = {"kind": "image", "latent_h": 4, "latent_w": 4, "latent": identity}
        condition = [[torch.zeros(1, 2, 8), {"minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": identity, "audio_latent": audio},
            {"resolved_frame_index": 0, "latent": multi},
            {"resolved_frame_index": 38, "latent": identity},
        ], "minimax_refs": [existing_ref]}]]
        shifted = sl.shift_conditioning(condition, 17)
        reference = torch.arange(5).view(1, 1, 5, 1, 1).expand(1, 24, 5, 4, 4).float()
        output = sl.add_motion_context(shifted, reference, 17)
        data = output[0][1]
        self.assertEqual([k["resolved_frame_index"] for k in data["minimax_keyframes"]], [17, 17, 55, 0])
        self.assertNotIn("latent", data["minimax_keyframes"][0])
        self.assertIs(data["minimax_keyframes"][0]["audio_latent"], audio)
        self.assertIs(data["minimax_keyframes"][1]["latent"], multi)
        self.assertEqual(len(data["minimax_refs"]), 2)
        self.assertIs(data["minimax_refs"][0], existing_ref)
        self.assertIs(data["minimax_refs"][1]["latent"], identity)
        self.assertEqual([k["resolved_frame_index"] for k in condition[0][1]["minimax_keyframes"]], [0, 0, 38])
        self.assertEqual(len(condition[0][1]["minimax_refs"]), 1)
        repeated = sl.add_motion_context(output, reference, 17)[0][1]
        self.assertEqual(sum(bool(k.get("h3kit_motion")) for k in repeated["minimax_keyframes"]), 1)
        self.assertEqual(len(repeated["minimax_refs"]), 2)
        self.assertIs(sl.add_motion_context(condition, reference[:, :, :0], 0), condition)

    def test_motion_guide_only_enters_positive_conditioning_with_cfg(self):
        condition = [[torch.zeros(1, 2, 8), {}]]
        prior = latent()
        prior[sl.LOW_CARRY] = torch.zeros(1, 24, 12, 2, 2)
        with patch.object(sl.latent_preview, "prepare_callback", return_value=lambda *args: None), \
                patch.object(sl.comfy.samplers, "sample", wraps=sl.comfy.samplers.sample) as sample:
            sl.progressive_sample(make_model(), condition, condition, latent(), torch.linspace(1, 0, 5),
                                  42, 2, 2, .5, "test", prior, lifter=nearest_lift)
        self.assertEqual(sample.call_count, 2)
        for call in sample.call_args_list:
            positive, negative = call.args[2:4]
            self.assertEqual(len(positive[0][1]["minimax_keyframes"]), 1)
            self.assertNotIn("minimax_keyframes", negative[0][1])
        self.assertNotIn("minimax_keyframes", condition[0][1])

    def test_context_vae_aligns_both_resolutions_from_delivered_pixels(self):
        class ContextVAE:
            def __init__(self):
                self.decoded = []
                self.encoded = []

            def decode(self, samples):
                self.decoded.append(tuple(samples.shape))
                return torch.full((sl.pixel_frames(samples.shape[2]), samples.shape[-2] * 16,
                                   samples.shape[-1] * 16, 3), float(samples.mean()))

            def encode(self, frames):
                self.encoded.append(tuple(frames.shape))
                tokens = (frames.shape[0] - 5) // 17 * 5 + 2
                return torch.full((1, 24, tokens, frames.shape[1] // 16, frames.shape[2] // 16), float(frames.mean()) + .1)

        prior = latent()
        prior["samples"].unbind()[0].fill_(3)
        prior[sl.LOW_CARRY] = torch.full((1, 24, 12, 2, 2), 9.)
        pixels = torch.full((39, 64, 64, 3), .7)
        for frames in (17, 34):
            vae = ContextVAE()
            video, audio, vm, am, low, prefix, info = sl.prepare_segment(latent(), prior, frames, True, vae, pixels)
            self.assertEqual(prefix, frames // 17 * 5)
            self.assertEqual(vae.decoded, [])
            self.assertEqual(vae.encoded, [(frames + 5, 64, 64, 3), (frames + 5, 32, 32, 3)])
            torch.testing.assert_close(video[:, :, :prefix], torch.full_like(video[:, :, :prefix], .8))
            torch.testing.assert_close(low, torch.full_like(low, .8))
            torch.testing.assert_close(prior[sl.LOW_CARRY], torch.full_like(prior[sl.LOW_CARRY], 9))
            self.assertEqual(info["frames"], 39 + frames)

        vae = ContextVAE()
        video, _, _, _, low, prefix, _ = sl.prepare_segment(latent(), prior, 17, True, vae)
        self.assertEqual(vae.decoded, [(1, 24, 12, 4, 4)])
        torch.testing.assert_close(video[:, :, :prefix], torch.full_like(video[:, :, :prefix], 3.1))
        torch.testing.assert_close(low, torch.full_like(low, 3.1))

    def test_low_context_resizes_actual_tail_pixels_before_encoding(self):
        class ContextVAE:
            def __init__(self):
                self.inputs = []

            def decode(self, samples):
                raise AssertionError("提供真实画面后不应解码旧低清状态")

            def encode(self, frames):
                self.inputs.append(frames.clone())
                tokens = (len(frames) - 5) // 17 * 5 + 2
                return torch.zeros(1, 24, tokens, frames.shape[1] // 16, frames.shape[2] // 16)

        prior = latent()
        prior[sl.LOW_CARRY] = torch.full((1, 24, 12, 2, 2), 9.)
        pixels = torch.arange(39 * 64 * 64 * 3).reshape(39, 64, 64, 3).float()
        original = pixels.clone()
        vae = ContextVAE()
        sl.continuation_window(prior, 17, vae, pixels)
        expected = torch.nn.functional.interpolate(pixels[-17:].movedim(-1, 1), size=(32, 32), mode="area").movedim(1, -1)
        torch.testing.assert_close(vae.inputs[0][:17], pixels[-17:])
        torch.testing.assert_close(vae.inputs[1][:17], expected)
        torch.testing.assert_close(vae.inputs[1][17:], expected[-1:].expand(5, -1, -1, -1))
        torch.testing.assert_close(pixels, original)

    def test_previous_frames_requires_context_vae(self):
        prior = latent()
        prior[sl.LOW_CARRY] = torch.zeros(1, 24, 12, 2, 2)
        with self.assertRaisesRegex(ValueError, "context_vae"):
            sl.prepare_segment(latent(), prior, 17, True, previous_frames=torch.zeros(39, 64, 64, 3))

    def test_standard_av_concat_square_audio_masks_match_native_sampler(self):
        video = torch.zeros(1, 24, 36, 4, 4)  # 120帧，对应200个音频时间步。
        audio = torch.randn(1, 32, 2, 200)
        for mask in (torch.zeros(1, 64, 64), torch.ones(1, 64, 64),
                     torch.linspace(0, 1, 64).view(1, 1, 64).expand(1, 64, 64)):
            with self.subTest(mask_mean=float(mask.mean())):
                masked = test_nodes.comfy_nodes.SetLatentNoiseMask().set_mask({"samples": audio}, mask)[0]
                av = LTXVConcatAVLatent.execute({"samples": video}, masked)[0]
                raw_video_mask, raw_audio_mask = av["noise_mask"].unbind()
                self.assertEqual(tuple(raw_audio_mask.shape), (1, 1, 64, 64))
                self.assertEqual(raw_video_mask.shape[1], 24)
                out_video, out_audio, vm, am, *_ = sl.prepare_segment(av, None, 17, True)
                self.assertEqual(tuple(vm.shape), (1, 1, 36, 4, 4))
                expected = comfy.sampler_helpers.prepare_mask(raw_audio_mask, audio.shape, audio.device)
                torch.testing.assert_close(am, expected)
                torch.testing.assert_close(out_audio, audio)
                torch.testing.assert_close(raw_audio_mask[:, 0], mask)

    def test_square_locked_audio_mask_survives_selflift_and_continuation(self):
        video = torch.zeros(1, 24, 37, 4, 4)  # 用户视频向上对齐到124帧，原音轨仅120帧。
        audio = torch.full((1, 32, 2, 200), .125)
        masked = test_nodes.comfy_nodes.SetLatentNoiseMask().set_mask({"samples": audio}, torch.zeros(1, 64, 64))[0]
        source = LTXVConcatAVLatent.execute({"samples": video}, masked)[0]
        model = make_model()
        condition = [[torch.zeros(1, 2, 8), {}]]
        with patch.object(sl.latent_preview, "prepare_callback", return_value=lambda *args: None):
            first = sl.progressive_sample(model, condition, condition, source, torch.linspace(1, 0, 5),
                                          42, 1, 2, .5, "test", lifter=nearest_lift)
            second = sl.progressive_sample(model, condition, condition, source, torch.linspace(1, 0, 5),
                                           42, 1, 2, .5, "test", first, lifter=nearest_lift)
        first_audio = sl.av_streams(first)[1]
        second_audio = sl.av_streams(second)[1]
        self.assertEqual(first_audio.shape[-1], round(124 * 40 / 24))
        self.assertEqual(second_audio.shape[-1], round(141 * 40 / 24))
        torch.testing.assert_close(first_audio[..., :200], audio)
        head = round(17 * 40 / 24)
        torch.testing.assert_close(second_audio[..., head:head + 200], audio)
        self.assertEqual(second[sl.SEGMENT]["delivery_frames"], 124)

    def test_high_stage_tiles_with_context_masks_keyframes_and_locked_audio(self):
        class CheckedH3(TinyH3):
            def __init__(self):
                super().__init__()
                self.masks = []

            def _forward(self, x, timestep, context, transformer_options, **kwargs):
                mask = kwargs["denoise_mask"]
                assert mask.shape[-3:] == x[0].shape[-3:]
                assert kwargs["audio_denoise_mask"].shape[-1] == x[1].shape[-1]
                for keyframe in kwargs["minimax_payload"].get("keyframes", []):
                    assert keyframe["latent"].shape[-2:] == x[0].shape[-2:]
                self.masks.append(mask.clone())
                return super()._forward(x, timestep, context, transformer_options, **kwargs)

        condition = [[torch.zeros(1, 2, 8), {"minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 12, 12)}]}]]
        source = latent(size=12)
        video, audio = sl.av_streams(source)
        audio.fill_(.125)
        video[:, :, :, :2, :2] = .25
        mask = torch.ones(1, 1, 12, 12, 12)
        mask[..., :4, :4] = 0
        source["noise_mask"] = NestedTensor([mask, torch.zeros_like(audio)])
        results = []
        for tiled in (False, True):
            model = make_model()
            network = CheckedH3()
            model.model.diffusion_model = network
            with patch.object(sl.latent_preview, "prepare_callback", return_value=lambda *args: None):
                first = sl.progressive_sample(model, condition, condition, source, torch.linspace(1, 0, 5),
                                              42, 1, 2, .5, "test", lifter=nearest_lift,
                                              spatial_tiles=tiled, minimum_tiles=2)
                second = sl.progressive_sample(model, condition, condition, latent(size=12), torch.linspace(1, 0, 5),
                                               42, 1, 2, .5, "test", first, 22, True, lifter=nearest_lift,
                                               spatial_tiles=tiled, minimum_tiles=2)
            high_shapes = [(1, 24, 12, 8, 12)] * 4 if tiled else [(1, 24, 12, 12, 12)] * 2
            second_high = [(1, 24, 17, *shape[-2:]) for shape in high_shapes]
            self.assertEqual(network.calls, [(1, 24, 12, 6, 6)] * 2 + high_shapes
                             + [(1, 24, 17, 6, 6)] * 2 + second_high)
            torch.testing.assert_close(sl.av_streams(first)[1], audio)
            torch.testing.assert_close(sl.av_streams(first)[0][..., :2, :2], video[..., :2, :2])
            torch.testing.assert_close(sl.av_streams(second)[0][:, :, 4], sl.av_streams(first)[0][:, :, -1])
            self.assertTrue(second[sl.SEGMENT]["soft_audio"])
            self.assertEqual(float(network.masks[-1][:, :, 4].max()), 0)
            self.assertNotIn("denoise_mask_function", model.model_options)
            self.assertFalse(model.get_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3kit_spatial_sampling"))
            results.append((first, second))
        # 轻量网络不含跨块注意力，拼接结果应与整幅计算相同；真实 H3 画质不作此保证。
        for reference, tiled in zip(*results):
            for expected, actual in zip(sl.av_streams(reference), sl.av_streams(tiled)):
                torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(tiled[sl.LOW_CARRY], reference[sl.LOW_CARRY])

    def test_node_denoise_schedule_and_external_override(self):
        args = dict(model=make_model(), positive=[], negative=[], latent_image=latent(), seed=0,
                    steps=8, cfg=1, scheduler="simple", high_resolution_steps=2, lowres_scale=.5,
                    upscale_weights="test", overlap_frames=22, continue_audio=True)
        node = selflift_nodes.H3KitSelfLiftSampler()
        with patch.object(selflift_nodes, "progressive_sample", return_value=args["latent_image"]) as sample:
            node.sample(**args, denoise=.5)
            reduced = sample.call_args.args[4]
            expected = selflift_nodes.BasicScheduler.execute(args["model"], "simple", 8, .5)[0]
            torch.testing.assert_close(reduced, expected)
            self.assertEqual(reduced.numel(), 9)
            self.assertLess(float(reduced[0]), 1)
            external = torch.tensor([.9, .6, .3, 0.])
            with patch.object(selflift_nodes.BasicScheduler, "execute", side_effect=AssertionError("外部调度不应被重建")):
                node.sample(**args, sigmas=external, denoise=0)
            self.assertIs(sample.call_args.args[4], external)

    def test_node_zero_denoise_returns_input_without_sampling(self):
        source = latent()
        with patch.object(selflift_nodes, "progressive_sample", side_effect=AssertionError("零降噪不应采样")):
            output = selflift_nodes.H3KitSelfLiftSampler().sample(
                make_model(), [], [], source, 0, 8, 1, "simple", 2, .5, "test", 22, True, denoise=0)
        self.assertIs(output[0], source)

    def test_real_euler_two_segments_and_native_low_carry(self):
        model = make_model()
        condition = [[torch.zeros(1, 2, 8), {}]]
        schedule = torch.linspace(1, 0, 9)
        first_input = latent()
        audio = first_input["samples"].unbind()[1]
        audio.fill_(0.125)
        first_input["noise_mask"] = NestedTensor([torch.ones(1, 1, 1, 1, 1), torch.zeros(1, 1, 1, audio.shape[-1])])
        with patch.object(sl.latent_preview, "prepare_callback", return_value=lambda *args: None):
            first = sl.progressive_sample(model, condition, condition, first_input, schedule,
                                          42, 1, 2, 0.5, "test", lifter=nearest_lift)
            original = first["samples"].unbind()[0].clone()
            low_original = first[sl.LOW_CARRY].clone()
            second = sl.progressive_sample(model, condition, condition, first_input, schedule,
                                           42, 1, 2, 0.5, "test", first, 22, True, lifter=nearest_lift)
        self.assertEqual(model.model.diffusion_model.calls, [(1, 24, 12, 2, 2)] * 6 + [(1, 24, 12, 4, 4)] * 2
                         + [(1, 24, 17, 2, 2)] * 6 + [(1, 24, 17, 4, 4)] * 2)
        torch.testing.assert_close(first["samples"].unbind()[1], audio)
        head = round(second[sl.SEGMENT]["overlap_frames"] * 40 / 24)
        torch.testing.assert_close(second["samples"].unbind()[1][..., head:head + audio.shape[-1]], audio)
        torch.testing.assert_close(first["samples"].unbind()[0], original)
        torch.testing.assert_close(first[sl.LOW_CARRY], low_original)
        self.assertEqual(second[sl.SEGMENT]["overlap_frames"], 17)
        self.assertTrue(torch.isfinite(second["samples"].unbind()[0]).all())
        torch.testing.assert_close(second["samples"].unbind()[0][:, :, 4], original[:, :, -1])
        self.assertNotIn("denoise_mask_function", model.model_options)

    def test_two_stage_state_matches_analytic_zero_velocity(self):
        model = make_model()
        condition = [[torch.zeros(1, 2, 8), {}]]
        source = latent()
        schedule = torch.tensor([1., 0.8, 0.5, 0.2, 0.])
        with patch.object(sl.latent_preview, "prepare_callback", return_value=lambda *args: None):
            output = sl.progressive_sample(model, condition, condition, source, schedule,
                                           7, 1, 1, 0.5, "test", lifter=nearest_lift)
        # 零视频速度：高清阶段的输入应精确等于 lift 后 x0 与新噪声在交接 sigma 的混合。
        fmt = model.model.latent_format
        endpoint = fmt.process_in(nearest_lift(output[sl.LOW_CARRY], (4, 4), ""))
        noise = comfy.sample.prepare_noise(endpoint, 8).to(endpoint)
        expected = model.model.model_sampling.noise_scaling(schedule[-2], noise, endpoint)
        torch.testing.assert_close(fmt.process_in(output["samples"].unbind()[0]), expected)

    def test_continuation_uses_low_tail_without_resizing_high_tail(self):
        prior = latent()
        prior[sl.LOW_CARRY] = torch.full((1, 24, 12, 2, 2), 9.)
        prior["samples"].unbind()[0].fill_(3.)
        video, audio, vm, am, low, prefix, info = sl.prepare_segment(latent(), prior, 24, True)
        self.assertEqual((prefix, info["overlap_frames"]), (5, 17))
        torch.testing.assert_close(video[:, :, :5], torch.full_like(video[:, :, :5], 3))
        torch.testing.assert_close(low, prior[sl.LOW_CARRY][:, :, -5:])
        self.assertEqual(float(am[..., 0].max()), 0)
        self.assertEqual(float(am[..., 27].min()), 1)
        self.assertTrue(torch.all(am[..., 20:28].diff() > 0))

    def test_following_segment_uses_delivered_endpoint_not_padding_tail(self):
        prior = latent(tokens=22)
        video, audio = sl.av_streams(prior)
        video[:] = torch.arange(22).view(1, 1, 22, 1, 1)
        audio[:] = torch.arange(audio.shape[-1]).view(1, 1, 1, -1)
        prior[sl.LOW_CARRY] = torch.arange(22).view(1, 1, 22, 1, 1).expand(1, 24, 22, 2, 2).float() + 100
        prior[sl.SEGMENT] = {"frames": 73, "overlap_frames": 17, "delivery_frames": 39}
        video, low, audio, overlap = sl.continuation_window(prior, 17)
        self.assertEqual(overlap, 17)
        torch.testing.assert_close(video[0, 0, :, 0, 0], torch.arange(12, 17).float())
        torch.testing.assert_close(low[0, 0, :, 0, 0], torch.arange(112, 117).float())
        self.assertEqual(float(audio[..., -1].max()), round(56 * 40 / 24) - 1)
        prior[sl.SEGMENT]["overlap_frames"] = 22
        with self.assertRaisesRegex(ValueError, "重新运行"):
            sl.continuation_window(prior, 17)

    def test_added_prefix_shifts_locked_audio_masks_and_guides(self):
        prior = latent()
        prior[sl.LOW_CARRY] = torch.zeros(1, 24, 12, 2, 2)
        source = latent()
        original_video, original_audio = sl.av_streams(source)
        original_video[:] = torch.arange(12).view(1, 1, 12, 1, 1)
        original_audio[:] = torch.arange(original_audio.shape[-1]).view(1, 1, 1, -1)
        source["noise_mask"] = NestedTensor([torch.zeros_like(original_video), torch.zeros_like(original_audio)])
        video, audio, vm, am, low, prefix, info = sl.prepare_segment(source, prior, 22, False)
        self.assertEqual(info, {"overlap_frames": 17, "soft_audio": False, "fps": 24,
                                "frames": 56, "delivery_frames": 39})
        torch.testing.assert_close(video[:, :, 5:], original_video)
        head = round(17 * 40 / 24)
        torch.testing.assert_close(audio[..., head:head + original_audio.shape[-1]], original_audio)
        self.assertTrue(torch.all(am[..., head:head + original_audio.shape[-1]] == 0))
        self.assertTrue(torch.all(am[..., :head] == 1))
        condition = [[torch.zeros(1), {"minimax_keyframes": [{"resolved_frame_index": 0},
                                                            {"resolved_frame_index": 38}],
                                      "minimax_refs": [{"tag": "unchanged"}]}]]
        moved = sl.shift_conditioning(condition, info["overlap_frames"])
        self.assertEqual([kf["resolved_frame_index"] for kf in moved[0][1]["minimax_keyframes"]], [17, 55])
        self.assertEqual(condition[0][1]["minimax_keyframes"][0]["resolved_frame_index"], 0)
        self.assertEqual(moved[0][1]["minimax_refs"], condition[0][1]["minimax_refs"])

    def test_missing_carry_and_different_grid_fail_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "低清状态"):
            sl.prepare_segment(latent(), latent(), 22, True)
        prior = latent()
        prior[sl.LOW_CARRY] = torch.zeros(1, 24, 12, 6, 6)
        model = make_model()
        with self.assertRaisesRegex(ValueError, "低清尺寸"):
            sl.progressive_sample(model, [], [], latent(), torch.linspace(1, 0, 9), 0, 1, 2, .5, "", prior)

    def test_context_mask_stays_fixed_and_model_matches_mask(self):
        video, audio = sl.av_streams(latent())
        video_mask = torch.ones(1, 1, 12, 4, 4)
        video_mask[:, :, :5] = 0
        state = sl.ContinuationMask(video, audio, video_mask, torch.ones_like(audio),
                                    make_model().model._pool_masks_to_token_grid)
        packed = comfy.utils.pack_latents([video, audio])[0]
        for sigma in (1., .8, .5, .2, 0.):
            timestep = torch.tensor([sigma])
            mask = state.denoise_mask(timestep, packed)
            vm, am = comfy.utils.unpack_latents(mask, [video.shape, audio.shape])
            torch.testing.assert_close(vm, video_mask.expand_as(video))
            result = state.apply_model(lambda *args, **kw: kw, packed, timestep)
            torch.testing.assert_close(result["denoise_mask"], video_mask)

    def test_seam_corrects_spatial_background_without_erasing_motion(self):
        for prefix in (5, 10):
            lifted = torch.zeros(2, 24, prefix + 12, 4, 4)
            lifted[..., 2:] = torch.arange(lifted.shape[2]).view(1, 1, -1, 1, 1) * .1
            offset = torch.zeros(2, 24, 1, 4, 4)
            offset[0, ..., :2] = .3
            offset[1, ..., :2] = -.2
            previous = lifted + offset
            original = lifted.clone()
            reference = previous.clone()
            mask = torch.ones(2, 1, lifted.shape[2], 4, 4)
            mask[:, :, :prefix] = 0
            corrected = sl.match_seam(lifted, previous, prefix, mask)
            torch.testing.assert_close(corrected, previous)
            torch.testing.assert_close(corrected[:, :, prefix + 1:] - corrected[:, :, prefix:-1],
                                       lifted[:, :, prefix + 1:] - lifted[:, :, prefix:-1])
            torch.testing.assert_close(lifted, original)
            torch.testing.assert_close(previous, reference)

    def test_seam_endpoint_releases_after_motion_and_honors_masks(self):
        # 五种不同的相位不能被误判成运动；第二个周期真正改变内容后才释放。
        lifted = torch.arange(1, 6).repeat(3).view(1, 1, 15, 1, 1).expand(1, 24, 15, 2, 2).float().clone()
        lifted[:, :, 10:] += 100
        previous = lifted.clone()
        previous[:, :, :5] += torch.tensor([-4., -2., 0., 2., 4.]).view(1, 1, 5, 1, 1)
        mask = torch.ones(1, 1, 15, 2, 2)
        mask[..., 0, 0] = 0
        mask[..., 0, 1] = .5
        corrected = sl.match_seam(lifted, previous, 5, mask)
        torch.testing.assert_close(corrected[:, :, :5], previous[:, :, :5])
        torch.testing.assert_close(corrected[:, :, 5:10], lifted[:, :, 5:10] + 4 * mask[:, :, 5:10])
        torch.testing.assert_close(corrected[:, :, 10:], lifted[:, :, 10:])
        self.assertIs(sl.match_seam(lifted, previous, 0, mask), lifted)

    def test_corrected_seam_reaches_high_sampler_without_changing_low_carry(self):
        condition = [[torch.zeros(1, 2, 8), {}]]
        prior = latent()
        sl.av_streams(prior)[0].fill_(.3)
        prior[sl.LOW_CARRY] = torch.zeros(1, 24, 12, 2, 2)
        source = latent()
        video, audio = sl.av_streams(source)
        video[..., 0, 0] = .7
        mask = torch.ones(1, 1, 12, 4, 4)
        mask[..., 0, 0] = 0
        source["noise_mask"] = NestedTensor([mask, torch.zeros_like(audio)])
        schedule = torch.tensor([1., .8, .5, .2, 0.])
        for tiled in (False, True):
            model = make_model()
            with patch.object(sl.latent_preview, "prepare_callback", return_value=lambda *args: None):
                output = sl.progressive_sample(model, condition, condition, source, schedule,
                    7, 1, 1, .5, "test", prior,
                    lifter=lambda low, size, weights: torch.zeros(low.shape[0], 24, low.shape[2], *size),
                    spatial_tiles=tiled, minimum_tiles=2)
                baseline = sl.progressive_sample(make_model(), condition, condition, source, schedule,
                    7, 1, 1, .5, "test", prior, lifter=nearest_lift)
            fmt = model.model.latent_format
            endpoint = fmt.process_in(torch.full((1, 24, 17, 4, 4), .3))
            noise = comfy.sample.prepare_noise(endpoint, 8).to(endpoint)
            expected = fmt.process_out(model.model.model_sampling.noise_scaling(schedule[-2], noise, endpoint))
            actual = sl.av_streams(output)[0]
            torch.testing.assert_close(actual[:, :, 5:, 1:, :], expected[:, :, 5:, 1:, :])
            torch.testing.assert_close(actual[:, :, :5], torch.full_like(actual[:, :, :5], .3))
            torch.testing.assert_close(actual[:, :, 5:, 0, 0], video[..., 0, 0])
            torch.testing.assert_close(output[sl.LOW_CARRY], baseline[sl.LOW_CARRY])
            torch.testing.assert_close(sl.av_streams(output)[1], sl.av_streams(baseline)[1])

    def test_lifter_preserves_real_temporal_groupnorm_statistics(self):
        class Capture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []
                self.conv_in = torch.nn.Conv3d(24, 32, 1)
                self.in_blocks = torch.nn.ModuleList()
                self.norm = torch.nn.GroupNorm(8, 24, affine=False)

            def forward(self, x, scale, target_size):
                self.inputs.append(x.clone())
                return torch.nn.functional.interpolate(self.norm(x), size=target_size, mode="nearest")
        network = Capture()
        low = torch.randn(1, 24, 42, 2, 2)
        low[:, :, :5] += 2
        mean, std = sl.make_latent_statistics(torch.device("cpu"), torch.float32)
        normalized = (low - mean) / std
        expected = torch.nn.functional.interpolate(network.norm(normalized), size=(42, 4, 4), mode="nearest") * std + mean
        with patch.object(sl, "load_upscale_network", return_value=network):
            output = sl.learned_lift(low, (4, 4), "test")
        self.assertEqual(len(network.inputs), 1)
        torch.testing.assert_close(network.inputs[0], normalized)
        torch.testing.assert_close(output, expected)


class JoinTests(unittest.TestCase):
    def test_17_frame_multiples_add_only_head_tokens_without_tail_padding(self):
        prior = latent(tokens=37)
        prior[sl.LOW_CARRY] = torch.randn(1, 24, 37, 2, 2)
        prior["samples"].unbind()[0][:] = torch.arange(37).view(1, 1, 37, 1, 1)
        for requested in (17, 34, 51, 68):
            with self.subTest(overlap=requested):
                video, audio, vm, am, low, tokens, info = sl.prepare_segment(latent(tokens=37), prior, requested, True)
                self.assertEqual(tokens, requested // 17 * 5)
                self.assertEqual(video.shape[2], 37 + tokens)
                self.assertEqual(info["frames"], 124 + requested)
                self.assertEqual(info["overlap_frames"], requested)
                self.assertEqual(info["delivery_frames"], 124)
                torch.testing.assert_close(video[:, :, :tokens], prior["samples"].unbind()[0][:, :, -tokens:])
                torch.testing.assert_close(low, prior[sl.LOW_CARRY][:, :, -tokens:])

    def test_auto_extension_three_segments_keep_124_new_frames_each(self):
        previous = None
        images = audio = None
        for segment in range(3):
            source = latent(tokens=37)
            video, sound, vm, am, low, prefix, info = sl.prepare_segment(source, previous, 22, True)
            previous = {"samples": NestedTensor([video, sound]), sl.LOW_CARRY: sl.resize_video(video, (2, 2)),
                        sl.SEGMENT: info}
            pixels = torch.arange(info["frames"]).view(-1, 1, 1, 1).expand(-1, 2, 2, 3).float()
            waveform = torch.ones(1, 2, round(info["frames"] * 44100 / 24))
            images, audio = H3KitSelfLiftAVJoin().join(previous, pixels,
                {"waveform": waveform, "sample_rate": 44100}, images, audio)
            self.assertEqual(info["delivery_frames"], 124)
            self.assertEqual(info["overlap_frames"], 17 if segment else 0)
            self.assertEqual(info["frames"], 141 if segment else 124)
            self.assertEqual(images.shape[0], (segment + 1) * 124)
            self.assertEqual(audio["waveform"].shape[-1], round(images.shape[0] * 44100 / 24))
            self.assertEqual(float(images[-1, 0, 0, 0]), info["overlap_frames"] + 123)
        self.assertEqual(images.shape[0], 372)

    def test_three_segments_video_ownership_soft_audio_and_exact_length(self):
        join = H3KitSelfLiftAVJoin()
        frames, audio = None, None
        for index in range(3):
            info = {"frames": 39, "overlap_frames": 22 if index else 0, "soft_audio": bool(index), "fps": 24}
            current_frames = torch.full((39, 2, 2, 3), float(index))
            current_audio = {"waveform": torch.full((1, 2, 390), float(index)), "sample_rate": 240}
            frames, audio = join.join({sl.SEGMENT: info}, current_frames, current_audio, frames, audio)
        self.assertEqual(frames.shape[0], 73)
        self.assertEqual(audio["waveform"].shape[-1], 730)
        self.assertTrue(torch.all(frames[:39] == 0))
        self.assertTrue(torch.all(frames[39:56] == 1))
        self.assertTrue(torch.all(audio["waveform"][..., :170] == 0))
        self.assertTrue(torch.all(audio["waveform"][..., 170:340] == 1))
        self.assertTrue(torch.all(audio["waveform"][..., 340:] == 2))

    def test_trim_without_prior_and_silent_segment(self):
        join = H3KitSelfLiftAVJoin()
        info = {"frames": 39, "overlap_frames": 22, "soft_audio": True, "fps": 24}
        frames = torch.ones(39, 2, 2, 3)
        result, audio = join.join({sl.SEGMENT: info}, frames)
        self.assertEqual(result.shape[0], 17)
        self.assertIsNone(audio)
        prior_audio = {"waveform": torch.ones(1, 2, 390), "sample_rate": 240}
        result, audio = join.join({sl.SEGMENT: info}, frames, previous_frames=frames, previous_audio=prior_audio)
        self.assertEqual(audio["waveform"].shape[-1], 560)
        self.assertTrue(torch.all(audio["waveform"][..., :390] == 1))
        self.assertTrue(torch.all(audio["waveform"][..., 390:] == 0))


if __name__ == "__main__":
    unittest.main()
