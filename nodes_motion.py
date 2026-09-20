# 移植自 ComfyUI-H3-Motion-Context；原始 GPL-3.0 许可见 LICENSE。
"""按 17 帧周期预留重叠区，或只用前段尾帧引导新片段。"""

import logging

import comfy.nested_tensor
import comfy.utils
import node_helpers
import torch

from .motion_layout import ensure_motion_layout

try:
    import torchaudio
except ImportError:
    torchaudio = None

H3KIT_MOTION_LOG = logging.getLogger("h3kit_motion")


def check_motion_layout():
    """Prove ComfyUI still places anchors the way this pack needs, once.

    This used to install two runtime patches. ComfyUI 0.33 does natively
    what they existed to do, so the node now builds plain keyframe dicts
    and only has to check that the arithmetic behind them still holds. See
    layout_contract.py for what is checked and why it is not free.

    Run on first use rather than at import, same as the patches were: the
    pack sitting in custom_nodes should change nothing at all until you
    actually chain a clip. The cost is that a failure shows up on the
    first render instead of in the startup log.
    """
    ensure_motion_layout("pinning a clip")


H3KIT_FRAMES_PER_TOKEN = (1, 4, 4, 4, 4)
H3KIT_NATIVE_FPS = 24  # H3's native rate; audio latents run at 40 Hz, hence FRAME_RESCALE 5/3
H3KIT_AUDIO_FRAME_SCALE = 5.0 / 3.0
H3KIT_AUDIO_RATE = 40.0

H3KIT_AUDIO_MODE = "timeline"
H3KIT_CROP = "disabled"


def count_pixel_frames(latent_t):
    """Pixel frames covered by latent_t latent steps."""
    return sum(H3KIT_FRAMES_PER_TOKEN[k % 5] for k in range(latent_t))


def resize_reference_frames(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]; matches the stock helper
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def encode_previous_audio(sound_vae, decoded_audio, seconds):
    """Encode the last `seconds` of a clip's audio with the H3 audio VAE.

    Returns ([1, 32, 2, T] latent, T) where T counts 40 Hz latent steps,
    matching what the layout calls ref_audio_t.
    """
    waveform = decoded_audio["waveform"]  # [B, C, L]
    sr = int(decoded_audio["sample_rate"])
    vae_sr = int(getattr(sound_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                "h3kit_motion: previous_audio is %d Hz but the VAE wants %d Hz "
                "and torchaudio is not available to resample." % (sr, vae_sr))
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    want = int(round(seconds * vae_sr))
    have = int(waveform.shape[-1])
    if have < want:
        H3KIT_MOTION_LOG.warning("h3kit_motion: previous_audio is %.3fs, shorter than the "
                     "%.3fs of pinned video. Pinning what there is.",
                     have / vae_sr, seconds)
    else:
        waveform = waveform[..., have - want:]
    z = sound_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, int(z.shape[-1])


def split_av_streams(target_latent):
    """Unpack an H3 AV latent into its contained streams.

    NestedTensor.__getitem__ broadcasts the index into every contained
    tensor rather than selecting one, so samples[0] would strip the batch
    dimension off both streams. unbind() returns the pair.
    """
    samples = target_latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3kit_motion: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r" % type(samples))
    if not parts:
        raise ValueError("h3kit_motion: AV latent contains no streams")
    return parts


def video_stream(target_latent):
    """Pull the video stream out of an H3 AV latent."""
    video = split_av_streams(target_latent)[0]
    if video.ndim == 4:  # unbatched [C,T,H,W]
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("h3kit_motion: expected video latent [B,C,T,H,W], "
                         "got shape %s" % (tuple(video.shape),))
    return video


def reserve_motion_prefix(target_latent, previous_frames=17):
    """Reserve a head on the H3 grid, keeping the original delivery length."""
    previous = target_latent.get("h3kit_motion_length")
    if previous is not None:
        if previous[1] != previous_frames:
            raise ValueError("重叠帧数与已扩展的 latent 不一致，请连接本段尚未添加重叠区的原始 latent。")
        return target_latent, previous[0]

    video, decoded_audio = split_av_streams(target_latent)
    delivery_frames = count_pixel_frames(int(video.shape[2]))
    if previous_frames == 0:
        return target_latent, delivery_frames
    frames = delivery_frames + previous_frames
    frames += (5 - frames % 17) % 17
    video_t = (frames - 5) // 17 * 5 + 2
    audio_t = round(frames / H3KIT_NATIVE_FPS * H3KIT_AUDIO_RATE)
    audio_head = round(previous_frames / H3KIT_NATIVE_FPS * H3KIT_AUDIO_RATE)

    # Video is the sampler's starting canvas; audio may contain a fixed song.
    # Move that song and its mask together so head trimming keeps its start.
    out = target_latent.copy()
    out["samples"] = comfy.nested_tensor.NestedTensor((
        torch.nn.functional.pad(video, (0, 0, 0, 0, video_t - video.shape[2], 0)),
        torch.nn.functional.pad(decoded_audio, (audio_head,
                                       audio_t - decoded_audio.shape[-1] - audio_head)),
    ))
    if target_latent.get("noise_mask") is not None:
        masks = target_latent["noise_mask"].unbind()
        video_mask = comfy.utils.reshape_mask(masks[0], video.shape)
        audio_mask = comfy.utils.reshape_mask(masks[1], decoded_audio.shape)
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((
            torch.nn.functional.pad(video_mask,
                                    (0, 0, 0, 0, video_t - video.shape[2], 0), value=1),
            torch.nn.functional.pad(audio_mask,
                                    (audio_head, audio_t - decoded_audio.shape[-1] - audio_head),
                                    value=1),
        ))
    out["h3kit_motion_length"] = (delivery_frames, previous_frames)
    H3KIT_MOTION_LOG.info("h3kit_motion: extended %d to %d frames; trim %d head "
              "frames and keep %d", delivery_frames, frames, previous_frames, delivery_frames)
    return out, delivery_frames


def encode_motion_prefix(video_vae, frames, width, height, overlap_frames=17):
    """Re-encode the actual tail at cycle position zero for the new prefix."""
    available = int(frames.shape[0])
    if available < 1:
        raise ValueError("h3kit_motion: no previous frames available")
    tail = frames[-max(1, overlap_frames):, ..., :3]
    if tail.shape[1:3] != (height, width):
        tail = resize_reference_frames(tail, width, height, H3KIT_CROP)
    if overlap_frames == 0:
        return video_vae.encode(tail)
    if available < overlap_frames:
        tail = torch.cat((tail[:1].repeat(overlap_frames - available, 1, 1, 1), tail), dim=0)
    # 额外编码 5 帧以取得完整的 17 帧周期，随后丢掉补充部分。
    window = torch.cat((tail, tail[-1:].repeat(5, 1, 1, 1)), dim=0)
    encoded = video_vae.encode(window)
    prefix_tokens = overlap_frames // 17 * 5
    if encoded.ndim != 5 or encoded.shape[2] != prefix_tokens + 2:
        raise ValueError("重叠区编码长度不符合 H3 的 17 帧周期，请检查视频 VAE。")
    return encoded[:, :, :prefix_tokens].clone()


def match_upscaled_motion_context(video, reference, video_mask):
    """以 context 末端衔接高清内容，画面离开原位置时退回稳定残差。"""
    prefix = reference.shape[2]
    residual = reference.to(device=video.device, dtype=torch.float32) - video[:, :, :prefix].float()
    offset = residual.median(dim=2, keepdim=True).values
    variation = (residual - offset).abs().mean(dim=2, keepdim=True)
    # 只传递多个时刻共有的差异；动作/遮挡造成的不稳定残差不用于校正。
    offset = offset.sign() * (offset.abs() - variation).clamp_min(0)
    endpoint = residual[:, :, -1:]
    # H3 每5个时间步循环一次；比较同相位内容，避免把周期本身当作运动。
    cycle = video[:, :, prefix:prefix + 5].float().clone()
    for index in range(prefix, video.shape[2]):
        baseline = cycle[:, :, (index - prefix) % 5:(index - prefix) % 5 + 1]
        current = video[:, :, index:index + 1].float()
        relative_change = (current - baseline).square().mean(dim=1, keepdim=True)
        relative_change /= baseline.square().mean(dim=1, keepdim=True).clamp_min(1e-6)
        # 相对RMS变化25%时末端补差权重降至e^-1，抑制运镜后的固定位置残影。
        confidence = torch.exp(-16 * relative_change)
        correction = offset + confidence * (endpoint - offset)
        video[:, :, index:index + 1].add_(correction.to(video) * video_mask[:, :, index:index + 1].to(video))


def lock_motion_prefix(target_latent, reference):
    """Preserve the phase-aligned prefix through ComfyUI's inpaint mask."""
    video, decoded_audio = split_av_streams(target_latent)
    video = video.clone()
    masks = target_latent.get("noise_mask")
    if masks is not None:
        video_mask, audio_mask = masks.unbind()
        video_mask = comfy.utils.reshape_mask(video_mask, video.shape).clone()
    else:
        video_mask = torch.ones_like(video)
        audio_mask = torch.ones_like(decoded_audio)
    if target_latent.get("h3kit_upscaled_motion"):
        match_upscaled_motion_context(video, reference, video_mask)
        H3KIT_MOTION_LOG.info("h3kit_motion: matched upscaled context before high-resolution sampling")
    prefix = reference.shape[2]
    video[:, :, :prefix].copy_(reference.to(video))
    video_mask[:, :, :prefix] = 0
    out = target_latent.copy()
    out.pop("h3kit_upscaled_motion", None)
    out["samples"] = comfy.nested_tensor.NestedTensor((video, decoded_audio))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return out


def previous_audio_tail(target_latent, a_frames):
    """Slice the last `a_frames` worth of audio steps straight out of a
    generated H3 latent, skipping the decode -> re-encode round trip.

    Returns (tail latent [1, C, 2, rt], rt, overhang) where rt counts
    40 Hz latent steps and overhang is the signed fraction of a step by
    which the clip's audio grid overshoots its last pixel frame.

    H3 rounds the audio grid to the NEAREST step, not up, so overhang is
    negative for a third of legal clip lengths. 5/3 of a frame count
    lands on .0, .333 or .667 and never on .5, so there are exactly three
    cases:

        frames % 3 == 0   243 wants 405.00, allocates 405, overhang    0
        frames % 3 == 1   124 wants 206.67, allocates 207, overhang +1/3
        frames % 3 == 2   260 wants 433.33, allocates 433, overhang -1/3

    A positive overhang means the latent's final step reaches past the
    last frame, a negative one means it stops short. Either way the
    caller compensates the placement with it, so the pinned content lands
    where its samples actually sit. The decoded-audio path never sees
    this because match_tail cuts at the frame.
    """
    parts = split_av_streams(target_latent)
    if len(parts) < 2:
        raise ValueError(
            "h3kit_motion: previous_latent has no audio stream. Wire the "
            "sampler output of an H3 AV graph, not a video-only latent.")
    video, decoded_audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if decoded_audio.ndim == 3:  # unbatched [C,2,T]
        decoded_audio = decoded_audio.unsqueeze(0)
    if decoded_audio.ndim != 4:
        raise ValueError("h3kit_motion: expected audio latent [B,C,2,T], "
                         "got shape %s" % (tuple(decoded_audio.shape),))
    total_t = int(decoded_audio.shape[-1])
    frames = count_pixel_frames(int(video.shape[2]))
    delivery = target_latent.get("h3kit_motion_length")
    if delivery is not None:
        frames = delivery[0] + delivery[1]
        total_t = min(total_t, round(frames / H3KIT_NATIVE_FPS * H3KIT_AUDIO_RATE))
    overhang = total_t - H3KIT_AUDIO_FRAME_SCALE * frames
    # legal values are exactly 0, +1/3 and -1/3; the band is the widest
    # one that admits all three and still rejects a grid that is out by a
    # whole step or more
    if not (-0.5 < overhang < 0.5):
        H3KIT_MOTION_LOG.warning(
            "h3kit_motion: previous_latent audio grid is unexpected "
            "(%d steps for %d frames); assuming no overhang.", total_t, frames)
        overhang = 0.0
    rt = int(round(a_frames / float(H3KIT_NATIVE_FPS) * H3KIT_AUDIO_RATE))
    if rt > total_t:
        H3KIT_MOTION_LOG.warning("h3kit_motion: asked for %d audio steps, the latent "
                     "has %d. Pinning all of it.", rt, total_t)
        rt = total_t
    if rt < 1:
        raise ValueError("h3kit_motion: audio window is empty")
    tail = decoded_audio[:1, ..., total_t - rt:total_t].clone()
    return tail, rt, float(overhang)


class H3KitMotionBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_conditioning": ("CONDITIONING",),
                "video_vae": ("VAE",),
                "target_latent": ("LATENT",),
                "overlap_frames": ("INT", {
                    "default": 17, "min": 0, "max": 4080, "step": 17,
                    "tooltip": "0：只参考前段最后一帧，不增加重叠区；17、34、51……：增加对应帧数的重叠区，并输出给裁剪节点。非整倍数向下对齐到 17 的倍数。音频自动参考 max(24, 重叠帧数) 帧；素材不足时使用现有尾音。"}),
            },
            "optional": {
                "previous_frames": ("IMAGE", {
                    "tooltip": "上一段最终输出的画面。优先使用以避免重复解码 previous_latent，"
                               "按 overlap_frames 编码末尾画面；0 时只取最后一帧。"}),
                "previous_latent": ("LATENT", {
                    "tooltip": "前段采样器的原始音视频 latent。未接 previous_frames 时先解码其尾部画面；音频直接取 latent 尾部作为参考。"}),
                "sound_vae": ("VAE", {
                    "tooltip": "H3 audio VAE. Supply with previous_audio to "
                               "carry the previous clip's tail sound across "
                               "the join. Not needed when previous_latent is "
                               "wired."}),
                "previous_audio": ("AUDIO", {
                    "tooltip": "Audio of the previous clip. The tail "
                               "matching the pinned frames is encoded and "
                               "pinned alongside them. Ignored when "
                               "previous_latent is wired."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "LATENT", "INT")
    RETURN_NAMES = ("positive_conditioning", "prefix_frames", "target_latent", "delivery_frames")
    FUNCTION = "bridge_motion"
    CATEGORY = "H3 Upgrade Kit/视频续接"
    DESCRIPTION = ("重叠帧数可选 0、17、34、51……；0 时只参考前段尾帧，不增加长度、不锁定重叠区。"
                   "大于 0 时编码并锁定对应长度的前段尾部。音频参考长度自动取 max(24, 重叠帧数)。"
                   "latent 输出接采样器，prefix_frames 和 delivery_frames 输出接‘音画裁剪与拼接’。"
                   "裁剪后保留原定新增长度。没有前段上下文时保持原样。")

    def bridge_motion(self, positive_conditioning, video_vae, target_latent, overlap_frames=17,
              previous_frames=None, previous_latent=None, sound_vae=None,
              previous_audio=None):
        if previous_latent is None and previous_frames is None:
            return (positive_conditioning, 0, target_latent, 0)
        check_motion_layout()
        encode_mode, anchor_mode = "video", "head"
        audio_mode = H3KIT_AUDIO_MODE
        span = max(0, int(overlap_frames)) // 17 * 17
        video = video_stream(target_latent)
        width, height = int(video.shape[4]) * 16, int(video.shape[3]) * 16

        if previous_latent is not None:
            src_video = video_stream(previous_latent)
            if src_video.shape[1] != video.shape[1] or src_video.shape[3:] != video.shape[3:]:
                raise ValueError("h3kit_motion: previous_latent and target must "
                                 "have the same channels and resolution.")
            if previous_frames is None:
                decoded = video_vae.decode(src_video[:1])
                decoded = decoded.reshape(-1, *decoded.shape[-3:])
                delivery = previous_latent.get("h3kit_motion_length")
                if delivery is not None:
                    delivered, head = delivery
                    previous_frames = decoded[head:head + delivered]
                else:
                    previous_frames = decoded

        reference = encode_motion_prefix(video_vae, previous_frames, width, height, span)
        video_src, n = "phase-aligned pixels" if span else "last frame", max(1, span)
        target_latent, delivery_frames = reserve_motion_prefix(target_latent, span)
        if span:
            target_latent = lock_motion_prefix(target_latent, reference)
        frame_count = count_pixel_frames(int(video_stream(target_latent).shape[2]))

        # H3 supplies the 0,1,5,9,13 positions for this complete temporal block.
        # Keep one clip so condition noise augmentation does not restart per step.
        keyframes = [{
            "resolved_frame_index": 0,
            "latent": reference,
            "h3kit_motion": True,
        }]

        ref_audio_t = 0
        audio_ref = None
        audio_kf = None
        audio_end_frame = None
        a_frames = 0
        audio_src = "off"
        if previous_latent is not None or previous_audio is not None:
            # the audio window is independent of the video one: audio cond
            # rows cost rows but never cost delivered frames
            a_frames = max(24, span)
            if previous_latent is not None:
                if previous_audio is not None:
                    H3KIT_MOTION_LOG.info("h3kit_motion: both previous_latent and "
                              "previous_audio wired; using the latent (skips "
                              "one VAE round trip).")
                audio_latent, ref_audio_t, overhang = previous_audio_tail(
                    previous_latent, a_frames)
                audio_src = "latent"
            else:
                if sound_vae is None:
                    raise ValueError(
                        "h3kit_motion: previous_audio supplied without "
                        "audio_vae. Wire the H3 audio VAE, or wire "
                        "previous_latent instead.")
                audio_latent, ref_audio_t = encode_previous_audio(
                    sound_vae, previous_audio, a_frames / float(H3KIT_NATIVE_FPS))
                overhang = 0.0  # decoded audio was match_tail-cut at the frame
                audio_src = "vae"
            if audio_mode == "timeline":
                # end-align the audio window with the pinned video: both are
                # the tail of clip A, so both must end at the same instant
                # of the new timeline, frame `span` in head mode (where
                # A's last frame sits), frame 0 in before mode. On the
                # latent path the sliced content overshoots A's last
                # frame by `overhang` of a step, signed, because H3
                # rounds its audio grid to the nearest step and so falls
                # short as often as it reaches past. The end coordinate
                # moves by exactly that much, and a keyframe index is a
                # plain multiplier so it takes a fractional frame.
                end_frame = float(span if anchor_mode == "head" else 0)
                end_frame += overhang / H3KIT_AUDIO_FRAME_SCALE
                # then snap the window onto the target's own audio grid.
                # The end coordinate is FRAME_RESCALE * end_frame, and
                # FRAME_RESCALE is 5/3, so unless that product happens to
                # be a whole number the pinned rows land between the
                # integer coordinates the target's audio rows occupy. A
                # third of a step is 8.3 ms, which is the size of the
                # constant late offset measured on chained clips. Whether
                # it lands on or off the grid depends on the window
                # length, the path, and the clip's own grid overhang, so
                # it cycles rather than staying put. Rounding the end
                # coordinate to the nearest integer costs at most a third
                # of a step of placement and puts the pinned content on
                # the same grid as the sound being generated from it.
                end_coord = round(H3KIT_AUDIO_FRAME_SCALE * end_frame)
                end_frame = end_coord / H3KIT_AUDIO_FRAME_SCALE
                # Stock places a keyframe's audio window STARTING at
                # FRAME_RESCALE * index past the target origin and running
                # forward. We need it to END at the join, so the index is
                # the start of a window `ref_audio_t` steps wide:
                #
                #   start coord = FRAME_RESCALE * end_frame - ref_audio_t
                #   index       = end_frame - ref_audio_t / FRAME_RESCALE
                #
                # which is fractional, and negative whenever the window is
                # longer than the pinned head, which it normally is. Both
                # are legal arithmetic in the layout and neither is
                # reachable through the stock Add Guide node, so
                # layout_contract checks them before the first render.
                audio_kf = {
                    "resolved_frame_index": (end_frame
                                             - ref_audio_t / H3KIT_AUDIO_FRAME_SCALE),
                    "audio_latent": audio_latent,
                    "h3kit_motion": True,
                }
                audio_end_frame = end_frame
            else:
                # stock reference placement: the window sits in its own
                # span ahead of the target, which is what makes the model
                # imitate the sound rather than continue it. Kept as the
                # comparison the timeline mode is measured against.
                audio_ref = {
                    "kind": "audio",
                    "ref_audio_t": ref_audio_t,
                    "audio_latent": audio_latent,
                }

        # MERGE with any keyframes already on the conditioning instead of
        # replacing them. A last_frame anchor from the upstream node, or an
        # Add Guide anchor, is a legitimate companion to a chained head: the
        # pinned run decides how the clip starts, the anchor decides where
        # it ends. They need no special handling now that every keyframe
        # carries its real index, ours included, and stock compensates all
        # of them for references the same way.
        #
        # Anchors inside the pinned head cannot stay as hard timeline
        # anchors: the pinned run already decides those frames, and a
        # second block at the same coordinate would fight it.  A stock
        # single-image first_frame is also commonly the subject/wardrobe
        # identity source, though.  Preserve that information as a soft
        # image reference while removing only its conflicting frame-0
        # placement.  Multi-frame/interior guides are still dropped.
        head_end = max(1, span)
        tail_kfs = [audio_kf] if audio_kf is not None else []
        out = []
        dropped = []
        converted_identity = []
        for emb, extra in positive_conditioning:
            d = extra.copy()
            prior = [kf for kf in (d.get("minimax_keyframes") or [])
                     if not kf.get("h3kit_motion")]
            kept = []
            identity_refs = []
            for kf in prior:
                p = kf.get("resolved_frame_index", 0)
                if p > 0 and not d.get("h3kit_motion_prefix"):
                    kf = dict(kf)
                    p += span
                    kf["resolved_frame_index"] = p
                if p >= frame_count:
                    raise ValueError(
                        "h3kit_motion: the conditioning carries a "
                        "keyframe anchored at frame %s, but this clip is "
                        "only %d frames. Wire the conditioning and the "
                        "latent from the same node." % (p, frame_count))
                if p < head_end:
                    identity_latent = kf.get("latent")
                    if (p == 0 and getattr(identity_latent, "ndim", 0) == 5
                            and int(identity_latent.shape[2]) == 1):
                        identity_refs.append({
                            "kind": "image",
                            "latent_h": int(identity_latent.shape[-2]),
                            "latent_w": int(identity_latent.shape[-1]),
                            "latent": identity_latent,
                        })
                        converted_identity.append(p)
                        continue
                    dropped.append(p)
                    continue
                kept.append(dict(kf))
            d["minimax_keyframes"] = kept + keyframes + tail_kfs
            d["h3kit_motion_prefix"] = span
            if identity_refs:
                # Keep any Ref2VA blocks already present and append the
                # former first frame as an identity/appearance reference.
                # It then guides the whole clip without replacing the
                # previous clip's pinned motion at the join.
                d["minimax_refs"] = (list(d.get("minimax_refs") or [])
                                     + identity_refs)
            out.append([emb, d])
        if converted_identity:
            H3KIT_MOTION_LOG.info(
                "h3kit_motion: converted %d first_frame anchor(s) "
                "inside the pinned head to soft image identity reference(s).",
                len(converted_identity))
        if dropped:
            H3KIT_MOTION_LOG.warning(
                "h3kit_motion: dropped %d keyframe anchor(s) at "
                "frame(s) %s: the pinned head already decides frames "
                "0..%d. A last_frame anchor is kept.",
                len(dropped), sorted(set(dropped)), head_end - 1)

        if audio_ref is not None:
            out = node_helpers.conditioning_set_values(
                out, {"minimax_refs": [audio_ref]}, append=True)

        trim = span if anchor_mode == "head" else 0
        H3KIT_MOTION_LOG.info("h3kit_motion: video from %s, %s/%s, %d frames -> "
                  "one condition block, %d frame clip at %dx%d, "
                  "trim %d, audio %s",
                  video_src, encode_mode, anchor_mode, n,
                  frame_count, width, height, trim,
                  ("%d frames -> %d latent steps (%.3fs) from %s, %s"
                   % (a_frames, ref_audio_t, ref_audio_t / H3KIT_AUDIO_RATE, audio_src,
                      "on the timeline ending at frame %.3f" % audio_end_frame
                      if audio_end_frame is not None
                      else "stock ref placement"))
                  if ref_audio_t else "off")
        return (out, trim, target_latent, delivery_frames)
