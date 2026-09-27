"""高清分块采样节点的完整生成模式：保留全局注意力，分批计算激活。"""

from functools import partial
import math

import comfy.model_base
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention


def chunked_mlp(forward, x, *, chunks):
    chunk_size = max(1, math.ceil(x.shape[0] / chunks))
    output = None
    for start in range(0, x.shape[0], chunk_size):
        prediction = forward(x[start:start + chunk_size])
        if output is None:
            output = prediction.new_empty((x.shape[0], *prediction.shape[1:]))
        output[start:start + prediction.shape[0]].copy_(prediction)
        del prediction
    return output


def chunked_attention(forward, q, k, v, heads, *, chunks, **kwargs):
    if isinstance(q, AttentionTensorContainer):
        q, k, v = q.take(), k.take(), v.take()
    # H3 的原生注意力没有 mask；外部补丁引入 mask 时保留它的原始语义。
    if kwargs.get("mask") is not None:
        return forward(q, k, v, heads, **kwargs)
    length = q.shape[-2]
    chunk_size = max(1, math.ceil(length / chunks))
    output = None
    for start in range(0, length, chunk_size):
        # 只切 query，每个 query 仍能看到所有视频、音频和条件的 key/value。
        prediction = forward(q[..., start:start + chunk_size, :], k, v, heads, **kwargs)
        if output is None:
            shape = list(prediction.shape)
            shape[-2] = length
            output = prediction.new_empty(shape)
        output[..., start:start + prediction.shape[-2], :].copy_(prediction)
        del prediction
    return output


def create_global_patcher(model, chunks=4):
    if not isinstance(model.model, comfy.model_base.MiniMaxH3):
        raise ValueError("H3Kit: full-context sampling requires a MiniMax H3 model")
    if not 2 <= chunks <= 8:
        raise ValueError("H3Kit: full-context sampling chunks must be between 2 and 8")
    patched = model.clone()
    blocks = model.get_model_object("diffusion_model.blocks")
    for index in range(len(blocks)):
        prefix = f"diffusion_model.blocks.{index}"
        mlp_path = f"{prefix}.mlp.forward"
        patched.add_object_patch(mlp_path, partial(
            chunked_mlp, model.get_model_object(mlp_path), chunks=chunks))
        attention_path = f"{prefix}.attn.comfy_attention.function"
        attention = model.get_model_object(attention_path)
        patched.add_object_patch(attention_path, partial(
            chunked_attention, optimized_attention if attention is None else attention, chunks=chunks))
    return patched
