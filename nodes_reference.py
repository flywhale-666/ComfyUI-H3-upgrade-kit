"""沿用参考转视频的输入，为后续片段替换提示词和长度。"""

from comfy_execution.graph_utils import GraphBuilder, is_link


class H3KitReferencePromptLength:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"rawLink": True, "lazy": True,
                    "tooltip": "接 MiniMax H3 参考转视频的正向输出，与 latent_image 来自同一个节点。"}),
                "latent_image": ("LATENT", {"rawLink": True, "lazy": True,
                    "tooltip": "接同一个参考转视频节点的 Latent 输出，不要接采样后的 latent。"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True,
                    "tooltip": "本段的新提示词，完整替换原提示词；参考素材及其编号沿用上游。"}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                    "tooltip": "本段新增帧数，24fps；按 H3 的 17k+5 规则向上对齐，例如 124 帧约 5.17 秒。"}),
            },
            "hidden": {"dynprompt": "DYNPROMPT"},
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "replace"
    CATEGORY = "H3 Upgrade Kit/条件"
    DESCRIPTION = "沿用上游 MiniMax H3 参考转视频的 CLIP、VAE、宽高和全部参考素材，只替换提示词与长度。输出接后段 K采的 positive 和 latent_image。会按新提示词重新编码参考条件，支持串联本节点。"

    def replace(self, positive, latent_image, prompt, length, dynprompt):
        visited = set()
        while True:
            if (not is_link(positive) or not is_link(latent_image)
                    or positive[0] != latent_image[0] or positive[1] != 0 or latent_image[1] != 1):
                raise ValueError("正向和 Latent 必须直接连接同一个 MiniMax H3 参考转视频节点，或同一个 H3Kit 提示词与长度替换节点。")
            source_id = positive[0]
            if source_id in visited:
                raise ValueError("提示词与长度替换节点存在循环连接，请检查正向和 Latent 的来源。")
            visited.add(source_id)
            source = dynprompt.get_node(source_id)
            if source["class_type"] == "MiniMaxH3ReferenceToVideo":
                break
            if source["class_type"] != "H3KitReferencePromptLength":
                raise ValueError("请连接 MiniMax H3 参考转视频的原始输出；不支持经过采样、条件合并或循环回传后的输出。")
            positive = source["inputs"]["positive"]
            latent_image = source["inputs"]["latent_image"]

        # 从原始输入重新编码，保留 Qwen 所需的参考画面、音频标签和顺序。
        inputs = source["inputs"].copy()
        inputs.update(prompt=prompt, length=length)
        graph = GraphBuilder()
        reference = graph.node("MiniMaxH3ReferenceToVideo", **inputs)
        return {"result": (reference.out(0), reference.out(1)), "expand": graph.finalize()}
