"""按选定标签拆分文本，支持首段独立输出和全部段落直接输出。"""

import re


TEXT_SPLIT_TAGS = {"空行": r"\n[^\S\n]*\n", "****": r"\*{4,}", "----": r"-{4,}", "~~~~": r"~{4,}"}


def text_segments(text, split_tag="空行"):
    if split_tag not in TEXT_SPLIT_TAGS:
        raise ValueError("分段标签请选择空行、****、---- 或 ~~~~。")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    segments = [part.strip() for part in re.split(TEXT_SPLIT_TAGS[split_tag], normalized) if part.strip()]
    if not segments:
        raise ValueError("输入文本为空，请填写至少一段文本。")
    return segments


class H3KitTextSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False,
                    "tooltip": "仅按选中的 split_tag 分段；段内换行和未选中的分隔符保留。"}),
                "split_tag": (list(TEXT_SPLIT_TAGS), {"default": "空行",
                    "tooltip": "分段标签：空行、****、----、~~~~。符号连续至少 4 个才生效；选择符号时，空行不分段。"}),
                "segment_index_from_second": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "从第二段开始的索引：0 是第二段，1 是第三段。第一段始终从 first_text 独立输出。"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("first_text", "text", "segment_count")
    FUNCTION = "split"
    CATEGORY = "H3 Upgrade Kit/文本"
    DESCRIPTION = "仅按选中的分段标签拆分文本：空行、****、---- 或 ~~~~，符号连续至少 4 个才生效。first_text 固定输出第一段，text 从第二段起按索引选择；索引 0 对应第二段。segment_count 仅统计后续段数；只有一段时 text 为空字符串。"

    def split(self, text, segment_index_from_second=0, split_tag="空行"):
        segments = text_segments(text, split_tag)
        remaining_count = len(segments) - 1
        if not isinstance(segment_index_from_second, int) or not 0 <= segment_index_from_second < max(1, remaining_count):
            raise ValueError(f"后续文本索引超出范围：共 {remaining_count} 段，请选择 0 至 {max(0, remaining_count - 1)}。")
        selected = segments[segment_index_from_second + 1] if remaining_count else ""
        return segments[0], selected, remaining_count


class TextSegmentOutputTypes(tuple):
    def __getitem__(self, index):
        if isinstance(index, int) and index >= 0:
            index = 0
        return super().__getitem__(index)


class H3KitTextSegments:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = H3KitTextSegment.INPUT_TYPES()["required"]
        return {"required": {name: inputs[name] for name in ("text", "split_tag")}}

    RETURN_TYPES = TextSegmentOutputTypes(("STRING",))
    RETURN_NAMES = ("1_text",)
    FUNCTION = "split"
    CATEGORY = "H3 Upgrade Kit/文本"
    OUTPUT_NODE = True
    DESCRIPTION = "仅按选中的分段标签拆分文本：空行、****、---- 或 ~~~~，符号连续至少 4 个才生效。直接输出全部段落，不需要循环或索引；首次运行后展开 1_text、2_text、3_text 等端口。不产生空段。"

    def split(self, text, split_tag="空行"):
        segments = text_segments(text, split_tag)
        return {"ui": {"segment_count": [len(segments)]}, "result": tuple(segments)}
