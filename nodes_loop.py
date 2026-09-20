"""基于 ComfyUI 原生循环的多路独立回传与收集。"""

from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder, is_link
from comfy_extras.nodes_loop import StartLoop, EndLoop, LoopProgress
from server import PromptServer


# 与 ComfyUI Autogrow 的端口上限一致；前端只显示已使用的路和一个空位。
MAX_CHANNELS = io.Autogrow._MaxNames


def channel_inputs(group, prefix, tooltip):
    return io.Autogrow.Input(
        group,
        template=io.Autogrow.TemplateNames(
            io.AnyType.Input(prefix, tooltip=tooltip),
            names=[f"{prefix}{i}" for i in range(1, MAX_CHANNELS + 1)],
            min=0,
        ),
        optional=True,
    )


def channel_sources(inputs, group, prefix):
    return {i: inputs[f"{group}.{prefix}{i}"] for i in range(1, MAX_CHANNELS + 1)
            if f"{group}.{prefix}{i}" in inputs}


def expand_loop(dynprompt, opener_id, body, close_id, values, list_items, reuse_cache):
    graph = GraphBuilder()
    loop_metadata = {}
    close_inputs = dynprompt.get_node(close_id)["inputs"]
    outputs = channel_sources(close_inputs, "output_values", "output_value")
    next_values = channel_sources(close_inputs, "next_values", "next_iteration_value")
    carry = channel_sources(dynprompt.get_node(opener_id)["inputs"], "initial_values", "initial_iteration_value")
    terminations = [value for name, value in close_inputs.items()
                    if name.startswith("terminations.") and is_link(value)]
    accumulate = close_inputs.get("accumulate", False)
    previous_dependencies = []
    previous_progress = None
    result_inputs = {"close_id": close_id}

    for position, value in enumerate(values):
        iteration = graph.node(
            "H3KitLoopIteration", f"iteration_{position}",
            iteration_index=value, is_first=position == 0, is_last=position == len(values) - 1,
            list_item=list_items[position] if list_items is not None else None,
            reuse_cache=reuse_cache,
            **{f"value{i}": source for i, source in carry.items()},
            **{f"dependency{i}": source for i, source in enumerate(previous_dependencies)},
        )
        iteration.set_override_display_id(opener_id)
        copies = {}
        for node_id in body:
            original = dynprompt.get_node(node_id)
            copy = graph.node(original["class_type"], f"{position}_{node_id}")
            copy.set_override_display_id(node_id)
            copies[node_id] = copy

        def copied_link(source):
            if not is_link(source):
                return source
            if source[0] == opener_id:
                return iteration.out(source[1])
            if source[0] in copies:
                return copies[source[0]].out(source[1])
            return source

        for node_id, copy in copies.items():
            original = dynprompt.get_node(node_id)
            for name, input_value in original.get("inputs", {}).items():
                copy.set_input(name, copied_link(input_value))
            if "_loop_end" in original:
                loop_metadata[copy.id] = {
                    "_loop_body": [copies[body_id].id for body_id in original["_loop_body"]],
                    "_loop_end": copies[original["_loop_end"]].id,
                }

        dependencies = []
        for channel, source in outputs.items():
            copied_output = copied_link(source)
            if accumulate or position == len(values) - 1:
                result_inputs[f"output_{channel}_{position}"] = copied_output
            if is_link(copied_output):
                dependencies.append(copied_output)
        for channel, source in next_values.items():
            carry[channel] = copied_link(source)
            if is_link(carry[channel]):
                dependencies.append(carry[channel])
        dependencies.extend(copied_link(source) for source in terminations)
        previous_dependencies = dependencies
        progress_inputs = {
            "start_id": opener_id, "position": position + 1, "total": len(values),
            **{f"dependency{i}": source for i, source in enumerate(dependencies)},
        }
        if previous_progress is not None:
            progress_inputs["previous_progress"] = previous_progress
        progress = graph.node("H3KitLoopProgress", f"progress_{position}", **progress_inputs)
        previous_progress = progress.out(0)

    if previous_progress is not None:
        result_inputs["progress"] = previous_progress
    result_inputs.update({f"dependency{i}": source for i, source in enumerate(previous_dependencies)})
    graph.node("H3KitLoopResult", "result", **result_inputs)
    expanded = graph.finalize()
    for node_id, metadata in loop_metadata.items():
        expanded[node_id].update(metadata)
    return expanded


class H3KitStartLoop(StartLoop):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3KitStartLoop"
        schema.display_name = "H3Kit Start Loop 多路循环开始"
        schema.category = "H3 Upgrade Kit/循环"
        schema.description = "接入一路初始值后自动增加空端口。current_iteration_valueN 对应初始值 N，后续轮次接收 End Loop 的 next_iteration_valueN。"
        schema.inputs = [item for item in schema.inputs if item.id != "initial_iteration_value"]
        schema.inputs.append(channel_inputs("initial_values", "initial_iteration_value", "第 N 路初始值；图片、音频、latent 可分别连接。"))
        schema.outputs = schema.outputs[:4] + [
            io.AnyType.Output(f"current_iteration_value{i}", is_output_list=True)
            for i in range(1, MAX_CHANNELS + 1)
        ]
        return schema

    @classmethod
    def execute(cls, mode, cache_iterations=False, parent_iteration=None, initial_values=None):
        selected_mode = mode.get("mode", ["simple"])[0]
        list_items = None
        if selected_mode == "simple":
            values = list(range(mode.get("num_iterations", [4])[0]))
        elif selected_mode == "For":
            step = mode.get("step", [1])[0]
            if step == 0:
                raise ValueError("循环步长不能为 0。")
            values = list(range(mode.get("start_iteration_index", [0])[0], mode.get("max_iteration", [4])[0], step))
        else:
            list_items = mode["list"]
            values = list(range(len(list_items)))

        dynprompt = cls.hidden.dynprompt
        execution_list = cls.hidden.execution_list
        unique_id = cls.hidden.unique_id
        loop = dynprompt.get_node(unique_id)
        body = set(loop["_loop_body"])
        close_id = loop["_loop_end"]
        close = dynprompt.get_node(close_id)
        if close["class_type"] != "H3KitEndLoop":
            raise ValueError("H3Kit Start Loop 必须配合 H3Kit End Loop 使用。")
        reuse_cache = cache_iterations[0] if isinstance(cache_iterations, list) else cache_iterations
        graph = expand_loop(dynprompt, unique_id, body, close_id, values, list_items, reuse_cache)
        close_inputs = {name: value for name, value in close["inputs"].items()
                        if not name.startswith(("output_values.", "next_values.", "terminations."))}
        execution_list.add_node(close_id)
        execution_list.add_external_block(close_id)
        execution_list.inhibit_nodes(body)
        dynprompt.override_node(close_id, {"class_type": close["class_type"], "inputs": close_inputs})
        PromptServer.instance.send_progress_text(f"Iteration 0 / {len(values)}", unique_id)
        return io.NodeOutput(None, False, not values, None, *([None] for _ in range(MAX_CHANNELS)), expand=graph)


class H3KitEndLoop(EndLoop):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3KitEndLoop"
        schema.display_name = "H3Kit End Loop 多路循环结束"
        schema.category = "H3 Upgrade Kit/循环"
        schema.description = "output_valueN 分别从 outputsN 输出；next_iteration_valueN 回传到开始节点的第 N 路。accumulate 开启时每路独立收集全部轮次，关闭时只输出最后一轮。"
        schema.inputs = [
            channel_inputs("output_values", "output_value", "本路的最终输出；accumulate 开启时按轮次收集为独立列表。"),
            channel_inputs("next_values", "next_iteration_value", "下一轮的第 N 路当前值；未连接的路保留原值。"),
            *[item for item in schema.inputs if item.id in ("accumulate", "terminations")],
        ]
        schema.outputs = [io.AnyType.Output(f"outputs{i}", is_output_list=True)
                          for i in range(1, MAX_CHANNELS + 1)]
        return schema

    @classmethod
    def execute(cls, accumulate, **kwargs):
        channels = cls.hidden.execution_list.get_external_block_result(cls.hidden.unique_id)
        collect = accumulate[0] if isinstance(accumulate, list) else accumulate
        return io.NodeOutput(*(
            [value for iteration in (outputs if collect else outputs[-1:]) for value in iteration]
            for outputs in channels
        ))


class H3KitLoopIteration(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3KitLoopIteration", is_input_list=True, is_dev_only=True, accept_all_inputs=True,
            inputs=[io.Int.Input("iteration_index"), io.Boolean.Input("is_first"),
                    io.Boolean.Input("is_last"), io.Boolean.Input("reuse_cache"),
                    io.AnyType.Input("list_item", optional=True)],
            outputs=[io.Int.Output(), io.Boolean.Output(), io.Boolean.Output(), io.AnyType.Output(),
                     *[io.AnyType.Output(f"value{i}", is_output_list=True) for i in range(1, MAX_CHANNELS + 1)]],
        )

    @classmethod
    def execute(cls, iteration_index, is_first, is_last, reuse_cache, list_item=None, **kwargs):
        return io.NodeOutput(iteration_index[0], is_first[0], is_last[0], list_item[0] if list_item else None,
                             *(kwargs.get(f"value{i}", [None]) for i in range(1, MAX_CHANNELS + 1)))

    @classmethod
    def fingerprint_inputs(cls, reuse_cache, **kwargs):
        return None if reuse_cache[0] else float("NaN")


class H3KitLoopProgress(LoopProgress):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3KitLoopProgress"
        return schema


class H3KitLoopResult(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3KitLoopResult", is_input_list=True, is_output_node=True,
            is_dev_only=True, accept_all_inputs=True, inputs=[io.String.Input("close_id")],
            outputs=[], hidden=[io.Hidden.execution_list],
        )

    @classmethod
    def execute(cls, close_id, **kwargs):
        channels = [[] for _ in range(MAX_CHANNELS)]
        output_names = sorted((name for name in kwargs if name.startswith("output_")),
                              key=lambda name: tuple(map(int, name.split("_")[1:])))
        for name in output_names:
            channel = int(name.split("_")[1])
            channels[channel - 1].append(kwargs[name])
        cls.hidden.execution_list.release_external_block(close_id[0], channels)
        return io.NodeOutput()

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("NaN")
