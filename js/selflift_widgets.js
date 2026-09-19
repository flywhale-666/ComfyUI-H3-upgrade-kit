import { app } from "../../scripts/app.js";

// 按旧版序列迁移，避免新增采样器、移动降噪后让已有工作流的参数错位。
export function migrateSelfLiftWidgets(graphData) {
    const graphs = [graphData, ...(graphData?.definitions?.subgraphs ?? [])];
    for (const graph of graphs) {
        for (const node of graph?.nodes ?? []) {
            if (node.type !== "H3KitSelfLiftSampler") continue;
            let values = node.widgets_values;
            if (Array.isArray(values) && typeof values[4] === "string"
                && typeof values[5] === "number" && typeof values[7] === "string") {
                const [seed, control, steps, cfg, scheduler, highSteps, lowScale, weights,
                    overlap, continueAudio, denoise = 1.0] = values;
                values = node.widgets_values = [seed, control, steps, cfg, "euler", scheduler,
                    denoise, highSteps, lowScale, weights, overlap, continueAudio];
                if (node.widgets_values_named) {
                    node.widgets_values_named.sampler_name ??= "euler";
                    node.widgets_values_named.denoise ??= denoise;
                }
            }
            if (Array.isArray(values) && typeof values[5] === "string" && Number.isFinite(values[10])) {
                values[10] = Math.max(17, Math.floor(values[10] / 17) * 17);
            }
            const named = node.widgets_values_named;
            // 模式已移除，旧工作流末尾的模式值不再对应任何控件。
            if (Array.isArray(values) && ["strict", "drift_control", "tail_guide"].includes(values[14])) {
                values.splice(14, 1);
            }
            if (named) delete named.continuation_mode;
            if (named && Number.isFinite(named.overlap_frames)) {
                named.overlap_frames = Math.max(17, Math.floor(named.overlap_frames / 17) * 17);
            }
        }
    }
}

app.registerExtension({
    name: "H3Kit.SelfLiftWidgetOrder",
    beforeConfigureGraph(graphData) {
        migrateSelfLiftWidgets(graphData);
    },
});
