import { app } from "../../../scripts/app.js";

const layouts = {
    H3KitAudioSegments: { types: ["AUDIO", "INT", "FLOAT"], names: ["audio", "frame_count", "duration"], property: "h3kit_audio_segment_count" },
    H3KitTextSegments: { types: ["STRING"], names: ["text"], property: "h3kit_text_segment_count" },
};

function syncOutputs(node, count, layout) {
    count = Number.isSafeInteger(count) && count > 0 ? count : 1;
    node.properties[layout.property] = count;
    const width = layout.types.length;
    const target = count * width;
    while (node.outputs.length > target) {
        node.removeOutput(node.outputs.length - 1);
    }
    while (node.outputs.length < target) {
        const slot = node.outputs.length;
        node.addOutput(`${Math.floor(slot / width) + 1}_${layout.names[slot % width]}`, layout.types[slot % width]);
    }
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "H3Kit.AudioSegments",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const layout = layouts[nodeData.name];
        if (!layout) return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            syncOutputs(this, message.segment_count?.[0], layout);
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            syncOutputs(this, this.properties[layout.property] ?? Math.ceil(this.outputs.length / layout.types.length), layout);
            return result;
        };
    },
});
