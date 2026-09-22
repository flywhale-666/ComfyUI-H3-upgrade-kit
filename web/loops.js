import { app } from "../../../scripts/app.js";

const layouts = {
    H3KitStartLoop: { input: "initial_iteration_value", output: "current_iteration_value", offset: 4, groups: ["initial_values"], inputOrder: ["parent_iteration", "initial_values"] },
    H3KitEndLoop: { input: "output_value", output: "outputs", offset: 0, groups: ["output_values", "next_values"], inputOrder: ["output_values", "next_values", "terminations"] },
};

function sortInputs(node, layout) {
    const rank = input => {
        const index = layout.inputOrder.indexOf(input.name.split(".")[0]);
        return index < 0 ? layout.inputOrder.length : index;
    };
    const sorted = [...node.inputs].sort((a, b) => {
        const groupOrder = rank(a) - rank(b);
        if (groupOrder) return groupOrder;
        if (rank(a) === layout.inputOrder.length) return 0;
        return a.name.localeCompare(b.name, undefined, { numeric: true });
    });
    if (sorted.every((input, index) => input === node.inputs[index])) return false;
    // 新版 input.link 按当前槽位查询，必须在重排前保存连线归属。
    const links = new Map(node.inputs.map((input, index) => [input, node.getInputLink(index)]));
    const moved = sorted.flatMap((input, index) => {
        const link = links.get(input);
        return link && link.target_slot !== index ? [{ link, index }] : [];
    });
    // 先移到空槽位，避免逐条更新时与尚未移动的连线争用目标槽位。
    for (const [temporary, { link }] of moved.entries()) {
        link.target_slot = node.inputs.length + temporary;
    }
    node.inputs.splice(0, node.inputs.length, ...sorted);
    for (const { link, index } of moved) {
        link.target_slot = index;
    }
    return true;
}

// 输入由 ComfyUI Autogrow 管理；按组整理新增端口，并同步输出数量。
function syncSlots(node, layout, nodeData) {
    let changed = sortInputs(node, layout);
    for (const group of layout.groups) {
        const slots = node.inputs.filter(input => input.name.startsWith(`${group}.`));
        const lastConnected = slots.findLastIndex(input => input.link != null);
        // 只收起末尾多余空位，不删除中间空位，不改变数据通道编号。
        for (let i = slots.length - 1; i > lastConnected + 1; i--) {
            node.removeInput(node.inputs.indexOf(slots[i]));
            changed = true;
        }
    }
    const pattern = new RegExp(`(?:^|\\.)${layout.input}(\\d+)$`);
    const limit = nodeData.output.length - layout.offset;
    let count = 1;
    for (const input of node.inputs ?? []) {
        const match = input.name.match(pattern);
        if (match) count = Math.max(count, Number(match[1]));
    }
    for (let slot = layout.offset; slot < (node.outputs?.length ?? 0); slot++) {
        if (node.outputs[slot].links?.length) count = Math.max(count, slot - layout.offset + 1);
    }
    count = Math.min(count, limit);
    const target = layout.offset + count;
    while (node.outputs.length > target) {
        node.removeOutput(node.outputs.length - 1);
        changed = true;
    }
    while (node.outputs.length < target) {
        const channel = node.outputs.length - layout.offset + 1;
        node.addOutput(`${layout.output}${channel}`, "*", { shape: 6 });
        changed = true;
    }
    if (changed) {
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "H3Kit.MultiChannelLoop",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const layout = layouts[nodeData.name];
        if (!layout) return;
        if (nodeData.name === "H3KitStartLoop") {
            const addCustomWidget = nodeType.prototype.addCustomWidget;
            nodeType.prototype.addCustomWidget = function (widget, ...args) {
                if (widget.type === "progressText") {
                    widget.props = {
                        ...widget.props,
                        style: [widget.props?.style, { color: "#e5e7eb" }],
                    };
                }
                return addCustomWidget.call(this, widget, ...args);
            };
        }
        function scheduleSync() {
            if (this._h3kitLoopSyncPending) return;
            this._h3kitLoopSyncPending = true;
            setTimeout(() => {
                this._h3kitLoopSyncPending = false;
                syncSlots(this, layout, nodeData);
            }, 0);
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = onNodeCreated?.apply(this, args);
            // 原生 Autogrow 在创建输入时安装实例回调，因此在节点创建完成后包装。
            const onConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function (type, slot, connected, ...rest) {
                const group = this.inputs[slot]?.name.split(".")[0];
                const autogrow = this.comfyDynamic?.autogrow;
                const template = autogrow?.[group];
                const preserveChannel = type === 1 && !connected && layout.groups.includes(group) && template;
                if (preserveChannel) delete autogrow[group];
                try {
                    return onConnectionsChange?.call(this, type, slot, connected, ...rest);
                } finally {
                    if (preserveChannel) autogrow[group] = template;
                    scheduleSync.call(this);
                }
            };
            scheduleSync.call(this);
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            scheduleSync.call(this);
            return result;
        };
    },
});
