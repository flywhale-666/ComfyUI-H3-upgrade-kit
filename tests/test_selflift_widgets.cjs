// 检查旧工作流在移动控件后是否仍保留原始参数，且重复载入不会再次改动。
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const source = fs.readFileSync(path.join(__dirname, "../js/selflift_widgets.js"), "utf8")
    .replace(/^import .*;\r?\n/m, "").replace("export function", "function");
let extension;
vm.runInNewContext(source, { app: { registerExtension(value) { extension = value; } } });
const base = [42, "fixed", 8, 1, "beta", 2, .5, "weights.safetensors", 22, true];
const node = (values) => ({ type: "H3KitSelfLiftSampler", widgets_values: values });
const old = node([...base]);
const withDenoise = node([...base, .65]);
const other = { type: "KSampler", widgets_values: [...base] };
const tiled = node([42, "fixed", 8, 1, "euler", "beta", .8, 2, .5, "weights.safetensors", 39, true, true, 4]);
const modes = ["strict", "drift_control", "tail_guide"].map((mode) => ({
    ...node([42, "fixed", 8, 1, "euler", "beta", 1, 4, .5, "weights.safetensors", 17, true, true, 4, mode]),
    widgets_values_named: { continuation_mode: mode, minimum_tiles: 4 },
}));
tiled.widgets_values_named = { overlap_frames: 39, spatial_tiles: true };
const graph = { nodes: [old, other, tiled, ...modes], definitions: { subgraphs: [{ nodes: [withDenoise] }] } };
extension.beforeConfigureGraph(graph);
assert.equal(JSON.stringify(old.widgets_values), JSON.stringify([42, "fixed", 8, 1, "euler", "beta", 1, 2, .5, "weights.safetensors", 17, true]));
assert.equal(tiled.widgets_values[10], 34);
assert.equal(tiled.widgets_values_named.overlap_frames, 34);
for (const migratedMode of modes) {
    assert.equal(migratedMode.widgets_values.length, 14);
    assert.equal("continuation_mode" in migratedMode.widgets_values_named, false);
    assert.equal(migratedMode.widgets_values_named.minimum_tiles, 4);
    assert.deepEqual(migratedMode.widgets_values.slice(11), [true, true, 4]);
}
assert.deepEqual(tiled.widgets_values.slice(11), [true, true, 4]);
assert.equal(withDenoise.widgets_values[6], .65);
assert.deepEqual(other.widgets_values, base);
const migrated = JSON.stringify(graph);
extension.beforeConfigureGraph(graph);
assert.equal(JSON.stringify(graph), migrated);
console.log("SelfLift widget migration: PASS (legacy, denoise, subgraph, unchanged other nodes, idempotent)");
