import assert from 'node:assert/strict';

/** Read the public TFLite FlatBuffer tensor contract without executing the model. */
export function inspectTflite(model) {
  assert.ok(model.byteLength >= 8, 'TFLite header is truncated');
  assert.equal(model.subarray(4, 8).toString('ascii'), 'TFL3');
  const view = new DataView(model.buffer, model.byteOffset, model.byteLength);
  const u16 = (p) => view.getUint16(p, true);
  const i32 = (p) => view.getInt32(p, true);
  const u32 = (p) => view.getUint32(p, true);
  const field = (table, index) => {
    const vtable = table - i32(table), entry = vtable + 4 + index * 2;
    return entry + 2 <= vtable + u16(vtable) ? u16(entry) : 0;
  };
  const target = (table, index) => {
    const offset = field(table, index);
    return offset ? table + offset + u32(table + offset) : null;
  };
  const vector = (table, index, read) => {
    const pos = target(table, index);
    if (pos === null) return [];
    const count = u32(pos);
    assert.ok(count <= Math.floor((model.byteLength - pos - 4) / 4), 'TFLite vector exceeds buffer');
    return Array.from({ length: count }, (_, i) => read(pos + 4 + i * 4));
  };
  const tableVector = (table, index) => vector(table, index, (p) => p + u32(p));
  const tensor = (table) => {
    const name = target(table, 3), dtype = field(table, 1);
    return {
      name: name === null ? null : model.subarray(name + 4, name + 4 + u32(name)).toString('utf8'),
      shape: vector(table, 0, i32),
      type: dtype ? view.getInt8(table + dtype) : 0,
    };
  };
  const root = u32(0), version = field(root, 0);
  const graphs = tableVector(root, 2);
  assert.ok(graphs.length > 0, 'TFLite has no subgraph');
  const graph = graphs[0], tensors = tableVector(graph, 0);
  const readTensor = (index) => {
    assert.ok(Number.isInteger(index) && index >= 0 && index < tensors.length, 'TFLite tensor index is invalid');
    return tensor(tensors[index]);
  };
  return {
    schemaVersion: version ? u32(root + version) : 0,
    inputs: vector(graph, 1, i32).map(readTensor),
    outputs: vector(graph, 2, i32).map(readTensor),
  };
}
