import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const modelDirectory = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "assets",
  "models",
  "braille-cell-classifier-v2",
);
const manifest = JSON.parse(
  readFileSync(resolve(modelDirectory, "manifest.json"), "utf8"),
);
const labels = JSON.parse(
  readFileSync(resolve(modelDirectory, manifest.outputs[0].labelsFile), "utf8"),
);
const model = readFileSync(resolve(modelDirectory, manifest.artifact.file));
const nativeSelfTest = readFileSync(
  resolve(modelDirectory, manifest.nativeSelfTest.file),
);
const view = new DataView(model.buffer, model.byteOffset, model.byteLength);
const decoder = new TextDecoder();

const uint16 = (position) => view.getUint16(position, true);
const int8 = (position) => view.getInt8(position);
const int32 = (position) => view.getInt32(position, true);
const uint32 = (position) => view.getUint32(position, true);

function field(table, index) {
  const vtable = table - int32(table);
  const entry = vtable + 4 + index * 2;
  return entry + 2 <= vtable + uint16(vtable) ? uint16(entry) : 0;
}

function scalar(table, index, read, fallback) {
  const offset = field(table, index);
  return offset ? read(table + offset) : fallback;
}

function target(table, index) {
  const offset = field(table, index);
  if (!offset) return null;

  const position = table + offset;
  return position + uint32(position);
}

function stringAt(table, index) {
  const position = target(table, index);
  if (position === null) return null;

  const length = uint32(position);
  return decoder.decode(model.subarray(position + 4, position + 4 + length));
}

function vector(table, index) {
  const position = target(table, index);
  return position === null
    ? null
    : { data: position + 4, length: uint32(position) };
}

function tableVector(table, index) {
  const values = vector(table, index);
  if (!values) return [];

  return Array.from({ length: values.length }, (_, itemIndex) => {
    const position = values.data + itemIndex * 4;
    return position + uint32(position);
  });
}

function int32Vector(table, index) {
  const values = vector(table, index);
  return values
    ? Array.from({ length: values.length }, (_, itemIndex) =>
        int32(values.data + itemIndex * 4),
      )
    : [];
}

assert.equal(model.subarray(4, 8).toString("ascii"), "TFL3");
assert.equal(model.length, manifest.artifact.sizeBytes);
assert.equal(
  createHash("sha256").update(model).digest("hex"),
  manifest.artifact.sha256,
);
assert.equal(
  createHash("sha256").update(nativeSelfTest).digest("hex"),
  manifest.nativeSelfTest.sha256,
  "Hash fixture self-test native tidak cocok.",
);

const root = uint32(0);
assert.equal(
  scalar(root, 0, uint32, 0),
  manifest.artifact.tfliteSchemaVersion,
);
const [subgraph] = tableVector(root, 2);
assert.notEqual(subgraph, undefined, "Model tidak memiliki subgraph.");
const tensors = tableVector(subgraph, 0);
const [inputIndex] = int32Vector(subgraph, 1);
const [outputIndex] = int32Vector(subgraph, 2);
assert.notEqual(inputIndex, undefined, "Model tidak memiliki input tensor.");
assert.notEqual(outputIndex, undefined, "Model tidak memiliki output tensor.");
const input = tensors[inputIndex];
const output = tensors[outputIndex];
assert.notEqual(input, undefined);
assert.notEqual(output, undefined);

assert.equal(stringAt(input, 3), manifest.inputs[0].name);
assert.deepEqual(int32Vector(input, 0), manifest.inputs[0].shape);
assert.equal(scalar(input, 1, int8, 0), 0, "Input bukan FLOAT32.");
assert.equal(stringAt(output, 3), manifest.outputs[0].name);
assert.deepEqual(int32Vector(output, 0), manifest.outputs[0].shape);
assert.equal(scalar(output, 1, int8, 0), 0, "Output bukan FLOAT32.");
assert.equal(labels.length, manifest.classCount);
assert.equal(labels.length, manifest.outputs[0].shape.at(-1));
assert.equal(new Set(labels).size, labels.length, "Label harus unik.");
assert.deepEqual(
  labels,
  [..."abcdefghijklmnopqrstuvwxyz"],
  "Urutan label harus tepat a-z sesuai kontrak decoder.",
);
assert.deepEqual(manifest.inputs[0].shape, [1, 28, 28, 3]);
assert.equal(manifest.inputs[0].layout, "NHWC");
assert.equal(manifest.inputs[0].dtype, "float32");
assert.equal(manifest.inputs[0].channelOrder, "RGB");
assert.deepEqual(manifest.inputs[0].expectedValueRange, [0, 1]);
assert.equal(manifest.inputs[0].normalization, "pixel / 255.0");
assert.equal(manifest.inputs[0].resizeInterpolation, "bilinear");
assert.equal(
  labels[manifest.nativeSelfTest.expectedClassIndex],
  manifest.nativeSelfTest.expectedLabel,
  "Label fixture self-test native tidak cocok.",
);
assert.ok(
  Number.isFinite(manifest.nativeSelfTest.minimumConfidence) &&
    manifest.nativeSelfTest.minimumConfidence >= 0 &&
    manifest.nativeSelfTest.minimumConfidence <= 1,
  "Ambang confidence fixture self-test native tidak valid.",
);

console.log(
  `Model assets valid: ${manifest.inputs[0].shape.join("x")} -> ${labels.length} kelas, SHA-256 ${manifest.artifact.sha256.slice(0, 12)}...`,
);
