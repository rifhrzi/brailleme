// Applies the actual worklet-safe implementation on native-captured crop bytes on Node.
// Actual Android Hermes timings/bridge tests are collected by a separate harness.
const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');
const { load, root } = require('./geometry-reference.cjs');
const { resizeWordRawPixels } = load(path.join(root, 'src/ml/word-image-processing.ts'));
const directory = process.argv[2];
if (!directory) throw new Error('Pass the native evidence directory');
const native = JSON.parse(fs.readFileSync(path.join(directory, 'native.json'), 'utf8'));
const timings = [];
for (const row of native.cases) {
  const [x1, y1, x2, y2] = row.context_crop_pixels;
  const raw = fs.readFileSync(path.join(directory, `${row.id}-crop.raw`));
  const start = performance.now();
  const result = resizeWordRawPixels({ width: x2 - x1, height: y2 - y1, pixelFormat: 'BGRA', buffer: raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) }, true);
  timings.push({ id: row.id, desktop_node_elapsed_ms: performance.now() - start });
  fs.writeFileSync(path.join(directory, `${row.id}.ts.f32`), new Uint8Array(result.tensor.buffer));
}
fs.writeFileSync(path.join(directory, 'typescript-pixel-reference.json'), `${JSON.stringify({ scope: 'Actual TypeScript resampler running on desktop Node with native-captured RGBA crop bytes; not emulator timing.', timings }, null, 2)}\n`);
process.stdout.write(`TypeScript pixel reference: ${timings.length} cases\n`);
