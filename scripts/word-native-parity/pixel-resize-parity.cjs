// Compare the exact production Java core with actual TS and independent Pillow.
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { Buffer } = require('node:buffer');
const { performance } = require('node:perf_hooks');
const { load, root, loadedSourcePaths } = require('./geometry-reference.cjs');
const [javac, java, python, work, reportPath] = process.argv.slice(2);
if (!reportPath) throw new Error('Usage: node pixel-resize-parity.cjs <javac> <java> <python> <new-work-dir> <new-report.json>');
assert.equal(os.endianness(), 'LE');
fs.mkdirSync(work, { recursive: false });
function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, encoding: 'utf8', windowsHide: true });
  assert.equal(result.status, 0, `${command}: ${result.stderr || result.error}`);
  return result.stdout;
}
const core = 'modules/braille-word-geometry/android/src/main/java/com/margelo/nitro/braillewordgeometry/WordPixelResize.java';
const cli = 'scripts/word-native-parity/WordPixelResizeCli.java';
run(javac, ['-d', work, path.join(root, core), path.join(root, cli)]);
const safety = JSON.parse(run(java, ['-cp', work, 'WordPixelResizeCli', '--self-test']));
const cases = [
  ['identity-rgb', 480, 192, 'RGB', false], ['identity-bgra', 480, 192, 'BGRA', false],
  ['android-bgra', 480, 192, 'BGRA', true], ['upscale-rgb', 3, 2, 'RGB', false],
  ['downscale-rgb', 960, 384, 'RGB', false], ['mixed-dimensions', 317, 209, 'RGB', false],
  ['prime-downscale', 641, 257, 'RGBA', false], ['constant-pixel', 1, 1, 'RGB', false],
  ['single-column', 1, 73, 'ARGB', false], ['single-row', 73, 1, 'ABGR', false],
  ['large-preview', 1920, 768, 'BGRA', true], ['bgr', 31, 23, 'BGR', false],
  ['xrgb', 23, 31, 'XRGB', false], ['xbgr', 23, 31, 'XBGR', false],
  ['bgrx', 31, 23, 'BGRX', true], ['rgbx', 31, 23, 'RGBX', false],
].map(([name, width, height, format, androidBitmap]) => ({ name, width, height, format, androidBitmap, input: `${name}.input` }));
let random = 123456789;
for (const row of cases) {
  const bytes = Buffer.alloc(row.width * row.height * (['RGB', 'BGR'].includes(row.format) ? 3 : 4));
  for (let i = 0; i < bytes.length; i++) { random ^= random << 13; random ^= random >>> 17; random ^= random << 5; bytes[i] = random & 255; }
  fs.writeFileSync(path.join(work, row.input), bytes);
}
const manifest = path.join(work, 'cases.json');
fs.writeFileSync(manifest, JSON.stringify(cases, null, 2));
const pillow = JSON.parse(run(python, [path.join(root, 'scripts/word-native-parity/pixel-resize-reference.py'), manifest]));
const { resizeWordRawPixels } = load(path.join(root, 'src/ml/word-image-processing.ts'));
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const rows = [];
for (const row of cases) {
  const raw = Uint8Array.from(fs.readFileSync(path.join(work, row.input)));
  const begin = performance.now();
  const ts = resizeWordRawPixels({ width: row.width, height: row.height, pixelFormat: row.format, buffer: raw.buffer }, row.androidBitmap);
  const nodeElapsed = performance.now() - begin;
  const result = JSON.parse(run(java, ['-cp', work, 'WordPixelResizeCli', String(row.width), String(row.height), row.format, String(row.androidBitmap), path.join(work, row.input), path.join(work, row.name)]));
  const actual = fs.readFileSync(path.join(work, `${row.name}.java.f32`));
  const tsBytes = Buffer.from(ts.tensor.buffer), pillowBytes = fs.readFileSync(path.join(work, `${row.name}.pillow.f32`));
  const rawJava = fs.readFileSync(path.join(work, `${row.name}.java.rgb`)), rawPillow = fs.readFileSync(path.join(work, `${row.name}.pillow.rgb`));
  rows.push({ ...row, input_sha256: sha(raw), java_float32_sha256: sha(actual), typescript_float32_sha256: sha(tsBytes), pillow_float32_sha256: sha(pillowBytes),
    java_typescript_exact: actual.equals(tsBytes), java_pillow_exact: actual.equals(pillowBytes), resized_uint8_pillow_exact: rawJava.equals(rawPillow),
    mean_luminance_abs_error: Math.abs(result.meanLuminance - ts.meanLuminance), contrast_abs_error: Math.abs(result.contrast - ts.contrast),
    desktop_node_single_ms: nodeElapsed, ...result });
}
const files = [core, cli, 'scripts/word-native-parity/pixel-resize-parity.cjs', 'scripts/word-native-parity/pixel-resize-reference.py',
  'scripts/word-native-parity/geometry-reference.cjs', 'modules/braille-word-geometry/src/WordPhotoGeometry.nitro.ts',
  'modules/braille-word-geometry/android/src/main/java/com/margelo/nitro/braillewordgeometry/HybridWordPhotoGeometry.kt',
  ...loadedSourcePaths().map((file) => path.relative(root, file).replaceAll('\\', '/'))];
const mismatches = rows.filter((row) => !row.java_typescript_exact || !row.java_pillow_exact || !row.resized_uint8_pillow_exact || row.mean_luminance_abs_error > 1e-13 || row.contrast_abs_error > 1e-13);
const report = { schema_version: 1, created_at_utc: new Date().toISOString(), scope: 'Production Java resizer core compiled on host JVM, compared to actual TypeScript and Pillow; no Android/Nitro bridge execution or phone timing.',
  ...pillow, ...safety, source_sha256: Object.fromEntries(files.map((file) => [file, sha(fs.readFileSync(path.join(root, file)))])),
  total: rows.length, parity_passed: mismatches.length === 0, mismatch_count: mismatches.length,
  largest_host_java_median_ms: Math.max(...rows.map((row) => row.host_java_median_ms)), rows };
fs.mkdirSync(path.dirname(reportPath), { recursive: true });
fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
const { rows: _rows, source_sha256: _sources, ...summary } = report;
process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
if (!report.parity_passed) process.exitCode = 1;
