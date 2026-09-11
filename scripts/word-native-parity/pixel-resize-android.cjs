// Execute the production Java core under Android ART, then compare actual TS/Pillow bytes.
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { Buffer } = require('node:buffer');
const { load, root, loadedSourcePaths } = require('./geometry-reference.cjs');
const [adb, serial, dex, hostCases, output, reportPath] = process.argv.slice(2);
if (!reportPath) throw new Error('Usage: node pixel-resize-android.cjs <adb> <serial> <classes.dex> <host-cases-dir> <new-output-dir> <new-report.json>');
assert.equal(serial, 'emulator-5554', 'This audit is scoped to the authorized emulator');
assert.equal(os.endianness(), 'LE');
assert.ok(!fs.existsSync(reportPath), 'Evidence is immutable');
fs.mkdirSync(output, { recursive: false });
const remote = `/data/local/tmp/braille-word-pixels-${Date.now()}`;
function run(args) {
  const result = spawnSync(adb, ['-s', serial, ...args], { cwd: root, encoding: 'utf8', windowsHide: true, timeout: 120000 });
  assert.equal(result.status, 0, `${args.join(' ')}: ${result.stderr || result.error}`);
  return result.stdout.trim();
}
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const property = (name) => run(['shell', 'getprop', name]);
const device = {
  serial, fingerprint: property('ro.build.fingerprint'), model: property('ro.product.model'),
  api: Number(property('ro.build.version.sdk')), abi: property('ro.product.cpu.abi'),
  emulator: property('ro.kernel.qemu') === '1',
};
assert.ok(device.emulator, 'Do not run this diagnostic on a physical device');
run(['shell', 'mkdir', remote]);
run(['push', dex, `${remote}/classes.dex`]);
function java(args) {
  return JSON.parse(run(['shell', 'dalvikvm', '-cp', `${remote}/classes.dex`, 'WordPixelResizeCli', ...args]));
}
const safety = java(['--self-test']);
const { resizeWordRawPixels } = load(path.join(root, 'src/ml/word-image-processing.ts'));
const cases = JSON.parse(fs.readFileSync(path.join(hostCases, 'cases.json'), 'utf8'));
const rows = [];
for (const row of cases) {
  assert.match(row.name, /^[a-z0-9-]+$/);
  assert.equal(row.input, `${row.name}.input`);
  const input = path.join(hostCases, row.input), prefix = `${remote}/${row.name}`;
  run(['push', input, `${remote}/${row.input}`]);
  const measured = java([String(row.width), String(row.height), row.format, String(row.androidBitmap), `${remote}/${row.input}`, prefix]);
  const floatPath = path.join(output, `${row.name}.android.f32`), rgbPath = path.join(output, `${row.name}.android.rgb`);
  run(['pull', `${prefix}.java.f32`, floatPath]);
  run(['pull', `${prefix}.java.rgb`, rgbPath]);
  const raw = Uint8Array.from(fs.readFileSync(input));
  const ts = resizeWordRawPixels({ width: row.width, height: row.height, pixelFormat: row.format, buffer: raw.buffer }, row.androidBitmap);
  const actual = fs.readFileSync(floatPath), actualRgb = fs.readFileSync(rgbPath);
  const tsBytes = Buffer.from(ts.tensor.buffer), pillowBytes = fs.readFileSync(path.join(hostCases, `${row.name}.pillow.f32`));
  const pillowRgb = fs.readFileSync(path.join(hostCases, `${row.name}.pillow.rgb`));
  rows.push({ ...row, input_sha256: sha(raw), android_float32_sha256: sha(actual), typescript_float32_sha256: sha(tsBytes), pillow_float32_sha256: sha(pillowBytes),
    android_typescript_exact: actual.equals(tsBytes), android_pillow_exact: actual.equals(pillowBytes), resized_uint8_pillow_exact: actualRgb.equals(pillowRgb),
    mean_luminance_abs_error: Math.abs(measured.meanLuminance - ts.meanLuminance), contrast_abs_error: Math.abs(measured.contrast - ts.contrast),
    android_core_median_ms: measured.host_java_median_ms, android_core_max_ms: measured.host_java_max_ms });
}
const files = ['modules/braille-word-geometry/android/src/main/java/com/margelo/nitro/braillewordgeometry/WordPixelResize.java',
  'scripts/word-native-parity/WordPixelResizeCli.java', 'scripts/word-native-parity/pixel-resize-android.cjs',
  'scripts/word-native-parity/pixel-resize-reference.py', 'scripts/word-native-parity/geometry-reference.cjs',
  ...loadedSourcePaths().map((file) => path.relative(root, file).replaceAll('\\', '/'))];
const mismatches = rows.filter((row) => !row.android_typescript_exact || !row.android_pillow_exact || !row.resized_uint8_pillow_exact || row.mean_luminance_abs_error > 1e-13 || row.contrast_abs_error > 1e-13);
const report = { schema_version: 1, created_at_utc: new Date().toISOString(),
  scope: 'Production Java resizer core executed via DEX/dalvikvm on Android emulator. No Nitro/JNI wrapper, camera callback, inference, APK or physical-phone timing.',
  device, dex_sha256: sha(fs.readFileSync(dex)), cases_manifest_sha256: sha(fs.readFileSync(path.join(hostCases, 'cases.json'))),
  remote_diagnostic_directory: remote, warmup_iterations: 3, measured_iterations: 5, ...safety,
  source_sha256: Object.fromEntries(files.map((file) => [file, sha(fs.readFileSync(path.join(root, file)))])),
  total: rows.length, parity_passed: mismatches.length === 0, mismatch_count: mismatches.length,
  largest_android_core_median_ms: Math.max(...rows.map((row) => row.android_core_median_ms)), rows };
fs.mkdirSync(path.dirname(reportPath), { recursive: true });
fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
const { rows: _rows, source_sha256: _source, ...summary } = report;
process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
if (!report.parity_passed) process.exitCode = 1;
