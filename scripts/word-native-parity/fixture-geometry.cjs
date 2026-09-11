// Calls the application's actual photo pipeline. Native pixels and inference are
// mocked here; Android pixel parity is separate evidence, not supplied by this tool.
const assert = require('node:assert/strict');
const { Buffer } = require('node:buffer');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { load, root, loadedSourcePaths } = require('./geometry-reference.cjs');
const image = load(path.join(root, 'src/ml/word-image-processing.ts'));
const decoder = load(path.join(root, 'src/ml/word-decoder.ts'));
const contract = load(path.join(root, 'src/ml/word-model-contract.ts'));
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');

function inspectFixture(fixture) {
  const pixels = fixture.rgb_base64 ? Uint8Array.from(Buffer.from(fixture.rgb_base64, 'base64'))
    : Uint8Array.from({ length: 480 * 192 * 3 }, (_, i) => i % 2 ? 192 : 64);
  assert.equal(pixels.length, 480 * 192 * 3);
  const output = fixture.output_float32_base64
    ? Uint8Array.from(Buffer.from(fixture.output_float32_base64, 'base64')).buffer : null;
  if (output) assert.equal(output.byteLength, 48 * 120 * 11 * 4);
  let crop = [0, 0, fixture.width, fixture.height];
  let selection = null, tensorSha256 = null;
  const nativePixels = () => ({ width: 480, height: 192, pixelFormat: 'RGB', buffer: pixels.buffer });
  const originalDecode = decoder.decodeWordOutput;
  decoder.decodeWordOutput = (actualOutput, thresholds, actualSelection) => {
    selection = actualSelection;
    if (output) return originalDecode(actualOutput, thresholds, actualSelection);
    return { status: 'rejected', reason: 'no_cells', cells: [], boxes: [] };
  };
  try {
    const result = image.recognizeWordImage({
      width: fixture.width, height: fixture.height, toRawPixelData: nativePixels,
      crop: (...coordinates) => {
        crop = coordinates;
        return { toRawPixelData: nativePixels, dispose() {} };
      },
    }, {
      runSync: ([tensor]) => {
        tensorSha256 = sha(Buffer.from(tensor));
        return [output ?? new ArrayBuffer(48 * 120 * 11 * 4)];
      },
    }, fixture.thresholds ?? contract.DEFAULT_WORD_MODEL_THRESHOLDS, fixture.roi);
    assert.notEqual(result.status, 'error', 'Actual runtime rejected fixture geometry');
    if (fixture.expectedText !== undefined) {
      assert.ok(output, 'Checking fixture text requires a real archived model output');
      assert.equal(result.status, 'success', 'Startup fixture must be accepted by the current runtime');
      assert.equal(result.text, fixture.expectedText, 'Startup fixture text differs from the current decoder');
    }
    const qualityReason = selection === null ? result.reason : '';
    if (!fixture.rgb_base64) assert.equal(qualityReason, '', 'Synthetic geometry pixels must pass quality');
    return { crop, selection, quality_reason: qualityReason, input_float32_sha256: tensorSha256,
      ...(output ? { cached_output_sha256: sha(Buffer.from(output)),
        decoded_text: result.status === 'success' ? result.text : '',
        rejection: result.status === 'success' ? '' : result.reason } : {}) };
  } finally {
    decoder.decodeWordOutput = originalDecode;
  }
}

function inspectFixtures(fixtures) {
  return {
    scope: 'Actual TypeScript crop/selection/quality/optional cached-output decoding with mocked native pixels and no inference; no native success claim.',
    source_sha256: Object.fromEntries([
      ...loadedSourcePaths().map((file) => path.relative(root, file).replaceAll('\\', '/')),
      'scripts/word-native-parity/geometry-reference.cjs', 'scripts/word-native-parity/fixture-geometry.cjs',
    ].map((file) => [file, sha(fs.readFileSync(path.join(root, file)))])),
    fixtures: fixtures.map(inspectFixture),
  };
}

module.exports = { inspectFixture, inspectFixtures };
if (require.main === module) {
  const request = JSON.parse(fs.readFileSync(0, 'utf8'));
  process.stdout.write(`${JSON.stringify(inspectFixtures(request.fixtures))}\n`);
}
