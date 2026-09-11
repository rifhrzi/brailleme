/* global __dirname */
// Runs the actual TypeScript geometry/preprocessing functions with native images mocked.
// The Android harness independently executes corresponding Bitmap primitives.
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const root = path.resolve(__dirname, '../..');
const modules = new Map();
function load(file) {
  if (modules.has(file)) return modules.get(file).exports;
  const instance = new Module(file, module);
  modules.set(file, instance);
  instance.filename = file;
  instance.paths = Module._nodeModulePaths(path.dirname(file));
  instance.require = (name) => name === 'react-native' ? { Platform: { OS: 'android' } } : name.startsWith('@/')
    ? load(path.join(root, 'src', `${name.slice(2)}.ts`))
    : Module.prototype.require.call(instance, name);
  const source = fs.readFileSync(file, 'utf8');
  instance._compile(ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText, file);
  return instance.exports;
}
function generateGeometryReference(output) {
const image = load(path.join(root, 'src/ml/word-image-processing.ts'));
const contract = load(path.join(root, 'src/ml/word-model-contract.ts'));
const cases = [];
for (const [rotation, mirror, aspect] of [[0, false, 1], [0, false, 4], [0, false, 8], [90, false, 4], [180, false, 4], [270, false, 4], [90, true, 4]]) {
  const width = rotation % 180 ? 1920 : 1440;
  const height = rotation % 180 ? 1440 : 1920;
  const roi = image.mapPreviewBoxToImage(image.getWordGuideBox(320, 320, aspect), 320, 320, width, height);
  let crop;
  const outcome = image.recognizeWordImage({
    width, height,
    crop: (...coordinates) => {
      crop = coordinates;
      return {
        dispose() {},
        toRawPixelData: () => ({ width: 480, height: 192, pixelFormat: 'RGB', buffer: new ArrayBuffer(480 * 192 * 3) }),
      };
    },
  }, { runSync() { throw new Error('Black synthetic pixels must fail exposure before model invocation'); } }, contract.DEFAULT_WORD_MODEL_THRESHOLDS, roi);
  if (!crop || outcome.reason !== 'dark') throw new Error('Cannot generate geometry reference from runtime');
  cases.push({
    id: `rotation-${rotation}-mirror-${mirror}-aspect-${aspect}`,
    selection_photo: [roi.x, roi.y, roi.width, roi.height],
    context_crop_pixels: crop,
  });
}
if (!output) throw new Error('Pass the output JSON file path');
fs.writeFileSync(output, `${JSON.stringify({ scope: 'Actual TypeScript math with mocked native pixels; not native evidence by itself.', cases }, null, 2)}\n`);
process.stdout.write(`TypeScript geometry: ${cases.length} cases\n`);
}
module.exports = { load, root, generateGeometryReference, loadedSourcePaths: () => [...modules.keys()] };
if (require.main === module) generateGeometryReference(process.argv[2]);
