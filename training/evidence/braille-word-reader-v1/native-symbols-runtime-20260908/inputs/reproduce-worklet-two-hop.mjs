import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import ts from 'typescript';

const artifactDirectory = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(artifactDirectory, '../..');
const unpackerPath = 'node_modules/react-native-worklets/src/memory/valueUnpacker.native.ts';
const serializerPath = 'node_modules/react-native-worklets/src/memory/serializable.native.ts';
function declaration(path, name) {
  const source = ts.createSourceFile(path, readFileSync(resolve(projectRoot, path), 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const node = source.statements.find((statement) => ts.isFunctionDeclaration(statement) && statement.name?.text === name);
  assert.ok(node, `${name} must exist in the installed library`);
  return node.getText(source).replace(/^export /, '');
}

// Execute the actual installed unpacker and worker serializer. Their native
// factory is stubbed to return the adapted object; the installed C++ factory's
// default object branch has the same observable type for this empty input.
const program = `
const __DEV__ = false;
const isHostObject = () => false;
globalThis.__customSerializationRegistry = [];
globalThis.__workletsModuleProxy = { createSerializableLEGACY: (adapted) => adapted };
${declaration(unpackerPath, 'installValueUnpacker')}
${declaration(serializerPath, 'makeShareableCloneOnUIRecursiveLEGACY')}
installValueUnpacker();
const descriptor = { __workletHash: 7, __closure: {}, __initData: { code: 'function helper() { return 42; }' } };
const first = globalThis.__valueUnpacker(descriptor);
const second = makeShareableCloneOnUIRecursiveLEGACY(first);
const observations = {
  firstRuntimeType: typeof first,
  firstRuntimeResult: first(),
  firstRuntimeEnumerableKeys: Object.keys(first),
  secondRuntimeType: typeof second,
  secondRuntimeEnumerableKeys: Object.keys(second),
};
try { second(); } catch (error) { observations.secondRuntimeErrorName = error.name; }
observations;
`;
const javascript = ts.transpileModule(program, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText;
const observations = JSON.parse(JSON.stringify(vm.runInNewContext(javascript)));
assert.deepEqual(observations, {
  firstRuntimeType: 'function', firstRuntimeResult: 42, firstRuntimeEnumerableKeys: [],
  secondRuntimeType: 'object', secondRuntimeEnumerableKeys: [], secondRuntimeErrorName: 'TypeError',
});

function source(path, excerpts) {
  const bytes = readFileSync(resolve(projectRoot, path));
  const lines = bytes.toString('utf8').split(/\r?\n/);
  return {
    path, sha256: createHash('sha256').update(bytes).digest('hex'),
    excerpts: excerpts.map((text) => {
      const index = lines.findIndex((line) => line.includes(text));
      assert.ok(index >= 0, `Missing evidence: ${text}`);
      return { line: index + 1, text: lines[index].trim() };
    }),
  };
}
const report = {
  schemaVersion: 1, executedAt: new Date().toISOString(), nodeVersion: process.version,
  status: 'reproduced', observations,
  limitation: 'Source-level Node reproduction with the native serialization factory stubbed. This is not a native camera or APK success test.',
  interpretation: 'The first worker binds the helper without enumerable worklet metadata. The installed worker-to-worker serializer converts that callable to an empty object, so the second runtime cannot call it.',
  fix: 'Keep admitted processing on the dedicated VisionCamera frame thread; avoid the second runtime transfer. Share interpreter ownership atomically across sessions and guard lease admission against unmount.',
  sources: [
    source(unpackerPath, ['const functionInstance = workletFun!.bind(objectToUnpack);', 'return functionInstance;']),
    source(serializerPath, ['function makeShareableCloneOnUIRecursiveLEGACY', 'for (const [key, element] of Object.entries(value))', 'globalThis.__workletsModuleProxy.createSerializableLEGACY(']),
    source('node_modules/react-native-worklets/Common/cpp/worklets/SharedItems/Serializable.cpp', ['if (!object.getProperty(rt, "__workletHash").isUndefined())', 'serializable = std::make_shared<SerializableObject>(rt, object, nativeStateSource);']),
    source('node_modules/react-native-vision-camera-worklets/src/createAsyncRunner.ts', ['scheduleOnRuntime(workletRuntime, () => {', 'task()']),
    source('node_modules/react-native-vision-camera/android/src/main/java/com/margelo/nitro/camera/hybrids/outputs/HybridFrameOutput.kt', ['private val executor = IdentifiableExecutor', 'imageAnalysis.setAnalyzer(executor)']),
    source('node_modules/react-native-vision-camera/android/src/main/java/com/margelo/nitro/camera/extensions/ImageAnalysisBuilder+setAllowDroppingLateFrames.kt', ['ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST']),
    source('src/features/live/use-word-camera-tracking.ts', ['const modelBusyFlags', 'busy.lock();', 'modelBusy.lock();', 'const result = detectWordImageCells']),
  ],
};
const path = resolve(artifactDirectory, 'worklet-two-hop-reproduction-v1.json');
writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ status: report.status, observations, reportPath: path }));
