import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { activeWordModelDirectory } from './word-model-active-path.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)),'..');
const temporary = mkdtempSync(resolve(tmpdir(),'braille-active-model-test-'));
let checks = 0;
try {
  mkdirSync(resolve(temporary,'src/ml'),{recursive:true});
  const put = (initializer) => writeFileSync(resolve(temporary,'src/ml/word-model-assets.ts'),
    `export const WORD_MODEL_ASSET: number | null = ${initializer};\n`);
  for (const name of ['braille-word-reader-v1','braille-word-reader-id-v5']) {
    put(`require('../../assets/models/${name}/word-reader.tflite')`);
    assert.equal(activeWordModelDirectory(temporary),resolve(temporary,'assets/models',name)); checks++;
  }
  for (const initializer of ['null',"require(assetPath)","require('../../assets/models/unapproved/word-reader.tflite')",
    "require('../../assets/models/braille-word-reader-id-v5/../word-reader.tflite')"]) {
    put(initializer); assert.throws(()=>activeWordModelDirectory(temporary)); checks++;
  }
} finally {
  // Only this process's resolved test fixture root; no project/build directory.
  assert.ok(temporary.startsWith(resolve(tmpdir())+'\\'));
  rmSync(temporary,{recursive:true,force:true});
}
for (const [script,environment,reason] of [
  ['verify-model-release.mjs',{APP_ENV:'production'},/Production ditolak/],
  ['verify-eas-build-profile.mjs',{EAS_BUILD_PROFILE:'preview'},/hak redistribusi|cloud builds/],
]) {
  const result = spawnSync(process.execPath,[resolve(root,'scripts',script)],{cwd:root,env:{...process.env,...environment},encoding:'utf8'});
  assert.notEqual(result.status,0); assert.match(result.stderr,reason); checks++;
}
console.log(`Active model path and blocked release checks: ${checks}/${checks} passed. No build or submission invoked.`);
