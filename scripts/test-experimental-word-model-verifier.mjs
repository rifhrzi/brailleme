import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { verifyExperimentalWordModel } from './verify-experimental-word-model-assets.mjs';

// Mutate in-memory copies of the retained experimental evidence. No asset or
// report is changed on disk, and no model inference is run by these regressions.
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const directory = resolve(root, 'assets/models/braille-word-reader-v1');
const original = JSON.parse(readFileSync(resolve(directory, 'manifest.json'), 'utf8'));
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const inside = (base, file) => {
  const path = resolve(base, file), rel = relative(base, path);
  assert.ok(rel && !rel.startsWith('..') && !rel.includes(':'));
  return path;
};
const originalReports = new Map(Object.values(original.evidence).map((entry) =>
  [entry.file, readFileSync(inside(root, entry.file))]));
let passed = 0;
function check(name, change, expectedError) {
  const manifest = structuredClone(original), buffers = new Map(originalReports), environment = {};
  const edit = (key, mutate, updateHash = true) => {
    const entry = manifest.evidence[key], report = JSON.parse(buffers.get(entry.file).toString('utf8'));
    mutate(report);
    const bytes = Buffer.from(JSON.stringify(report)); buffers.set(entry.file, bytes);
    if (updateHash) entry.sha256 = hash(bytes);
  };
  change({ manifest, environment, edit });
  const verify = () => verifyExperimentalWordModel({ root, directory, manifest, hash, inside, environment,
    evidence(entry) {
      const bytes = buffers.get(entry.file);
      assert.equal(hash(bytes), entry.sha256, 'Evidence hash mismatch');
      return JSON.parse(bytes.toString('utf8'));
    },
  });
  if (expectedError) assert.throws(verify, expectedError, name);
  else {
    const result = verify();
    assert.equal(result.correct, 323); assert.equal(result.total, 607); assert.equal(result.accepted, 351);
    assert.equal(manifest.internalAcceptance.passed, false);
  }
  passed++; console.log(`PASS ${name}`);
}
check('authorized local experiment preserves actual failed metrics', () => {});
check('missing explicit user authorization rejected', ({ manifest }) => { delete manifest.experimentalUse; }, /explicitly authorized/);
check('production rejected even with authorization', ({ environment }) => { environment.APP_ENV = 'production'; }, /cannot enter production/);
check('cloud preview rejected even with authorization', ({ environment }) => { environment.EAS_BUILD_PROFILE = 'preview'; }, /cloud builds/);
check('fabricated manifest acceptance rejected', ({ manifest }) => { manifest.internalAcceptance.passed = true; }, /preserve failed acceptance/);
check('changed evidence without hash update rejected', ({ edit }) => edit('publicBenchmark', (r) => { r.whole_words.correct++; }, false), /Evidence hash mismatch/);
check('rehashing false benchmark metrics does not bypass row recount', ({ edit }) => edit('publicBenchmark', (r) => { r.whole_words.correct++; }), /Misreported whole-word correct/);
check('rejection diagnostic failures cannot be hidden', ({ edit }) => edit('rejectionDiagnostics', (r) => { r.accepted = 0; }), /strictly equal/);
check('old decoder report rejected', ({ edit }) => edit('publicBenchmark', (r) => { r.decoder_version = 'six-dot-word-v1'; }), /decoder identity mismatch/);
check('stale startup source proof rejected', ({ manifest }) => { manifest.nativeSelfTest.runtime_input_reference.source_sha256['src/ml/word-decoder.ts'] = '0'.repeat(64); }, /Stale source proof/);
console.log(`Experimental integrity regressions: ${passed}/${passed} passed.`);
