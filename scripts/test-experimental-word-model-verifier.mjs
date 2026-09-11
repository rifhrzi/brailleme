import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { verifyExperimentalWordModel } from './verify-experimental-word-model-assets.mjs';
import { activeWordModelManifest } from './word-model-active-path.mjs';

// Mutate in-memory copies of the retained experimental evidence. No asset or
// report is changed on disk, and no model inference is run by these regressions.
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
if (activeWordModelManifest(root).experimentVersion === 'overall-exact-text-v5-20260911') {
  await import('./test-overall-word-model-verifier.mjs');
} else {
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
    if (updateHash) {
      // Baseline/public evidence may intentionally reference the same immutable file.
      for (const current of Object.values(manifest.evidence)) if (current.file === entry.file) current.sha256 = hash(bytes);
    }
  };
  // Keep duplicated readiness fields and report hashes consistent while testing
  // the independent selection rules, rather than stopping at a stale outer hash.
  const editSelection = (mutate) => {
    let selection;
    edit('checkpointSelection', (report) => { mutate(report); selection = report; });
    edit('symbolReadiness', (report) => {
      report.provenance.checkpoint_selection_sha256 = manifest.evidence.checkpointSelection.sha256;
      report.baseline_groups = selection.baseline;
      report.candidate_comparisons = selection.candidates;
      report.training.baseline_checkpoint_sha256 = selection.configuration.baseline_checkpoint_sha256;
      report.training.baseline_retained = selection.baseline_retained;
    });
    edit('deploymentSelection', (report) => { report.training_selection_sha256 = manifest.evidence.checkpointSelection.sha256; });
  };
  change({ manifest, environment, edit, editSelection });
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
    assert.equal(result.total, 607);
    assert.equal(result.correct / result.total, manifest.internalAcceptance.exactWordSuccessRate);
    assert.equal(result.correct / result.accepted, manifest.internalAcceptance.acceptedWordAccuracy);
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
check('rejection diagnostic counts cannot be fabricated', ({ edit }) => edit('rejectionDiagnostics', (r) => { r.accepted++; }), /strictly equal/);
check('old decoder report rejected', ({ edit }) => edit('publicBenchmark', (r) => { r.decoder_version = 'six-dot-word-v1'; }), /decoder identity mismatch/);
check('stale startup source proof rejected', ({ manifest }) => { manifest.nativeSelfTest.runtime_input_reference.source_sha256['src/ml/word-decoder.ts'] = '0'.repeat(64); }, /Stale source proof/);
check('wrong printed/physical limit rejected', ({ manifest }) => { manifest.symbolContract.maxPhysicalCells = 12; }, /symbol contract/);
check('fabricated capital precision rejected', ({ edit }) => edit('symbolReadiness', (r) => { r.groups.capital_single.accepted_accuracy = 1; }), /Misreported symbol group/);
check('physical claim rejected in synthetic symbol report', ({ edit }) => edit('symbolReadiness', (r) => { r.physical_camera_validated = true; }), /strictly equal/);
check('hidden parser/model parity mismatch rejected', ({ edit }) => edit('symbolDecoderParity', (r) => { r.rows[0].pipeline_text_match = false; }), /Failed TypeScript\/Python/);
check('unbounded training evidence rejected', ({ edit }) => edit('symbolReadiness', (r) => { r.training_epochs_completed = 11; }), /falsy/);
check('changed native resizer proof rejected', ({ edit }) => edit('nativePixelParity', (r) => { r.source_sha256['modules/braille-word-geometry/android/src/main/java/com/margelo/nitro/braillewordgeometry/WordPixelResize.java'] = '0'.repeat(64); }), /Stale source proof/);
check('omitting the native Java source does not bypass provenance', ({ edit }) => edit('nativePixelParity', (r) => { delete r.source_sha256['modules/braille-word-geometry/android/src/main/java/com/margelo/nitro/braillewordgeometry/WordPixelResize.java']; }), /Missing source proof/);
check('omitting the native Kotlin bridge source rejected', ({ edit }) => edit('nativePixelParity', (r) => { delete r.source_sha256['modules/braille-word-geometry/android/src/main/java/com/margelo/nitro/braillewordgeometry/HybridWordPhotoGeometry.kt']; }), /Missing source proof/);
check('empty equal tensor hashes are not parity evidence', ({ edit }) => edit('nativePixelParity', (r) => {
  for (const field of ['java_float32_sha256', 'typescript_float32_sha256', 'pillow_float32_sha256']) r.rows[0][field] = '';
}), /Missing pixel parity hash/);
check('native uint8 mismatch cannot hide behind summary pass', ({ edit }) => edit('nativePixelParity', (r) => { r.rows[0].resized_uint8_pillow_exact = false; }), /strictly equal/);
check('native luminance mismatch cannot hide behind tensor parity', ({ edit }) => edit('nativePixelParity', (r) => { r.rows[0].mean_luminance_abs_error = .01; }), /Invalid pixel statistics parity/);
check('native contrast evidence must be finite', ({ edit }) => edit('nativePixelParity', (r) => { r.rows[0].contrast_abs_error = null; }), /Invalid pixel statistics parity/);
check('duplicating native cases cannot meet coverage count', ({ edit }) => edit('nativePixelParity', (r) => { r.rows[1] = r.rows[0]; }), /strictly equal/);
check('hidden semantic decoder mismatch rejected', ({ edit }) => edit('symbolSemantics', (r) => { r.rows[0].decoder_match = false; }), /Failed symbol semantics row/);
check('omitting semantic source identity rejected', ({ edit }) => edit('symbolSemantics', (r) => { delete r.source_sha256['src/ml/word-symbols.ts']; }), /Missing source proof/);
check('symbol corpus hash must bind actual evidence', ({ edit }) => edit('symbolReadiness', (r) => { r.provenance.corpus_audit_sha256 = '0'.repeat(64); }), /Symbol provenance mismatch/);
check('symbol replay hash must bind compared tensors', ({ edit }) => edit('symbolReadiness', (r) => { r.provenance.python_reference_sha256 = '0'.repeat(64); }), /Symbol replay provenance mismatch/);
check('readiness cannot silently drop mandatory rejections', ({ edit }) => edit('symbolReadiness', (r) => { r.mandatory_rejections.pop(); }), /Mandatory rejection rows differ/);
check('readiness cannot rewrite bounded training metadata', ({ edit }) => edit('symbolReadiness', (r) => { r.training.single_bounded_round = false; }), /Symbol training provenance mismatch/);
check('baseline retention cannot substitute an unevaluated checkpoint', ({ editSelection }) => editSelection((r) => {
  r.baseline_retained = true;
  r.configuration.baseline_checkpoint_sha256 = '0'.repeat(64);
}), /Selected checkpoint is not the best|Baseline retention differs/);
check('candidate rank must be recomputed from real metrics', ({ editSelection }) => editSelection((r) => {
  assert.ok(r.candidates.length > 0, 'This fixture requires an actual completed training epoch');
  r.candidates[0].rank[2] += .01;
}), /Checkpoint rank differs/);
check('candidate regression flags cannot be hidden', ({ editSelection }) => editSelection((r) => {
  assert.ok(r.candidates.length > 0, 'This fixture requires an actual completed training epoch');
  r.candidates[0].regressions = ['invented'];
}), /Checkpoint regression flags differ/);
check('missing training protocol source proof rejected', ({ editSelection }) => editSelection((r) => { delete r.configuration.source_sha256['word_symbol_train.py']; }), /Missing source proof/);
check('external regression veto cannot be waived despite better validation', ({ edit }) => edit('deploymentSelection', (r) => { r.status = 'candidate_promoted'; }), /respect external regression veto/);
check('external exact regression cannot be omitted from decision reasons', ({ edit }) => edit('deploymentSelection', (r) => { r.reason_codes = []; }), /regression reasons differ/);
check('deployment must retain baseline model bytes after veto', ({ edit }) => edit('deploymentSelection', (r) => { r.selected_model_sha256 = r.candidate.model_sha256; }), /Deployment selected the wrong model/);
check('deployment must retain baseline checkpoint after veto', ({ edit }) => edit('deploymentSelection', (r) => { r.selected_checkpoint_sha256 = r.candidate.checkpoint_sha256; }), /Deployment selected the wrong checkpoint/);
check('deployment must bind immutable ten-epoch training history', ({ edit }) => edit('deploymentSelection', (r) => { r.training_selection_sha256 = '0'.repeat(64); }), /Deployment training history mismatch/);
check('external baseline must use identical decoder', ({ edit, manifest }) => {
  edit('baselineBenchmark', (r) => { r.decoder_version = 'six-dot-word-v2'; });
  edit('deploymentSelection', (r) => { r.baseline.benchmark_sha256 = manifest.evidence.baselineBenchmark.sha256; });
}, /decoder identity mismatch|Deployment benchmark decoder mismatch/);
check('external candidate must use identical thresholds', ({ edit, manifest }) => {
  edit('candidateBenchmark', (r) => { r.thresholds.acceptScore += .01; });
  edit('deploymentSelection', (r) => { r.candidate.benchmark_sha256 = manifest.evidence.candidateBenchmark.sha256; });
}, /Deployment benchmark thresholds mismatch/);
check('external candidate metrics are independently recounted', ({ edit, manifest }) => {
  edit('candidateBenchmark', (r) => { r.whole_words.correct++; });
  edit('deploymentSelection', (r) => { r.candidate.benchmark_sha256 = manifest.evidence.candidateBenchmark.sha256; });
}, /Misreported deployment benchmark/);
check('external comparison cannot substitute other fixture labels', ({ edit, manifest }) => {
  edit('candidateBenchmark', (r) => { r.words[0].expected = 'invented'; });
  edit('deploymentSelection', (r) => { r.candidate.benchmark_sha256 = manifest.evidence.candidateBenchmark.sha256; });
}, /Deployment benchmark row differs/);
console.log(`Experimental integrity regressions: ${passed}/${passed} passed.`);
}
