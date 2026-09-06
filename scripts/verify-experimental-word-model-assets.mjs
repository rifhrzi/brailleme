import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import ts from 'typescript';

// Integrity of the explicitly authorized local experiment, not an accuracy waiver
// for a release. The caller first checks the actual TFLite tensor/normalization contract.
export function verifyExperimentalWordModel({ root, directory, manifest, evidence, inside, hash, environment = process.env }) {
  assert.equal(manifest.artifact.status, 'experimental_candidate');
  assert.deepEqual(manifest.experimentalUse, {
    enabled: true, scope: 'local_internal_only', authorizedBy: 'user',
    reason: 'Use highest measured existing whole-word model; original accuracy gates remain failed.',
  }, 'Experimental use must be explicitly authorized for local internal use');
  assert.notEqual(environment.APP_ENV, 'production', 'Experimental model cannot enter production');
  assert.ok(!environment.EAS_BUILD_PROFILE && environment.EAS_BUILD !== 'true', 'Experimental model cannot enter cloud builds');
  assert.equal(manifest.internalAcceptance.passed, false, 'Experimental use must preserve failed acceptance');
  assert.equal(manifest.physicalCameraValidated, false);
  const reports = Object.fromEntries(Object.entries(manifest.evidence).map(([key, entry]) => [key, evidence(entry)]));
  const { publicBenchmark: benchmark, wholeWordFixtures: fixtures, conversion, validationSelection: validation } = reports;
  const bound = (report, decoder = true) => {
    assert.equal(report.model_sha256, manifest.artifact.sha256, 'Report model identity mismatch');
    assert.deepEqual(report.thresholds, manifest.thresholds, 'Report thresholds mismatch');
    if (decoder) assert.equal(report.decoder_version, manifest.decoderVersion, 'Report decoder identity mismatch');
  };
  const prediction = (row) => {
    assert.match(row.predicted, /^(?:[A-Z]{1,12})?$/);
    assert.equal(typeof row.rejection, 'string');
    assert.equal(Boolean(row.predicted), !row.rejection, 'Accepted words and rejections must be mutually exclusive');
  };
  const counts = (rows) => ({ total: rows.length, accepted: rows.filter((r) => r.predicted).length,
    correct: rows.filter((r) => r.predicted === r.expected).length });
  const metrics = (rows) => {
    const result = counts(rows);
    return { ...result, exact_word_accuracy: result.correct / result.total,
      accepted_accuracy: result.accepted ? result.correct / result.accepted : null };
  };
  const currentSources = (sources, prefix = '') => {
    assert.ok(sources && Object.keys(sources).length > 0, 'Missing source identity');
    for (const [file, digest] of Object.entries(sources))
      assert.equal(hash(readFileSync(inside(root, prefix + file))), digest, `Stale source proof: ${file}`);
  };
  bound(benchmark);
  assert.equal(benchmark.fixture_manifest_sha256, manifest.evidence.wholeWordFixtures.sha256);
  assert.equal(benchmark.words.length, 607);
  assert.equal(fixtures.records.length, benchmark.words.length);
  assert.equal(new Set(benchmark.words.map((r) => r.image)).size, benchmark.words.length);
  for (const [index, row] of benchmark.words.entries()) {
    const { predicted, rejection, detected_cells: cells, ...record } = row;
    assert.deepEqual(record, fixtures.records[index], `Benchmark fixture mismatch: ${index}`);
    assert.match(row.expected, /^[A-Z]{1,12}$/);
    prediction({ predicted, rejection });
    assert.ok(Array.isArray(cells));
    if (predicted) {
      assert.equal(cells.length, predicted.length);
      for (const cell of cells) {
        assert.ok(cell.box.length === 4 && cell.box.every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
        assert.ok(cell.box[0] < cell.box[2] && cell.box[1] < cell.box[3]);
        assert.ok(cell.dots.length === 6 && cell.dots.every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
        assert.ok(Number.isFinite(cell.score) && cell.score >= manifest.thresholds.acceptScore && cell.score <= 1);
      }
    }
  }
  const whole = metrics(benchmark.words);
  for (const [key, value] of Object.entries(whole)) assert.equal(benchmark.whole_words[key], value, `Misreported whole-word ${key}`);
  const byLength = {};
  for (let length = 1; length <= 12; length++) {
    const rows = benchmark.words.filter((r) => r.expected.length === length);
    if (rows.length) byLength[length] = metrics(rows);
  }
  assert.deepEqual(benchmark.whole_words.by_length, byLength, 'Misreported accuracy by length');
  const reasons = {};
  for (const row of benchmark.words.filter((r) => !r.predicted)) reasons[row.rejection] = (reasons[row.rejection] ?? 0) + 1;
  assert.deepEqual(benchmark.whole_words.rejection_reasons, reasons);
  assert.equal(benchmark.negative_rows.length, 44);
  assert.equal(new Set(benchmark.negative_rows.map((r) => r.source)).size, 44);
  benchmark.negative_rows.forEach(prediction);
  const negativeAccepted = benchmark.negative_rows.filter((r) => r.predicted).length;
  assert.equal(benchmark.negative_regression.total, 44);
  assert.equal(benchmark.negative_regression.accepted, negativeAccepted);
  assert.equal(benchmark.negative_regression.independent_holdout, false);
  assert.equal(benchmark.legacy_rows.length, 66);
  benchmark.legacy_rows.forEach(prediction);
  for (const [key, value] of Object.entries(counts(benchmark.legacy_rows)))
    assert.equal(benchmark.legacy_arbitrary_block_regression[key], value, `Misreported legacy ${key}`);
  assert.equal(benchmark.legacy_arbitrary_block_regression.is_complete_word_benchmark, false);
  const numericalPass = whole.exact_word_accuracy >= .75 && whole.accepted > 0 && whole.accepted_accuracy >= .95 && negativeAccepted === 0;
  assert.equal(benchmark.numerical_internal_gates_pass, numericalPass);
  assert.equal(numericalPass, false, 'A passing candidate must use the original acceptance path');
  assert.equal(benchmark.release_ready, false);
  assert.equal(benchmark.external_independent_holdout, false);
  assert.deepEqual(manifest.internalAcceptance, { passed: false, exactWordSuccessRate: whole.exact_word_accuracy,
    acceptedWordAccuracy: whole.accepted_accuracy, evaluatedWordCount: whole.total, acceptedWordCount: whole.accepted,
    negativeImageCount: 44, negativeFalseAccepts: negativeAccepted });

  assert.equal(conversion.artifact_sha256, manifest.artifact.sha256);
  assert.equal(conversion.artifact_bytes, manifest.artifact.sizeBytes);
  assert.equal(conversion.source_checkpoint_sha256, manifest.provenance.checkpointSha256);
  assert.deepEqual(conversion.input_shape, manifest.input.shape);
  assert.deepEqual(conversion.output_shape, manifest.output.shape);
  assert.equal(conversion.decoder_version, manifest.decoderVersion);
  assert.deepEqual(conversion.thresholds, manifest.thresholds);
  assert.deepEqual(conversion.imageNormalization, manifest.imageNormalization);
  for (const key of ['parity_pass', 'decoded_text_parity', 'decoded_rejection_parity']) assert.equal(conversion[key], true);
  assert.equal(conversion.parity_samples, 405);
  for (const key of ['training_backend_vs_tensorflow_max_abs', 'tensorflow_vs_tflite_max_abs', 'training_backend_vs_tflite_max_abs'])
    assert.ok(Number.isFinite(conversion[key]) && conversion[key] >= 0 && conversion[key] <= .0005, `Invalid numeric parity: ${key}`);
  assert.equal(validation.checkpoint_sha256, conversion.source_checkpoint_sha256);
  assert.equal(validation.oracle.checkpoint_sha256, validation.checkpoint_sha256);
  assert.equal(validation.oracle.fixture_sha256, conversion.parity_fixture_sha256);
  assert.equal(validation.decoder_version, manifest.decoderVersion);
  assert.equal(validation.oracle.decoder_version, manifest.decoderVersion);
  assert.deepEqual(validation.thresholds, manifest.thresholds);
  assert.deepEqual(validation.imageNormalization, manifest.imageNormalization);
  assert.equal(validation.words.total, 405);
  const parity = reports.decoderParity;
  assert.equal(parity.status, 'passed');
  assert.equal(parity.decoder_version, manifest.decoderVersion);
  assert.equal(parity.model_sha256, manifest.artifact.sha256);
  assert.equal(parity.selection_sha256, manifest.evidence.validationSelection.sha256);
  assert.equal(parity.samples, conversion.parity_samples);
  assert.equal(parity.raw_decisions.length, parity.samples);
  assert.equal(parity.effective_decisions.length, parity.samples);
  assert.ok(Number.isFinite(parity.max_abs) && parity.max_abs >= 0 && parity.max_abs <= .0005);
  currentSources(parity.source_sha256, 'training/scripts/');

  for (const [key, expectedTotal, rejecting] of [['repetitionDiagnostics', 12, false], ['rejectionDiagnostics', 14, true]]) {
    const report = reports[key];
    bound(report, false);
    currentSources(report.source_sha256);
    assert.equal(report.rows.length, expectedTotal);
    assert.equal(new Set(report.rows.map((r) => r.image)).size, expectedTotal);
    const categories = {};
    for (const row of report.rows) {
      prediction(row);
      assert.equal(row.expected_reject, rejecting);
      assert.match(row.image_sha256, /^[a-f0-9]{64}$/);
      assert.equal(row.correct, rejecting ? !row.predicted : row.predicted === row.expected);
      const category = categories[row.category] ??= { total: 0, correct: 0, accepted: 0 };
      category.total++; category.correct += Number(row.correct); category.accepted += Number(Boolean(row.predicted));
    }
    assert.equal(report.total, expectedTotal);
    assert.equal(report.correct, report.rows.filter((r) => r.correct).length);
    assert.equal(report.accepted, report.rows.filter((r) => r.predicted).length);
    assert.deepEqual(report.by_category, categories);
    assert.equal(report.is_natural_whole_word_benchmark, false);
  }

  const fixture = manifest.nativeSelfTest;
  const png = readFileSync(inside(directory, fixture.file));
  assert.equal(hash(png), fixture.sha256);
  assert.equal(png.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
  assert.equal(png.readUInt32BE(16), fixture.width);
  assert.equal(png.readUInt32BE(20), fixture.height);
  assert.match(fixture.expectedText, /^[A-Z]{1,12}$/);
  assert.equal(fixture.native_validated, false);
  const { x, y, width, height } = fixture.roi;
  assert.ok([x, y, width, height].every(Number.isFinite) && x >= 0 && y >= 0 && width > 0 && height > 0 && x + width <= 1 && y + height <= 1);
  const reference = fixture.runtime_input_reference;
  bound(reference, false);
  assert.equal(reference.decoded_text, fixture.expectedText);
  assert.equal(reference.rejection, '');
  assert.equal(reference.quality_reason, '');
  assert.equal(reference.exact_cached_pixels, true);
  for (const key of ['input_float32_sha256', 'input_rgb_sha256', 'cached_output_sha256']) assert.match(reference[key], /^[a-f0-9]{64}$/);
  for (const file of ['src/ml/word-image-processing.ts', 'src/ml/word-model-contract.ts', 'src/ml/word-decoder.ts', 'src/types/word-scan.ts'])
    assert.ok(reference.source_sha256[file], `Missing startup runtime proof: ${file}`);
  currentSources(reference.source_sha256);
  const geometry = reports.nativeFixtureGeometry;
  currentSources(geometry.source_sha256);
  const geometryRow = geometry.rows.find((r) => r.image === reference.cached_validation_image);
  assert.ok(geometryRow?.eligible);
  assert.equal(geometryRow.expected, fixture.expectedText);
  assert.equal(geometryRow.input_rgb_sha256, reference.input_rgb_sha256);
  assert.deepEqual(geometryRow.actual_crop, reference.crop);
  assert.deepEqual(geometryRow.expected_crop, reference.crop);
  assert.equal(parity.effective_decisions[geometryRow.index].text, fixture.expectedText);
  assert.equal(parity.effective_decisions[geometryRow.index].reason, '');

  const syntax = ts.createSourceFile('word-model-assets.ts', readFileSync(resolve(root, 'src/ml/word-model-assets.ts'), 'utf8'), ts.ScriptTarget.Latest, true);
  const declarations = syntax.statements.filter((s) => ts.isVariableStatement(s) && s.modifiers?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword))
    .flatMap((s) => s.declarationList.declarations);
  for (const [name, file] of [['WORD_MODEL_ASSET', manifest.artifact.file], ['WORD_NATIVE_FIXTURE_ASSET', fixture.file]]) {
    const init = declarations.find((d) => ts.isIdentifier(d.name) && d.name.text === name)?.initializer;
    assert.ok(init && ts.isCallExpression(init) && ts.isIdentifier(init.expression) && init.expression.text === 'require' && init.arguments.length === 1 && ts.isStringLiteral(init.arguments[0]), `${name} must be a literal Metro require`);
    assert.equal(init.arguments[0].text, `../../assets/models/braille-word-reader-v1/${file}`);
  }
  const release = declarations.find((d) => ts.isIdentifier(d.name) && d.name.text === 'WORD_MODEL_RELEASE_VERIFIED')?.initializer;
  assert.equal(release?.kind, ts.SyntaxKind.FalseKeyword, 'Experimental model must remain unverified for production');
  return whole;
}
