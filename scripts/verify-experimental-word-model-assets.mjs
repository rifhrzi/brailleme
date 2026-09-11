import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';
import ts from 'typescript';
import { verifyOverallWordModel } from './verify-overall-word-model-assets.mjs';

const { load } = createRequire(import.meta.url)('./word-native-parity/geometry-reference.cjs');

// Integrity of the explicitly authorized local experiment, not an accuracy waiver
// for a release. The caller first checks the actual TFLite tensor/normalization contract.
export function verifyExperimentalWordModel({ root, directory, manifest, evidence, inside, hash, environment = process.env }) {
  if (manifest.experimentVersion === 'overall-exact-text-v5-20260911') {
    return verifyOverallWordModel({ root, directory, manifest, evidence, inside, hash, environment });
  }
  assert.equal(manifest.artifact.status, 'experimental_candidate');
  assert.deepEqual(manifest.experimentalUse, {
    enabled: true, scope: 'local_internal_only', authorizedBy: 'user',
    reason: 'User-authorized Indonesian symbols and automatic tracking experiment; physical acceptance remains unverified.',
  }, 'Experimental use must be explicitly authorized for local internal use');
  assert.notEqual(environment.APP_ENV, 'production', 'Experimental model cannot enter production');
  assert.ok(!environment.EAS_BUILD_PROFILE && environment.EAS_BUILD !== 'true', 'Experimental model cannot enter cloud builds');
  assert.equal(manifest.internalAcceptance.passed, false, 'Experimental use must preserve failed acceptance');
  assert.equal(manifest.physicalCameraValidated, false);
  assert.deepEqual(manifest.symbolContract, {
    locale: 'id-basic', eventSchemaVersion: 4, capitalMask: 32, numberMask: 60,
    maxCharacters: 12, maxPhysicalCells: 24, preserveLeadingZeros: true,
    mixedAlphanumeric: false,
  }, 'Indonesian symbol contract must match the application');
  const { parseWordMasks } = load(resolve(root, 'src/ml/word-symbols.ts'));
  const reports = Object.fromEntries(Object.entries(manifest.evidence).map(([key, entry]) => [key, evidence(entry)]));
  const { publicBenchmark: benchmark, wholeWordFixtures: fixtures, conversion, validationSelection: validation } = reports;
  const bound = (report, decoder = true) => {
    assert.equal(report.model_sha256, manifest.artifact.sha256, 'Report model identity mismatch');
    assert.deepEqual(report.thresholds, manifest.thresholds, 'Report thresholds mismatch');
    if (decoder) assert.equal(report.decoder_version, manifest.decoderVersion, 'Report decoder identity mismatch');
  };
  const prediction = (row) => {
    assert.match(row.predicted, /^(?:[A-Za-z]{1,12}|[0-9]{1,12})?$/);
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
  const currentSources = (sources, prefix = '', required = []) => {
    assert.ok(sources && Object.keys(sources).length > 0, 'Missing source identity');
    for (const file of required) assert.ok(sources[file], `Missing source proof: ${file}`);
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
    assert.match(row.expected, /^[a-z]{1,12}$/);
    prediction({ predicted, rejection });
    assert.ok(Array.isArray(cells));
    if (predicted) {
      assert.ok(cells.length > 0 && cells.length <= 24);
      for (const cell of cells) {
        assert.ok(cell.box.length === 4 && cell.box.every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
        assert.ok(cell.box[0] < cell.box[2] && cell.box[1] < cell.box[3]);
        assert.ok(cell.dots.length === 6 && cell.dots.every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
        assert.ok(cell.dots.every((v) => Math.max(v, 1 - v) >= manifest.thresholds.dotConfidence));
        assert.ok(Number.isFinite(cell.score) && cell.score >= manifest.thresholds.acceptScore && cell.score <= 1);
      }
      const reconstructed = parseWordMasks(cells.map((cell) => cell.dots.reduce((mask, probability, dot) => mask | (probability >= .5 ? 1 << dot : 0), 0)));
      assert.equal(reconstructed.status, 'success', 'Accepted physical cells must form a supported token');
      assert.equal(reconstructed.text, predicted, 'Accepted text differs from physical cells');
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
  assert.match(fixture.expectedText, /^(?:[A-Za-z]{1,12}|[0-9]{1,12})$/);
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
  for (const file of ['src/ml/word-image-processing.ts', 'src/ml/word-model-contract.ts', 'src/ml/word-decoder.ts', 'src/ml/word-symbols.ts', 'src/types/word-scan.ts'])
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

  const symbols = reports.symbolReadiness;
  bound(symbols);
  assert.equal(symbols.physical_camera_validated, false);
  assert.equal(symbols.release_ready, false);
  assert.equal(symbols.natural_indonesian_capital_number_samples, 0);
  assert.equal(symbols.source_kind, 'synthetic_recomposed_public_cell_photographs');
  for (const [field, key] of [['corpus_audit_sha256', 'symbolCorpusAudit'], ['transform_audit_sha256', 'symbolTransformAudit'],
    ['fixture_sha256', 'symbolFixtures'], ['checkpoint_selection_sha256', 'checkpointSelection']]) {
    assert.ok(manifest.evidence[key], `Missing symbol provenance evidence: ${key}`);
    assert.equal(symbols.provenance[field], manifest.evidence[key].sha256, `Symbol provenance mismatch: ${field}`);
  }
  assert.equal(symbols.provenance.python_reference_sha256, reports.symbolDecoderParity.python_reference_sha256, 'Symbol replay provenance mismatch');
  assert.equal(reports.symbolDecoderParity.fixture_manifest_sha256, manifest.evidence.symbolFixtures.sha256, 'Symbol fixture replay mismatch');
  const corpus = reports.symbolCorpusAudit;
  assert.equal(corpus.status, 'passed');
  assert.equal(corpus.physical_camera_validated, false);
  assert.equal(corpus.split_before_crops, true);
  assert.deepEqual(corpus.source_overlap, []);
  const sourceSplits = [corpus.train_sources, corpus.validation_sources, corpus.external_sources_excluded];
  for (const sources of sourceSplits) assert.ok(Array.isArray(sources) && sources.length > 0 && new Set(sources).size === sources.length, 'Invalid corpus source split');
  const [trainSources, validationSources, externalSources] = sourceSplits.map((sources) => new Set(sources));
  assert.ok([...trainSources].every((source) => !validationSources.has(source) && !externalSources.has(source)) &&
    [...validationSources].every((source) => !externalSources.has(source)), 'Symbol corpus source splits overlap');
  const symbolFixtures = reports.symbolFixtures.records;
  assert.ok(Array.isArray(symbols.rows) && symbols.rows.length > 0, 'Missing symbol inference rows');
  assert.equal(symbols.rows.length, symbolFixtures.length);
  assert.equal(new Set(symbols.rows.map((row) => row.image)).size, symbols.rows.length);
  const symbolCounts = {};
  for (const [index, row] of symbols.rows.entries()) {
    const fixture = symbolFixtures[index];
    assert.deepEqual([row.image, row.expected, row.group, row.source_kind],
      [fixture.image, fixture.expected, fixture.symbol_group, fixture.origin], 'Symbol row differs from its fixture');
    assert.ok(fixture.source_cells.length > 0 && fixture.source_cells.every((cell) => validationSources.has(cell.source)), 'Symbol validation cell escaped held source split');
    symbolCounts[row.group] = (symbolCounts[row.group] ?? 0) + 1;
  }
  assert.deepEqual(corpus.composition_counts.validation, symbolCounts, 'Symbol corpus composition count mismatch');
  for (const group of ['lowercase', 'capital_single', 'capital_word', 'numbers']) {
    const rows = symbols.rows.filter((row) => row.group === group);
    assert.ok(rows.length > 0, `Missing symbol group: ${group}`);
    for (const row of rows) {
      assert.equal(row.source_kind, symbols.source_kind);
      assert.match(row.expected, /^(?:[A-Za-z]{1,12}|[0-9]{1,12})$/);
      prediction(row);
    }
    assert.deepEqual(symbols.groups[group], metrics(rows), `Misreported symbol group: ${group}`);
  }
  const invalid = symbols.mandatory_rejections;
  assert.ok(Array.isArray(invalid) && invalid.length > 0, 'Missing mandatory rejection evidence');
  assert.deepEqual(invalid, symbols.rows.filter((row) => row.group === 'rejection'), 'Mandatory rejection rows differ from symbol replay');
  invalid.forEach(prediction);
  assert.deepEqual(symbols.groups.rejection, { total: invalid.length,
    false_acceptances: invalid.filter((row) => row.predicted).length,
    all_rejected: invalid.every((row) => !row.predicted) });
  assert.equal(symbols.numeric_symbol_gates_pass, symbols.groups.rejection.all_rejected &&
    ['lowercase', 'capital_single', 'capital_word', 'numbers'].every((group) => symbols.groups[group].exact_word_accuracy >= .75 && (symbols.groups[group].accepted_accuracy ?? 0) >= .95));
  assert.ok(Number.isInteger(symbols.training_epochs_completed) && symbols.training_epochs_completed >= 0 && symbols.training_epochs_completed <= 10);
  assert.equal(symbols.initial_learning_rate, .00002);
  assert.ok([24, 12].includes(symbols.batch_size));
  const selection = reports.checkpointSelection;
  assert.equal(typeof selection.baseline_retained, 'boolean');
  assert.match(selection.configuration.baseline_checkpoint_sha256, /^[a-f0-9]{64}$/);
  assert.equal(selection.configuration.learning_rate, .00002);
  assert.equal(selection.configuration.batch_size, symbols.batch_size);
  assert.ok(Number.isInteger(selection.configuration.epochs_limit) && selection.configuration.epochs_limit <= 10 && selection.configuration.epochs_limit >= 1);
  assert.deepEqual(selection.configuration.thresholds, manifest.thresholds);
  assert.equal(selection.configuration.decoder_version, manifest.decoderVersion);
  currentSources(selection.configuration.source_sha256, 'training/scripts/', ['word_symbol_train.py', 'word_symbol_prepare.py', 'word_symbols.py', 'word_decode.py']);
  assert.equal(selection.candidates.length, symbols.training_epochs_completed);
  assert.ok(selection.candidates.length <= selection.configuration.epochs_limit, 'Training exceeded configured epoch limit');
  assert.deepEqual(symbols.baseline_groups, selection.baseline, 'Symbol baseline metrics differ from checkpoint selection');
  assert.deepEqual(symbols.candidate_comparisons, selection.candidates, 'Symbol candidate history differs from checkpoint selection');
  assert.deepEqual(symbols.training, { epochs_completed: selection.candidates.length,
    epochs_limit: selection.configuration.epochs_limit, learning_rate: .00002, batch_size: symbols.batch_size,
    single_bounded_round: true, baseline_checkpoint_sha256: selection.configuration.baseline_checkpoint_sha256,
    baseline_preserved: true, selected_checkpoint_sha256: selection.selected_sha256, baseline_retained: selection.baseline_retained }, 'Symbol training provenance mismatch');
  const positiveGroups = ['legacy_natural', 'legacy_stress', 'lowercase', 'capital_single', 'capital_word', 'numbers'];
  const rank = (groups) => {
    for (const name of positiveGroups) {
      const group = groups[name];
      assert.ok(group && Number.isInteger(group.total) && group.total > 0 && Number.isInteger(group.accepted) &&
        Number.isInteger(group.correct) && group.correct >= 0 && group.correct <= group.accepted && group.accepted <= group.total, `Invalid selection group counts: ${name}`);
      assert.equal(group.exact_word_accuracy, group.correct / group.total, `Invalid selection exact accuracy: ${name}`);
      assert.equal(group.accepted_accuracy, group.accepted ? group.correct / group.accepted : 0, `Invalid selection precision: ${name}`);
    }
    assert.ok(Number.isInteger(groups.rejection.total) && groups.rejection.total > 0 && Number.isInteger(groups.rejection.false_acceptances) &&
      groups.rejection.false_acceptances >= 0 && groups.rejection.false_acceptances <= groups.rejection.total, 'Invalid selection rejection counts');
    assert.equal(groups.rejection.all_rejected, groups.rejection.false_acceptances === 0);
    const regressions = ['legacy_natural', 'legacy_stress'].filter((name) => groups[name].exact_word_accuracy < selection.baseline[name].exact_word_accuracy ||
      groups[name].accepted_accuracy < selection.baseline[name].accepted_accuracy);
    const precision = Math.min(...positiveGroups.map((name) => groups[name].accepted_accuracy));
    const exact = positiveGroups.reduce((sum, name) => sum + groups[name].exact_word_accuracy, 0) / positiveGroups.length;
    const eligible = precision >= .95 && groups.rejection.all_rejected;
    return { regressions, values: [!regressions.length, eligible, eligible ? exact : precision, exact] };
  };
  const compareRank = (a, b) => {
    for (let index = 0; index < a.length; index++) {
      const difference = Number(a[index]) - Number(b[index]);
      if (Math.abs(difference) > 1e-12) return Math.sign(difference);
    }
    return 0;
  };
  let best = { sha: selection.configuration.baseline_checkpoint_sha256, groups: selection.baseline, values: rank(selection.baseline).values, baseline: true };
  for (const [index, candidate] of selection.candidates.entries()) {
    assert.equal(candidate.epoch, index + 1, 'Checkpoint candidate epoch sequence differs');
    assert.match(candidate.checkpoint_sha256, /^[a-f0-9]{64}$/);
    const computed = rank(candidate.groups);
    assert.deepEqual(candidate.regressions, computed.regressions, 'Checkpoint regression flags differ from metrics');
    assert.ok(Array.isArray(candidate.rank) && candidate.rank.length === 4 && candidate.rank.every((value) => typeof value === 'boolean' || Number.isFinite(value)) &&
      compareRank(candidate.rank, computed.values) === 0, 'Checkpoint rank differs from evaluated metrics');
    if (compareRank(computed.values, best.values) > 0) best = { sha: candidate.checkpoint_sha256, groups: candidate.groups, values: computed.values, baseline: false };
  }
  assert.equal(selection.selected_sha256, best.sha, 'Selected checkpoint is not the best eligible evaluated candidate');
  assert.equal(selection.baseline_retained, best.baseline, 'Baseline retention differs from checkpoint ranking');
  // Training selection is immutable history. Deployment adds a separate external
  // regression veto; a rejected training winner must never be relabelled baseline.
  const deployment = reports.deploymentSelection;
  assert.ok(deployment, 'Missing deployment selection evidence');
  assert.equal(deployment.schema_version, 1);
  assert.equal(deployment.decoder_version, manifest.decoderVersion);
  assert.deepEqual(deployment.thresholds, manifest.thresholds);
  assert.equal(deployment.fixture_manifest_sha256, manifest.evidence.wholeWordFixtures.sha256, 'Deployment comparison fixture mismatch');
  assert.equal(deployment.training_selection_sha256, manifest.evidence.checkpointSelection.sha256, 'Deployment training history mismatch');
  assert.equal(deployment.baseline.checkpoint_sha256, selection.configuration.baseline_checkpoint_sha256, 'Deployment baseline differs from frozen training baseline');
  assert.equal(deployment.candidate.checkpoint_sha256, selection.selected_sha256, 'Deployment candidate differs from training winner');
  const externalMetrics = (key, identity) => {
    const report = reports[key];
    assert.ok(report && manifest.evidence[key], `Missing deployment benchmark: ${key}`);
    assert.equal(identity.benchmark_sha256, manifest.evidence[key].sha256, `Deployment benchmark hash mismatch: ${key}`);
    assert.match(identity.model_sha256, /^[a-f0-9]{64}$/);
    assert.equal(report.model_sha256, identity.model_sha256, `Deployment benchmark model mismatch: ${key}`);
    assert.equal(report.decoder_version, manifest.decoderVersion, `Deployment benchmark decoder mismatch: ${key}`);
    assert.deepEqual(report.thresholds, manifest.thresholds, `Deployment benchmark thresholds mismatch: ${key}`);
    assert.equal(report.fixture_manifest_sha256, manifest.evidence.wholeWordFixtures.sha256, `Deployment benchmark fixture mismatch: ${key}`);
    assert.equal(report.words.length, fixtures.records.length);
    for (const [index, row] of report.words.entries()) {
      const { predicted, rejection, detected_cells: _cells, ...record } = row;
      assert.deepEqual(record, fixtures.records[index], `Deployment benchmark row differs: ${key}/${index}`);
      prediction({ predicted, rejection });
    }
    const result = metrics(report.words);
    for (const [field, value] of Object.entries(result)) assert.equal(report.whole_words[field], value, `Misreported deployment benchmark: ${key}/${field}`);
    assert.equal(report.negative_rows.length, 44);
    assert.equal(new Set(report.negative_rows.map((row) => row.source)).size, 44);
    report.negative_rows.forEach(prediction);
    const falseAcceptances = report.negative_rows.filter((row) => row.predicted).length;
    assert.equal(report.negative_regression.accepted, falseAcceptances);
    return { ...result, falseAcceptances };
  };
  const baselineExternal = externalMetrics('baselineBenchmark', deployment.baseline);
  const candidateExternal = externalMetrics('candidateBenchmark', deployment.candidate);
  assert.deepEqual(reports.baselineBenchmark.negative_rows.map((row) => row.source), reports.candidateBenchmark.negative_rows.map((row) => row.source), 'Deployment negative fixtures differ');
  const deploymentRegressions = [];
  if (candidateExternal.exact_word_accuracy < baselineExternal.exact_word_accuracy) deploymentRegressions.push('external_exact_accuracy_regressed');
  if ((candidateExternal.accepted_accuracy ?? 0) < (baselineExternal.accepted_accuracy ?? 0)) deploymentRegressions.push('external_accepted_accuracy_regressed');
  if (candidateExternal.falseAcceptances > baselineExternal.falseAcceptances) deploymentRegressions.push('negative_false_acceptances_increased');
  assert.deepEqual(deployment.reason_codes, deploymentRegressions, 'Deployment regression reasons differ from actual benchmark rows');
  const vetoed = deploymentRegressions.length > 0;
  assert.equal(deployment.status, vetoed ? 'baseline_retained_after_external_regression' : 'candidate_promoted', 'Deployment must respect external regression veto');
  const deployed = vetoed ? deployment.baseline : deployment.candidate;
  assert.equal(deployment.selected_checkpoint_sha256, deployed.checkpoint_sha256, 'Deployment selected the wrong checkpoint');
  assert.equal(deployment.selected_model_sha256, deployed.model_sha256, 'Deployment selected the wrong model');
  assert.equal(manifest.provenance.checkpointSha256, deployed.checkpoint_sha256, 'Bundled checkpoint bypasses deployment decision');
  assert.equal(manifest.artifact.sha256, deployed.model_sha256, 'Bundled model bypasses deployment decision');
  assert.equal(manifest.evidence.publicBenchmark.sha256, deployed.benchmark_sha256, 'Public metrics describe a model other than the deployed artifact');
  assert.equal(vetoed ? baselineExternal.falseAcceptances : candidateExternal.falseAcceptances, 0, 'Deployed model must retain zero negative false acceptances');
  assert.deepEqual(validation.groups, vetoed ? selection.baseline : best.groups, 'Deployed validation metrics differ from its evaluated checkpoint');
  const semantics = reports.symbolSemantics;
  assert.equal(semantics.decoder_version, manifest.decoderVersion);
  assert.equal(semantics.parity_passed, true);
  assert.equal(semantics.mismatch_count, 0);
  assert.ok(semantics.total >= 87);
  assert.equal(semantics.rows.length, semantics.total);
  assert.equal(new Set(semantics.rows.map((row) => row.name)).size, semantics.total);
  currentSources(semantics.source_sha256, '', ['src/ml/word-symbols.ts', 'src/ml/word-decoder.ts', 'src/types/word-scan.ts', 'training/fixtures/word-symbols-id-v1.json', 'scripts/word-decoder-parity/replay-id.cjs']);
  assert.equal(semantics.fixture_sha256, semantics.source_sha256['training/fixtures/word-symbols-id-v1.json']);
  for (const row of semantics.rows) {
    for (const field of ['parser_match', 'decoder_match', 'event_valid']) assert.equal(row[field], true, `Failed symbol semantics row: ${field}`);
    assert.match(row.tensor_sha256, /^[a-f0-9]{64}$/);
    assert.equal(row.decoder_text, row.expected);
    assert.equal(row.decoder_reason, row.masks.length ? row.expected_reason : 'no_grid');
    if (row.parser) assert.deepEqual(row.parser, parseWordMasks(row.masks), 'Recorded parser result differs from current parser');
  }
  for (const [key, expectedCount] of [['typescriptParity', 405], ['symbolDecoderParity', 516]]) {
    const report = reports[key];
    bound(report);
    currentSources(report.source_sha256, '', ['src/ml/word-symbols.ts', 'src/ml/word-decoder.ts', 'src/ml/word-image-processing.ts', 'training/scripts/word_symbols.py', 'scripts/word-decoder-parity/replay-id.cjs']);
    assert.equal(report.total, expectedCount);
    assert.equal(report.rows.length, expectedCount);
    assert.equal(report.parity_passed, true);
    assert.equal(report.mismatch_count, 0);
    for (const row of report.rows) {
      for (const field of ['quality_match', 'decoder_text_match', 'decoder_reason_match', 'pipeline_text_match', 'pipeline_reason_match'])
        assert.equal(row[field], true, `Failed TypeScript/Python ${key}: ${field}`);
      assert.notEqual(row.cell_semantics_and_boxes_match, false);
      assert.equal(row.typescript_text, row.python_text);
      assert.equal(row.typescript_reason, row.python_reason);
    }
  }
  const pixels = reports.nativePixelParity;
  currentSources(pixels.source_sha256, '', ['modules/braille-word-geometry/android/src/main/java/com/margelo/nitro/braillewordgeometry/WordPixelResize.java',
    'modules/braille-word-geometry/android/src/main/java/com/margelo/nitro/braillewordgeometry/HybridWordPhotoGeometry.kt',
    'modules/braille-word-geometry/src/WordPhotoGeometry.nitro.ts', 'src/ml/word-image-processing.ts',
    'scripts/word-native-parity/pixel-resize-parity.cjs', 'scripts/word-native-parity/pixel-resize-reference.py', 'scripts/word-native-parity/WordPixelResizeCli.java']);
  assert.equal(pixels.parity_passed, true);
  assert.equal(pixels.mismatch_count, 0);
  assert.ok(Number.isInteger(pixels.total) && pixels.total >= 16);
  assert.equal(pixels.invalid_input_and_ownership_passed, true);
  assert.equal(pixels.rows.length, pixels.total);
  assert.equal(new Set(pixels.rows.map((row) => row.name)).size, pixels.total);
  for (const row of pixels.rows) {
    for (const field of ['input_sha256', 'java_float32_sha256', 'typescript_float32_sha256', 'pillow_float32_sha256']) assert.match(row[field], /^[a-f0-9]{64}$/, `Missing pixel parity hash: ${field}`);
    assert.equal(row.java_typescript_exact, true);
    assert.equal(row.java_pillow_exact, true);
    assert.equal(row.resized_uint8_pillow_exact, true);
    for (const field of ['mean_luminance_abs_error', 'contrast_abs_error']) assert.ok(Number.isFinite(row[field]) && row[field] >= 0 && row[field] <= 1e-13, `Invalid pixel statistics parity: ${field}`);
    assert.equal(row.java_float32_sha256, row.typescript_float32_sha256);
    assert.equal(row.java_float32_sha256, row.pillow_float32_sha256);
  }

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
