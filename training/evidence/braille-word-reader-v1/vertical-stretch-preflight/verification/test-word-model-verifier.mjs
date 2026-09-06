import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { inspectTflite } from './inspect-tflite.mjs';

// These synthetic documents exercise verification rules only. The tiny FlatBuffer
// declares tensor metadata but cannot execute inference and is never an app asset.
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const temporaryRoot = resolve(tmpdir());
const stage = mkdtempSync(join(temporaryRoot, 'braille-word-verifier-'));
const sha = (value) => createHash('sha256').update(value).digest('hex');
const json = (value) => JSON.stringify(value, null, 2);
const put = (file, bytes) => {
  const target = join(stage, file);
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, bytes);
};

function metadataOnlyModel() {
  const bytes = Buffer.alloc(184);
  const shorts = (offset, values) => values.forEach((value, i) => bytes.writeUInt16LE(value, offset + i * 2));
  const ints = (offset, values) => values.forEach((value, i) => bytes.writeInt32LE(value, offset + i * 4));
  ints(0, [24]); bytes.write('TFL3', 4);
  shorts(8, [10, 16, 4, 0, 12]); ints(24, [16, 3, 0, 8]);
  ints(44, [1, 16]); shorts(52, [10, 16, 4, 8, 12]);
  ints(64, [12, 12, 56, 60]); ints(80, [2, 20, 32]);
  shorts(92, [8, 8, 4, 0]); ints(104, [12, 36]);
  shorts(112, [8, 8, 4, 0]); ints(120, [8, 40]);
  ints(128, [1, 0]); ints(136, [1, 1]);
  ints(144, [4, 1, 192, 480, 3]); ints(164, [4, 1, 48, 120, 11]);
  return bytes;
}

function metrics(rows) {
  const total = rows.length;
  const accepted = rows.filter((row) => row.predicted).length;
  const correct = rows.filter((row) => row.predicted === row.expected).length;
  return { total, accepted, correct, exact_word_accuracy: correct / total, accepted_accuracy: accepted ? correct / accepted : null };
}

function diagnosticFixtures(model, thresholds) {
  const reports = {};
  for (const kind of ['repetition', 'rejection']) {
    const mustReject = kind === 'rejection';
    const words = ['A', 'MAMA', 'KAKAK', 'MAMAMAMAMAMA'].flatMap((word) => [word, word, word]);
    const categories = ['unsupported_pattern', 'multiple_rows', 'selection_cut', 'over_limit'].flatMap((category) => [category, category, category]).concat(['blank', 'blank']);
    const records = Array.from({ length: mustReject ? 14 : 12 }, (_, index) => ({
      image: `${kind}/${index}.png`, image_sha256: sha(`${kind}/${index}`),
      expected: mustReject ? '' : words[index], expected_reject: mustReject,
      category: mustReject ? categories[index] : 'repetition',
    }));
    const fixtures = { scope: 'synthetic verification rules only', model_predictions_used: false, records };
    const rows = records.map((row) => ({ ...row, predicted: row.expected, rejection: mustReject ? 'no_grid' : '', detected_cell_count: mustReject ? 0 : row.expected.length, correct: true }));
    reports[`${kind}Fixtures`] = fixtures;
    reports[`${kind}Diagnostics`] = { scope: fixtures.scope, model_sha256: sha(model), fixture_manifest_sha256: sha(json(fixtures)),
      thresholds: { ...thresholds }, used_for_external_checkpoint_or_threshold_selection: false, is_natural_whole_word_benchmark: false,
      original_rgb_quality_checked_before_inference: true, all_frozen_image_hashes_verified: true,
      total: rows.length, correct: rows.length, accepted: mustReject ? 0 : rows.length, rows,
      by_category: Object.fromEntries([...new Set(records.map((row) => row.category))].map((category) => {
        const count = records.filter((row) => row.category === category).length;
        return [category, { total: count, correct: count, accepted: mustReject ? 0 : count }];
      })),
    };
  }
  return reports;
}

function fixture() {
  const model = metadataOnlyModel();
  const manifest = JSON.parse(readFileSync(join(root, 'assets/models/braille-word-reader-v1/manifest.json'), 'utf8'));
  manifest.artifact = { name: 'verification-unit-fixture', version: 'unit-test', status: 'unverified_candidate', file: 'unit.tflite', sha256: sha(model), sizeBytes: model.length };
  manifest.imageNormalization = { kind: 'none' };
  manifest.datasets = (manifest.datasets ?? []).filter((dataset) => dataset.name !== 'DSBI');
  // Keep tests independent of whichever candidate is currently being trained.
  const thresholds = { proposalScore: .25, acceptScore: .65, dotConfidence: .9, nmsIou: .25, minCellWidth: .008, minCellHeight: .06, edgeMargin: .008, maxRowDeviation: .55, maxGapRatio: 1.2, maxOverlapRatio: .5 };
  manifest.thresholds = { ...thresholds };
  const records = Array.from({ length: 48 }, (_, index) => ({ image: `unit/${index}.png`, source: 'unit-only', expected: 'A'.repeat(index % 12 + 1) }));
  const wholeWordFixtures = { records };
  const words = records.map((row) => ({ ...row, predicted: row.expected, rejection: '', detected_cells: [] }));
  const whole_words = { ...metrics(words), by_length: {} };
  for (let length = 1; length <= 12; length++) whole_words.by_length[length] = metrics(words.filter((row) => row.expected.length === length));
  const publicBenchmark = { model_sha256: manifest.artifact.sha256, fixture_manifest_sha256: sha(json(wholeWordFixtures)), thresholds: { ...thresholds }, numerical_internal_gates_pass: true, words, whole_words, negative_rows: Array.from({ length: 44 }, (_, index) => ({ source: `negative-${index}`, predicted: '' })), legacy_rows: Array.from({ length: 66 }, () => ({})), negative_regression: { independent_holdout: false } };
  const conversion = { artifact_sha256: sha(model), artifact_bytes: model.length, imageNormalization: { kind: 'none' }, input_shape: [1, 192, 480, 3], output_shape: [1, 48, 120, 11], thresholds: { ...thresholds }, parity_pass: true, parity_samples: 24, training_backend_vs_tensorflow_max_abs: 0, tensorflow_vs_tflite_max_abs: 0, decoded_text_parity: true, decoded_rejection_parity: true, source_checkpoint_sha256: 'a'.repeat(64), parity_fixture_sha256: 'b'.repeat(64) };
  const validationSelection = { imageNormalization: { kind: 'none' }, thresholds: { ...thresholds }, oracle: { checkpoint_sha256: conversion.source_checkpoint_sha256, fixture_sha256: conversion.parity_fixture_sha256 }, words: { total: 24 } };
  const boundaryReview = { fixture_manifest_sha256: publicBenchmark.fixture_manifest_sha256, reviewed_count: 48, automatic_geometry_passed_count: 48, automatic_geometry_failures: [], visual_reviewed_count: 48, visual_reviewed_ids: Array.from({ length: 48 }, (_, i) => i), passed: true };
  const unitResizer = 'function convertWordRawPixels() { return 0; }\nfunction bilinearWeights() { return 0; }\nfunction resizeWordRawPixels() { return 0; }';
  const sourceFiles = { 'src/ml/word-image-processing.ts': unitResizer, 'src/ml/word-model-contract.ts': '// unit-test constants', 'src/ml/word-decoder.ts': '// unit-test decoder', 'src/types/word-scan.ts': '// unit-test DTO', 'harness.template': '// unit-test harness', 'tested-source.txt': unitResizer };
  const nativePreprocessing = { source_code_sha256: Object.fromEntries(Object.entries(sourceFiles).filter(([name]) => ['src/ml/word-image-processing.ts', 'src/ml/word-model-contract.ts'].includes(name)).map(([name, source]) => [name, sha(source)])), rotation_mirroring_geometry_passed: true, typescript_antialias_pillow_pixel_parity_passed: true, cases: Array.from({ length: 7 }, (_, index) => ({ id: String(index), rotation_and_mirroring: { max_channel_error_0_255: 0 }, crop_matches_typescript: true, selection_max_error: 0, native_crop_pixel_error: { max_channel_error_0_255: 0 }, typescript_antialias_vs_pillow: { max_channel_error_0_255: 0 }, typescript_float32_vs_pillow_max_error: 0, raw_as_rgba_vs_get_pixels: { max_channel_error_0_255: 0 } })) };
  const nativeRuntime = { hermes: true, harness_source: 'harness.template', harness_entry_sha256: sha(sourceFiles['harness.template']), resizer_source_archive: 'tested-source.txt', resizer_source_sha256: sha(unitResizer), resizer_contract_sha256: sha(sourceFiles['src/ml/word-model-contract.ts']), results: Array.from({ length: 4 }, () => ({ runtimeKind: 3, tensorLength: 276480, mismatchByteCount: 0, maxAbsError: 0, jsHeartbeatCount: 5 })) };
  const pngHeader = Buffer.from('89504e470d0a1a0a', 'hex');
  manifest.nativeSelfTest = { expectedText: 'A', file: 'unit.png', sha256: sha(pngHeader), roi: { x: .1, y: .1, width: .8, height: .8 } };
  manifest.internalAcceptance = { passed: true, evaluatedWordCount: 48, acceptedWordCount: 48, exactWordSuccessRate: 1, acceptedWordAccuracy: 1, negativeImageCount: 44, negativeFalseAccepts: 0 };
  const assetModule = "export const WORD_MODEL_ASSET: number | null = require('../../assets/models/braille-word-reader-v1/unit.tflite');\nexport const WORD_NATIVE_FIXTURE_ASSET: number | null = require('../../assets/models/braille-word-reader-v1/unit.png');\n";
  const parityRows = words.map((row, index) => ({ index, image: row.image, expected: row.expected, typescript_text: row.predicted, python_text: row.predicted, typescript_reason: '', python_reason: '', quality_bypassed_decoder: false, saved_output_is_zero_placeholder: false, quality_match: true, decoder_text_match: true, decoder_reason_match: true, pipeline_text_match: true, pipeline_reason_match: true }));
  const decoderParity = { schema_version: 1, model_sha256: sha(model), benchmark_sha256: sha(json(publicBenchmark)), fixture_manifest_sha256: publicBenchmark.fixture_manifest_sha256, thresholds: { ...thresholds }, parity_passed: true, first_mismatches: [], total_pipeline_cases: 48, actual_model_output_decoder_cases: 48, quality_bypass_cases: 0, zero_placeholder_cases: 0, quality_matches: 48, actual_decoder_text_matches: 48, actual_decoder_rejection_matches: 48, pipeline_text_matches: 48, pipeline_rejection_matches: 48, source_sha256: Object.fromEntries(Object.entries(sourceFiles).filter(([file]) => file.endsWith('.ts')).map(([file, source]) => [file, sha(source)])), rows: parityRows, outcomes_sha256: sha(JSON.stringify(parityRows)) };
  const diagnostics = diagnosticFixtures(model, thresholds);
  const diagnosticSources = Object.fromEntries(['word_diagnostics.py', 'word_benchmark.py', 'word_decode.py', 'word_data.py']
    .map((name) => [name, `# test-only ${name}`]));
  for (const [name, source] of Object.entries(diagnosticSources)) sourceFiles[`training/scripts/${name}`] = source;
  for (const kind of ['repetition', 'rejection']) {
    diagnostics[`${kind}Diagnostics`].source_sha256 = Object.fromEntries(Object.entries(diagnosticSources).map(([name, source]) => [`training/scripts/${name}`, sha(source)]));
    for (const [index, record] of diagnostics[`${kind}Fixtures`].records.entries()) {
      sourceFiles[`evidence/${record.image}`] = `${kind}/${index}`;
    }
  }
  const trainingSource = '// original training source';
  sourceFiles['retained/word_model.py'] = trainingSource;
  const sourceEntry = { file: 'retained/word_model.py', sha256: sha(trainingSource) };
  validationSelection.source_checkpoint = 'unit-run/epoch-06.keras';
  validationSelection.oracle.source_sha256 = { 'word_model.py': sourceEntry.sha256 };
  const trainingManifest = json({ records: ['unit training record'] });
  const validationManifest = json({ records: ['unit validation record'] });
  conversion.parity_fixture_sha256 = sha(validationManifest);
  validationSelection.oracle.fixture_sha256 = conversion.parity_fixture_sha256;
  const dataManifestIndex = { files: [trainingManifest, validationManifest, json(wholeWordFixtures)].map((original, index) => {
    const compressed = gzipSync(original), file = `data-${index}.json.gz`;
    sourceFiles[`evidence/${file}`] = compressed;
    return { file, sha256: sha(original), sizeBytes: Buffer.byteLength(original), gzipSha256: sha(compressed), gzipSizeBytes: compressed.length };
  }) };
  const trainingConfiguration = { source_sha256: { 'word_model.py': sourceEntry.sha256 }, data_sha256: { 'training.json': sha(trainingManifest) } };
  const samplingHistory = [{ epoch: 6, complete: true }];
  const samplingLineage = { status: 'passed', actual: { config_sha256: sha(json(trainingConfiguration)), manifest_sha256: sha(trainingManifest), sampling_history_sha256: sha(json(samplingHistory)), observed_epochs: [{ epoch: 6, checkpoint_sha256: conversion.source_checkpoint_sha256, eligible_by_sampling_lineage: true, all_positive_records_seen_in_this_run: true }] }, eligible_retained_epochs: [6] };
  const checkpointSelection = { selected_checkpoint: validationSelection.source_checkpoint, selected_checkpoint_sha256: conversion.source_checkpoint_sha256, fixture_sha256: conversion.parity_fixture_sha256, validation_words: 24, external_outputs_used: false, candidates: [{ checkpoint_sha256: conversion.source_checkpoint_sha256, source_checkpoint: validationSelection.source_checkpoint, thresholds: { ...thresholds }, imageNormalization: { kind: 'none' }, both_group_precision_gates_pass: true, validation_groups: { natural: { total: 12, accepted: 12, correct: 12, exact_word_accuracy: 1, accepted_accuracy: 1 }, stress: { total: 12, accepted: 12, correct: 12, exact_word_accuracy: 1, accepted_accuracy: 1 } } }] };
  const reportLink = (name, report) => ({ file: `evidence/${name}.json`, sha256: sha(json(report)) });
  const trainingProvenance = { selectedCheckpointSha256: conversion.source_checkpoint_sha256, selectedSourceCheckpoint: validationSelection.source_checkpoint, selectionSources: [sourceEntry], selectionReport: reportLink('checkpointSelection', checkpointSelection), trainingRuns: [{ run: 'unit-run', configuration: reportLink('trainingConfiguration', trainingConfiguration), sources: [sourceEntry], unavailable_original_sources: [], 'sampling-history.json': reportLink('samplingHistory', samplingHistory), 'sampling-lineage-final.json': reportLink('samplingLineage', samplingLineage) }] };
  manifest.provenance = { checkpointSha256: conversion.source_checkpoint_sha256, validationFixtureSha256: conversion.parity_fixture_sha256, externalFixtureSha256: publicBenchmark.fixture_manifest_sha256, retainedTrainingProvenance: reportLink('trainingProvenance', trainingProvenance), retainedDataManifests: reportLink('dataManifestIndex', dataManifestIndex), selectionSources: [sourceEntry], selectionReport: 'evidence/checkpointSelection.json' };
  return { manifest, model, pngHeader, assetModule, sourceFiles, reports: { publicBenchmark, wholeWordFixtures, conversion, validationSelection, boundaryReview, nativePreprocessing, nativeRuntime, decoderParity, ...diagnostics, trainingProvenance, trainingConfiguration, samplingHistory, samplingLineage, checkpointSelection, dataManifestIndex } };
}

function withSupplementalDSBI(data, mutate = () => {}) {
  const revision = '6937eb8973efe5ff337e5c018485cb626303f5c2';
  const coordinateSpace = 'provided_deskewed_recto_image_pixels';
  const members = [
    ...Array.from({ length: 10 }, (_, i) => `Massage/M+${i + 1}.jpg`),
    ...Array.from({ length: 10 }, (_, i) => `Math/math+${i + 1}.jpg`),
    ...Array.from({ length: 3 }, (_, i) => `Ordinary Printed Document/OPD+${i + 1}.jpg`),
    ...Array.from({ length: 3 }, (_, i) => `Shaver Yang Fengting/SYF+${i + 3}.jpg`),
  ];
  // Synthetic metadata only: these do not claim to reproduce the real DSBI cells.
  const sources = members.map((member) => {
    const recto = member.replace(/\.jpg$/, '+recto.jpg');
    return { source: `dsbi/${recto}`, source_family: `dsbi/${member.slice(0, member.lastIndexOf('/'))}`,
      split: 'train', train_member: member, original_image: `data/${recto}`,
      original_annotation: `data/${recto.replace(/\.jpg$/, '.txt')}`, image_sha256: sha(`source ${member}`),
      annotation_sha256: sha(`annotation ${member}`), source_coordinate_space: coordinateSpace,
      positive_cell_count: 1, generated_record_count: 1 };
  });
  const records = sources.map((source, i) => ({ image: `dsbi/${i}.png`, image_sha256: sha(`PNG ${i}`),
    source: source.source, source_family: source.source_family, source_coordinate_space: coordinateSpace,
    boxes: [[10, 20, 30, 60]], masks: [1], originally_complete: [true], ignore_boxes: [[40, 20, 60, 60]] }));
  const preparationSources = ['word_dsbi.py', 'word_dsbi_prepare.py'].map((name) => {
    const file = `retained/dsbi/source/${name}`, bytes = `# synthetic original ${name}`;
    data.sourceFiles[file] = bytes;
    return { file, sha256: sha(bytes) };
  });
  const protocol = { dataset_revision: revision, source_coordinate_space: coordinateSpace,
    parser_sha256: preparationSources[0].sha256, generator_sha256: preparationSources[1].sha256,
    created_at_utc: '2026-09-05T00:00:00Z' };
  const index = { schemaVersion: 1, dataset: 'DSBI', revision, source_coordinate_space: coordinateSpace,
    testImagePixelsAccessed: false, trainList: 'train.txt', testList: 'test.txt',
    trainListSha256: '3a679245733600ee78640254f00b850d7ea7de3f413e7274d0e34ccaf9d3da9c',
    testListSha256: '347836183ea3cd9a285f80eb0902cc894173e184ed6ac2b7e425581e702a02e3', sources,
    generatedManifest: { path: 'train.json', recordCount: 26 }, preparationProtocol: {} };
  const snapshot = { files: preparationSources.map((source) => ({ archived: `source/${basename(source.file)}`, sha256: source.sha256 })) };
  const audit = { status: 'passed', conservative_boundary_flags: [], original_sources: 26, contexts: 26,
    unique_positive_source_cells: 26, complete_unique_source_cells_seen: 26,
    all_intersecting_positive_occurrences_checked: 26, all_unknown_ignore_boxes_checked: 26,
    pixel_bytes_compared: 26 * 480 * 192 * 3, pixel_mismatches: 0, max_coordinate_error: 0 };
  const docs = { index, records, protocol, snapshot, audit, omitArchive: null };
  mutate(docs);
  const link = (name, value) => {
    data.reports[name] = value;
    return { file: `evidence/${name}.json`, sha256: sha(json(value)) };
  };
  const protocolLink = link('dsbiPreparationProtocol', protocol);
  index.generatedManifest.sha256 ??= sha(json(records));
  index.preparationProtocol.sha256 ??= protocolLink.sha256;
  const indexLink = link('dsbiSourceIndex', index);
  snapshot.source_index_sha256 ??= indexLink.sha256;
  snapshot.manifest_sha256 ??= index.generatedManifest.sha256;
  snapshot.preparation_protocol_sha256 ??= protocolLink.sha256;
  audit.source_index_sha256 ??= indexLink.sha256;
  audit.manifest_sha256 ??= index.generatedManifest.sha256;
  const preparation = { dataset: 'DSBI', revision, sourceIndex: indexLink,
    generatedManifestSha256: index.generatedManifest.sha256, sources: preparationSources,
    'source-archive.json': link('dsbiSourceArchive', snapshot), 'preparation-protocol.json': protocolLink,
    'independent-context-audit.json': link('dsbiContextAudit', audit) };
  data.reports.trainingProvenance.supplementalPreparations = [preparation];
  data.manifest.datasets.push({ name: 'DSBI', repository: 'https://github.com/yeluo1994/DSBI', revision });
  const baseRecords = [{ image: 'base.png', source: 'original TRAIN', boxes: [[10, 20, 30, 60]], masks: [1] }];
  const combined = [...baseRecords, ...records];
  docs.mutateCombined?.(combined);
  const indexPath = '../dsbi/source-index.json';
  const config = data.reports.trainingConfiguration;
  Object.assign(config, { train_manifest: 'combined.json', base_train_manifest: 'base.json',
    started_at_utc: '2026-09-05T00:02:00Z', supplemental_source_indexes: [indexPath],
    data_sha256: { 'combined.json': sha(json(combined)), 'base.json': sha(json(baseRecords)), [indexPath]: indexLink.sha256 } });
  const trainingProtocol = { created_at_utc: '2026-09-05T00:01:00Z', source_sha256: config.source_sha256,
    supplemental_source_indexes: [indexPath], training_manifest_sha256: config.data_sha256['combined.json'],
    base_manifest_sha256: config.data_sha256['base.json'], supplemental_source_index_sha256: indexLink.sha256,
    supplemental_manifest_sha256: index.generatedManifest.sha256,
    independent_context_audit_sha256: preparation['independent-context-audit.json'].sha256 };
  const run = data.reports.trainingProvenance.trainingRuns[0];
  run.configuration = link('trainingConfiguration', config);
  run['dsbi-addition-protocol.json'] = link('dsbiTrainingProtocol', trainingProtocol);
  data.reports.samplingLineage.actual.config_sha256 = run.configuration.sha256;
  data.reports.samplingLineage.actual.manifest_sha256 = config.data_sha256['combined.json'];
  for (const [name, document] of Object.entries({ index, records, base: baseRecords, combined })) {
    if (docs.omitArchive === name) continue;
    const original = json(document), compressed = gzipSync(original), file = `dsbi-${name}.json.gz`;
    data.sourceFiles[`evidence/${file}`] = compressed;
    data.reports.dataManifestIndex.files.push({ file, sha256: sha(original), sizeBytes: Buffer.byteLength(original),
      gzipSha256: sha(compressed), gzipSizeBytes: compressed.length });
  }
  data.manifest.provenance.retainedDataManifests.sha256 = sha(json(data.reports.dataManifestIndex));
}

function rebindSupplementalRunReports(data) {
  const run = data.reports.trainingProvenance.trainingRuns[0];
  run.configuration.sha256 = sha(json(data.reports.trainingConfiguration));
  run['dsbi-addition-protocol.json'].sha256 = sha(json(data.reports.dsbiTrainingProtocol));
  data.reports.samplingLineage.actual.config_sha256 = run.configuration.sha256;
}

function withSupplementalPoints(data, mutate = () => {}) {
  let supplemental;
  const raw = '0\n10 30\n20 37 60\n1 1 1 0 0 0 0 0\n';
  withSupplementalDSBI(data, (docs) => {
    supplemental = docs;
    for (const source of docs.index.sources) Object.assign(source, { image_size: [480, 192], annotation_sha256: sha(raw) });
    for (const row of docs.records) Object.assign(row, { crop: [0, 0, 480, 192], source_cells: [{ row: 1, col: 1, annotation_line: 4 }] });
  });
  const preparation = data.reports.trainingProvenance.supplementalPreparations[0];
  preparation.sourceAnnotations = supplemental.index.sources.map((source, i) => {
    const file = `retained/dsbi/annotations/${i}.txt`;
    data.sourceFiles[file] = raw;
    return { source: source.source, file, sha256: sha(raw) };
  });
  const metadata = { schemaVersion: 1, coordinate_space: 'source_pixel_index_centers',
    indexes: [{ path: 'original/source-index.json', sha256: preparation.sourceIndex.sha256, manifest_sha256: preparation.generatedManifestSha256 }],
    records: Object.fromEntries(supplemental.records.map((row) => [row.image, {
      source: row.source, image_sha256: row.image_sha256, crop: [...row.crop], masks: [...row.masks],
      source_cells: row.source_cells.map((identity) => ({ ...identity })),
      source_points: [[[10, 20], [10, 37], [10, 60], [30, 20], [30, 37], [30, 60]]],
    }])) };
  const config = data.reports.trainingConfiguration, run = data.reports.trainingProvenance.trainingRuns[0];
  run.sources.push(preparation.sources.find((entry) => basename(entry.file) === 'word_dsbi.py'));
  for (const name of ['word_point_metadata.py', 'word_point_geometry.py']) {
    const file = `retained/${name}`, bytes = `# synthetic original ${name}`;
    data.sourceFiles[file] = bytes;
    run.sources.push({ file, sha256: sha(bytes) });
  }
  for (const entry of run.sources) config.source_sha256[basename(entry.file)] = entry.sha256;
  Object.assign(config, { epochs: 10, resume_sha256: 'a'.repeat(64), dsbi_point_metadata: 'points.json',
    partial_geometry_supervision: { enabled: false }, dsbi_visible_partial_supervision: { enabled: true,
      minimum_visible_size_pixels: [4, 12], metadata: { records: 26, known_cell_occurrences: 26 } } });
  const protocol = { created_at_utc: '2026-09-05T00:01:00Z', enabled: true, metadata_path: 'points.json',
    source_sha256: { ...config.source_sha256 }, training_manifest_sha256: config.data_sha256['combined.json'],
    supplemental_source_index_sha256: preparation.sourceIndex.sha256, supplemental_manifest_sha256: preparation.generatedManifestSha256,
    parent_checkpoint_sha256: config.resume_sha256, planned_epochs: 10, minimum_visible_size_pixels: [4, 12],
    minimum_surviving_occupied_centers: 1, full_span_canvas_dimensions_excluded: true };
  const docs = { metadata, protocol, preparation, config, run, omitArchive: false };
  mutate(docs);
  const bytes = json(metadata), digest = sha(bytes);
  data.reports.dsbiPointMetadata = metadata;
  run.dsbiPointMetadata = { file: 'evidence/dsbiPointMetadata.json', sha256: digest };
  config.data_sha256['points.json'] = digest;
  config.dsbi_visible_partial_supervision.metadata.sha256 ??= digest;
  protocol.point_metadata_sha256 ??= digest;
  data.reports.dsbiPointProtocol = protocol;
  run['dsbi-visible-partial-protocol.json'] = { file: 'evidence/dsbiPointProtocol.json', sha256: sha(json(protocol)) };
  if (!docs.omitArchive) {
    const compressed = gzipSync(bytes), file = 'dsbi-points.json.gz';
    data.sourceFiles[`evidence/${file}`] = compressed;
    data.reports.dataManifestIndex.files.push({ file, sha256: digest, sizeBytes: Buffer.byteLength(bytes),
      gzipSha256: sha(compressed), gzipSizeBytes: compressed.length });
  }
  data.manifest.provenance.retainedDataManifests.sha256 = sha(json(data.reports.dataManifestIndex));
  rebindSupplementalRunReports(data);
}

function withGeometryInitialization(data, mutate = () => {}) {
  withSupplementalPoints(data);
  withImageNormalization(data);
  const config = data.reports.trainingConfiguration, run = data.reports.trainingProvenance.trainingRuns[0];
  const base = 'b'.repeat(64), zero = 'c'.repeat(64), diagnostic = 'd'.repeat(64), latticeCheckpoint = 'e'.repeat(64), oracle = 'f'.repeat(64);
  const link = (name, value) => { data.reports[name] = value; return { file: `evidence/${name}.json`, sha256: sha(json(value)) }; };
  const sources = [];
  for (const name of ['word_data.py', 'word_model.py', 'word_geometry_head.py', 'word_geometry_init.py', 'word_geometry_parity.py',
    'audit_slot_priors.py', 'audit_feature_lattice.py', 'independent-coordinate-preflight.py', 'independent-geometry-head-audit.py',
    'test_word_geometry_head.py', 'probe_geometry_training_batch.py']) {
    let entry = run.sources.find((row) => basename(row.file) === name);
    if (!entry) {
      const file = `retained/geometry/${name}`, bytes = `# synthetic geometry ${name}`;
      data.sourceFiles[file] = bytes;
      entry = { file, sha256: sha(bytes) };
      if (name.startsWith('word_')) run.sources.push(entry);
    }
    sources.push(entry);
  }
  for (const entry of run.sources) config.source_sha256[basename(entry.file)] = entry.sha256;
  const sourceHash = (name) => sources.find((entry) => basename(entry.file) === name).sha256;
  const sourceSubset = Object.fromEntries(['word_data.py', 'word_model.py', 'word_geometry_head.py', 'word_geometry_init.py'].map((name) => [name, sourceHash(name)]));
  const priors = [[6.5 / 32, 6.5 / 52], [6.5 / 32, 23.5 / 52], [6.5 / 32, 46.5 / 52],
    [26.5 / 32, 6.5 / 52], [26.5 / 32, 23.5 / 52], [26.5 / 32, 46.5 / 52]];
  const sampling = { slot_fractions: priors, feature_origin: 3, stride: 4 };
  const geometry = { enabled: true, base_parent_checkpoint_sha256: base, zero_checkpoint_sha256: zero,
    sampling, residual_channels: 11, projection_channels: 8, hidden_units: 32, added_parameters: 2451 };
  Object.assign(config, { resume_sha256: zero, geometry_conditioned_residual: geometry, batch_size: 24,
    learning_rate: .00002, geometry_loss: 'relative_size_huber', horizontal_mirror_augmentation: false,
    image_normalization: 'channel_gray_std05', imageNormalization: structuredClone(data.manifest.imageNormalization),
    batch_normalization: { layers: 52, trainable_affine_layers: 0, backbone_call_training_keywords: [false] },
    negative_expansion: { base_repeat: 27, supplemental_repeat: 2 } });
  const preparation = data.reports.trainingProvenance.supplementalPreparations[0];
  const pointHash = run.dsbiPointMetadata.sha256;
  const slots = { status: 'passed', script_sha256: sourceHash('audit_slot_priors.py'), point_metadata_sha256: pointHash,
    manifest_sha256: preparation.generatedManifestSha256, unique_cells: 26, median_corner_fractions: structuredClone(priors),
    complete_cached_occurrences: 26, max_exact_cached_reconstruction_error_pixels: 0, max_same_source_fraction_difference_across_crops: 0,
    slot_order: ['left_top', 'left_middle', 'left_bottom', 'right_top', 'right_middle', 'right_bottom'], axis_order: ['x', 'y'] };
  const lattice = { status: 'passed', script_sha256: sourceHash('audit_feature_lattice.py'), checkpoint_sha256: latticeCheckpoint,
    backbone_outputs: { block_3_expand_relu: { size: [48, 120], jump: [4, 4], origin: [3, 3] } } };
  const identity = { status: 'passed', candidates: [latticeCheckpoint, base].map((checkpoint_sha256) => ({ checkpoint_sha256, backbone_config_sha256: sha('same graph') })) };
  const coordinates = { status: 'passed', script_sha256: sourceHash('independent-coordinate-preflight.py'),
    point_metadata_sha256: pointHash, manifest_sha256: preparation.generatedManifestSha256,
    source_index_sha256: preparation.sourceIndex.sha256, checkpoint_sha256: latticeCheckpoint,
    unique_raw_nonzero_cells: 26, raw_source_medians: structuredClone(priors), defects: [],
    unique_complete_seen: 26, cached_occurrences: 26, complete_cached_occurrences: 26,
    raw_annotations: preparation.sourceAnnotations.map(({ source, sha256 }) => ({ source, sha256 })),
    all_reported_fraction_and_pixel_error_percentiles_match: true };
  const init = { status: 'passed', created_at_utc: '2026-09-05T00:00:20Z', parent_checkpoint_sha256: base,
    zero_checkpoint_sha256: zero, nonzero_diagnostic_sha256: diagnostic, nonzero_diagnostic_must_never_be_resumed: true,
    source_sha256: sourceSubset, zero_outputs_equal_parent: true, zero_save_reload_all_weights_equal: true,
    initial_final_residual_gradient_l1_by_channel: Array(11).fill(1), zero_branch_gradients: { projection_l1: 0, hidden_l1: 0 },
    after_isolated_final_step_gradients: { projection_l1: 1, hidden_l1: 1 }, nonzero_max_output_delta: .01,
    oracle_sha256: oracle, added_parameters: 2451, sampling: structuredClone(sampling) };
  const proof = { status: 'passed', created_at_utc: '2026-09-05T00:00:40Z', source_sha256: sourceHash('word_geometry_parity.py'),
    oracle_sha256: oracle, conversion_probes_must_not_be_activated_or_resumed: true, trained_candidate_requires_full405_parity: true,
    absolute_tolerance: .0001, relative_tolerance: .0001, probes: Object.fromEntries([['zero', zero], ['nonzero', diagnostic]].map(([name, digest]) =>
      [name, { checkpoint_sha256: digest, artifact_sha256: sha(name), samples: 6, all_finite: true, builtins_only: true,
        operators: ['GATHER', 'FLOOR', 'FULLY_CONNECTED'], torch_vs_tf_max_abs: .00001, tf_vs_tflite_max_abs: .00001, torch_vs_tflite_max_abs: .00001 }])) };
  const head = { status: 'passed', script_sha256: sourceHash('independent-geometry-head-audit.py'),
    test_source_sha256: sourceHash('test_word_geometry_head.py'), source_sha256: sourceHash('word_geometry_head.py'),
    parent_checkpoint_sha256: base, zero_checkpoint_sha256: zero, oracle_sha256: oracle, added_parameters: 2451,
    final_projection_all_zero: true, head_kernel_duplication: false, inverse_sigmoid_reconstruction: false,
    backbone_called_training_false: true, unchanged_old_normalization_and_feature_graph: true,
    inherited_head_filters: [1, 4, 6], input_shape: [null, 192, 480, 3], output_shape: [null, 48, 120, 11],
    all_inherited_arrays_equal: 280, frozen_batchnorm_layers: 52, stored_zero_oracle_equal_inputs: 6,
    stored_nonzero_max_delta: .01, defects: [], multi_batch_scalar_oracle: {
      batch: 3, out_of_range_slots_clamped: 7, batched_equals_separate_calls: true, max_absolute_error: .000001 } };
  const gpu = { status: 'passed', script_sha256: sourceHash('probe_geometry_training_batch.py'),
    source_sha256: { ...sourceSubset }, parent_sha256: zero, batch_size: 24, loss: .2, gradient_tensors: 78,
    point_metadata_sha256: pointHash, negative_expansion: structuredClone(config.negative_expansion) };
  const protocol = { created_at_utc: '2026-09-05T00:01:00Z', geometry_conditioned_residual: true,
    source_sha256: { ...config.source_sha256 }, base_parent_checkpoint_sha256: base, zero_parent_checkpoint_sha256: zero,
    point_metadata_sha256: pointHash, training_manifest_sha256: config.data_sha256[config.train_manifest],
    sampling: structuredClone(sampling), added_parameters: 2451, batch_size: 24, planned_epochs: 10,
    planned_eligible_epochs: [8, 9, 10], initial_learning_rate: .00002, geometry_loss: 'relative_size_huber',
    dsbi_visible_partial_supervision: true, horizontal_mirror_augmentation: false };
  const retained = { baseParentCheckpointSha256: base, zeroParentCheckpointSha256: zero,
    forbiddenDiagnosticCheckpointSha256: diagnostic, sources };
  const ancestorConfig = { source_sha256: {}, data_sha256: { train: config.data_sha256[config.train_manifest] } };
  const ancestorHistory = [{ epoch: 9, complete: true }];
  const ancestor = { run: 'base-parent-run', sources: [], configuration: link('geometryAncestorConfig', ancestorConfig),
    'sampling-history.json': link('geometryAncestorHistory', ancestorHistory) };
  const ancestorAudit = { status: 'passed', eligible_retained_epochs: [9], actual: {
    config_sha256: ancestor.configuration.sha256, sampling_history_sha256: ancestor['sampling-history.json'].sha256,
    manifest_sha256: ancestorConfig.data_sha256.train, observed_epochs: [{ epoch: 9, checkpoint_sha256: base, eligible_by_sampling_lineage: true }] } };
  const docs = { retained, geometry, config, init, proof, slots, lattice, identity, coordinates, head, gpu, protocol, ancestorAudit, omitProof: null };
  mutate(docs);
  const add = (key, doc) => { retained[key] = link(`geometry${key}`, doc); return retained[key].sha256; };
  const slotHash = add('slotPriorAudit', slots), latticeHash = add('featureLatticeAudit', lattice);
  init.slot_prior_audit_sha256 ??= slotHash;
  geometry.slot_prior_audit_sha256 ??= slotHash;
  protocol.slot_prior_audit_sha256 ??= slotHash;
  identity.lattice_audit_sha256 ??= latticeHash;
  coordinates.audited_files ??= { 'audit_feature_lattice.py': lattice.script_sha256, 'feature-lattice-audit.json': latticeHash,
    'audit_slot_priors.py': slots.script_sha256, 'slot-prior-audit.json': slotHash };
  const initHash = add('initializationReport', init);
  geometry.initialization_report_sha256 ??= initHash;
  protocol.initialization_report_sha256 ??= initHash;
  proof.initialization_report_sha256 ??= initHash;
  head.initialization_report_sha256 ??= initHash;
  const proofHash = add('nonzeroConversionProof', proof);
  geometry.conversion_report_sha256 ??= proofHash;
  protocol.conversion_report_sha256 ??= proofHash;
  add('backboneLatticeIdentity', identity); add('independentCoordinateAudit', coordinates);
  add('independentHeadAudit', head); add('gpuBatchPreflight', gpu);
  protocol.preflight_audits ??= ['initializationReport', 'nonzeroConversionProof', 'independentCoordinateAudit', 'independentHeadAudit', 'gpuBatchPreflight']
    .map((key) => ({ path: retained[key].file, sha256: retained[key].sha256 }));
  add('protocol', protocol);
  run['geometry-residual-protocol.json'] = { ...retained.protocol };
  if (docs.omitProof) delete retained[docs.omitProof];
  run.geometryInitialization = retained;
  ancestor['sampling-lineage-final.json'] = link('geometryAncestorAudit', ancestorAudit);
  data.reports.trainingProvenance.trainingRuns.push(ancestor);
  data.reports.dsbiPointProtocol.source_sha256 = { ...config.source_sha256 };
  data.reports.dsbiPointProtocol.parent_checkpoint_sha256 = config.resume_sha256;
  run['dsbi-visible-partial-protocol.json'].sha256 = sha(json(data.reports.dsbiPointProtocol));
  rebindSupplementalRunReports(data);
}

function withHorizontalMirror(data, mutate = () => {}) {
  // Reuse the synthetic retained ancestor and point-data setup; this run does
  // not use the geometry branch or its initialization protocol.
  withGeometryInitialization(data);
  const config = data.reports.trainingConfiguration, run = data.reports.trainingProvenance.trainingRuns[0];
  const ancestor = data.reports.trainingProvenance.trainingRuns[1];
  const parent = run.geometryInitialization.baseParentCheckpointSha256;
  delete run.geometryInitialization;
  delete run['geometry-residual-protocol.json'];
  const link = (name, value) => { data.reports[name] = value; return { file: `evidence/${name}.json`, sha256: sha(json(value)) }; };
  for (const name of ['word_sampling.py', 'word_train.py']) {
    const source = { file: `retained/${name}`, sha256: sha(`# synthetic ${name}`) };
    data.sourceFiles[source.file] = `# synthetic ${name}`;
    run.sources.push(source);
    config.source_sha256[name] = source.sha256;
  }
  Object.assign(config, { resume_sha256: parent, geometry_conditioned_residual: { enabled: false },
    horizontal_mirror_augmentation: true, half_turn_augmentation: false, polarity_augmentation: false,
    illumination_augmentation: true, seed: 20260905, augmentation_protocol: { horizontal_mirror_probability: .5,
      horizontal_mirror_bit_mapping: [4, 5, 6, 1, 2, 3], half_turn_probability: 0, polarity_inversion_probability: 0 } });
  Object.assign(config.negative_expansion, { base_unique_negative_records: 44, supplemental_unique_negative_records: 47 });
  const script = { file: 'retained/audit_run22_mirror.py', sha256: sha('# original mirror audit') };
  data.sourceFiles[script.file] = '# original mirror audit';
  const audit = { status: 'passed', defects: [], script_sha256: script.sha256, source_sha256: { ...config.source_sha256 },
    run20_config_sha256: ancestor.configuration.sha256, training_manifest_sha256: config.data_sha256[config.train_manifest],
    point_metadata_sha256: run.dsbiPointMetadata.sha256, all_known_mask_mirror_involutions_and_target_labels: 63,
    partial_unknown_bit_checks: 63, stage_visibility_cases: 252, point_slot_permutation: [3, 4, 5, 0, 1, 2],
    run20_data_point_metadata_sampling_sources_byte_identical: true, run20_patches_ast_identical: true,
    mirror_pixel_operation_exact: true, full_balanced_order_identical: true, off_mode_rgb_targets_rng_exact_vs_run20: true,
    on_off_random_calls_and_final_rng_states_equal: true, metadata_and_cached_point_states_unchanged: true,
    real_batches: [{}], mirror_invocations: 1, expanded_negative_exposure: { base: [44, 27], supplemental: [47, 2] } };
  const protocol = { created_at_utc: '2026-09-05T00:01:00Z', source_sha256: { ...config.source_sha256 },
    parent_checkpoint_sha256: parent, parent_training_config_sha256: ancestor.configuration.sha256,
    parent_sampling_lineage_sha256: ancestor['sampling-lineage-final.json'].sha256,
    training_manifest_sha256: config.data_sha256[config.train_manifest], point_metadata_sha256: run.dsbiPointMetadata.sha256,
    horizontal_mirror_augmentation: true, half_turn_augmentation: false, illumination_augmentation: true,
    planned_epochs: 10, planned_eligible_epochs: [8, 9, 10], batch_size: 24, learning_rate: .00002, seed: 20260905,
    geometry_loss: 'relative_size_huber', negative_repeat: 27, supplemental_negative_repeat: 2,
    dsbi_visible_partial_supervision: true, partial_geometry_supervision: false, geometry_conditioned_residual: false,
    frozen_batchnorm: true, no_activation_before_original_gates: true,
    mirror: { probability: .5, slot_permutation_zero_based: [3, 4, 5, 0, 1, 2] } };
  const retained = { parentCheckpointSha256: parent, sources: [script] };
  const docs = { config, protocol, audit, retained, omitAudit: false };
  mutate(docs);
  const auditLink = link('mirrorIndependentAudit', audit);
  retained.audits = docs.omitAudit ? [] : [auditLink];
  protocol.preflight_audits ??= [{ path: auditLink.file, sha256: auditLink.sha256 }];
  retained.protocol = link('mirrorProtocol', protocol);
  run['horizontal-mirror-protocol.json'] = { ...retained.protocol };
  run.horizontalMirrorPreflight = retained;
  data.reports.dsbiPointProtocol.source_sha256 = { ...config.source_sha256 };
  data.reports.dsbiPointProtocol.parent_checkpoint_sha256 = config.resume_sha256;
  run['dsbi-visible-partial-protocol.json'].sha256 = sha(json(data.reports.dsbiPointProtocol));
  rebindSupplementalRunReports(data);
}

function withVerticalStretch(data, mutate = () => {}) {
  withHorizontalMirror(data);
  const config = data.reports.trainingConfiguration, run = data.reports.trainingProvenance.trainingRuns[0];
  const protocol = structuredClone(data.reports.mirrorProtocol);
  delete run.horizontalMirrorPreflight;
  delete run['horizontal-mirror-protocol.json'];
  delete protocol.mirror;
  const source = { file: 'retained/word_vertical_stretch.py', sha256: sha('# synthetic vertical stream') };
  data.sourceFiles[source.file] = '# synthetic vertical stream';
  run.sources.push(source); config.source_sha256['word_vertical_stretch.py'] = source.sha256;
  Object.assign(config, { horizontal_mirror_augmentation: false, vertical_stretch_augmentation: true,
    augmentation_protocol: { vertical_stretch_range: [.9, 1.1], vertical_stretch_affine_probability: .85, vertical_stretch_stream_id: 0x56535452 } });
  const script = { file: 'retained/audit_vertical_stretch_independent.py', sha256: sha('# original vertical audit') };
  data.sourceFiles[script.file] = '# original vertical audit';
  Object.assign(protocol, { horizontal_mirror_augmentation: false, polarity_augmentation: false, vertical_stretch_augmentation: true,
    source_sha256: { ...config.source_sha256 }, reviewed_plan_sha256: 'a'.repeat(64),
    vertical_stretch: { range: [.9, 1.1], distribution: 'uniform', existing_affine_probability: .85,
      base_zoom: [.8, 1.7], scale_x: 'zoom', scale_y: 'zoom*height_factor', stream_id: 0x56535452,
      seed_sequence: ['seed', 'dataset_epoch', 'batch_index', 0x56535452] } });
  const audit = { status: 'passed', defects: [], source_sha256: { ...config.source_sha256 }, script_sha256: script.sha256,
    plan_sha256: protocol.reviewed_plan_sha256, training_manifest_sha256: protocol.training_manifest_sha256,
    point_metadata_sha256: protocol.point_metadata_sha256, all63_stage_point_visibility_checks: 252,
    plan_source_flags_and_range_verified: true, stream_identity_and_prefixes_passed: true, global_numpy_rng_unchanged: true,
    run20_off_pixels_targets_rng_byte_identical: true, on_off_main_rng_calls_and_final_state_equal: true,
    actual_augment_cut_then_rotation_return_never_recovers_complete_or_occupied_point: true,
    actual_patches_factor_order_matches_independent_stream: true, factor_does_nothing_when_existing_affine_is_skipped: true,
    evaluation_mode_ignores_flag_and_matches_run20: true, repeated_batch_order_independent: true,
    full_balanced_order_identical: true, cached_point_states_unchanged: true, real_batches: [{}],
    target_checks: { complete: 1, geometry_only: 1, unknown_boxes: 1 },
    pillow_marker_checks: [.9, 1, 1.1].map((factor) => ({ factor, max_centroid_error_pixels: .01 })) };
  const retained = { parentCheckpointSha256: protocol.parent_checkpoint_sha256, sources: [script] };
  const docs = { config, protocol, audit, retained };
  mutate(docs);
  const link = (name, value) => { data.reports[name] = value; return { file: `evidence/${name}.json`, sha256: sha(json(value)) }; };
  retained.audits = [link('verticalIndependentAudit', audit)];
  protocol.preflight_audits = retained.audits.map((entry) => ({ path: entry.file, sha256: entry.sha256 }));
  retained.protocol = link('verticalProtocol', protocol);
  run['vertical-stretch-protocol.json'] = { ...retained.protocol };
  run.verticalStretchPreflight = retained;
  data.reports.dsbiPointProtocol.source_sha256 = { ...config.source_sha256 };
  run['dsbi-visible-partial-protocol.json'].sha256 = sha(json(data.reports.dsbiPointProtocol));
  rebindSupplementalRunReports(data);
}

function withQualityBypass(data, zeroPlaceholder) {
  const benchmark = data.reports.publicBenchmark;
  const parity = data.reports.decoderParity;
  benchmark.words[0].predicted = '';
  benchmark.words[0].rejection = 'low_confidence';
  Object.assign(benchmark.whole_words, metrics(benchmark.words));
  benchmark.whole_words.by_length['1'] = metrics(benchmark.words.filter((row) => row.expected.length === 1));
  data.manifest.internalAcceptance.acceptedWordCount = 47;
  data.manifest.internalAcceptance.exactWordSuccessRate = 47 / 48;
  Object.assign(parity.rows[0], { typescript_text: '', python_text: '', typescript_reason: 'low_confidence', python_reason: 'low_confidence', quality_bypassed_decoder: true, saved_output_is_zero_placeholder: zeroPlaceholder });
  Object.assign(parity, { benchmark_sha256: sha(json(benchmark)), quality_bypass_cases: 1, zero_placeholder_cases: zeroPlaceholder ? 1 : 0, actual_model_output_decoder_cases: zeroPlaceholder ? 47 : 48, actual_decoder_text_matches: zeroPlaceholder ? 47 : 48, actual_decoder_rejection_matches: zeroPlaceholder ? 47 : 48, outcomes_sha256: sha(JSON.stringify(parity.rows)) });
}

function withImageNormalization(data) {
  const identity = { kind: 'per_channel_standardize_grayscale', location: 'inside_model_before_backbone_rescaling', variance: 'population', stdFloor: .001, targetMean: .5, targetStd: .05, clip: [0, 1], grayscale: 'mean_of_standardized_channels', outputChannels: 3 };
  data.manifest.imageNormalization = { ...identity };
  data.reports.conversion.imageNormalization = { ...identity };
  data.reports.validationSelection.imageNormalization = { ...identity };
  data.reports.checkpointSelection.candidates[0].imageNormalization = { ...identity };
}

function writeFixture(data) {
  const directory = 'assets/models/braille-word-reader-v1/';
  data.manifest.evidence = {};
  const retained = data.reports.trainingProvenance;
  retained.selectionReport.sha256 = sha(json(data.reports.checkpointSelection));
  retained.trainingRuns[0]['sampling-lineage-final.json'].sha256 = sha(json(data.reports.samplingLineage));
  if (data.manifest.provenance) data.manifest.provenance.retainedTrainingProvenance.sha256 = sha(json(retained));
  for (const [name, report] of Object.entries(data.reports)) {
    const bytes = json(report), file = `evidence/${name}.json`;
    put(file, bytes);
    data.manifest.evidence[name] = { file, sha256: sha(bytes) };
  }
  for (const [file, source] of Object.entries(data.sourceFiles)) put(file, source);
  put(directory + 'manifest.json', json(data.manifest));
  put(directory + 'unit.tflite', data.model);
  put(directory + 'unit.png', data.pngHeader);
  put('src/ml/word-model-assets.ts', data.assetModule);
  put('package.json', json({ dependencies: { 'react-native-nitro-image': '0.15.1', 'react-native-vision-camera': '5.2.2', 'react-native-nitro-modules': '0.36.5' } }));
}

try {
  mkdirSync(join(stage, 'scripts'));
  for (const name of ['verify-word-model-assets.mjs', 'inspect-tflite.mjs']) copyFileSync(join(root, 'scripts', name), join(stage, 'scripts', name));
  symlinkSync(join(root, 'node_modules'), join(stage, 'node_modules'), 'junction');
  const model = metadataOnlyModel();
  assert.deepEqual(inspectTflite(model).inputs[0].shape, [1, 192, 480, 3]);
  const invalidIndex = Buffer.from(model); invalidIndex.writeInt32LE(-1, 132);
  assert.throws(() => inspectTflite(invalidIndex), /tensor index/);
  const oversizedVector = Buffer.from(model); oversizedVector.writeUInt32LE(0xffffffff, 144);
  assert.throws(() => inspectTflite(oversizedVector), /vector exceeds/);
  assert.throws(() => inspectTflite(Buffer.alloc(4)), /truncated/);
  const cases = [
    ['consistent synthetic gate fixture', () => {}, null],
    ['vertical stretch preflight binds isolated augmentation', (d) => withVerticalStretch(d), null],
    ['vertical enabled run requires preflight', (d) => { withVerticalStretch(d); delete d.reports.trainingProvenance.trainingRuns[0].verticalStretchPreflight; }, /Missing vertical stretch preflight/],
    ['vertical range remains paper-bounded', (d) => withVerticalStretch(d, (x) => { x.protocol.vertical_stretch.range = [.5, 1.5]; }), /Vertical stretch range changed/],
    ['vertical stream identity is frozen', (d) => withVerticalStretch(d, (x) => { x.protocol.vertical_stretch.stream_id++; }), /Vertical random stream changed/],
    ['vertical cannot restore previously lost points', (d) => withVerticalStretch(d, (x) => { x.audit.actual_augment_cut_then_rotation_return_never_recovers_complete_or_occupied_point = false; }), /Vertical independent audit failed/],
    ['vertical must preserve old OFF pixels and RNG', (d) => withVerticalStretch(d, (x) => { x.audit.run20_off_pixels_targets_rng_byte_identical = false; }), /Vertical independent audit failed/],
    ['vertical source evidence must match exact helper', (d) => withVerticalStretch(d, (x) => { x.audit.source_sha256['word_vertical_stretch.py'] = '0'.repeat(64); }), /Vertical audit sources differ/],
    ['vertical parent requires retained eligible ancestry', (d) => withVerticalStretch(d, (x) => { x.protocol.parent_sampling_lineage_sha256 = '0'.repeat(64); }), /Vertical parent has no matching retained ancestor/],
    ['vertical must retain original audit script', (d) => withVerticalStretch(d, (x) => { x.retained.sources = []; }), /Missing exact original vertical audit script/],
    ['vertical rejects failed actual Pillow point parity', (d) => withVerticalStretch(d, (x) => { x.audit.pillow_marker_checks[0].max_centroid_error_pixels = .5; }), /Vertical Pillow point parity failed/],
    ['mirror preflight preserves inherited TRAIN semantics', (d) => withHorizontalMirror(d), null],
    ['mirror enabled run requires retained preflight', (d) => { withHorizontalMirror(d); delete d.reports.trainingProvenance.trainingRuns[0].horizontalMirrorPreflight; }, /Missing horizontal mirror preflight/],
    ['mirror cannot invert relief polarity', (d) => withHorizontalMirror(d, (x) => { x.config.polarity_augmentation = true; }), /cannot invert relief polarity/],
    ['mirror probability is frozen', (d) => withHorizontalMirror(d, (x) => { x.protocol.mirror.probability = 1; }), /probability must remain one half/],
    ['mirror slot mapping cannot become a half turn', (d) => withHorizontalMirror(d, (x) => { x.protocol.mirror.slot_permutation_zero_based = [5, 4, 3, 2, 1, 0]; }), /slot permutation changed/],
    ['mirror audit preserves paired RNG', (d) => withHorizontalMirror(d, (x) => { x.audit.on_off_random_calls_and_final_rng_states_equal = false; }), /independent audit failed/],
    ['mirror prelaunch audit cannot be replaced by addendum', (d) => withHorizontalMirror(d, (x) => { x.audit.post_launch_addendum = true; }), /Postlaunch mirror addendum/],
    ['mirror audit script must be original', (d) => withHorizontalMirror(d, (x) => { x.retained.sources = []; }), /Missing exact original mirror audit script/],
    ['mirror audit binds point source snapshot', (d) => withHorizontalMirror(d, (x) => { x.audit.source_sha256['word_point_geometry.py'] = '0'.repeat(64); }), /Mirror audit source differs/],
    ['mirror parent audit must be the retained ancestor', (d) => withHorizontalMirror(d, (x) => { x.protocol.parent_sampling_lineage_sha256 = '0'.repeat(64); }), /no matching retained ancestor/],
    ['mirror audit list cannot omit independent proof', (d) => withHorizontalMirror(d, (x) => { x.omitAudit = true; }), /audit links changed/],
    ['geometry zero parent and nonzero proof bind to raw TRAIN priors', (d) => withGeometryInitialization(d), null],
    ['geometry enabled branch needs provenance', (d) => { withGeometryInitialization(d); delete d.reports.trainingProvenance.trainingRuns[0].geometryInitialization; }, /Missing geometry initialization/],
    ['geometry provenance cannot hide behind disabled flag', (d) => withGeometryInitialization(d, (x) => { x.geometry.enabled = false; }), /requires enabled training/],
    ['geometry cannot resume diagnostic checkpoint', (d) => withGeometryInitialization(d, (x) => { x.config.resume_sha256 = x.init.nonzero_diagnostic_sha256; }), /must resume the zero parent/],
    ['geometry base needs an eligible ancestor', (d) => withGeometryInitialization(d, (x) => { x.ancestorAudit.actual.observed_epochs[0].eligible_by_sampling_lineage = false; }), /no retained eligible ancestor/],
    ['geometry cannot substitute initialization base', (d) => withGeometryInitialization(d, (x) => { x.init.parent_checkpoint_sha256 = '0'.repeat(64); }), /different base parent/],
    ['geometry needs a genuinely nonzero conversion input', (d) => withGeometryInitialization(d, (x) => { x.init.nonzero_max_output_delta = 0; }), /requires actual nonzero outputs/],
    ['geometry nonzero proof cannot use zero checkpoint', (d) => withGeometryInitialization(d, (x) => { x.proof.probes.nonzero.checkpoint_sha256 = x.init.zero_checkpoint_sha256; }), /nonzero conversion checkpoint differs/],
    ['geometry nonzero artifact cannot be the zero export', (d) => withGeometryInitialization(d, (x) => { x.proof.probes.nonzero.artifact_sha256 = x.proof.probes.zero.artifact_sha256; }), /nonzero artifact must differ/],
    ['geometry nonzero conversion cannot require Flex', (d) => withGeometryInitialization(d, (x) => { x.proof.probes.nonzero.operators.push('FlexGather'); }), /unsupported operators/],
    ['geometry conversion error cannot exceed frozen tolerance', (d) => withGeometryInitialization(d, (x) => { x.proof.probes.nonzero.tf_vs_tflite_max_abs = .001; }), /exceeds tolerance/],
    ['geometry every inherited channel must receive gradients', (d) => withGeometryInitialization(d, (x) => { x.init.initial_final_residual_gradient_l1_by_channel[4] = 0; }), /Every geometry residual channel/],
    ['geometry exact head source cannot be omitted', (d) => withGeometryInitialization(d, (x) => { x.retained.sources = x.retained.sources.filter((r) => basename(r.file) !== 'word_geometry_head.py'); }), /Missing exact geometry source/],
    ['geometry coordinated priors cannot shift source points', (d) => withGeometryInitialization(d, (x) => {
      for (const priors of [x.slots.median_corner_fractions, x.coordinates.raw_source_medians, x.geometry.sampling.slot_fractions, x.init.sampling.slot_fractions, x.protocol.sampling.slot_fractions]) priors[1][1] += .01;
    }), /differ from raw TRAIN grids/],
    ['geometry priors count unique cells instead of occurrences', (d) => withGeometryInitialization(d, (x) => { x.slots.unique_cells = 52; }), /weight unique source cells/],
    ['geometry uses the asymmetric SAME feature origin', (d) => withGeometryInitialization(d, (x) => { x.geometry.sampling.feature_origin = 0; }), /feature origin must be 3/],
    ['geometry lattice must match base graph', (d) => withGeometryInitialization(d, (x) => { x.identity.candidates[1].backbone_config_sha256 = '0'.repeat(64); }), /backbone lattice graph changed/],
    ['geometry proof cannot switch point metadata', (d) => withGeometryInitialization(d, (x) => { x.slots.point_metadata_sha256 = '0'.repeat(64); }), /different DSBI points/],
    ['geometry coordinate audit binds raw annotations', (d) => withGeometryInitialization(d, (x) => { x.coordinates.raw_annotations[0].sha256 = '0'.repeat(64); }), /different raw annotations/],
    ['geometry cached reconstruction must be exact', (d) => withGeometryInitialization(d, (x) => { x.slots.max_exact_cached_reconstruction_error_pixels = .001; }), /cached coordinate audit failed/],
    ['geometry independent head audit is required', (d) => withGeometryInitialization(d, (x) => { x.omitProof = 'independentHeadAudit'; }), /Missing geometry proof: independentHeadAudit/],
    ['geometry protocol sources are frozen', (d) => withGeometryInitialization(d, (x) => { x.protocol.source_sha256['word_model.py'] = '0'.repeat(64); }), /protocol sources differ/],
    ['geometry protocol cannot be written after training', (d) => withGeometryInitialization(d, (x) => { x.protocol.created_at_utc = '2026-09-06T00:01:00Z'; }), /must precede training/],
    ['geometry preflight protocol binds every audit', (d) => withGeometryInitialization(d, (x) => { x.protocol.preflight_audits = []; }), /preflight proofs changed/],
    ['geometry batch proof preserves negative exposure', (d) => withGeometryInitialization(d, (x) => { x.gpu.negative_expansion.base_repeat = 2; }), /different negative exposure/],
    ['geometry inherited batch normalization remains frozen', (d) => withGeometryInitialization(d, (x) => { x.config.batch_normalization.trainable_affine_layers = 52; }), /must remain frozen/],
    ['geometry inherited normalization cannot change', (d) => withGeometryInitialization(d, (x) => { x.config.imageNormalization.targetStd = .1; }), /normalization differs from the selected model/],
    ['DSBI points reconstruct from retained raw annotations', (d) => withSupplementalPoints(d), null],
    ['DSBI point metadata needs its gzip archive', (d) => withSupplementalPoints(d, (x) => { x.omitArchive = true; }), /DSBI point metadata data archive is missing/],
    ['DSBI points need the enabled training flag', (d) => withSupplementalPoints(d, (x) => { x.config.dsbi_visible_partial_supervision.enabled = false; }), /requires explicit enabled supervision/],
    ['DSBI point coordinates cannot use a reconstructed midpoint', (d) => withSupplementalPoints(d, (x) => { x.metadata.records['dsbi/0.png'].source_points[0][1][1] = 40; }), /DSBI points differ from the raw source grid/],
    ['DSBI points cannot silently change source masks', (d) => withSupplementalPoints(d, (x) => { x.metadata.records['dsbi/0.png'].masks[0] = 63; }), /DSBI point record masks differs/],
    ['DSBI points cannot silently change source identities', (d) => withSupplementalPoints(d, (x) => { x.metadata.records['dsbi/0.png'].source_cells[0].annotation_line = 5; }), /DSBI point record source_cells differs/],
    ['DSBI points cannot move to a different crop', (d) => withSupplementalPoints(d, (x) => { x.metadata.records['dsbi/0.png'].crop[0] = 1; }), /DSBI point record crop differs/],
    ['DSBI points require every original annotation', (d) => withSupplementalPoints(d, (x) => { x.preparation.sourceAnnotations.pop(); }), /all 26 original raw annotations/],
    ['DSBI raw annotation bytes cannot change', (d) => { withSupplementalPoints(d); d.sourceFiles['retained/dsbi/annotations/0.txt'] += 'changed'; }, /Evidence hash mismatch/],
    ['DSBI points cannot add an unregistered record', (d) => withSupplementalPoints(d, (x) => { x.metadata.records['unknown.png'] = { source_points: [] }; }), /exactly cover frozen supplemental data/],
    ['DSBI point protocol must use matching sources', (d) => withSupplementalPoints(d, (x) => { x.protocol.source_sha256['word_point_geometry.py'] = 'f'.repeat(64); }), /DSBI point protocol uses different training sources/],
    ['DSBI point protocol must bind metadata bytes', (d) => withSupplementalPoints(d, (x) => { x.protocol.point_metadata_sha256 = 'f'.repeat(64); }), /DSBI point protocol metadata hash changed/],
    ['DSBI point source snapshots cannot be missing', (d) => withSupplementalPoints(d, (x) => {
      const name = 'word_point_geometry.py';
      x.run.sources = x.run.sources.filter((entry) => basename(entry.file) !== name);
      x.run.unavailable_original_sources.push({ file: name, sha256_recorded_at_training: x.config.source_sha256[name] });
    }), /Missing original DSBI point source snapshot/],
    ['retained DSBI preparation and exact append pass', (d) => withSupplementalDSBI(d), null],
    ['DSBI source index archive is required', (d) => withSupplementalDSBI(d, (x) => { x.omitArchive = 'index'; }), /DSBI source index data archive is missing/],
    ['DSBI generated record archive is required', (d) => withSupplementalDSBI(d, (x) => { x.omitArchive = 'records'; }), /DSBI generated manifest data archive is missing/],
    ['DSBI TEST pixels cannot be used', (d) => withSupplementalDSBI(d, (x) => { x.index.testImagePixelsAccessed = true; }), /DSBI TEST pixels/],
    ['DSBI pinned membership is enforced', (d) => withSupplementalDSBI(d, (x) => { x.index.sources[0].train_member = 'Massage/M+11.jpg'; }), /pinned 26 TRAIN identities/],
    ['DSBI TEST source cannot masquerade as TRAIN', (d) => withSupplementalDSBI(d, (x) => { x.index.sources[0].split = 'test'; }), /DSBI source is not TRAIN/],
    ['DSBI original image must match its member', (d) => withSupplementalDSBI(d, (x) => { x.index.sources[0].original_image = 'data/other.jpg'; }), /original image differs/],
    ['DSBI mask zero cannot become a positive object', (d) => withSupplementalDSBI(d, (x) => { x.records[0].masks[0] = 0; }), /zero or unknown masks/],
    ['DSBI source snapshot parser hash is bound', (d) => withSupplementalDSBI(d, (x) => { x.protocol.parser_sha256 = 'f'.repeat(64); }), /DSBI parser snapshot differs/],
    ['DSBI missing source snapshot cannot be disclosed away', (d) => withSupplementalDSBI(d, (x) => { x.snapshot.files.pop(); }), /Missing DSBI preparation source snapshots/],
    ['DSBI failed independent crop audit is rejected', (d) => withSupplementalDSBI(d, (x) => { x.audit.status = 'failed'; }), /DSBI independent context audit did not pass/],
    ['DSBI unresolved boundary audit flags are rejected', (d) => withSupplementalDSBI(d, (x) => { x.audit.conservative_boundary_flags = [{ record: 1 }]; }), /unresolved boundary flags/],
    ['DSBI audit must bind the current index', (d) => withSupplementalDSBI(d, (x) => { x.audit.source_index_sha256 = 'f'.repeat(64); }), /DSBI context audit uses a different index/],
    ['DSBI audit must cover every cached pixel', (d) => withSupplementalDSBI(d, (x) => { x.audit.pixel_bytes_compared -= 3; }), /DSBI pixel audit coverage is incomplete/],
    ['DSBI manifest disclosure is required', (d) => { withSupplementalDSBI(d); d.manifest.datasets = []; }, /Manifest DSBI datasets differ/],
    ['DSBI combined manifest cannot drop an original record', (d) => withSupplementalDSBI(d, (x) => { x.mutateCombined = (rows) => rows.shift(); }), /preserve every base and supplemental record/],
    ['DSBI training configuration needs its retained preparation', (d) => {
      withSupplementalDSBI(d); d.reports.trainingProvenance.supplementalPreparations = []; d.manifest.datasets = [];
    }, /DSBI training index has no matching retained preparation/],
    ['DSBI configuration index hash must match preparation', (d) => {
      withSupplementalDSBI(d); d.reports.trainingConfiguration.data_sha256['../dsbi/source-index.json'] = 'f'.repeat(64);
      rebindSupplementalRunReports(d);
    }, /DSBI training index has no matching retained preparation/],
    ['DSBI run needs its predeclared protocol', (d) => {
      withSupplementalDSBI(d); delete d.reports.trainingProvenance.trainingRuns[0]['dsbi-addition-protocol.json'];
    }, /Missing evidence/],
    ['DSBI run protocol must bind its original source snapshots', (d) => {
      withSupplementalDSBI(d); d.reports.dsbiTrainingProtocol.source_sha256 = { 'word_model.py': 'f'.repeat(64) };
      rebindSupplementalRunReports(d);
    }, /DSBI training protocol uses different source snapshots/],
    ['DSBI run protocol must precede training', (d) => {
      withSupplementalDSBI(d); d.reports.dsbiTrainingProtocol.created_at_utc = '2026-09-06T00:00:00Z';
      rebindSupplementalRunReports(d);
    }, /DSBI protocol must precede training/],
    ['supported internal normalization identity', (d) => withImageNormalization(d), null],
    ['missing normalization identity', (d) => { delete d.manifest.imageNormalization; }, /Unsupported imageNormalization/],
    ['unknown normalization identity', (d) => { d.manifest.imageNormalization.kind = 'external'; }, /Unsupported imageNormalization/],
    ['wrong normalization variance', (d) => { withImageNormalization(d); d.manifest.imageNormalization.variance = 'sample'; }, /supported model contract/],
    ['invalid normalization standard deviation', (d) => { withImageNormalization(d); d.manifest.imageNormalization.targetStd = null; }, /supported model contract/],
    ['wrong normalization clipping', (d) => { withImageNormalization(d); d.manifest.imageNormalization.clip = [-1, 1]; }, /supported model contract/],
    ['normalization outside the model', (d) => { withImageNormalization(d); d.manifest.imageNormalization.location = 'native_preprocessing'; }, /supported model contract/],
    ['different conversion normalization', (d) => { withImageNormalization(d); d.reports.conversion.imageNormalization = { kind: 'none' }; }, /Conversion image normalization differs/],
    ['different validation normalization', (d) => { withImageNormalization(d); d.reports.validationSelection.imageNormalization.stdFloor = .01; }, /Validation selection image normalization differs/],
    ['legacy quality placeholders counted separately', (d) => withQualityBypass(d, true), null],
    ['real tensors retained after quality rejection', (d) => withQualityBypass(d, false), null],
    ['training candidate', (d) => { d.manifest.artifact.status = 'training'; }, /belum lulus/],
    ['missing row threshold', (d) => { delete d.manifest.thresholds.maxRowDeviation; }, /Threshold keys/],
    ['invalid runtime edge margin', (d) => { d.manifest.thresholds.edgeMargin = .6; }, /edgeMargin/],
    ['changed benchmark record', (d) => { d.reports.publicBenchmark.words[0].expected = 'B'; }, /Benchmark fixture changed/],
    ['stale model digest', (d) => { d.model[100] = 1; }, /AssertionError/],
    ['stale selected checkpoint', (d) => { d.reports.validationSelection.oracle.checkpoint_sha256 = 'c'.repeat(64); }, /selected checkpoint/],
    ['missing retained training provenance', (d) => { delete d.manifest.provenance; }, /Missing evidence/],
    ['different provenance checkpoint', (d) => { d.reports.trainingProvenance.selectedCheckpointSha256 = 'd'.repeat(64); }, /different selected checkpoint/],
    ['explicitly missing ancestor original is preserved', (d) => {
      const missingHash = sha('unavailable original source');
      const config = { source_sha256: { 'old_source.py': missingHash } };
      d.reports.ancestorConfiguration = config;
      d.reports.trainingProvenance.trainingRuns.push({ run: 'ancestor', configuration: { file: 'evidence/ancestorConfiguration.json', sha256: sha(json(config)) }, sources: [], unavailable_original_sources: [{ file: 'old_source.py', sha256_recorded_at_training: missingHash }] });
    }, null],
    ['stale archived training source', (d) => { d.sourceFiles['retained/word_model.py'] = '// edited archive'; }, /Evidence hash mismatch/],
    ['unmarked missing original training source', (d) => { d.reports.trainingProvenance.trainingRuns[0].sources = []; }, /missing without an explicit archive gap/],
    ['identical selection report copied for review', (d) => { const file = 'evidence/copied-selection.json'; d.manifest.provenance.selectionReport = file; d.sourceFiles[file] = json(d.reports.checkpointSelection); }, null],
    ['selected candidate absent from archive', (d) => { d.reports.checkpointSelection.candidates = []; }, /absent from retained validation candidates/],
    ['external outputs used for selection', (d) => { d.reports.checkpointSelection.external_outputs_used = true; }, /must not select the checkpoint/],
    ['misreported group precision gate', (d) => { Object.assign(d.reports.checkpointSelection.candidates[0].validation_groups.stress, { correct: 11, exact_word_accuracy: 11 / 12, accepted_accuracy: 11 / 12 }); }, /failed validation group precision gates/],
    ['stale retained gzip archive', (d) => { d.sourceFiles['evidence/data-0.json.gz'] = gzipSync('changed records'); }, /Retained compressed data changed/],
    ['selected checkpoint ineligible by sampling', (d) => { d.reports.samplingLineage.actual.observed_epochs[0].eligible_by_sampling_lineage = false; }, /not eligible by sampling lineage/],
    ['selected checkpoint missing from eligible epochs', (d) => { d.reports.samplingLineage.eligible_retained_epochs = []; }, /absent from eligible retained epochs/],
    ['selected checkpoint has no sampling proof', (d) => { d.reports.samplingLineage.actual.observed_epochs[0].checkpoint_sha256 = 'e'.repeat(64); }, /no retained sampling eligibility proof/],
    ['sampling proof uses stale configuration', (d) => { d.reports.samplingLineage.actual.config_sha256 = 'f'.repeat(64); }, /different training configuration/],
    ['sampling proof uses stale history', (d) => { d.reports.samplingLineage.actual.sampling_history_sha256 = 'f'.repeat(64); }, /different history/],
    ['sampling proof uses different training records', (d) => { d.reports.samplingLineage.actual.manifest_sha256 = 'f'.repeat(64); }, /different training data/],
    ['thresholds changed after selection', (d) => { d.reports.validationSelection.thresholds.acceptScore = .8; }, /selected on validation/],
    ['null conversion error', (d) => { d.reports.conversion.tensorflow_vs_tflite_max_abs = null; }, /Invalid conversion error/],
    ['failed conversion', (d) => { d.reports.conversion.parity_pass = false; }, /AssertionError/],
    ['stale preprocessing source', (d) => { d.sourceFiles['src/ml/word-image-processing.ts'] += '\n// changed'; }, /evidence is stale/],
    ['changed Hermes harness', (d) => { d.sourceFiles['harness.template'] += '\n// changed'; }, /harness entry changed/],
    ['resampler changed after Hermes test', (d) => { const file = 'src/ml/word-image-processing.ts'; d.sourceFiles[file] = d.sourceFiles[file].replace('return 0', 'return 1'); d.reports.nativePreprocessing.source_code_sha256[file] = sha(d.sourceFiles[file]); }, /Hermes resampler implementation changed/],
    ['null native error', (d) => { d.reports.nativeRuntime.results[0].maxAbsError = null; }, /maxAbsError/],
    ['wrong worklet runtime', (d) => { d.reports.nativeRuntime.results[0].runtimeKind = 1; }, /background Worklets/],
    ['negative false acceptance', (d) => { d.reports.publicBenchmark.negative_rows[0].predicted = 'A'; }, /explicitly rejected/],
    ['missing decoder parity evidence', (d) => { delete d.reports.decoderParity; }, /Missing evidence/],
    ['decoder parity model mismatch', (d) => { d.reports.decoderParity.model_sha256 = 'd'.repeat(64); }, /different model/],
    ['decoder parity benchmark mismatch', (d) => { d.reports.decoderParity.benchmark_sha256 = 'e'.repeat(64); }, /different benchmark/],
    ['decoder parity fixture mismatch', (d) => { d.reports.decoderParity.fixture_manifest_sha256 = 'f'.repeat(64); }, /different fixtures/],
    ['decoder parity threshold mismatch', (d) => { d.reports.decoderParity.thresholds.acceptScore = .7; }, /different thresholds/],
    ['malformed decoder parity count', (d) => { d.reports.decoderParity.quality_matches = null; }, /Invalid decoder parity count/],
    ['incomplete decoder parity coverage', (d) => { d.reports.decoderParity.total_pipeline_cases = 47; }, /every benchmark word/],
    ['decoder parity reported mismatch', (d) => { d.reports.decoderParity.first_mismatches = [{ index: 1 }]; }, /zero mismatches/],
    ['stale TypeScript decoder evidence', (d) => { d.sourceFiles['src/ml/word-decoder.ts'] += '\n// changed'; }, /Decoder parity source is stale/],
    ['incorrect parity count of real outputs', (d) => { d.reports.decoderParity.actual_model_output_decoder_cases = 47; }, /AssertionError/],
    ['hidden decoder mismatch in rows', (d) => { d.reports.decoderParity.rows[0].decoder_reason_match = false; d.reports.decoderParity.outcomes_sha256 = sha(JSON.stringify(d.reports.decoderParity.rows)); }, /Real-output decoder rejection parity failed/],
    ['missing real rejection diagnostics', (d) => { delete d.reports.rejectionDiagnostics; }, /Missing evidence/],
    ['missing diagnostic Python source identity', (d) => { delete d.reports.rejectionDiagnostics.source_sha256['training/scripts/word_decode.py']; }, /Missing rejection diagnostic source hash/],
    ['stale diagnostic Python source', (d) => { d.sourceFiles['training/scripts/word_decode.py'] += '\n# changed'; }, /diagnostic source is stale/],
    ['repetition diagnostics model mismatch', (d) => { d.reports.repetitionDiagnostics.model_sha256 = 'f'.repeat(64); }, /different model/],
    ['diagnostic threshold mismatch', (d) => { d.reports.rejectionDiagnostics.thresholds.acceptScore = .8; }, /different thresholds/],
    ['incomplete diagnostic coverage', (d) => { d.reports.rejectionDiagnostics.rows.pop(); }, /diagnostic coverage/],
    ['changed diagnostic fixture pixels', (d) => { d.reports.repetitionDiagnostics.rows[0].image_sha256 = 'e'.repeat(64); }, /frozen fixture/],
    ['stale actual diagnostic PNG bytes', (d) => { d.sourceFiles['evidence/repetition/0.png'] = 'changed pixels'; }, /diagnostic fixture pixels changed/],
    ['accepted diagnostic cell count mismatch', (d) => { d.reports.repetitionDiagnostics.rows[0].detected_cell_count = 2; }, /cell count differs/],
    ['misreported diagnostic correctness', (d) => { d.reports.rejectionDiagnostics.rows[0].correct = false; }, /correctness is misreported/],
    ['accepted word from rejection diagnostic', (d) => { Object.assign(d.reports.rejectionDiagnostics.rows[0], { predicted: 'A', rejection: '', correct: false }); }, /reject the entire capture/],
    ['misreported repetition totals', (d) => { d.reports.repetitionDiagnostics.accepted = 13; }, /AssertionError/],
    ['commented Metro require', (d) => { d.assetModule = 'export const WORD_MODEL_ASSET = null;\n/* ' + d.assetModule + ' */'; }, /actual literal Metro asset/],
  ];
  for (const [label, mutate, error] of cases) {
    const data = fixture(); mutate(data); writeFixture(data);
    const result = spawnSync(process.execPath, [join(stage, 'scripts/verify-word-model-assets.mjs')], { encoding: 'utf8' });
    if (error === null) assert.equal(result.status, 0, `${label}: ${result.stderr}`);
    else { assert.notEqual(result.status, 0, label); assert.match(result.stderr, error, label); }
    console.log(`PASS ${label}`);
  }
  console.log(`${cases.length} asset gate cases and 4 FlatBuffer checks passed; synthetic rule tests only.`);
} finally {
  const rel = relative(temporaryRoot, resolve(stage));
  assert.ok(rel && !rel.startsWith('..') && basename(stage).startsWith('braille-word-verifier-'), 'Refuse cleanup outside generated temporary directory');
  rmSync(stage, { recursive: true, force: true });
}
