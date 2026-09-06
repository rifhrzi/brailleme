import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { basename, dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import ts from 'typescript';
import { inspectTflite } from './inspect-tflite.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const directory = resolve(root, 'assets/models/braille-word-reader-v1');
const manifest = JSON.parse(readFileSync(resolve(directory, 'manifest.json'), 'utf8'));
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const inside = (base, file) => {
  assert.ok(typeof file === 'string' && file.length > 0, 'Missing evidence or asset path');
  const path = resolve(base, file), rel = relative(base, path);
  assert.ok(rel && !rel.startsWith('..') && !rel.includes(':'), 'Path must stay inside its directory');
  return path;
};
const verifiedBytes = (entry) => {
  const bytes = readFileSync(inside(root, entry?.file));
  assert.equal(hash(bytes), entry.sha256, `Evidence hash mismatch: ${entry.file}`);
  return bytes;
};
const evidence = (entry) => JSON.parse(verifiedBytes(entry).toString('utf8'));

assert.equal(manifest.schemaVersion, 1);
assert.ok(['unverified_candidate', 'verified'].includes(manifest.artifact.status),
  'Model kata belum lulus pengujian; APK pemindai kata belum boleh dibangun.');
const model = readFileSync(inside(directory, manifest.artifact.file));
assert.equal(model.length, manifest.artifact.sizeBytes);
assert.equal(hash(model), manifest.artifact.sha256);
const contract = inspectTflite(model);
assert.equal(contract.schemaVersion, 3);
assert.equal(contract.inputs.length, 1);
assert.equal(contract.outputs.length, 1);
assert.deepEqual(contract.inputs[0].shape, [1, 192, 480, 3]);
assert.deepEqual(contract.outputs[0].shape, [1, 48, 120, 11]);
assert.equal(contract.inputs[0].type, 0, 'Input must be float32');
assert.equal(contract.outputs[0].type, 0, 'Output must be float32');
assert.deepEqual(manifest.input.shape, contract.inputs[0].shape);
assert.deepEqual(manifest.output.shape, contract.outputs[0].shape);
assert.equal(manifest.input.dtype, 'float32');
assert.equal(manifest.output.dtype, 'float32');
assert.equal(manifest.input.layout, 'NHWC');
assert.equal(manifest.input.channelOrder, 'RGB');
assert.equal(manifest.input.normalization, 'pixel / 255.0');
assert.equal(manifest.input.resizeInterpolation, 'pillow_bilinear_antialias_u8_22bit');
assert.equal(manifest.input.androidRawPixelCorrection, 'nitro-image-0.15.1-argb8888-is-rgba');
const packageConfig = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'));
assert.equal(packageConfig.dependencies['react-native-nitro-image'], '0.15.1', 'Native channel correction must be revalidated when Nitro Image changes');
assert.equal(packageConfig.dependencies['react-native-vision-camera'], '5.2.2', 'Photo sensor geometry must be revalidated when VisionCamera changes');
assert.equal(packageConfig.dependencies['react-native-nitro-modules'], '0.36.5', 'Photo geometry bindings must be regenerated and revalidated when Nitro Modules changes');
assert.equal(manifest.input.backboneNormalization, 'Rescaling(2, -1) inside model');
const imageNormalization = manifest.imageNormalization;
assert.ok(imageNormalization && ['none', 'per_channel_standardize_grayscale'].includes(imageNormalization.kind), 'Unsupported imageNormalization identity');
const supportedNormalization = imageNormalization.kind === 'none' ? { kind: 'none' } : {
  kind: 'per_channel_standardize_grayscale',
  location: 'inside_model_before_backbone_rescaling',
  variance: 'population',
  stdFloor: 0.001,
  targetMean: 0.5,
  targetStd: 0.05,
  clip: [0, 1],
  grayscale: 'mean_of_standardized_channels',
  outputChannels: 3,
};
assert.deepEqual(imageNormalization, supportedNormalization, 'Image normalization differs from the supported model contract');
assert.deepEqual(manifest.output.layout, ['objectness', 'offset_x', 'offset_y', 'width', 'height', 'dot1', 'dot2', 'dot3', 'dot4', 'dot5', 'dot6']);
assert.deepEqual(manifest.latinMasks, [1,3,9,25,17,11,27,19,10,26,5,7,13,29,21,15,31,23,14,30,37,39,58,45,61,53]);
const t = manifest.thresholds;
const thresholdKeys = ['proposalScore','acceptScore','dotConfidence','nmsIou','minCellWidth','minCellHeight','edgeMargin','maxRowDeviation','maxGapRatio','maxOverlapRatio'];
assert.ok(t && typeof t === 'object' && !Array.isArray(t), 'Missing threshold object');
assert.deepEqual(Object.keys(t).sort(), [...thresholdKeys].sort(), 'Threshold keys must match the runtime contract');
for (const key of thresholdKeys) assert.ok(Number.isFinite(t[key]), `Invalid threshold ${key}`);
for (const key of ['nmsIou','minCellWidth','minCellHeight']) assert.ok(t[key] > 0 && t[key] < 1, `Invalid ratio ${key}`);
assert.ok(t.proposalScore > 0 && t.acceptScore >= t.proposalScore && t.acceptScore <= 1);
assert.ok(t.dotConfidence > .5 && t.dotConfidence <= 1);
assert.ok(t.edgeMargin >= 0 && t.edgeMargin < .5);
assert.ok(t.maxRowDeviation > 0 && t.maxRowDeviation <= 1);
assert.ok(t.maxOverlapRatio >= 0 && t.maxOverlapRatio < 1);
assert.ok(t.maxGapRatio > 0);
assert.ok(t.maxGapRatio <= 1.2, 'Gap gate must catch a skipped interior cell at observed validation spacing');

const benchmark = evidence(manifest.evidence?.publicBenchmark);
const wholeWordFixtures = evidence(manifest.evidence?.wholeWordFixtures);
assert.equal(manifest.evidence.wholeWordFixtures.sha256, benchmark.fixture_manifest_sha256);
assert.ok(Array.isArray(wholeWordFixtures.records) && wholeWordFixtures.records.length > 0);
assert.equal(benchmark.model_sha256, manifest.artifact.sha256);
assert.deepEqual(benchmark.thresholds, t);
assert.equal(benchmark.numerical_internal_gates_pass, true);
assert.equal(benchmark.words.length, wholeWordFixtures.records.length);
for (const [index, row] of benchmark.words.entries()) {
  const record = { ...row };
  delete record.predicted;
  delete record.rejection;
  delete record.detected_cells;
  assert.deepEqual(record, wholeWordFixtures.records[index], `Benchmark fixture changed or was reordered: ${index}`);
  assert.match(row.expected, /^[A-Z]{1,12}$/, 'Expected benchmark word must contain 1..12 letters');
  assert.ok(typeof row.predicted === 'string' && /^(?:[A-Z]{1,12})?$/.test(row.predicted), 'Prediction must be an accepted A-Z word or an explicit empty rejection');
}
assert.equal(new Set(benchmark.words.map((row) => row.image)).size, benchmark.words.length, 'Benchmark word fixtures must not be duplicated');
const accepted = benchmark.words.filter((row) => row.predicted).length;
const correct = benchmark.words.filter((row) => row.predicted === row.expected).length;
const total = benchmark.words.length;
assert.equal(benchmark.whole_words.total, total);
assert.equal(benchmark.whole_words.accepted, accepted);
assert.equal(benchmark.whole_words.correct, correct);
assert.equal(benchmark.whole_words.exact_word_accuracy, correct / total);
assert.equal(benchmark.whole_words.accepted_accuracy, accepted ? correct / accepted : null);
assert.ok(correct / total >= .75, 'Exact whole word accuracy must be at least 75%, including rejects');
assert.ok(accepted > 0 && correct / accepted >= .95, 'Accepted word accuracy must be at least 95%');
assert.equal(benchmark.negative_rows.length, 44);
for (const row of benchmark.negative_rows) {
  assert.ok(typeof row.source === 'string' && row.source.length > 0);
  assert.equal(row.predicted, '', 'Every negative must be explicitly rejected');
}
assert.equal(new Set(benchmark.negative_rows.map((row) => row.source)).size, 44);
assert.equal(benchmark.negative_rows.filter((row) => row.predicted).length, 0);
assert.equal(benchmark.legacy_rows.length, 66);
assert.equal(benchmark.negative_regression.independent_holdout, false);
for (let length = 1; length <= 12; length++) {
  const rows = benchmark.words.filter((row) => row.expected.length === length);
  if (!rows.length) continue;
  const metrics = benchmark.whole_words.by_length[String(length)];
  assert.equal(metrics.total, rows.length);
  assert.equal(metrics.accepted, rows.filter((row) => row.predicted).length);
  assert.equal(metrics.correct, rows.filter((row) => row.predicted === row.expected).length);
  assert.equal(metrics.exact_word_accuracy, metrics.correct / metrics.total);
  assert.equal(metrics.accepted_accuracy, metrics.accepted ? metrics.correct / metrics.accepted : null);
}
const acceptance = manifest.internalAcceptance;
assert.equal(acceptance.passed, true);
assert.equal(acceptance.evaluatedWordCount, total);
assert.equal(acceptance.acceptedWordCount, accepted);
assert.equal(acceptance.exactWordSuccessRate, correct / total);
assert.equal(acceptance.acceptedWordAccuracy, correct / accepted);
assert.equal(acceptance.negativeImageCount, 44);
assert.equal(acceptance.negativeFalseAccepts, 0);
const conversion = evidence(manifest.evidence?.conversion);
assert.equal(conversion.artifact_sha256, manifest.artifact.sha256);
assert.deepEqual(conversion.imageNormalization, imageNormalization, 'Conversion image normalization differs from the manifest');
assert.equal(conversion.artifact_bytes, model.length);
assert.deepEqual(conversion.input_shape, contract.inputs[0].shape);
assert.deepEqual(conversion.output_shape, contract.outputs[0].shape);
assert.deepEqual(conversion.thresholds, t);
assert.equal(conversion.parity_pass, true);
assert.ok(Number.isInteger(conversion.parity_samples) && conversion.parity_samples >= 24);
for (const key of ['training_backend_vs_tensorflow_max_abs', 'tensorflow_vs_tflite_max_abs']) {
  assert.ok(Number.isFinite(conversion[key]) && conversion[key] >= 0 && conversion[key] <= 5e-4, `Invalid conversion error: ${key}`);
}
assert.equal(conversion.decoded_text_parity, true);
assert.equal(conversion.decoded_rejection_parity, true);
const validation = evidence(manifest.evidence?.validationSelection);
assert.deepEqual(validation.imageNormalization, imageNormalization, 'Validation selection image normalization differs from the manifest');
assert.deepEqual(validation.thresholds, t, 'Active thresholds must be selected on validation');
assert.match(validation.oracle?.checkpoint_sha256, /^[a-f0-9]{64}$/);
assert.match(validation.oracle?.fixture_sha256, /^[a-f0-9]{64}$/);
assert.equal(validation.oracle.checkpoint_sha256, conversion.source_checkpoint_sha256, 'Conversion must use the selected checkpoint');
assert.equal(validation.oracle.fixture_sha256, conversion.parity_fixture_sha256, 'Conversion parity must use the frozen validation fixtures');
assert.equal(validation.words.total, conversion.parity_samples);
const provenanceIdentity = manifest.provenance;
const provenance = evidence(provenanceIdentity?.retainedTrainingProvenance);
assert.equal(provenanceIdentity.checkpointSha256, validation.oracle.checkpoint_sha256, 'Manifest provenance uses a different selected checkpoint');
assert.equal(provenanceIdentity.validationFixtureSha256, validation.oracle.fixture_sha256);
assert.equal(provenanceIdentity.externalFixtureSha256, benchmark.fixture_manifest_sha256);
assert.equal(provenance.selectedCheckpointSha256, validation.oracle.checkpoint_sha256, 'Retained provenance uses a different selected checkpoint');
assert.equal(provenance.selectedSourceCheckpoint, validation.source_checkpoint, 'Retained provenance names a different source checkpoint');
// Every available original archive stays bound to its digest. Explicitly recorded
// missing ancestor sources are preserved; current files cannot replace them.
const verifyArchiveTree = (value) => {
  if (!value || typeof value !== 'object') return;
  if (typeof value.file === 'string' && typeof value.sha256 === 'string') verifiedBytes(value);
  for (const child of Object.values(value)) verifyArchiveTree(child);
};
verifyArchiveTree(provenance);
assert.deepEqual(provenanceIdentity.selectionSources, provenance.selectionSources, 'Selection source links differ from retained provenance');
const verifySourceSet = (entries, expected, unavailable = []) => {
  assert.ok(Array.isArray(entries) && expected && typeof expected === 'object', 'Missing retained source identities');
  assert.equal(new Set(entries.map((entry) => basename(entry.file))).size, entries.length, 'Duplicate retained source');
  for (const entry of entries) {
    assert.equal(entry.sha256, expected[basename(entry.file)], `Retained source differs from recorded hash: ${entry.file}`);
  }
  for (const [name, digest] of Object.entries(expected)) {
    assert.match(digest, /^[a-f0-9]{64}$/);
    assert.ok(entries.some((entry) => basename(entry.file) === name && entry.sha256 === digest) ||
      unavailable.some((entry) => entry.file === name && entry.sha256_recorded_at_training === digest),
    `Original source is missing without an explicit archive gap: ${name}`);
  }
};
verifySourceSet(provenance.selectionSources, validation.oracle.source_sha256);
assert.equal(hash(readFileSync(inside(root, provenanceIdentity.selectionReport))), provenance.selectionReport.sha256,
  'Selection report differs from retained provenance');
const selectionReport = evidence(provenance.selectionReport);
assert.equal(selectionReport.selected_checkpoint_sha256, validation.oracle.checkpoint_sha256, 'Selection report uses a different selected checkpoint');
assert.equal(selectionReport.selected_checkpoint, validation.source_checkpoint);
assert.equal(selectionReport.fixture_sha256, validation.oracle.fixture_sha256);
assert.equal(selectionReport.external_outputs_used, false, 'External model outputs must not select the checkpoint');
const selectedCandidate = selectionReport.candidates.find((candidate) =>
  candidate.checkpoint_sha256 === validation.oracle.checkpoint_sha256 && candidate.source_checkpoint === validation.source_checkpoint);
assert.ok(selectedCandidate, 'Selected checkpoint is absent from retained validation candidates');
assert.deepEqual(selectedCandidate.thresholds, t, 'Selected candidate thresholds differ from the active thresholds');
assert.equal(selectedCandidate.both_group_precision_gates_pass, true, 'Selected candidate failed validation group precision gates');
assert.deepEqual(selectedCandidate.imageNormalization, imageNormalization, 'Selected candidate normalization differs');
assert.equal(selectionReport.validation_words, validation.words.total);
for (const name of ['natural', 'stress']) {
  const group = selectedCandidate.validation_groups?.[name];
  assert.ok(group && ['total', 'accepted', 'correct'].every((key) => Number.isSafeInteger(group[key]) && group[key] >= 0), 'Invalid selected validation group counts');
  assert.ok(group.accepted > 0 && group.correct <= group.accepted && group.accepted <= group.total);
  assert.equal(group.exact_word_accuracy, group.correct / group.total);
  assert.equal(group.accepted_accuracy, group.correct / group.accepted);
  assert.ok(group.accepted_accuracy >= .95, 'Selected candidate failed validation group precision gates');
}

const retainedData = evidence(provenanceIdentity.retainedDataManifests);
const dataDirectory = dirname(inside(root, provenanceIdentity.retainedDataManifests.file));
assert.ok(Array.isArray(retainedData.files) && retainedData.files.length > 0, 'Missing retained data archives');
const retainedDataHashes = new Set();
const retainedDataBytes = new Map();
for (const entry of retainedData.files) {
  const compressed = readFileSync(inside(dataDirectory, entry.file));
  assert.equal(hash(compressed), entry.gzipSha256, `Retained compressed data changed: ${entry.file}`);
  assert.equal(compressed.length, entry.gzipSizeBytes);
  const original = gunzipSync(compressed);
  assert.equal(hash(original), entry.sha256, `Retained data differs from its original: ${entry.file}`);
  assert.equal(original.length, entry.sizeBytes);
  retainedDataHashes.add(entry.sha256);
  retainedDataBytes.set(entry.sha256, original);
}
assert.ok(retainedDataHashes.has(validation.oracle.fixture_sha256), 'Frozen validation data archive is missing');
assert.ok(retainedDataHashes.has(benchmark.fixture_manifest_sha256), 'Frozen external data archive is missing');
const archivedData = (digest, label) => {
  assert.ok(retainedDataBytes.has(digest), `DSBI ${label} data archive is missing`);
  return JSON.parse(retainedDataBytes.get(digest).toString('utf8'));
};
const dsbiRevision = '6937eb8973efe5ff337e5c018485cb626303f5c2';
const dsbiCoordinateSpace = 'provided_deskewed_recto_image_pixels';
const dsbiTrainMembers = [
  ...Array.from({ length: 10 }, (_, i) => `Massage/M+${i + 1}.jpg`),
  ...Array.from({ length: 10 }, (_, i) => `Math/math+${i + 1}.jpg`),
  ...Array.from({ length: 3 }, (_, i) => `Ordinary Printed Document/OPD+${i + 1}.jpg`),
  ...Array.from({ length: 3 }, (_, i) => `Shaver Yang Fengting/SYF+${i + 3}.jpg`),
];
const supplements = provenance.supplementalPreparations ?? [];
assert.ok(Array.isArray(supplements), 'Invalid DSBI preparations');
const supplementalByHash = new Map();
for (const preparation of supplements) {
  assert.equal(preparation.dataset, 'DSBI', 'Unsupported supplemental dataset');
  assert.equal(preparation.revision, dsbiRevision, 'DSBI revision changed');
  const index = evidence(preparation.sourceIndex);
  assert.deepEqual(archivedData(preparation.sourceIndex.sha256, 'source index'), index);
  assert.ok(!supplementalByHash.has(preparation.sourceIndex.sha256), 'Duplicate DSBI preparation');
  assert.equal(index.schemaVersion, 1);
  assert.equal(index.dataset, 'DSBI');
  assert.equal(index.revision, dsbiRevision, 'DSBI source index revision changed');
  assert.equal(index.source_coordinate_space, dsbiCoordinateSpace, 'DSBI coordinates must refer to the supplied deskewed image');
  assert.equal(index.testImagePixelsAccessed, false, 'DSBI TEST pixels must remain unused');
  assert.equal(index.trainList, 'train.txt');
  assert.equal(index.testList, 'test.txt');
  assert.equal(index.trainListSha256, '3a679245733600ee78640254f00b850d7ea7de3f413e7274d0e34ccaf9d3da9c', 'DSBI pinned TRAIN metadata changed');
  assert.equal(index.testListSha256, '347836183ea3cd9a285f80eb0902cc894173e184ed6ac2b7e425581e702a02e3', 'DSBI pinned TEST metadata changed');
  assert.ok(Array.isArray(index.sources) && index.sources.length === 26, 'DSBI must retain all 26 TRAIN sources');
  const members = index.sources.map((source) => source.train_member?.replaceAll('\\', '/'));
  assert.deepEqual([...members].sort(), [...dsbiTrainMembers].sort(), 'DSBI sources differ from the pinned 26 TRAIN identities');
  const sourceMap = new Map();
  for (const [number, source] of index.sources.entries()) {
    const member = members[number], recto = member.replace(/\.jpg$/, '+recto.jpg');
    assert.equal(source.split, 'train', 'DSBI source is not TRAIN');
    assert.equal(source.source, `dsbi/${recto}`, 'DSBI source identity differs from its TRAIN member');
    assert.equal(source.source_family, `dsbi/${member.slice(0, member.lastIndexOf('/'))}`, 'DSBI source family changed');
    assert.equal(source.original_image, `data/${recto}`, 'DSBI original image differs from its TRAIN member');
    assert.equal(source.original_annotation, `data/${recto.replace(/\.jpg$/, '.txt')}`, 'DSBI annotation differs from its TRAIN member');
    for (const key of ['image_sha256', 'annotation_sha256']) assert.match(source[key], /^[a-f0-9]{64}$/, `Missing DSBI original ${key}`);
    assert.equal(source.source_coordinate_space, dsbiCoordinateSpace);
    assert.ok(Number.isSafeInteger(source.positive_cell_count) && source.positive_cell_count >= 0, 'Invalid DSBI source cell count');
    sourceMap.set(source.source, source);
  }
  assert.equal(preparation.generatedManifestSha256, index.generatedManifest?.sha256, 'DSBI generated manifest identity changed');
  const records = archivedData(preparation.generatedManifestSha256, 'generated manifest');
  assert.ok(Array.isArray(records) && records.length === index.generatedManifest.recordCount && records.length > 0, 'DSBI generated record count changed');
  const counts = new Map();
  for (const record of records) {
    const source = sourceMap.get(record.source);
    assert.ok(source, 'DSBI generated record has an unregistered TRAIN source');
    assert.equal(record.source_family, source.source_family, 'DSBI generated source family changed');
    assert.equal(record.source_coordinate_space, dsbiCoordinateSpace);
    assert.match(record.image_sha256, /^[a-f0-9]{64}$/, 'DSBI cached PNG hash is missing');
    assert.ok(Array.isArray(record.boxes) && Array.isArray(record.masks) && Array.isArray(record.originally_complete) && Array.isArray(record.ignore_boxes), 'DSBI cell and unknown-region arrays are required');
    assert.equal(record.boxes.length, record.masks.length, 'DSBI box/mask counts differ');
    assert.equal(record.boxes.length, record.originally_complete.length, 'DSBI completeness count differs');
    assert.ok(record.masks.every((mask) => Number.isSafeInteger(mask) && mask >= 1 && mask <= 63), 'DSBI zero or unknown masks cannot become positive targets');
    assert.ok(record.originally_complete.every((value) => typeof value === 'boolean'), 'Invalid DSBI completeness flag');
    for (const box of [...record.boxes, ...record.ignore_boxes]) {
      assert.ok(Array.isArray(box) && box.length === 4 && box.every(Number.isFinite) && box[0] < box[2] && box[1] < box[3], 'Invalid DSBI full source box');
    }
    counts.set(record.source, (counts.get(record.source) ?? 0) + 1);
  }
  assert.equal(new Set(records.map((record) => record.image)).size, records.length, 'Duplicate DSBI cached record');
  for (const source of index.sources) assert.equal(counts.get(source.source), source.generated_record_count, 'DSBI source record coverage changed');
  const snapshot = evidence(preparation['source-archive.json']);
  const protocol = evidence(preparation['preparation-protocol.json']);
  const audit = evidence(preparation['independent-context-audit.json']);
  assert.equal(snapshot.source_index_sha256, preparation.sourceIndex.sha256, 'DSBI source snapshot uses a different index');
  assert.equal(snapshot.manifest_sha256, preparation.generatedManifestSha256, 'DSBI source snapshot uses different records');
  assert.equal(snapshot.preparation_protocol_sha256, preparation['preparation-protocol.json'].sha256, 'DSBI source snapshot uses a different protocol');
  assert.equal(index.preparationProtocol?.sha256, preparation['preparation-protocol.json'].sha256, 'DSBI index uses a different preparation protocol');
  assert.ok(Array.isArray(snapshot.files) && snapshot.files.length >= 2, 'Missing DSBI preparation source snapshots');
  assert.equal(new Set(snapshot.files.map((source) => basename(source.archived))).size, snapshot.files.length, 'Duplicate DSBI source snapshot');
  const preparationSources = Object.fromEntries(snapshot.files.map((source) => [basename(source.archived), source.sha256]));
  verifySourceSet(preparation.sources, preparationSources);
  assert.equal(protocol.dataset_revision, dsbiRevision, 'DSBI preparation protocol revision changed');
  assert.equal(protocol.source_coordinate_space, dsbiCoordinateSpace);
  assert.equal(protocol.parser_sha256, preparationSources['word_dsbi.py'], 'DSBI parser snapshot differs from its preparation protocol');
  assert.equal(protocol.generator_sha256, preparationSources['word_dsbi_prepare.py'], 'DSBI generator snapshot differs from its preparation protocol');
  assert.match(protocol.parser_sha256, /^[a-f0-9]{64}$/);
  assert.match(protocol.generator_sha256, /^[a-f0-9]{64}$/);
  assert.equal(audit.status, 'passed', 'DSBI independent context audit did not pass');
  assert.equal(audit.source_index_sha256, preparation.sourceIndex.sha256, 'DSBI context audit uses a different index');
  assert.equal(audit.manifest_sha256, preparation.generatedManifestSha256, 'DSBI context audit uses different records');
  assert.deepEqual(audit.conservative_boundary_flags, [], 'DSBI context audit has unresolved boundary flags');
  assert.equal(audit.original_sources, 26);
  assert.equal(audit.contexts, records.length, 'DSBI context audit coverage is incomplete');
  const positiveCells = index.sources.reduce((sum, source) => sum + source.positive_cell_count, 0);
  assert.equal(audit.unique_positive_source_cells, positiveCells, 'DSBI source cell audit coverage is incomplete');
  assert.equal(audit.complete_unique_source_cells_seen, positiveCells, 'DSBI source cells lack complete contexts');
  assert.equal(audit.all_intersecting_positive_occurrences_checked, records.reduce((sum, record) => sum + record.boxes.length, 0));
  assert.equal(audit.all_unknown_ignore_boxes_checked, records.reduce((sum, record) => sum + record.ignore_boxes.length, 0));
  assert.equal(audit.pixel_bytes_compared, records.length * 480 * 192 * 3, 'DSBI pixel audit coverage is incomplete');
  assert.equal(audit.pixel_mismatches, 0, 'DSBI cached pixels differ from their source crops');
  assert.ok(Number.isFinite(audit.max_coordinate_error) && audit.max_coordinate_error >= 0 && audit.max_coordinate_error <= 1e-9, 'DSBI source coordinate audit failed');
  supplementalByHash.set(preparation.sourceIndex.sha256, { preparation, records, index });
}
const dsbiDatasets = (manifest.datasets ?? []).filter((dataset) => dataset.name === 'DSBI');
assert.equal(dsbiDatasets.length, supplements.length, 'Manifest DSBI datasets differ from retained preparations');
for (const dataset of dsbiDatasets) {
  assert.equal(dataset.revision, dsbiRevision);
  assert.equal(dataset.repository, 'https://github.com/yeluo1994/DSBI');
}
const parseDSBIPointSource = (entry, source) => {
  assert.equal(entry.sha256, source.annotation_sha256, 'DSBI point annotation hash differs from its source index');
  const text = verifiedBytes(entry).toString('utf8').replace(/^\uFEFF/, '');
  const cells = new Map();
  if (!text.trim()) {
    assert.equal(source.positive_cell_count, 0, 'DSBI empty annotation has declared positive cells');
    return cells;
  }
  const lines = text.split(/\r\n|\n|\r/);
  const numbers = (line) => {
    const tokens = (line ?? '').trim().split(/\s+/);
    assert.ok(tokens.length && tokens.every((token) => /^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$/.test(token)), 'DSBI raw grid must contain plain numeric tokens');
    const values = tokens.map(Number);
    assert.ok(values.every(Number.isFinite), 'DSBI raw grid must be finite');
    return values;
  };
  assert.equal(numbers(lines[0]).length, 1, 'DSBI raw angle must be scalar');
  const xs = numbers(lines[1]), ys = numbers(lines[2]);
  assert.ok(xs.length % 2 === 0 && ys.length % 3 === 0, 'DSBI raw grid dimensions are invalid');
  for (const values of [xs, ys]) assert.ok(values[0] >= 0 && values.every((value, i) => i === 0 || value > values[i - 1]), 'DSBI raw grid must increase strictly');
  assert.ok(Array.isArray(source.image_size) && source.image_size.length === 2 && source.image_size.every((value) => Number.isSafeInteger(value) && value > 0), 'Missing DSBI source raster size');
  assert.ok(xs.every((x) => x < source.image_size[0]) && ys.every((y) => y < source.image_size[1]), 'DSBI raw grid exceeds its source raster');
  const slots = new Set();
  for (let i = 3; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const tokens = lines[i].trim().split(/\s+/);
    assert.ok(tokens.length === 8 && tokens.slice(0, 2).every((token) => /^[1-9][0-9]*$/.test(token)) && tokens.slice(2).every((token) => token === '0' || token === '1'), 'DSBI raw cell needs one-based row/column and six bits');
    const [row, col] = tokens.map(Number);
    assert.ok(Number.isSafeInteger(row) && Number.isSafeInteger(col) && row <= ys.length / 3 && col <= xs.length / 2, 'DSBI raw cell is outside its grid');
    const slot = `${row}:${col}`;
    assert.ok(!slots.has(slot), 'DSBI raw annotation repeats a cell');
    slots.add(slot);
    const mask = tokens.slice(2).reduce((value, bit, bitIndex) => value | (Number(bit) << bitIndex), 0);
    if (!mask) continue;
    const points = xs.slice((col - 1) * 2, col * 2).flatMap((x) => ys.slice((row - 1) * 3, row * 3).map((y) => [x, y]));
    cells.set(`${slot}:${i + 1}`, { mask, points });
  }
  assert.equal(cells.size, source.positive_cell_count, 'DSBI raw positive cell count differs from its index');
  return cells;
};
const verifyDSBIPointMetadata = (run, config, indexes) => {
  const enabled = config.dsbi_visible_partial_supervision?.enabled ?? false;
  assert.equal(typeof enabled, 'boolean', 'Invalid DSBI point supervision flag');
  assert.equal(Boolean(config.dsbi_point_metadata), enabled, 'DSBI point metadata requires explicit enabled supervision');
  if (!enabled) {
    assert.ok(!run.dsbiPointMetadata, 'Disabled DSBI point supervision cannot retain active metadata');
    return;
  }
  assert.ok(indexes.length > 0, 'DSBI point supervision requires retained source indexes');
  assert.equal(config.partial_geometry_supervision?.enabled, false, 'DSBI visible and full-source partial supervision cannot be combined');
  assert.deepEqual(config.dsbi_visible_partial_supervision.minimum_visible_size_pixels, [4, 12]);
  const digest = config.data_sha256?.[config.dsbi_point_metadata];
  const metadata = evidence(run.dsbiPointMetadata);
  assert.equal(run.dsbiPointMetadata.sha256, digest, 'DSBI point metadata differs from its training configuration');
  assert.deepEqual(archivedData(digest, 'point metadata'), metadata);
  assert.equal(metadata.schemaVersion, 1);
  assert.equal(metadata.coordinate_space, 'source_pixel_index_centers', 'DSBI points must use source pixel-index centers');
  assert.ok(Array.isArray(metadata.indexes) && metadata.indexes.length === indexes.length, 'DSBI point index coverage differs');
  assert.deepEqual(metadata.indexes.map((entry) => entry.sha256).sort(), indexes.map((path) => config.data_sha256[path]).sort(), 'DSBI point metadata uses different source indexes');
  assert.ok(metadata.records && typeof metadata.records === 'object' && !Array.isArray(metadata.records), 'Missing DSBI point records');
  const trainingProtocol = evidence(run['dsbi-visible-partial-protocol.json']);
  assert.equal(trainingProtocol.enabled, true, 'DSBI point protocol is not enabled');
  assert.equal(trainingProtocol.metadata_path, config.dsbi_point_metadata, 'DSBI point protocol metadata path changed');
  assert.equal(trainingProtocol.point_metadata_sha256, digest, 'DSBI point protocol metadata hash changed');
  assert.deepEqual(trainingProtocol.source_sha256, config.source_sha256, 'DSBI point protocol uses different training sources');
  assert.equal(trainingProtocol.training_manifest_sha256, config.data_sha256[config.train_manifest], 'DSBI point protocol uses different training records');
  assert.equal(trainingProtocol.parent_checkpoint_sha256, config.resume_sha256, 'DSBI point protocol uses a different parent checkpoint');
  assert.equal(trainingProtocol.planned_epochs, config.epochs, 'DSBI point protocol epoch count differs');
  assert.deepEqual(trainingProtocol.minimum_visible_size_pixels, [4, 12]);
  assert.equal(trainingProtocol.minimum_surviving_occupied_centers, 1);
  assert.equal(trainingProtocol.full_span_canvas_dimensions_excluded, true);
  assert.ok(Date.parse(trainingProtocol.created_at_utc) < Date.parse(config.started_at_utc), 'DSBI point protocol must precede training');
  for (const name of ['word_dsbi.py', 'word_point_metadata.py', 'word_point_geometry.py']) {
    assert.match(config.source_sha256?.[name], /^[a-f0-9]{64}$/, `Missing DSBI point source hash: ${name}`);
    assert.ok(run.sources.some((entry) => basename(entry.file) === name && entry.sha256 === config.source_sha256[name]), `Missing original DSBI point source snapshot: ${name}`);
  }
  const expectedImages = [], occurrenceCounts = [];
  for (const path of indexes) {
    const retained = supplementalByHash.get(config.data_sha256[path]);
    assert.ok(retained, 'DSBI point source preparation is missing');
    const { preparation, records, index } = retained;
    assert.equal(metadata.indexes.find((entry) => entry.sha256 === preparation.sourceIndex.sha256).manifest_sha256, preparation.generatedManifestSha256, 'DSBI point metadata uses different generated records');
    assert.equal(trainingProtocol.supplemental_source_index_sha256, preparation.sourceIndex.sha256, 'DSBI point protocol source index changed');
    assert.equal(trainingProtocol.supplemental_manifest_sha256, preparation.generatedManifestSha256, 'DSBI point protocol supplemental records changed');
    const annotations = preparation.sourceAnnotations;
    assert.ok(Array.isArray(annotations) && annotations.length === 26, 'DSBI points require all 26 original raw annotations');
    assert.equal(new Set(annotations.map((entry) => entry.source)).size, 26, 'Duplicate DSBI point source annotation');
    assert.deepEqual(annotations.map((entry) => entry.source).sort(), index.sources.map((source) => source.source).sort(), 'DSBI point annotation identities differ from TRAIN');
    const sourceCells = new Map(index.sources.map((source) => [source.source,
      parseDSBIPointSource(annotations.find((entry) => entry.source === source.source), source)]));
    for (const record of records) {
      expectedImages.push(record.image);
      const row = metadata.records[record.image];
      assert.ok(row, 'DSBI point metadata dropped a generated record');
      for (const key of ['source', 'image_sha256', 'crop', 'masks', 'source_cells']) assert.deepEqual(row[key], record[key], `DSBI point record ${key} differs from frozen data`);
      assert.ok(Array.isArray(row.source_cells) && row.source_cells.length === record.masks.length && Array.isArray(row.source_points) && row.source_points.length === record.masks.length, 'DSBI point cell coverage differs');
      for (const [number, identity] of row.source_cells.entries()) {
        assert.ok(['row', 'col', 'annotation_line'].every((key) => Number.isSafeInteger(identity[key]) && identity[key] > 0), 'Invalid DSBI point source-cell identity');
        const original = sourceCells.get(record.source).get(`${identity.row}:${identity.col}:${identity.annotation_line}`);
        assert.ok(original && original.mask === record.masks[number], 'DSBI occupied points do not match a known raw source mask');
        assert.deepEqual(row.source_points[number], original.points, 'DSBI points differ from the raw source grid');
      }
      occurrenceCounts.push(record.masks.length);
    }
  }
  assert.deepEqual(Object.keys(metadata.records).sort(), expectedImages.sort(), 'DSBI point records must exactly cover frozen supplemental data');
  assert.equal(config.dsbi_visible_partial_supervision.metadata?.sha256, digest, 'DSBI point startup audit uses different metadata');
  assert.equal(config.dsbi_visible_partial_supervision.metadata.records, expectedImages.length, 'DSBI point startup record count differs');
  assert.equal(config.dsbi_visible_partial_supervision.metadata.known_cell_occurrences, occurrenceCounts.reduce((a, b) => a + b, 0), 'DSBI point startup cell count differs');
};
const verifyGeometryInitialization = (run, config, indexes) => {
  const branch = config.geometry_conditioned_residual;
  assert.ok(branch === undefined || typeof branch.enabled === 'boolean', 'Invalid geometry residual training flag');
  if (!branch?.enabled) {
    assert.equal(run.geometryInitialization, undefined, 'Geometry initialization requires enabled training');
    return;
  }
  const retained = run.geometryInitialization;
  assert.ok(retained, 'Missing geometry initialization provenance');
  verifySourceSet(run.sources, config.source_sha256);
  assert.equal(config.dsbi_visible_partial_supervision?.enabled, true, 'Geometry priors require verified DSBI point metadata');
  assert.equal(indexes.length, 1, 'Geometry priors require one frozen DSBI source index');
  const documents = {};
  for (const key of ['initializationReport', 'nonzeroConversionProof', 'slotPriorAudit', 'featureLatticeAudit',
    'backboneLatticeIdentity', 'independentCoordinateAudit', 'independentHeadAudit', 'gpuBatchPreflight', 'protocol']) {
    assert.ok(retained[key], `Missing geometry proof: ${key}`);
    documents[key] = evidence(retained[key]);
    if (key !== 'protocol') assert.equal(documents[key].status, 'passed', `Geometry proof failed: ${key}`);
  }
  const { initializationReport: init, nonzeroConversionProof: conversionProof, slotPriorAudit: slots,
    featureLatticeAudit: lattice, backboneLatticeIdentity: identity, independentCoordinateAudit: coordinates,
    independentHeadAudit: head, gpuBatchPreflight: gpu, protocol } = documents;
  const base = retained.baseParentCheckpointSha256, zero = retained.zeroParentCheckpointSha256,
    diagnostic = retained.forbiddenDiagnosticCheckpointSha256;
  for (const digest of [base, zero, diagnostic]) assert.match(digest, /^[a-f0-9]{64}$/);
  assert.equal(new Set([base, zero, diagnostic]).size, 3, 'Geometry parent and diagnostic hashes must be distinct');
  assert.equal(branch.base_parent_checkpoint_sha256, base, 'Geometry base parent differs');
  assert.equal(branch.zero_checkpoint_sha256, zero, 'Geometry zero parent differs');
  assert.equal(config.resume_sha256, zero, 'Geometry training must resume the zero parent');
  assert.ok(![zero, diagnostic].includes(validation.oracle.checkpoint_sha256), 'Geometry initialization probes cannot be selected');
  for (const ancestor of provenance.trainingRuns) assert.notEqual(evidence(ancestor.configuration).resume_sha256, diagnostic, 'Geometry diagnostic checkpoint cannot enter training ancestry');
  const eligibleBase = provenance.trainingRuns.some((ancestor) => {
    if (ancestor === run || !ancestor['sampling-lineage-final.json']) return false;
    const audit = evidence(ancestor['sampling-lineage-final.json']);
    return audit.status === 'passed' && audit.actual.config_sha256 === ancestor.configuration.sha256 &&
      audit.actual.sampling_history_sha256 === ancestor['sampling-history.json']?.sha256 &&
      audit.actual.observed_epochs.some((row) => row.checkpoint_sha256 === base &&
        row.eligible_by_sampling_lineage === true && audit.eligible_retained_epochs.includes(row.epoch));
  });
  assert.ok(eligibleBase, 'Geometry base parent has no retained eligible ancestor');
  assert.equal(init.parent_checkpoint_sha256, base, 'Geometry initialization uses a different base parent');
  assert.equal(init.zero_checkpoint_sha256, zero, 'Geometry initialization uses a different zero parent');
  assert.equal(init.nonzero_diagnostic_sha256, diagnostic, 'Geometry diagnostic identity differs');
  assert.equal(init.nonzero_diagnostic_must_never_be_resumed, true);
  assert.equal(branch.initialization_report_sha256, retained.initializationReport.sha256, 'Geometry initialization report changed');
  assert.equal(branch.conversion_report_sha256, retained.nonzeroConversionProof.sha256, 'Geometry conversion report changed');
  assert.equal(branch.slot_prior_audit_sha256, retained.slotPriorAudit.sha256, 'Geometry slot audit changed');
  assert.equal(init.slot_prior_audit_sha256, retained.slotPriorAudit.sha256);
  const requiredSource = (name, digest, inTraining = false) => {
    assert.match(digest, /^[a-f0-9]{64}$/);
    const entry = retained.sources?.find((item) => basename(item.file) === name && item.sha256 === digest);
    assert.ok(entry, `Missing exact geometry source: ${name}`);
    verifiedBytes(entry);
    if (inTraining) {
      assert.equal(config.source_sha256[name], digest, `Geometry source differs from training: ${name}`);
      assert.ok(run.sources.some((item) => basename(item.file) === name && item.sha256 === digest), `Geometry training source cannot use an archive gap: ${name}`);
    }
  };
  assert.equal(new Set(retained.sources?.map((entry) => basename(entry.file))).size, retained.sources?.length, 'Duplicate geometry source');
  for (const name of ['word_data.py', 'word_model.py', 'word_geometry_head.py', 'word_geometry_init.py']) requiredSource(name, init.source_sha256?.[name], true);
  requiredSource('word_geometry_parity.py', conversionProof.source_sha256, true);
  for (const [name, digest] of Object.entries(gpu.source_sha256)) {
    assert.equal(config.source_sha256[name], digest, `Geometry GPU source differs: ${name}`);
    assert.ok(run.sources.some((entry) => basename(entry.file) === name && entry.sha256 === digest));
  }
  for (const [name, digest] of [['audit_slot_priors.py', slots.script_sha256], ['audit_feature_lattice.py', lattice.script_sha256],
    ['independent-coordinate-preflight.py', coordinates.script_sha256], ['independent-geometry-head-audit.py', head.script_sha256],
    ['test_word_geometry_head.py', head.test_source_sha256], ['probe_geometry_training_batch.py', gpu.script_sha256]]) requiredSource(name, digest);
  assert.equal(head.source_sha256, config.source_sha256['word_geometry_head.py']);
  assert.deepEqual([branch.residual_channels, branch.projection_channels, branch.hidden_units, branch.added_parameters], [11, 8, 32, 2451]);
  for (const document of [init, protocol, head]) assert.equal(document.added_parameters, branch.added_parameters);
  assert.equal(init.zero_outputs_equal_parent, true, 'Geometry zero outputs must equal the parent');
  assert.equal(init.zero_save_reload_all_weights_equal, true);
  const positive = (value) => Number.isFinite(value) && value > 0;
  assert.equal(init.initial_final_residual_gradient_l1_by_channel?.length, 11);
  assert.ok(init.initial_final_residual_gradient_l1_by_channel.every(positive), 'Every geometry residual channel needs a direct gradient');
  for (const key of ['projection_l1', 'hidden_l1']) {
    assert.equal(init.zero_branch_gradients[key], 0, 'Geometry zero projection must block hidden gradients initially');
    assert.ok(positive(init.after_isolated_final_step_gradients[key]), 'Geometry hidden gradients must work after the diagnostic step');
  }
  assert.ok(positive(init.nonzero_max_output_delta), 'Geometry conversion requires actual nonzero outputs');
  assert.match(init.oracle_sha256, /^[a-f0-9]{64}$/);
  assert.equal(conversionProof.initialization_report_sha256, retained.initializationReport.sha256);
  assert.equal(conversionProof.oracle_sha256, init.oracle_sha256, 'Geometry conversion uses a different oracle');
  assert.equal(conversionProof.conversion_probes_must_not_be_activated_or_resumed, true);
  assert.equal(conversionProof.trained_candidate_requires_full405_parity, true);
  assert.notEqual(conversionProof.probes?.zero?.artifact_sha256, conversionProof.probes?.nonzero?.artifact_sha256, 'Geometry nonzero artifact must differ from the zero probe');
  assert.deepEqual([conversionProof.absolute_tolerance, conversionProof.relative_tolerance], [.0001, .0001]);
  for (const [name, digest] of [['zero', zero], ['nonzero', diagnostic]]) {
    const probe = conversionProof.probes?.[name];
    assert.equal(probe?.checkpoint_sha256, digest, `Geometry ${name} conversion checkpoint differs`);
    assert.match(probe.artifact_sha256, /^[a-f0-9]{64}$/);
    assert.notEqual(probe.artifact_sha256, manifest.artifact.sha256, 'Geometry conversion probes cannot be activated');
    assert.equal(probe.samples, 6);
    assert.equal(probe.all_finite, true);
    assert.equal(probe.builtins_only, true, 'Geometry nonzero and zero conversion must use builtins');
    assert.ok(probe.operators.length > 0 && probe.operators.every((op) => typeof op === 'string' && !/CUSTOM|Flex/i.test(op)), 'Geometry conversion contains unsupported operators');
    for (const key of ['torch_vs_tf_max_abs', 'tf_vs_tflite_max_abs', 'torch_vs_tflite_max_abs']) {
      assert.ok(Number.isFinite(probe[key]) && probe[key] >= 0 && probe[key] <= conversionProof.absolute_tolerance, `Geometry conversion error exceeds tolerance: ${name}/${key}`);
    }
  }
  const supplemental = supplementalByHash.get(config.data_sha256[indexes[0]]);
  const metadataHash = run.dsbiPointMetadata.sha256;
  for (const document of [slots, coordinates, gpu, protocol]) assert.equal(document.point_metadata_sha256, metadataHash, 'Geometry proof uses different DSBI points');
  for (const document of [slots, coordinates]) assert.equal(document.manifest_sha256, supplemental.preparation.generatedManifestSha256);
  assert.equal(coordinates.source_index_sha256, supplemental.preparation.sourceIndex.sha256);
  assert.deepEqual(coordinates.raw_annotations, supplemental.preparation.sourceAnnotations.map(({ source, sha256 }) => ({ source, sha256 })), 'Geometry coordinate audit uses different raw annotations');
  // One vote per raw TRAIN cell, not per cached crop occurrence. Source coordinates
  // use Pillow pixel indices; the .5 converts them to the box corner convention.
  const fractions = Array.from({ length: 6 }, () => [[], []]);
  let unique = 0;
  for (const source of supplemental.index.sources) {
    const annotation = supplemental.preparation.sourceAnnotations.find((entry) => entry.source === source.source);
    const cells = parseDSBIPointSource(annotation, source);
    for (const { points } of cells.values()) {
      const margin = .3 * (points[3][0] - points[0][0]);
      const left = points[0][0] - margin, top = points[0][1] - margin;
      const width = points[3][0] + margin - left, height = points[2][1] + margin - top;
      for (let slot = 0; slot < 6; slot++) {
        fractions[slot][0].push((points[slot][0] + .5 - left) / width);
        fractions[slot][1].push((points[slot][1] + .5 - top) / height);
      }
      unique++;
    }
  }
  const median = (values) => { values.sort((a, b) => a - b); const half = Math.floor(values.length / 2); return values.length % 2 ? values[half] : (values[half - 1] + values[half]) / 2; };
  const expectedFractions = fractions.map((slot) => slot.map(median));
  assert.equal(slots.unique_cells, unique, 'Geometry priors must weight unique source cells');
  assert.equal(coordinates.unique_raw_nonzero_cells, unique);
  assert.equal(coordinates.unique_complete_seen, unique);
  const occurrences = supplemental.records.reduce((sum, record) => sum + record.masks.length, 0);
  const completeOccurrences = supplemental.records.reduce((sum, record) => sum + record.originally_complete.filter(Boolean).length, 0);
  assert.equal(coordinates.cached_occurrences, occurrences);
  assert.equal(coordinates.complete_cached_occurrences, completeOccurrences);
  assert.equal(slots.complete_cached_occurrences, completeOccurrences);
  for (const key of ['max_exact_cached_reconstruction_error_pixels', 'max_same_source_fraction_difference_across_crops']) {
    assert.ok(Number.isFinite(slots[key]) && slots[key] >= 0 && slots[key] < 1e-10, `Geometry cached coordinate audit failed: ${key}`);
  }
  for (const actual of [slots.median_corner_fractions, coordinates.raw_source_medians, branch.sampling?.slot_fractions]) {
    assert.equal(actual?.length, 6);
    for (let slot = 0; slot < 6; slot++) {
      assert.equal(actual[slot].length, 2);
      for (let axis = 0; axis < 2; axis++) assert.ok(Number.isFinite(actual[slot][axis]) && Math.abs(actual[slot][axis] - expectedFractions[slot][axis]) < 1e-12, 'Geometry slot priors differ from raw TRAIN grids');
    }
  }
  assert.deepEqual(slots.slot_order, ['left_top', 'left_middle', 'left_bottom', 'right_top', 'right_middle', 'right_bottom']);
  assert.deepEqual(slots.axis_order, ['x', 'y']);
  assert.equal(branch.sampling.feature_origin, 3, 'Geometry feature origin must be 3');
  assert.equal(branch.sampling.stride, 4);
  assert.deepEqual(init.sampling, branch.sampling, 'Geometry initialization sampling changed');
  assert.deepEqual(protocol.sampling, branch.sampling, 'Geometry protocol sampling changed');
  assert.deepEqual(lattice.backbone_outputs.block_3_expand_relu.size, [48, 120]);
  assert.deepEqual(lattice.backbone_outputs.block_3_expand_relu.jump, [4, 4]);
  assert.deepEqual(lattice.backbone_outputs.block_3_expand_relu.origin, [3, 3]);
  assert.equal(identity.lattice_audit_sha256, retained.featureLatticeAudit.sha256);
  const originalGraph = identity.candidates.find((row) => row.checkpoint_sha256 === lattice.checkpoint_sha256);
  const baseGraph = identity.candidates.find((row) => row.checkpoint_sha256 === base);
  assert.ok(originalGraph && baseGraph, 'Geometry lattice audit does not cover the base parent');
  assert.match(originalGraph.backbone_config_sha256, /^[a-f0-9]{64}$/);
  assert.equal(originalGraph.backbone_config_sha256, baseGraph.backbone_config_sha256, 'Geometry backbone lattice graph changed');
  assert.equal(coordinates.checkpoint_sha256, lattice.checkpoint_sha256);
  assert.deepEqual(coordinates.defects, []);
  for (const [name, digest] of Object.entries({ 'audit_feature_lattice.py': lattice.script_sha256,
    'feature-lattice-audit.json': retained.featureLatticeAudit.sha256, 'audit_slot_priors.py': slots.script_sha256,
    'slot-prior-audit.json': retained.slotPriorAudit.sha256 })) assert.equal(coordinates.audited_files[name], digest);
  assert.equal(coordinates.all_reported_fraction_and_pixel_error_percentiles_match, true);
  assert.deepEqual(head.defects, []);
  assert.equal(head.initialization_report_sha256, retained.initializationReport.sha256);
  assert.equal(head.parent_checkpoint_sha256, base);
  assert.equal(head.zero_checkpoint_sha256, zero);
  assert.equal(head.oracle_sha256, init.oracle_sha256);
  assert.equal(head.final_projection_all_zero, true);
  assert.equal(head.head_kernel_duplication, false);
  assert.equal(head.inverse_sigmoid_reconstruction, false);
  assert.equal(head.backbone_called_training_false, true);
  assert.equal(head.unchanged_old_normalization_and_feature_graph, true);
  assert.equal(config.image_normalization, 'channel_gray_std05', 'Geometry training changed inherited image normalization');
  assert.deepEqual(config.imageNormalization, imageNormalization, 'Geometry training normalization differs from the selected model');
  assert.equal(config.batch_normalization?.layers, head.frozen_batchnorm_layers);
  assert.equal(config.batch_normalization.trainable_affine_layers, 0, 'Geometry inherited batch normalization must remain frozen');
  assert.deepEqual(config.batch_normalization.backbone_call_training_keywords, [false]);
  assert.deepEqual(head.inherited_head_filters, [1, 4, 6]);
  assert.deepEqual(head.input_shape, [null, 192, 480, 3]);
  assert.deepEqual(head.output_shape, [null, 48, 120, 11]);
  assert.ok(head.all_inherited_arrays_equal > 0 && head.frozen_batchnorm_layers > 0);
  assert.equal(head.stored_zero_oracle_equal_inputs, 6);
  assert.equal(head.stored_nonzero_max_delta, init.nonzero_max_output_delta);
  assert.ok(head.multi_batch_scalar_oracle.batch > 1 && head.multi_batch_scalar_oracle.out_of_range_slots_clamped > 0);
  assert.equal(head.multi_batch_scalar_oracle.batched_equals_separate_calls, true);
  assert.ok(Number.isFinite(head.multi_batch_scalar_oracle.max_absolute_error) && head.multi_batch_scalar_oracle.max_absolute_error <= 1e-5);
  assert.equal(gpu.parent_sha256, zero);
  assert.equal(gpu.batch_size, config.batch_size);
  assert.ok(Number.isFinite(gpu.loss) && gpu.gradient_tensors > 0);
  assert.deepEqual(gpu.negative_expansion, config.negative_expansion, 'Geometry GPU probe used different negative exposure');
  assert.equal(protocol.geometry_conditioned_residual, true);
  assert.equal(run['geometry-residual-protocol.json']?.sha256, retained.protocol.sha256, 'Geometry retained protocol links differ');
  assert.deepEqual(protocol.source_sha256, config.source_sha256, 'Geometry protocol sources differ from training');
  for (const [key, expected] of Object.entries({ base_parent_checkpoint_sha256: base, zero_parent_checkpoint_sha256: zero,
    initialization_report_sha256: retained.initializationReport.sha256, conversion_report_sha256: retained.nonzeroConversionProof.sha256,
    slot_prior_audit_sha256: retained.slotPriorAudit.sha256, training_manifest_sha256: config.data_sha256[config.train_manifest],
    batch_size: config.batch_size, planned_epochs: config.epochs, initial_learning_rate: config.learning_rate,
    geometry_loss: config.geometry_loss, dsbi_visible_partial_supervision: true, horizontal_mirror_augmentation: config.horizontal_mirror_augmentation })) {
    assert.equal(protocol[key], expected, `Geometry protocol differs: ${key}`);
  }
  assert.deepEqual(protocol.planned_eligible_epochs, [8, 9, 10]);
  assert.equal(protocol.geometry_loss, 'relative_size_huber');
  assert.equal(protocol.horizontal_mirror_augmentation, false);
  for (const doc of [init, conversionProof, protocol]) assert.ok(Date.parse(doc.created_at_utc) < Date.parse(config.started_at_utc), 'Geometry proofs must precede training');
  const preflights = ['initializationReport', 'nonzeroConversionProof', 'independentCoordinateAudit', 'independentHeadAudit', 'gpuBatchPreflight'];
  assert.deepEqual(protocol.preflight_audits.map((entry) => entry.sha256).sort(), preflights.map((key) => retained[key].sha256).sort(), 'Geometry protocol preflight proofs changed');
};
const verifyHorizontalMirrorPreflight = (run, config) => {
  assert.ok(config.horizontal_mirror_augmentation === undefined || typeof config.horizontal_mirror_augmentation === 'boolean', 'Invalid horizontal mirror flag');
  if (!config.horizontal_mirror_augmentation) {
    assert.equal(run.horizontalMirrorPreflight, undefined, 'Mirror preflight requires enabled augmentation');
    return;
  }
  const retained = run.horizontalMirrorPreflight;
  assert.ok(retained, 'Missing horizontal mirror preflight');
  verifySourceSet(run.sources, config.source_sha256);
  const protocol = evidence(retained.protocol), parent = retained.parentCheckpointSha256;
  assert.match(parent, /^[a-f0-9]{64}$/);
  assert.equal(config.resume_sha256, parent, 'Mirror training resumed a different parent');
  assert.equal(protocol.parent_checkpoint_sha256, parent);
  if (run['horizontal-mirror-protocol.json']) assert.equal(run['horizontal-mirror-protocol.json'].sha256, retained.protocol.sha256, 'Mirror retained protocol links differ');
  assert.ok(Date.parse(protocol.created_at_utc) < Date.parse(config.started_at_utc), 'Mirror protocol must precede training');
  assert.deepEqual(protocol.source_sha256, config.source_sha256, 'Mirror protocol sources differ');
  assert.equal(protocol.training_manifest_sha256, config.data_sha256?.[config.train_manifest], 'Mirror training data changed');
  assert.equal(protocol.point_metadata_sha256, run.dsbiPointMetadata?.sha256, 'Mirror point metadata changed');
  for (const [key, expected] of Object.entries({ horizontal_mirror_augmentation: true, half_turn_augmentation: false,
    illumination_augmentation: true, planned_epochs: config.epochs, batch_size: config.batch_size, learning_rate: config.learning_rate,
    seed: config.seed, geometry_loss: config.geometry_loss, negative_repeat: config.negative_expansion?.base_repeat,
    supplemental_negative_repeat: config.negative_expansion?.supplemental_repeat })) assert.equal(protocol[key], expected, `Mirror protocol differs: ${key}`);
  assert.equal(config.half_turn_augmentation, false, 'Mirror cannot combine with a half turn');
  assert.equal(config.polarity_augmentation, false, 'Mirror cannot invert relief polarity');
  assert.equal(config.illumination_augmentation, true);
  for (const [key, enabled] of [['dsbi_visible_partial_supervision', true], ['partial_geometry_supervision', false], ['geometry_conditioned_residual', false]]) {
    assert.equal(config[key]?.enabled, enabled, `Mirror training flag differs: ${key}`);
    assert.equal(protocol[key], enabled);
  }
  assert.equal(protocol.frozen_batchnorm, true);
  assert.equal(config.batch_normalization?.trainable_affine_layers, 0);
  assert.deepEqual(config.batch_normalization.backbone_call_training_keywords, [false]);
  assert.equal(protocol.no_activation_before_original_gates, true);
  assert.deepEqual(protocol.planned_eligible_epochs, [8, 9, 10]);
  assert.equal(protocol.mirror?.probability, .5, 'Mirror probability must remain one half');
  assert.deepEqual(protocol.mirror.slot_permutation_zero_based, [3, 4, 5, 0, 1, 2], 'Mirror slot permutation changed');
  assert.equal(config.augmentation_protocol?.horizontal_mirror_probability, .5);
  assert.deepEqual(config.augmentation_protocol.horizontal_mirror_bit_mapping, [4, 5, 6, 1, 2, 3]);
  assert.equal(config.augmentation_protocol.half_turn_probability, 0);
  assert.equal(config.augmentation_protocol.polarity_inversion_probability, 0);
  const ancestor = provenance.trainingRuns.find((row) => row !== run && row.configuration.sha256 === protocol.parent_training_config_sha256 &&
    row['sampling-lineage-final.json']?.sha256 === protocol.parent_sampling_lineage_sha256);
  assert.ok(ancestor, 'Mirror parent has no matching retained ancestor');
  const lineage = evidence(ancestor['sampling-lineage-final.json']);
  assert.equal(lineage.status, 'passed');
  assert.equal(lineage.actual.config_sha256, ancestor.configuration.sha256);
  assert.equal(lineage.actual.sampling_history_sha256, ancestor['sampling-history.json']?.sha256);
  assert.ok(lineage.actual.observed_epochs.some((row) => row.checkpoint_sha256 === parent && row.eligible_by_sampling_lineage === true &&
    lineage.eligible_retained_epochs.includes(row.epoch)), 'Mirror parent is not sampling eligible');
  assert.deepEqual(retained.audits.map((entry) => entry.sha256).sort(), protocol.preflight_audits.map((entry) => entry.sha256).sort(), 'Mirror preflight audit links changed');
  const audits = retained.audits.map(evidence);
  const independent = audits.filter((audit) => audit.all_known_mask_mirror_involutions_and_target_labels !== undefined);
  assert.equal(independent.length, 1, 'Missing unique independent mirror audit');
  const audit = independent[0];
  assert.equal(audit.status, 'passed');
  assert.deepEqual(audit.defects, []);
  assert.equal(audit.post_launch_addendum, undefined, 'Postlaunch mirror addendum cannot replace preflight');
  assert.equal(audit.run20_config_sha256, ancestor.configuration.sha256);
  assert.equal(audit.training_manifest_sha256, protocol.training_manifest_sha256);
  assert.equal(audit.point_metadata_sha256, protocol.point_metadata_sha256);
  for (const name of ['word_data.py', 'word_train.py', 'word_point_geometry.py', 'word_point_metadata.py', 'word_sampling.py']) {
    assert.equal(audit.source_sha256?.[name], config.source_sha256[name], `Mirror audit source differs: ${name}`);
    assert.match(audit.source_sha256[name], /^[a-f0-9]{64}$/);
  }
  const script = retained.sources?.find((entry) => basename(entry.file) === 'audit_run22_mirror.py' && entry.sha256 === audit.script_sha256);
  assert.ok(script, 'Missing exact original mirror audit script');
  verifiedBytes(script);
  assert.equal(audit.all_known_mask_mirror_involutions_and_target_labels, 63);
  assert.equal(audit.partial_unknown_bit_checks, 63);
  assert.equal(audit.stage_visibility_cases, 252);
  assert.deepEqual(audit.point_slot_permutation, protocol.mirror.slot_permutation_zero_based);
  for (const key of ['run20_data_point_metadata_sampling_sources_byte_identical', 'run20_patches_ast_identical', 'mirror_pixel_operation_exact',
    'full_balanced_order_identical', 'off_mode_rgb_targets_rng_exact_vs_run20', 'on_off_random_calls_and_final_rng_states_equal',
    'metadata_and_cached_point_states_unchanged']) assert.equal(audit[key], true, `Mirror independent audit failed: ${key}`);
  assert.ok(audit.real_batches?.length > 0 && audit.mirror_invocations > 0);
  assert.deepEqual(audit.expanded_negative_exposure, { base: [config.negative_expansion.base_unique_negative_records, config.negative_expansion.base_repeat],
    supplemental: [config.negative_expansion.supplemental_unique_negative_records, config.negative_expansion.supplemental_repeat] });
};
const verifyVerticalStretchPreflight = (run, config) => {
  assert.ok(config.vertical_stretch_augmentation === undefined || typeof config.vertical_stretch_augmentation === 'boolean', 'Invalid vertical stretch flag');
  if (!config.vertical_stretch_augmentation) {
    assert.equal(run.verticalStretchPreflight, undefined, 'Vertical preflight requires enabled augmentation');
    return;
  }
  const retained = run.verticalStretchPreflight;
  assert.ok(retained, 'Missing vertical stretch preflight');
  verifySourceSet(run.sources, config.source_sha256);
  for (const name of ['word_data.py', 'word_train.py', 'word_vertical_stretch.py', 'word_point_geometry.py', 'word_point_metadata.py', 'word_sampling.py']) {
    assert.match(config.source_sha256[name] ?? '', /^[a-f0-9]{64}$/, `Missing vertical source snapshot: ${name}`);
  }
  const protocol = evidence(retained.protocol), parent = retained.parentCheckpointSha256;
  assert.match(parent, /^[a-f0-9]{64}$/);
  assert.equal(config.resume_sha256, parent, 'Vertical training resumed a different parent');
  assert.equal(protocol.parent_checkpoint_sha256, parent);
  if (run['vertical-stretch-protocol.json']) assert.equal(run['vertical-stretch-protocol.json'].sha256, retained.protocol.sha256);
  assert.ok(Date.parse(protocol.created_at_utc) < Date.parse(config.started_at_utc), 'Vertical protocol must precede training');
  assert.deepEqual(protocol.source_sha256, config.source_sha256, 'Vertical protocol sources differ');
  assert.equal(protocol.training_manifest_sha256, config.data_sha256?.[config.train_manifest], 'Vertical training data changed');
  assert.equal(protocol.point_metadata_sha256, run.dsbiPointMetadata?.sha256, 'Vertical point metadata changed');
  for (const [key, expected] of Object.entries({ vertical_stretch_augmentation: true, horizontal_mirror_augmentation: false,
    half_turn_augmentation: false, polarity_augmentation: false, illumination_augmentation: true })) {
    assert.equal(config[key], expected, `Vertical training flag differs: ${key}`);
    assert.equal(protocol[key], expected, `Vertical protocol differs: ${key}`);
  }
  for (const [key, enabled] of [['dsbi_visible_partial_supervision', true], ['partial_geometry_supervision', false], ['geometry_conditioned_residual', false]]) {
    assert.equal(config[key]?.enabled, enabled, `Vertical training flag differs: ${key}`);
    assert.equal(protocol[key], enabled);
  }
  for (const [key, expected] of Object.entries({ planned_epochs: config.epochs, batch_size: config.batch_size, learning_rate: config.learning_rate,
    seed: config.seed, geometry_loss: config.geometry_loss, negative_repeat: config.negative_expansion?.base_repeat,
    supplemental_negative_repeat: config.negative_expansion?.supplemental_repeat })) assert.equal(protocol[key], expected, `Vertical protocol differs: ${key}`);
  assert.equal(protocol.frozen_batchnorm, true);
  assert.equal(config.batch_normalization?.trainable_affine_layers, 0);
  assert.deepEqual(config.batch_normalization.backbone_call_training_keywords, [false]);
  assert.equal(protocol.no_activation_before_original_gates, true);
  assert.deepEqual(protocol.planned_eligible_epochs, [8, 9, 10]);
  const view = protocol.vertical_stretch;
  assert.deepEqual(view?.range, [.9, 1.1], 'Vertical stretch range changed');
  assert.equal(view.distribution, 'uniform');
  assert.equal(view.existing_affine_probability, .85, 'Vertical affine probability changed');
  assert.deepEqual(view.base_zoom, [.8, 1.7]);
  assert.equal(view.scale_x, 'zoom');
  assert.equal(view.scale_y, 'zoom*height_factor');
  assert.equal(view.stream_id, 0x56535452, 'Vertical random stream changed');
  assert.deepEqual(view.seed_sequence, ['seed', 'dataset_epoch', 'batch_index', 0x56535452]);
  assert.deepEqual(config.augmentation_protocol?.vertical_stretch_range, view.range);
  assert.equal(config.augmentation_protocol.vertical_stretch_affine_probability, .85);
  assert.equal(config.augmentation_protocol.vertical_stretch_stream_id, view.stream_id);
  const ancestor = provenance.trainingRuns.find((row) => row !== run && row.configuration.sha256 === protocol.parent_training_config_sha256 &&
    row['sampling-lineage-final.json']?.sha256 === protocol.parent_sampling_lineage_sha256);
  assert.ok(ancestor, 'Vertical parent has no matching retained ancestor');
  const lineage = evidence(ancestor['sampling-lineage-final.json']);
  assert.equal(lineage.status, 'passed');
  assert.equal(lineage.actual.config_sha256, ancestor.configuration.sha256);
  assert.equal(lineage.actual.sampling_history_sha256, ancestor['sampling-history.json']?.sha256);
  assert.ok(lineage.actual.observed_epochs.some((row) => row.checkpoint_sha256 === parent && row.eligible_by_sampling_lineage === true &&
    lineage.eligible_retained_epochs.includes(row.epoch)), 'Vertical parent is not sampling eligible');
  assert.deepEqual(retained.audits.map((entry) => entry.sha256).sort(), protocol.preflight_audits.map((entry) => entry.sha256).sort(), 'Vertical preflight audit links changed');
  const independent = retained.audits.map(evidence).filter((audit) => audit.all63_stage_point_visibility_checks !== undefined);
  assert.equal(independent.length, 1, 'Missing unique independent vertical audit');
  const audit = independent[0];
  assert.equal(audit.status, 'passed');
  assert.deepEqual(audit.defects, []);
  assert.equal(audit.post_launch_addendum, undefined, 'Postlaunch vertical addendum cannot replace preflight');
  assert.equal(audit.plan_sha256, protocol.reviewed_plan_sha256, 'Vertical reviewed plan changed');
  assert.match(audit.plan_sha256, /^[a-f0-9]{64}$/);
  assert.equal(audit.training_manifest_sha256, protocol.training_manifest_sha256);
  assert.equal(audit.point_metadata_sha256, protocol.point_metadata_sha256);
  assert.deepEqual(audit.source_sha256, config.source_sha256, 'Vertical audit sources differ');
  const script = retained.sources?.find((entry) => basename(entry.file) === 'audit_vertical_stretch_independent.py' && entry.sha256 === audit.script_sha256);
  assert.ok(script, 'Missing exact original vertical audit script');
  verifiedBytes(script);
  assert.equal(audit.all63_stage_point_visibility_checks, 252);
  for (const key of ['plan_source_flags_and_range_verified', 'stream_identity_and_prefixes_passed', 'global_numpy_rng_unchanged',
    'run20_off_pixels_targets_rng_byte_identical', 'on_off_main_rng_calls_and_final_state_equal',
    'actual_augment_cut_then_rotation_return_never_recovers_complete_or_occupied_point', 'actual_patches_factor_order_matches_independent_stream',
    'factor_does_nothing_when_existing_affine_is_skipped', 'evaluation_mode_ignores_flag_and_matches_run20',
    'repeated_batch_order_independent', 'full_balanced_order_identical', 'cached_point_states_unchanged']) assert.equal(audit[key], true, `Vertical independent audit failed: ${key}`);
  assert.ok(audit.real_batches?.length > 0 && audit.target_checks?.geometry_only > 0 && audit.target_checks?.unknown_boxes > 0);
  assert.deepEqual(audit.pillow_marker_checks?.map((row) => row.factor), [.9, 1, 1.1]);
  assert.ok(audit.pillow_marker_checks.every((row) => Number.isFinite(row.max_centroid_error_pixels) && row.max_centroid_error_pixels >= 0 && row.max_centroid_error_pixels < .04), 'Vertical Pillow point parity failed');
};
assert.ok(Array.isArray(provenance.trainingRuns) && provenance.trainingRuns.length > 0, 'Missing retained training runs');
let eligibleSelectedCheckpoint = false;
const usedSupplementalHashes = new Set();
for (const run of provenance.trainingRuns) {
  const config = evidence(run.configuration);
  verifySourceSet(run.sources, config.source_sha256 ?? {}, run.unavailable_original_sources ?? []);
  const indexes = config.supplemental_source_indexes ?? [];
  assert.ok(Array.isArray(indexes), 'Invalid DSBI training source index list');
  if (indexes.length) {
    assert.equal(new Set(indexes).size, indexes.length, 'Duplicate DSBI training source index');
    const trainingProtocol = evidence(run['dsbi-addition-protocol.json']);
    assert.deepEqual(trainingProtocol.supplemental_source_indexes, indexes, 'DSBI training protocol uses different indexes');
    assert.deepEqual(trainingProtocol.source_sha256, config.source_sha256, 'DSBI training protocol uses different source snapshots');
    assert.equal(trainingProtocol.training_manifest_sha256, config.data_sha256?.[config.train_manifest], 'DSBI training protocol uses different combined data');
    assert.equal(trainingProtocol.base_manifest_sha256, config.data_sha256?.[config.base_train_manifest], 'DSBI training protocol uses different base data');
    assert.ok(Date.parse(trainingProtocol.created_at_utc) < Date.parse(config.started_at_utc), 'DSBI protocol must precede training');
    const baseRecords = archivedData(config.data_sha256?.[config.base_train_manifest], 'preserved base manifest');
    const combinedRecords = archivedData(config.data_sha256?.[config.train_manifest], 'combined manifest');
    assert.ok(Array.isArray(baseRecords) && Array.isArray(combinedRecords), 'DSBI training manifests must contain record arrays');
    const expectedRecords = [...baseRecords];
    for (const path of indexes) {
      const digest = config.data_sha256?.[path], retained = supplementalByHash.get(digest);
      assert.ok(retained, 'DSBI training index has no matching retained preparation');
      assert.equal(trainingProtocol.supplemental_source_index_sha256, digest, 'DSBI training protocol index hash changed');
      assert.equal(trainingProtocol.supplemental_manifest_sha256, retained.preparation.generatedManifestSha256, 'DSBI training protocol records changed');
      assert.equal(trainingProtocol.independent_context_audit_sha256, retained.preparation['independent-context-audit.json'].sha256, 'DSBI training protocol uses a different independent audit');
      expectedRecords.push(...retained.records);
      usedSupplementalHashes.add(digest);
    }
    assert.deepEqual(combinedRecords, expectedRecords, 'DSBI combined manifest must preserve every base and supplemental record in order');
  }
  verifyDSBIPointMetadata(run, config, indexes);
  verifyGeometryInitialization(run, config, indexes);
  verifyHorizontalMirrorPreflight(run, config);
  verifyVerticalStretchPreflight(run, config);
  if (!run['sampling-lineage-final.json']) continue;
  const lineage = evidence(run['sampling-lineage-final.json']);
  assert.equal(lineage.status, 'passed', 'Sampling lineage audit did not pass');
  assert.equal(lineage.actual.config_sha256, run.configuration.sha256, 'Sampling lineage uses a different training configuration');
  assert.equal(lineage.actual.sampling_history_sha256, run['sampling-history.json']?.sha256, 'Sampling lineage uses a different history');
  assert.ok(Object.values(config.data_sha256 ?? {}).includes(lineage.actual.manifest_sha256), 'Sampling lineage uses different training data');
  assert.ok(retainedDataHashes.has(lineage.actual.manifest_sha256), 'Sampling lineage training data archive is missing');
  for (const row of lineage.actual.observed_epochs) {
    if (row.checkpoint_sha256 !== validation.oracle.checkpoint_sha256) continue;
    assert.ok(Number.isSafeInteger(row.epoch) && row.epoch > 0);
    assert.equal(row.eligible_by_sampling_lineage, true, 'Selected checkpoint is not eligible by sampling lineage');
    assert.ok(lineage.eligible_retained_epochs.includes(row.epoch), 'Selected checkpoint epoch is absent from eligible retained epochs');
    eligibleSelectedCheckpoint = true;
  }
}
assert.equal(usedSupplementalHashes.size, supplements.length, 'Retained DSBI preparation is not used by any training run');
assert.ok(eligibleSelectedCheckpoint, 'Selected checkpoint has no retained sampling eligibility proof');
const review = evidence(manifest.evidence?.boundaryReview);
assert.match(benchmark.fixture_manifest_sha256, /^[a-f0-9]{64}$/);
assert.equal(review.fixture_manifest_sha256, benchmark.fixture_manifest_sha256);
assert.equal(review.reviewed_count, total);
assert.equal(review.automatic_geometry_passed_count, total);
assert.deepEqual(review.automatic_geometry_failures, []);
assert.ok(review.visual_reviewed_count >= 48, 'Visual boundary review must cover a stratified sample');
assert.equal(new Set(review.visual_reviewed_ids).size, review.visual_reviewed_count);
assert.ok(review.visual_reviewed_ids.every((id) => Number.isInteger(id) && id >= 0 && id < total));
assert.equal(review.passed, true, 'Whole-word boundaries must be reviewed before promotion');
const pixels = evidence(manifest.evidence?.nativePreprocessing);
assert.ok(pixels.source_code_sha256 && typeof pixels.source_code_sha256 === 'object');
for (const file of ['src/ml/word-image-processing.ts', 'src/ml/word-model-contract.ts']) {
  assert.match(pixels.source_code_sha256[file], /^[a-f0-9]{64}$/, `Missing tested source hash: ${file}`);
}
for (const [file, expectedHash] of Object.entries(pixels.source_code_sha256)) {
  assert.equal(hash(readFileSync(inside(root, file))), expectedHash, `Native preprocessing evidence is stale: ${file}`);
}
assert.equal(pixels.rotation_mirroring_geometry_passed, true);
assert.equal(pixels.typescript_antialias_pillow_pixel_parity_passed, true);
assert.equal(pixels.cases.length, 7);
assert.equal(new Set(pixels.cases.map((sample) => sample.id)).size, 7);
for (const sample of pixels.cases) {
  assert.equal(sample.rotation_and_mirroring.max_channel_error_0_255, 0);
  assert.equal(sample.crop_matches_typescript, true);
  assert.equal(sample.selection_max_error, 0);
  assert.equal(sample.native_crop_pixel_error.max_channel_error_0_255, 0);
  assert.equal(sample.typescript_antialias_vs_pillow.max_channel_error_0_255, 0);
  assert.equal(sample.typescript_float32_vs_pillow_max_error, 0);
  assert.equal(sample.raw_as_rgba_vs_get_pixels.max_channel_error_0_255, 0);
}
const nativeRuntime = evidence(manifest.evidence?.nativeRuntime);
assert.equal(nativeRuntime.hermes, true);
assert.equal(hash(readFileSync(inside(root, nativeRuntime.harness_source))), nativeRuntime.harness_entry_sha256, 'Tested Hermes harness entry changed');
const testedResizer = readFileSync(inside(root, nativeRuntime.resizer_source_archive));
assert.equal(hash(testedResizer), nativeRuntime.resizer_source_sha256, 'Tested Hermes source archive changed');
assert.equal(hash(readFileSync(resolve(root, 'src/ml/word-model-contract.ts'))), nativeRuntime.resizer_contract_sha256, 'Tested Hermes tensor constants changed');
const resizerFunctions = (source) => {
  const syntax = ts.createSourceFile('resizer.ts', source, ts.ScriptTarget.Latest, true);
  return ['convertWordRawPixels', 'bilinearWeights', 'resizeWordRawPixels'].map((name) => {
    const declaration = syntax.statements.find((statement) => ts.isFunctionDeclaration(statement) && statement.name?.text === name);
    assert.ok(declaration?.body, `Missing tested resizer function: ${name}`);
    return declaration.getText(syntax).replace(/\r\n/g, '\n');
  });
};
assert.deepEqual(resizerFunctions(readFileSync(resolve(root, 'src/ml/word-image-processing.ts'), 'utf8')),
  resizerFunctions(testedResizer.toString('utf8')), 'Hermes resampler implementation changed after its test');
assert.ok(nativeRuntime.results.length >= 4);
for (const sample of nativeRuntime.results) {
  assert.equal(sample.runtimeKind, 3, 'Preprocessing must run on a background Worklets runtime');
  assert.equal(sample.tensorLength, 480 * 192 * 3);
  assert.equal(sample.mismatchByteCount, 0);
  assert.ok(Number.isFinite(sample.maxAbsError) && sample.maxAbsError >= 0 && sample.maxAbsError <= 6e-8);
  assert.ok(Number.isInteger(sample.jsHeartbeatCount) && sample.jsHeartbeatCount > 0, 'The main JS thread must remain responsive');
}
const decoderParity = evidence(manifest.evidence?.decoderParity);
assert.equal(decoderParity.schema_version, 1);
assert.equal(decoderParity.model_sha256, manifest.artifact.sha256, 'Decoder parity uses a different model');
assert.equal(decoderParity.benchmark_sha256, manifest.evidence.publicBenchmark.sha256, 'Decoder parity uses a different benchmark');
assert.equal(decoderParity.fixture_manifest_sha256, benchmark.fixture_manifest_sha256, 'Decoder parity uses different fixtures');
assert.deepEqual(decoderParity.thresholds, t, 'Decoder parity uses different thresholds');
assert.equal(decoderParity.parity_passed, true);
assert.deepEqual(decoderParity.first_mismatches, [], 'Decoder parity must have zero mismatches');
for (const key of ['total_pipeline_cases', 'actual_model_output_decoder_cases', 'quality_bypass_cases', 'zero_placeholder_cases', 'quality_matches', 'actual_decoder_text_matches', 'actual_decoder_rejection_matches', 'pipeline_text_matches', 'pipeline_rejection_matches']) {
  assert.ok(Number.isSafeInteger(decoderParity[key]) && decoderParity[key] >= 0, `Invalid decoder parity count: ${key}`);
}
assert.equal(decoderParity.total_pipeline_cases, total, 'Decoder parity must cover every benchmark word');
assert.equal(decoderParity.quality_matches, total);
assert.equal(decoderParity.pipeline_text_matches, total);
assert.equal(decoderParity.pipeline_rejection_matches, total);
assert.equal(decoderParity.actual_decoder_text_matches, decoderParity.actual_model_output_decoder_cases);
assert.equal(decoderParity.actual_decoder_rejection_matches, decoderParity.actual_model_output_decoder_cases);
assert.ok(decoderParity.actual_model_output_decoder_cases > 0);
assert.equal(decoderParity.actual_model_output_decoder_cases + decoderParity.zero_placeholder_cases, total);
assert.ok(decoderParity.zero_placeholder_cases <= decoderParity.quality_bypass_cases && decoderParity.quality_bypass_cases <= total);
assert.ok(Array.isArray(decoderParity.rows) && decoderParity.rows.length === total, 'Decoder parity row coverage is incomplete');
assert.equal(hash(JSON.stringify(decoderParity.rows)), decoderParity.outcomes_sha256, 'Decoder parity outcomes changed');
let qualityBypasses = 0, zeroPlaceholders = 0;
for (const [index, row] of decoderParity.rows.entries()) {
  const expected = benchmark.words[index];
  assert.equal(row.index, index);
  assert.equal(row.image, expected.image);
  assert.equal(row.expected, expected.expected);
  assert.equal(row.python_text, expected.predicted, 'Decoder parity reference differs from the benchmark');
  assert.equal(row.python_reason, expected.rejection, 'Decoder parity rejection differs from the benchmark');
  assert.equal(row.typescript_text, row.python_text, 'TypeScript/Python decoded text differs');
  assert.equal(row.typescript_reason, row.python_reason, 'TypeScript/Python rejection differs');
  for (const key of ['quality_match', 'pipeline_text_match', 'pipeline_reason_match']) assert.equal(row[key], true, `Decoder parity mismatch: ${key}`);
  assert.equal(typeof row.quality_bypassed_decoder, 'boolean');
  assert.equal(typeof row.saved_output_is_zero_placeholder, 'boolean');
  if (row.quality_bypassed_decoder) {
    qualityBypasses += 1;
    assert.equal(row.python_text, '');
    assert.ok(['dark', 'glare', 'low_confidence'].includes(row.python_reason));
  }
  if (row.saved_output_is_zero_placeholder) {
    zeroPlaceholders += 1;
    assert.equal(row.quality_bypassed_decoder, true, 'Only precheck-rejected rows may contain output placeholders');
  } else {
    assert.equal(row.decoder_text_match, true, 'Real-output decoder text parity failed');
    assert.equal(row.decoder_reason_match, true, 'Real-output decoder rejection parity failed');
  }
}
assert.equal(decoderParity.quality_bypass_cases, qualityBypasses);
assert.equal(decoderParity.zero_placeholder_cases, zeroPlaceholders);
for (const file of ['src/ml/word-decoder.ts', 'src/ml/word-model-contract.ts', 'src/ml/word-image-processing.ts', 'src/types/word-scan.ts']) {
  assert.match(decoderParity.source_sha256?.[file], /^[a-f0-9]{64}$/, `Missing decoder parity source hash: ${file}`);
  assert.equal(hash(readFileSync(resolve(root, file))), decoderParity.source_sha256[file], `Decoder parity source is stale: ${file}`);
}
// These diagnostics stay separate from the whole-word accuracy denominator.
for (const [kind, count, mustReject] of [['repetition', 12, false], ['rejection', 14, true]]) {
  const fixturesKey = `${kind}Fixtures`;
  const diagnostic = evidence(manifest.evidence?.[`${kind}Diagnostics`]);
  const fixtures = evidence(manifest.evidence?.[fixturesKey]);
  assert.equal(diagnostic.model_sha256, manifest.artifact.sha256, `${kind} diagnostics use a different model`);
  assert.equal(diagnostic.fixture_manifest_sha256, manifest.evidence[fixturesKey].sha256, `${kind} diagnostic fixtures changed`);
  assert.deepEqual(diagnostic.thresholds, t, `${kind} diagnostics use different thresholds`);
  assert.equal(diagnostic.scope, fixtures.scope);
  assert.equal(fixtures.model_predictions_used, false);
  assert.equal(diagnostic.used_for_external_checkpoint_or_threshold_selection, false);
  assert.equal(diagnostic.is_natural_whole_word_benchmark, false);
  assert.equal(diagnostic.original_rgb_quality_checked_before_inference, true);
  assert.equal(diagnostic.all_frozen_image_hashes_verified, true);
  for (const name of ['word_diagnostics.py', 'word_benchmark.py', 'word_decode.py', 'word_data.py']) {
    const file = `training/scripts/${name}`;
    assert.match(diagnostic.source_sha256?.[file], /^[a-f0-9]{64}$/, `Missing ${kind} diagnostic source hash: ${file}`);
    assert.equal(hash(readFileSync(resolve(root, file))), diagnostic.source_sha256[file],
      `${kind} diagnostic source is stale: ${file}`);
  }
  assert.equal(fixtures.records.length, count, `${kind} fixture count changed`);
  assert.equal(diagnostic.rows.length, count, `${kind} diagnostic coverage is incomplete`);
  assert.equal(new Set(fixtures.records.map((row) => row.image)).size, count);
  const categories = {};
  let diagnosticCorrect = 0, diagnosticAccepted = 0;
  for (const [index, row] of diagnostic.rows.entries()) {
    const expected = fixtures.records[index];
    for (const key of ['image', 'image_sha256', 'expected']) {
      assert.equal(row[key], expected[key], `${kind} diagnostic row differs from the frozen fixture: ${index}/${key}`);
    }
    assert.match(row.image_sha256, /^[a-f0-9]{64}$/);
    const fixtureDirectory = dirname(inside(root, manifest.evidence[fixturesKey].file));
    assert.equal(hash(readFileSync(inside(fixtureDirectory, row.image))), row.image_sha256,
      `${kind} diagnostic fixture pixels changed: ${row.image}`);
    assert.equal(row.category, expected.category ?? 'repetition');
    assert.equal(row.expected_reject, Boolean(expected.expected_reject));
    assert.equal(row.expected_reject, mustReject);
    assert.equal(typeof row.predicted, 'string');
    assert.match(row.predicted, /^(?:[A-Z]{1,12})?$/);
    assert.equal(typeof row.rejection, 'string');
    assert.equal(Boolean(row.predicted), !Boolean(row.rejection), `${kind} diagnostic must have a result or rejection`);
    assert.ok(Number.isSafeInteger(row.detected_cell_count) && row.detected_cell_count >= 0);
    const rowCorrect = mustReject ? !row.predicted && Boolean(row.rejection) : row.predicted === row.expected;
    assert.equal(row.correct, rowCorrect, `${kind} diagnostic correctness is misreported`);
    if (mustReject) assert.equal(row.predicted, '', 'Every rejection diagnostic must reject the entire capture');
    if (row.predicted) assert.equal(row.detected_cell_count, row.predicted.length,
      `${kind} accepted diagnostic cell count differs from its text`);
    diagnosticCorrect += Number(rowCorrect);
    diagnosticAccepted += Number(Boolean(row.predicted));
    const category = categories[row.category] ??= { total: 0, correct: 0, accepted: 0 };
    category.total += 1;
    category.correct += Number(rowCorrect);
    category.accepted += Number(Boolean(row.predicted));
  }
  assert.equal(diagnostic.total, count);
  assert.equal(diagnostic.correct, diagnosticCorrect);
  assert.equal(diagnostic.accepted, diagnosticAccepted);
  assert.deepEqual(diagnostic.by_category, categories, `${kind} diagnostic category totals are incorrect`);
  if (mustReject) {
    assert.deepEqual(Object.fromEntries(Object.entries(categories).map(([key, value]) => [key, value.total])),
      { unsupported_pattern: 3, multiple_rows: 3, selection_cut: 3, over_limit: 3, blank: 2 });
  } else {
    assert.deepEqual(fixtures.records.map((row) => row.expected).sort(),
      ['A', 'MAMA', 'KAKAK', 'MAMAMAMAMAMA'].flatMap((word) => [word, word, word]).sort());
  }
}
const fixture = manifest.nativeSelfTest;
assert.match(fixture.expectedText, /^[A-Z]{1,12}$/);
const png = readFileSync(inside(directory, fixture.file));
assert.equal(hash(png), fixture.sha256);
assert.equal(png.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
for (const key of ['x', 'y', 'width', 'height']) assert.ok(Number.isFinite(fixture.roi[key]));
assert.ok(fixture.roi.x >= 0 && fixture.roi.y >= 0 && fixture.roi.width > 0 && fixture.roi.height > 0 && fixture.roi.x + fixture.roi.width <= 1 && fixture.roi.y + fixture.roi.height <= 1);
const assetModule = readFileSync(resolve(root, 'src/ml/word-model-assets.ts'), 'utf8');
const assetSyntax = ts.createSourceFile('word-model-assets.ts', assetModule, ts.ScriptTarget.Latest, true);
const assertLiteralAsset = (name, file) => {
  const declaration = assetSyntax.statements.filter((statement) =>
    ts.isVariableStatement(statement) && statement.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword))
    .flatMap((statement) => statement.declarationList.declarations)
    .find((item) => ts.isIdentifier(item.name) && item.name.text === name);
  const value = declaration?.initializer;
  assert.ok(value && ts.isCallExpression(value) && ts.isIdentifier(value.expression) && value.expression.text === 'require' &&
    value.arguments.length === 1 && ts.isStringLiteral(value.arguments[0]) &&
    value.arguments[0].text === `../../assets/models/braille-word-reader-v1/${file}`, `${name} must export the actual literal Metro asset`);
};
assertLiteralAsset('WORD_MODEL_ASSET', manifest.artifact.file);
assertLiteralAsset('WORD_NATIVE_FIXTURE_ASSET', fixture.file);
console.log(`Word assets valid: ${total} complete blocks, ${(100*correct/total).toFixed(2)}% exact, ${(100*correct/accepted).toFixed(2)}% accepted accuracy, 0/44 negative accepts, SHA-256 ${manifest.artifact.sha256}`);
