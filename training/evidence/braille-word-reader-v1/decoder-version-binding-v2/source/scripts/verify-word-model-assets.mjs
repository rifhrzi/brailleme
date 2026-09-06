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
const decoderContractSyntax = ts.createSourceFile('word-model-contract.ts',
  readFileSync(resolve(root, 'src/ml/word-model-contract.ts'), 'utf8'), ts.ScriptTarget.Latest, true);
const decoderVersionDeclaration = decoderContractSyntax.statements.filter((statement) =>
  ts.isVariableStatement(statement) && statement.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword))
  .flatMap((statement) => statement.declarationList.declarations)
  .find((declaration) => ts.isIdentifier(declaration.name) && declaration.name.text === 'WORD_DECODER_VERSION');
assert.ok(decoderVersionDeclaration?.initializer && ts.isStringLiteral(decoderVersionDeclaration.initializer),
  'Current decoder version must be an exported literal');
const decoderVersion = decoderVersionDeclaration.initializer.text;
assert.equal(manifest.decoderVersion, decoderVersion, 'Manifest decoder version differs from the app');
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
assert.equal(conversion.decoder_version, decoderVersion, 'Conversion decoder version differs from the app');
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
assert.equal(validation.decoder_version, decoderVersion, 'Validation decoder version differs from the app');
assert.equal(validation.oracle?.decoder_version, decoderVersion, 'Validation oracle decoder version differs from the app');
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
assert.equal(selectionReport.decoder_version, decoderVersion, 'Selection decoder version differs from the app');
for (const candidate of selectionReport.candidates) {
  assert.equal(candidate.decoder_version, decoderVersion, 'Candidate decoder version differs from the app');
}
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
const verifyHorizontalAspectContinuation = (run, config) => {
  const retained = run.horizontalAspectContinuation;
  assert.equal(run.horizontalAspectPreflight, undefined, 'Continuation cannot relabel the initial horizontal aspect audit');
  const protocol = evidence(retained.protocol), guard = evidence(retained.guardAudit), artifact = evidence(retained.parentArtifactAudit);
  const inherited = protocol.continuation, parent = retained.parentCheckpointSha256;
  assert.equal(inherited?.kind, 'weight_only_same_sample_realizations', 'Invalid horizontal aspect continuation kind');
  assert.equal(inherited.parent_retained_epoch, 10);
  assert.equal(inherited.parent_is_selected_control, false);
  assert.equal(config.resume_sha256, parent, 'Continuation resumed a different parent');
  assert.equal(protocol.parent_checkpoint_sha256, parent);
  assert.equal(config.resume, protocol.parent_checkpoint);
  assert.equal(retained.inheritedRunConfigurationSha256, inherited.parent_training_config.sha256);
  assert.equal(retained.inheritedHorizontalAspectProtocolSha256, inherited.inherited_horizontal_aspect_protocol.sha256);
  const ancestor = provenance.trainingRuns.find((row) => row !== run && row.configuration.sha256 === retained.inheritedRunConfigurationSha256);
  assert.ok(ancestor?.horizontalAspectPreflight && !ancestor.horizontalAspectContinuation, 'Continuation needs its original retained horizontal aspect run');
  assert.equal(ancestor.horizontalAspectPreflight.protocol.sha256, retained.inheritedHorizontalAspectProtocolSha256);
  const previous = evidence(ancestor.configuration), originalProtocol = evidence(ancestor.horizontalAspectPreflight.protocol);
  const withoutResume = (value) => Object.fromEntries(Object.entries(value).filter(([key]) => !['resume', 'resume_sha256', 'started_at_utc'].includes(key)));
  assert.deepEqual(withoutResume(config), withoutResume(previous), 'Continuation changed inherited configuration');
  assert.ok(Date.parse(previous.started_at_utc) < Date.parse(config.started_at_utc));
  verifySourceSet(run.sources, config.source_sha256);
  assert.equal(protocol.parent_training_config_sha256, ancestor.configuration.sha256);
  assert.equal(protocol.parent_sampling_lineage_sha256, ancestor['sampling-lineage-final.json']?.sha256);
  assert.equal(inherited.parent_sampling_lineage.sha256, ancestor['sampling-lineage-final.json']?.sha256);
  assert.equal(inherited.parent_sampling_history.sha256, ancestor['sampling-history.json']?.sha256);
  assert.equal(inherited.original_initial_checkpoint_sha256, previous.resume_sha256, 'Continuation relabeled historical initial parent');
  assert.notEqual(parent, previous.resume_sha256);
  assert.equal(protocol.control_checkpoint_sha256, previous.resume_sha256);
  assert.ok(ancestor.horizontalAspectPreflight.audits.some((row) => row.sha256 === inherited.inherited_transform_audit.sha256));
  assert.ok(ancestor.horizontalAspectPreflight.audits.some((row) => row.sha256 === inherited.inherited_source_bound_tests.sha256));
  assert.deepEqual(protocol.source_sha256, config.source_sha256);
  assert.equal(protocol.training_manifest_sha256, config.data_sha256[config.train_manifest]);
  assert.equal(protocol.point_metadata_sha256, run.dsbiPointMetadata?.sha256);
  assert.deepEqual(protocol.horizontal_aspect, originalProtocol.horizontal_aspect, 'Continuation changed horizontal aspect transform');
  assert.deepEqual(protocol.supplemental_source_indexes, config.supplemental_source_indexes);
  for (const key of ['learning_rate', 'batch_size', 'seed', 'negative_repeat', 'supplemental_negative_repeat', 'geometry_loss',
    'illumination_augmentation', 'horizontal_aspect_augmentation', 'horizontal_mirror_augmentation', 'vertical_stretch_augmentation',
    'camera_augmentation', 'half_turn_augmentation', 'polarity_augmentation']) assert.equal(protocol[key], config[key], `Continuation protocol differs: ${key}`);
  for (const key of ['dsbi_visible_partial_supervision', 'partial_geometry_supervision', 'geometry_conditioned_residual', 'batchnorm_affine', 'fine_detail_branch']) assert.equal(protocol[key], config[key]?.enabled);
  assert.equal(protocol.planned_epochs, config.epochs);
  assert.equal(config.epochs, 10);
  assert.deepEqual(protocol.planned_eligible_epochs, [8, 9, 10]);
  assert.equal(protocol.no_activation_before_original_gates, true);
  const lineage = evidence(ancestor['sampling-lineage-final.json']), history = evidence(ancestor['sampling-history.json']);
  assert.equal(lineage.status, 'passed');
  assert.equal(lineage.actual.config_sha256, ancestor.configuration.sha256);
  assert.equal(lineage.actual.sampling_history_sha256, ancestor['sampling-history.json'].sha256);
  const epoch = lineage.actual.observed_epochs.find((row) => row.epoch === 10 && row.checkpoint_sha256 === parent);
  assert.ok(epoch?.eligible_by_sampling_lineage && epoch.all_positive_records_seen_in_this_run && lineage.eligible_retained_epochs.includes(10), 'Continuation parent epoch10 is not eligible');
  assert.equal(epoch.completed_batches, 1037);
  for (const doc of [guard, artifact]) {
    assert.equal(doc.status, 'passed', 'Continuation independent audit failed');
    assert.equal(doc.parent_checkpoint_sha256, parent);
    assert.deepEqual(doc.source_sha256, config.source_sha256, 'Continuation audit source mismatch');
    assert.deepEqual(doc.backend_modules_imported, []);
  }
  for (const time of [protocol.created_at_utc, guard.created_at_utc, artifact.completed_at_utc]) assert.ok(Date.parse(time) < Date.parse(config.started_at_utc), 'Continuation preflight must precede training');
  assert.deepEqual(protocol.preflight_audits.map((row) => row.sha256).sort(), [retained.guardAudit.sha256, retained.parentArtifactAudit.sha256].sort());
  assert.equal(guard.plan_sha256, protocol.reviewed_plan_sha256);
  assert.equal(guard.parent_training_config_sha256, ancestor.configuration.sha256);
  assert.equal(guard.parent_sampling_lineage_sha256, ancestor['sampling-lineage-final.json'].sha256);
  assert.equal(guard.inherited_horizontal_aspect_protocol_sha256, retained.inheritedHorizontalAspectProtocolSha256);
  assert.equal(guard.inherited_transform_audit_sha256, inherited.inherited_transform_audit.sha256);
  assert.deepEqual(guard.data_sha256, config.data_sha256);
  for (const key of ['historical_transform_audit_original_parent_preserved', 'sources_equal_run26_current_and_snapshots', 'data_hashes_equal_run26',
    'profile_equal_run26', 'new_parent_epoch10_sampling_eligible', 'actual_source_mock_load_compile_false_fresh_adam', 'fresh_scheduler_same_policy',
    'no_initial_epoch_or_optimizer_restore', 'fresh_dataset_starts_epoch1', 'expected_pre_fit_reset_to_epoch2_matches_parent_history',
    'sample_width_factors_replay_exact', 'legacy_pixel_rng_expression_unchanged', 'parent_coverage_does_not_qualify_early_local_epochs']) assert.equal(guard[key], true, `Continuation guard failed: ${key}`);
  assert.deepEqual(guard.own_cycle_required_local_eligible_epochs, [8, 9, 10]);
  assert.deepEqual(guard.command_changed_arguments_only, ['--output', '--resume']);
  assert.deepEqual(guard.negative_expansion, config.negative_expansion);
  assert.deepEqual(guard.negative_guard_cases, ['parent_sha', 'learning_rate', 'partial_geometry_metadata', 'point_metadata_hash',
    'early_eligibility', 'optimizer_description', 'historical_parent_relabel', 'parent_launch_hash']);
  assert.equal(guard.new_training_or_model_execution, false);
  assert.equal(guard.expected_ten_epoch_order_replay.length, 10);
  assert.equal(history.length, 10);
  guard.expected_ten_epoch_order_replay.forEach((row, index) => {
    assert.equal(row.local_epoch, index + 1);
    assert.equal(row.expected_dataset_epoch, index + 2);
    assert.equal(row.expected_dataset_epoch, history[index].dataset_epoch);
    assert.equal(row.order_int64_sha256, history[index].order_int64_sha256, 'Continuation sample order differs');
    assert.equal(row.sample_width_factor_sha256.length, 3);
    for (const digest of row.sample_width_factor_sha256) assert.match(digest, /^[a-f0-9]{64}$/);
  });
  const retainedByHash = (digest) => {
    assert.match(digest ?? '', /^[a-f0-9]{64}$/);
    const entry = [...(retained.sources ?? []), ...run.sources].find((row) => row.sha256 === digest);
    assert.ok(entry, 'Missing exact continuation audit source or command');
    verifiedBytes(entry);
    return entry;
  };
  for (const digest of [guard.script_sha256, guard.launcher_sha256, guard.preparation_script_sha256, artifact.script_sha256]) retainedByHash(digest);
  const parentCommand = evidence(retainedByHash(guard.parent_launch_sha256)).command;
  const changed = parentCommand.flatMap((value, index) => value === guard.command[index] ? [] : [index]);
  assert.equal(guard.command.length, parentCommand.length);
  assert.deepEqual(changed, [parentCommand.indexOf('--output') + 1, parentCommand.indexOf('--resume') + 1].sort((a, b) => a - b));
  assert.equal(guard.command[guard.command.indexOf('--resume') + 1], protocol.parent_checkpoint);
  assert.equal(guard.command[guard.command.indexOf('--output') + 1], protocol.run);
  assert.equal(artifact.control_checkpoint_sha256, previous.resume_sha256);
  assert.equal(artifact.bindings.config.sha256, ancestor.configuration.sha256);
  assert.equal(artifact.bindings.lineage.sha256, ancestor['sampling-lineage-final.json'].sha256);
  assert.equal(artifact.bindings.sampling.sha256, ancestor['sampling-history.json'].sha256);
  assert.equal(artifact.identical_complete_serialized_model_config, true, 'Continuation parent graph changed');
  assert.equal(artifact.control_config_bytes_sha256, artifact.parent_config_bytes_sha256);
  assert.equal(artifact.all_model_array_names_shapes_dtypes_identical, true);
  assert.equal(artifact.model_arrays, 280);
  assert.equal(artifact.model_parameters, 829307);
  assert.equal(artifact.frozen_batchnorm_arrays_byte_identical, 208, 'Continuation changed frozen BN');
  assert.equal(artifact.arrays.length, 280);
  assert.equal(new Set(artifact.arrays.map((row) => row.path)).size, 280);
  const bn = artifact.arrays.filter((row) => ['bn_moving', 'bn_gamma_beta'].includes(row.category));
  assert.equal(bn.length, 208);
  for (const row of bn) { assert.equal(row.identical_bytes, true, 'Continuation changed a BN array'); assert.equal(row.control_sha256, row.parent_sha256); }
  for (const kind of ['bn_moving', 'bn_gamma_beta']) assert.deepEqual(artifact.counts[kind], { arrays: 104, changed_arrays: 0 });
  assert.deepEqual(artifact.sampling_epoch10, epoch);
  assert.deepEqual(artifact.sampling_callback_epoch10, history[9]);
  if (run['sampling-lineage-final.json']) {
    const actual = evidence(run['sampling-history.json']);
    assert.equal(actual.length, 10, 'Continuation did not complete its declared training cycle');
    actual.forEach((row, index) => {
      assert.equal(row.dataset_epoch, guard.expected_ten_epoch_order_replay[index].expected_dataset_epoch);
      assert.equal(row.order_int64_sha256, guard.expected_ten_epoch_order_replay[index].order_int64_sha256, 'Actual continuation sample replay changed');
      assert.equal(row.completed_batches, 1037);
      assert.equal(row.complete_epoch, true);
    });
  }
};
const verifyHorizontalAspectPreflight = (run, config) => {
  assert.ok(config.horizontal_aspect_augmentation === undefined || typeof config.horizontal_aspect_augmentation === 'boolean', 'Invalid horizontal aspect flag');
  if (!config.horizontal_aspect_augmentation) {
    assert.equal(run.horizontalAspectPreflight, undefined, 'Horizontal aspect preflight requires enabled augmentation');
    assert.equal(run.horizontalAspectContinuation, undefined, 'Horizontal aspect continuation requires enabled augmentation');
    return;
  }
  if (run.horizontalAspectContinuation) return verifyHorizontalAspectContinuation(run, config);
  const retained = run.horizontalAspectPreflight;
  assert.ok(retained, 'Missing horizontal aspect preflight');
  verifySourceSet(run.sources, config.source_sha256);
  for (const name of ['word_data.py', 'word_train.py', 'word_horizontal_aspect.py', 'word_point_geometry.py', 'word_point_metadata.py', 'word_sampling.py']) {
    assert.match(config.source_sha256[name] ?? '', /^[a-f0-9]{64}$/, `Missing horizontal aspect source: ${name}`);
  }
  const protocol = evidence(retained.protocol), parent = retained.parentCheckpointSha256;
  assert.match(parent, /^[a-f0-9]{64}$/);
  assert.equal(config.resume_sha256, parent, 'Horizontal aspect resumed a different parent');
  assert.equal(protocol.parent_checkpoint_sha256, parent);
  if (run['horizontal-aspect-protocol.json']) assert.equal(run['horizontal-aspect-protocol.json'].sha256, retained.protocol.sha256, 'Horizontal aspect protocol links differ');
  assert.ok(Date.parse(protocol.created_at_utc) < Date.parse(config.started_at_utc), 'Horizontal aspect protocol must precede training');
  assert.deepEqual(protocol.source_sha256, config.source_sha256, 'Horizontal aspect protocol sources differ');
  assert.equal(protocol.training_manifest_sha256, config.data_sha256?.[config.train_manifest], 'Horizontal aspect training data changed');
  assert.equal(protocol.point_metadata_sha256, run.dsbiPointMetadata?.sha256, 'Horizontal aspect point metadata changed');
  assert.deepEqual(protocol.supplemental_source_indexes, config.supplemental_source_indexes);
  for (const [key, expected] of Object.entries({ horizontal_aspect_augmentation: true, illumination_augmentation: true,
    camera_augmentation: false, horizontal_mirror_augmentation: false, vertical_stretch_augmentation: false,
    half_turn_augmentation: false, polarity_augmentation: false })) {
    assert.equal(config[key], expected, `Horizontal aspect incompatible profile: ${key}`);
    assert.equal(protocol[key], expected, `Horizontal aspect protocol flag differs: ${key}`);
  }
  for (const [key, enabled] of [['dsbi_visible_partial_supervision', true], ['partial_geometry_supervision', false],
    ['geometry_conditioned_residual', false], ['batchnorm_affine', false], ['fine_detail_branch', false]]) {
    assert.equal(config[key]?.enabled, enabled, `Horizontal aspect incompatible branch: ${key}`);
    assert.equal(protocol[key], enabled);
  }
  for (const [key, expected] of Object.entries({ planned_epochs: config.epochs, batch_size: config.batch_size, learning_rate: config.learning_rate,
    seed: config.seed, geometry_loss: config.geometry_loss, negative_repeat: config.negative_expansion?.base_repeat,
    supplemental_negative_repeat: config.negative_expansion?.supplemental_repeat })) assert.equal(protocol[key], expected, `Horizontal aspect protocol differs: ${key}`);
  assert.equal(protocol.frozen_batchnorm, true);
  assert.equal(config.batch_normalization?.layers, 52);
  assert.equal(config.batch_normalization.trainable_affine_layers, 0);
  assert.deepEqual(config.batch_normalization.backbone_call_training_keywords, [false]);
  assert.equal(config.image_normalization, 'channel_gray_std05', 'Horizontal aspect changed inherited normalization');
  assert.deepEqual(config.imageNormalization, imageNormalization);
  assert.equal(protocol.no_activation_before_original_gates, true);
  assert.deepEqual(protocol.planned_eligible_epochs, [8, 9, 10]);
  const view = protocol.horizontal_aspect;
  assert.deepEqual(view?.range, [.55, 1.25], 'Horizontal aspect range changed');
  assert.equal(view.distribution, 'uniform');
  assert.equal(view.existing_affine_probability, .85, 'Horizontal aspect affine probability changed');
  assert.deepEqual(view.base_zoom, [.8, 1.7]);
  assert.equal(view.scale_x, 'zoom*width_factor');
  assert.equal(view.scale_y, 'zoom');
  assert.equal(view.stream_id, 0x48415350, 'Horizontal aspect random stream changed');
  assert.deepEqual(view.seed_sequence, ['seed', 'dataset_epoch', 'expanded_record_index', 0x48415350], 'Horizontal aspect must use per-record RNG');
  assert.deepEqual(config.augmentation_protocol?.horizontal_aspect_range, view.range);
  assert.equal(config.augmentation_protocol.horizontal_aspect_affine_probability, .85);
  assert.equal(config.augmentation_protocol.horizontal_aspect_stream_id, view.stream_id);
  const ancestor = provenance.trainingRuns.find((row) => row !== run && row.configuration.sha256 === protocol.parent_training_config_sha256 &&
    row['sampling-lineage-final.json']?.sha256 === protocol.parent_sampling_lineage_sha256);
  assert.ok(ancestor, 'Horizontal aspect parent has no matching retained ancestor');
  const lineage = evidence(ancestor['sampling-lineage-final.json']);
  assert.equal(lineage.status, 'passed');
  assert.equal(lineage.actual.config_sha256, ancestor.configuration.sha256);
  assert.equal(lineage.actual.sampling_history_sha256, ancestor['sampling-history.json']?.sha256);
  assert.ok(lineage.actual.observed_epochs.some((row) => row.checkpoint_sha256 === parent && row.eligible_by_sampling_lineage === true &&
    lineage.eligible_retained_epochs.includes(row.epoch)), 'Horizontal aspect parent is not sampling eligible');
  assert.deepEqual(retained.audits.map((entry) => entry.sha256).sort(), protocol.preflight_audits.map((entry) => entry.sha256).sort(), 'Horizontal aspect preflight audit links changed');
  const audits = retained.audits.map((entry) => ({ entry, doc: evidence(entry) }));
  const matching = audits.filter(({ doc }) => doc.patches_private_ast_exact !== undefined);
  assert.equal(matching.length, 1, 'Missing unique horizontal aspect integration audit');
  const audit = matching[0].doc;
  assert.equal(audit.status, 'passed');
  assert.ok(Date.parse(audit.created_at_utc) < Date.parse(config.started_at_utc), 'Horizontal aspect integration audit must precede training');
  assert.equal(audit.plan_sha256, protocol.reviewed_plan_sha256, 'Horizontal aspect reviewed plan changed');
  assert.match(audit.plan_sha256, /^[a-f0-9]{64}$/);
  assert.equal(audit.parent_checkpoint_sha256, parent);
  assert.deepEqual(audit.source_sha256, config.source_sha256, 'Horizontal aspect audit sources differ');
  const retainedByHash = (digest, label) => {
    assert.match(digest ?? '', /^[a-f0-9]{64}$/, `Invalid horizontal aspect dependency: ${label}`);
    const entry = [...(retained.sources ?? []), ...run.sources, run.preflightTestLog, run.integrationTestLog]
      .find((row) => row?.sha256 === digest);
    assert.ok(entry, `Missing exact horizontal aspect dependency: ${label}`);
    verifiedBytes(entry);
    return entry;
  };
  retainedByHash(audit.script_sha256, 'integration audit script');
  retainedByHash(audit.tests_log_sha256, 'integration test log');
  for (const key of ['patches_private_ast_exact', 'rng_private_ast_exact', 'raster_private_ast_exact_except_profile_guard',
    'all63_masks_actual_patches_off_equal', 'actual_single_affine_and_same_legacy_rng',
    'known_points_all_stage_survival_and_partial_unknown', 'unknown_not_positive', 'active_sources_unchanged']) {
    assert.equal(audit[key], true, `Horizontal aspect integration audit failed: ${key}`);
  }
  assert.equal(audit.actual_cpu_tests, 7);
  assert.equal(audit.cli_and_direct_api_rejections?.length, 14, 'Horizontal aspect profile guards not verified');
  assert.deepEqual(audit.backend_modules_imported, []);
  const suite = audits.find(({ entry }) => entry.sha256 === audit.full_word_suite?.summary_sha256)?.doc;
  assert.ok(suite, 'Missing horizontal aspect source-bound test suite');
  assert.equal(suite.status, 'passed');
  assert.equal(suite.exit_code, 0);
  assert.equal(suite.test_sources_unchanged_before_after, true);
  assert.ok(Date.parse(suite.completed_at_utc) < Date.parse(config.started_at_utc), 'Horizontal aspect test suite must precede training');
  assert.equal(audit.full_word_suite.tests, 141);
  assert.equal(audit.full_word_suite.log_sha256, suite.log_sha256);
  assert.equal(audit.full_word_suite.test_sources_verified, Object.keys(suite.tests_sha256).length);
  const suiteLog = verifiedBytes(retainedByHash(suite.log_sha256, 'full suite log')).toString('utf8');
  assert.match(suiteLog, /Ran 141 tests/);
  assert.match(suiteLog, /^OK\s*$/m);
  for (const [name, digest] of Object.entries(suite.tests_sha256)) retainedByHash(digest, name);
  assert.equal(audit.prototype_report_sha256, protocol.private_prototype?.proof_sha256, 'Horizontal aspect prototype proof changed');
  const prototype = evidence(retainedByHash(audit.prototype_report_sha256, 'private proof'));
  assert.equal(prototype.status, 'passed');
  for (const [name, digest] of Object.entries(prototype.source_sha256)) retainedByHash(digest, name);
  const independent = evidence(retainedByHash(audit.independent_prototype_review_sha256, 'independent private review'));
  assert.equal(independent.status, 'passed');
  assert.equal(independent.prototype_report.sha256, audit.prototype_report_sha256);
  assert.equal(independent.prototype_index.sha256, protocol.private_prototype.index_sha256);
  retainedByHash(independent.audit_source.sha256, 'independent private review script');
  const index = evidence(retainedByHash(protocol.private_prototype.index_sha256, 'private source index'));
  for (const [name, entry] of Object.entries(index.files)) retainedByHash(entry.sha256, name);
  assert.equal(index.files['prototype-spec.json']?.sha256, protocol.private_prototype.spec_sha256);
};
const verifyBatchnormAffineInitialization = (run, config) => {
  const branch = config.batchnorm_affine;
  assert.ok(branch === undefined || typeof branch.enabled === 'boolean', 'Invalid BN affine flag');
  if (!branch?.enabled) {
    assert.equal(run.batchnormAffineInitialization, undefined, 'BN affine provenance requires enabled training');
    return;
  }
  const retained = run.batchnormAffineInitialization;
  assert.ok(retained, 'Missing BN affine initialization provenance');
  verifySourceSet(run.sources, config.source_sha256);
  const docs = {};
  for (const key of ['initializationReport', 'nonzeroConversionProof', 'sixInputOracle', 'activationReport',
    'independentActivationReview', 'independentTrainingAudit', 'independentGuardAudit', 'paritySourceReview', 'protocol']) {
    assert.ok(retained[key], `Missing BN affine proof: ${key}`);
    docs[key] = evidence(retained[key]);
  }
  const { initializationReport: init, nonzeroConversionProof: proof, sixInputOracle: oracle,
    independentTrainingAudit: runtime, independentGuardAudit: guard, protocol } = docs;
  for (const key of ['initializationReport', 'nonzeroConversionProof', 'sixInputOracle', 'independentTrainingAudit', 'independentGuardAudit']) {
    assert.equal(docs[key].status, 'passed', `BN affine proof failed: ${key}`);
  }
  const base = retained.baseParentCheckpointSha256, initial = retained.initialParentCheckpointSha256,
    diagnostic = retained.forbiddenDiagnosticCheckpointSha256;
  for (const digest of [base, initial, diagnostic, branch.moving_moments_sha256]) assert.match(digest, /^[a-f0-9]{64}$/);
  assert.equal(new Set([base, initial, diagnostic]).size, 3, 'BN affine checkpoint identities must be distinct');
  assert.equal(config.resume_sha256, initial, 'BN affine must resume the exact initial parent');
  assert.equal(branch.base_parent_checkpoint_sha256, base);
  assert.equal(branch.initial_checkpoint_sha256, initial);
  assert.equal(branch.nonzero_diagnostic_sha256, diagnostic);
  assert.equal(branch.nonzero_diagnostic_forbidden, true);
  assert.ok(![initial, diagnostic].includes(validation.oracle.checkpoint_sha256), 'BN affine probes cannot be selected');
  for (const ancestor of provenance.trainingRuns) assert.notEqual(evidence(ancestor.configuration).resume_sha256, diagnostic, 'BN affine diagnostic cannot enter training ancestry');
  assert.equal(init.parent_checkpoint_sha256, base);
  assert.equal(init.initial_checkpoint_sha256, initial);
  assert.equal(init.nonzero_diagnostic_sha256, diagnostic);
  assert.equal(init.nonzero_diagnostic_must_never_be_resumed, true);
  assert.equal(branch.initialization_report_sha256, retained.initializationReport.sha256, 'BN affine initialization changed');
  assert.equal(branch.conversion_report_sha256, retained.nonzeroConversionProof.sha256, 'BN affine conversion changed');
  for (const report of [proof, oracle, guard]) assert.equal(report.initialization_report_sha256, retained.initializationReport.sha256, 'BN affine proof pair changed');
  assert.equal(proof.oracle_report_sha256, retained.sixInputOracle.sha256);
  assert.equal(proof.oracle_sha256, oracle.oracle_sha256);
  assert.match(oracle.oracle_sha256, /^[a-f0-9]{64}$/);
  assert.equal(proof.selected_validation_sha256, protocol.parent_selected_validation_sha256);
  assert.equal(oracle.selected_validation_sha256, proof.selected_validation_sha256);
  assert.deepEqual(oracle.thresholds, proof.thresholds);
  const requiredSource = (name, digest, training = false) => {
    assert.match(digest ?? '', /^[a-f0-9]{64}$/, `Missing BN affine source hash: ${name}`);
    const entry = retained.sources?.find((row) => basename(row.file) === name && row.sha256 === digest);
    assert.ok(entry, `Missing exact BN affine source: ${name}`);
    verifiedBytes(entry);
    if (training) assert.equal(config.source_sha256[name], digest, `BN affine training source differs: ${name}`);
  };
  for (const name of ['word_bn_affine.py', 'word_bn_affine_init.py', 'word_model.py', 'word_data.py']) requiredSource(name, init.source_sha256?.[name], true);
  requiredSource('word_bn_affine_parity.py', proof.source_sha256, true);
  assert.equal(oracle.source_sha256, proof.source_sha256);
  assert.deepEqual(oracle.decision_sources_sha256, proof.decision_sources_sha256);
  for (const [name, digest] of Object.entries(proof.decision_sources_sha256)) requiredSource(name, digest, name in config.source_sha256);
  for (const report of Object.values(docs)) if (report.script_sha256) {
    const source = retained.sources?.find((row) => row.sha256 === report.script_sha256);
    assert.ok(source, 'Missing original BN affine audit script'); verifiedBytes(source);
  }
  const spec = branch.spec;
  assert.deepEqual(spec, init.batchnorm_affine, 'BN affine layer spec changed');
  assert.equal(spec?.mode, 'affine_only_frozen_moments');
  for (const key of ['batchnorm_layers', 'trainable_affine_layers', 'layer_forced_frozen_moments']) assert.equal(spec[key], 52, 'Every BN must force frozen moments inside call');
  assert.equal(spec.affine_parameters, 18544);
  assert.equal(spec.layer_names?.length, 52); assert.equal(new Set(spec.layer_names).size, 52);
  assert.deepEqual(spec.backbone_call_training_keywords, [false]);
  assert.equal(config.image_normalization, 'channel_gray_std05', 'BN affine changed inherited image normalization');
  assert.deepEqual(config.imageNormalization, imageNormalization, 'BN affine normalization differs from selected model');
  assert.deepEqual(init.imageNormalization, config.imageNormalization, 'BN affine initialization normalization changed');
  assert.equal(config.batch_normalization?.layers, 52);
  assert.equal(config.batch_normalization.trainable_affine_layers, 52);
  assert.deepEqual(config.batch_normalization.backbone_call_training_keywords, [false]);
  assert.equal(branch.layer_enforcement, 'FrozenMomentBatchNormalization.call always passes training=False to BN; existing convolution/head trainability is preserved.');
  assert.equal(branch.epoch_guard, 'Hash all stored moving means/variances before saving any epoch checkpoint; mismatch aborts.');
  for (const key of ['all_initial_arrays_equal_parent', 'initial_train_eval_parent_outputs_equal', 'save_reload_values_and_outputs_equal',
    'all_arrays_unchanged_after_backward', 'parent_unchanged', 'diagnostic_other_weights_and_moments_unchanged']) assert.equal(init[key], true, `BN affine initialization check failed: ${key}`);
  assert.equal(init.array_count, 280); assert.equal(init.parameters, 829307); assert.equal(init.diagnostic_changed_affine_tensors, 104);
  assert.deepEqual(init.gradient_rows?.map((row) => row.layer), spec.layer_names);
  assert.ok(init.gradient_rows.every((row) => [row.gamma_gradient_l1, row.beta_gradient_l1].every((value) => Number.isFinite(value) && value > 0)), 'BN affine gradients must reach every gamma and beta');
  assert.equal(oracle.samples, 6);
  for (const key of ['initial_and_parent_exact', 'train_and_eval_exact', 'all_weights_unchanged']) assert.equal(oracle[key], true);
  assert.ok(Number.isFinite(oracle.nonzero_max_output_delta) && oracle.nonzero_max_output_delta > 1e-6, 'BN affine nonzero probe has no measured effect');
  assert.equal(proof.absolute_tolerance, 1e-4); assert.equal(proof.relative_tolerance, 1e-4);
  assert.equal(proof.conversion_probes_must_not_be_activated_or_resumed, true);
  assert.equal(proof.trained_candidate_requires_full405_parity, true, 'BN affine preflight cannot replace trained parity');
  for (const [name, checkpoint] of [['initial', initial], ['nonzero', diagnostic]]) {
    const probe = proof.probes?.[name];
    assert.equal(probe?.checkpoint_sha256, checkpoint); assert.equal(probe.samples, 6);
    assert.match(probe.artifact_sha256 ?? '', /^[a-f0-9]{64}$/);
    for (const key of ['builtins_only', 'all_finite', 'eager_train_eval_exact', 'all_moments_and_weights_unchanged', 'raw_and_quality_effective_decisions_equal']) assert.equal(probe[key], true, `BN affine conversion check failed: ${key}`);
    assert.deepEqual(probe.input, { count: 1, shape: [1, 192, 480, 3], dtype: 'float32' });
    assert.deepEqual(probe.output, { count: 1, shape: [1, 48, 120, 11], dtype: 'float32' });
    assert.ok(probe.operators?.length > 0 && probe.operators.every((op) => typeof op === 'string' && !/CUSTOM|Flex/i.test(op)), 'BN affine conversion needs builtins only');
    assert.equal(probe.decisions?.length, 6);
    assert.deepEqual(probe.decisions, oracle.decisions?.[name], 'BN affine raw/effective decisions differ');
    for (const key of ['torch_vs_tf_max_abs', 'tf_vs_tflite_max_abs', 'torch_vs_tflite_max_abs', 'compiled_vs_eager_max_abs']) assert.ok(Number.isFinite(probe[key]) && probe[key] >= 0 && probe[key] <= 2e-4, 'BN affine numeric parity summary contradicts tolerance');
  }
  assert.deepEqual(proof.probes.initial.operators, proof.probes.nonzero.operators);
  assert.deepEqual(oracle.decisions.parent, oracle.decisions.initial);
  assert.equal(init.activation_report_sha256, retained.activationReport.sha256);
  assert.equal(docs.activationReport.checkpoint_sha256, base);
  assert.equal(docs.independentActivationReview.status, 'passed_with_descriptive_limits');
  assert.ok(Object.values(docs.independentActivationReview.reviewed_files).includes(retained.activationReport.sha256));
  assert.equal(docs.paritySourceReview.status, 'passed_bounded_source_delta_review');
  assert.equal(docs.paritySourceReview.source_sha256, proof.source_sha256);
  assert.equal(docs.paritySourceReview.new_torch_oracle_report_sha256, retained.sixInputOracle.sha256);
  assert.equal(runtime.helper_sha256, init.source_sha256['word_bn_affine.py']);
  assert.deepEqual(runtime.defects, []); assert.deepEqual(guard.defects, []);
  for (const key of ['initial_parent_clone_weights_and_train_eval_outputs_exact', 'variables_do_not_alias_parent',
    'trained_save_load_weights_train_eval_outputs_exact', 'loaded_classes_force_false_inside_call',
    'keras_train_on_batch_updates_weights_without_moving_moments', 'original_parent_weights_and_flags_unchanged']) assert.equal(runtime[key], true, `BN affine runtime audit failed: ${key}`);
  assert.equal(runtime.three_real_sgd_steps?.all4_affine_tensors_changed, true);
  assert.equal(runtime.three_real_sgd_steps.all_moving_and_non_affine_arrays_exact, true);
  assert.equal(guard.independent_helper_runtime_audit_sha256, retained.independentTrainingAudit.sha256);
  assert.equal(guard.conversion_report_sha256, retained.nonzeroConversionProof.sha256);
  assert.equal(guard.base_parent_checkpoint_sha256, base); assert.equal(guard.initial_checkpoint_sha256, initial);
  assert.equal(guard.forbidden_diagnostic_sha256, diagnostic);
  assert.equal(guard.moving_moments_sha256, branch.moving_moments_sha256, 'BN affine moment digest changed');
  assert.deepEqual(guard.source_sha256, config.source_sha256, 'BN affine guard sources differ');
  for (const key of ['actual52BN_checkpoint_metadata_and_spec_passed', 'exact_actual_proof_pair_and_initial_resume_passed', 'default_disabled_behavior_unchanged',
    'patches_ast_identical_run23', 'data_targets_sampler_point_sources_identical_run23', 'plain_trainable_BN_rejected_despite_saved_false',
    'moving_moment_corruption_aborts_before_any_checkpoint_write', 'unchanged_moments_allow_last_and_retained_save',
    'gamma_beta_changes_do_not_change_moment_digest']) assert.equal(guard[key], true, `BN affine guard audit failed: ${key}`);
  assert.equal(protocol.reviewed_plan_sha256, guard.reviewed_plan_sha256);
  assert.ok(protocol.preflight_audits.some((row) => row.sha256 === retained.independentGuardAudit.sha256), 'BN affine guard is absent from prelaunch protocol');
  assert.ok(Date.parse(protocol.created_at_utc) < Date.parse(config.started_at_utc), 'BN affine protocol must precede training');
  if (run['bn-affine-protocol.json']) assert.equal(run['bn-affine-protocol.json'].sha256, retained.protocol.sha256);
  assert.deepEqual(protocol.source_sha256, config.source_sha256, 'BN affine protocol sources differ');
  assert.equal(protocol.parent_checkpoint_sha256, initial); assert.equal(protocol.base_parent_checkpoint_sha256, base);
  assert.equal(protocol.bn_affine_initialization_report_sha256, retained.initializationReport.sha256);
  assert.equal(protocol.bn_affine_conversion_report_sha256, retained.nonzeroConversionProof.sha256);
  assert.equal(protocol.moving_moments_sha256, branch.moving_moments_sha256);
  assert.equal(protocol.nonzero_diagnostic_sha256, diagnostic); assert.equal(protocol.nonzero_diagnostic_forbidden, true);
  assert.equal(protocol.training_manifest_sha256, config.data_sha256?.[config.train_manifest]);
  assert.equal(protocol.point_metadata_sha256, run.dsbiPointMetadata?.sha256);
  for (const [key, expected] of Object.entries({ illumination_augmentation: true, horizontal_mirror_augmentation: false,
    vertical_stretch_augmentation: false, half_turn_augmentation: false, polarity_augmentation: false })) {
    assert.equal(config[key], expected, `BN affine training flag changed: ${key}`); assert.equal(protocol[key], expected);
  }
  for (const [key, expected] of [['batchnorm_affine', true], ['dsbi_visible_partial_supervision', true], ['geometry_conditioned_residual', false], ['partial_geometry_supervision', false]]) {
    assert.equal(config[key]?.enabled, expected); assert.equal(protocol[key], expected);
  }
  for (const [key, value] of Object.entries({ planned_epochs: config.epochs, batch_size: config.batch_size, learning_rate: config.learning_rate,
    seed: config.seed, geometry_loss: config.geometry_loss, negative_repeat: config.negative_expansion?.base_repeat,
    supplemental_negative_repeat: config.negative_expansion?.supplemental_repeat })) assert.equal(protocol[key], value);
  assert.deepEqual(protocol.planned_eligible_epochs, [8, 9, 10]); assert.equal(protocol.no_activation_before_original_gates, true);
  const ancestor = provenance.trainingRuns.find((row) => row !== run && row.configuration.sha256 === protocol.parent_training_config_sha256 && row['sampling-lineage-final.json']?.sha256 === protocol.parent_sampling_lineage_sha256);
  assert.ok(ancestor, 'BN affine base has no matching retained ancestor');
  const lineage = evidence(ancestor['sampling-lineage-final.json']);
  assert.equal(lineage.status, 'passed'); assert.equal(lineage.actual.config_sha256, ancestor.configuration.sha256);
  assert.equal(lineage.actual.sampling_history_sha256, ancestor['sampling-history.json']?.sha256);
  assert.ok(lineage.actual.observed_epochs.some((row) => row.checkpoint_sha256 === base && row.eligible_by_sampling_lineage === true && lineage.eligible_retained_epochs.includes(row.epoch)), 'BN affine base checkpoint is not eligible');
  if (run['sampling-lineage-final.json']) {
    assert.ok(run['validation-history.json'], 'BN affine trained epochs require moving-moment evidence');
    const history = evidence(run['validation-history.json']);
    assert.ok(Array.isArray(history) && history.length > 0);
    for (const row of history) assert.equal(row.batchnorm_moving_moments_sha256, branch.moving_moments_sha256, 'BN affine epoch changed frozen moments');
    for (const row of evidence(run['sampling-lineage-final.json']).actual.observed_epochs) assert.equal(history.filter((item) => item.epoch === row.epoch).length, 1, 'BN affine epoch moment evidence missing or duplicate');
  }
};
const verifyFineDetailInitialization = (run, config) => {
  const branch = config.fine_detail_branch;
  assert.ok(branch === undefined || typeof branch.enabled === 'boolean', 'Invalid fine detail flag');
  if (!branch?.enabled) {
    assert.equal(run.fineDetailInitialization, undefined, 'Fine detail provenance requires enabled training');
    return;
  }
  const retained = run.fineDetailInitialization;
  assert.ok(retained, 'Missing fine detail initialization provenance');
  verifySourceSet(run.sources, config.source_sha256);
  const docs = {};
  for (const key of ['initializationReport', 'nonzeroConversionProof', 'independentStaticAudit', 'independentPreflightAudit', 'independentGuardAudit', 'protocol']) {
    assert.ok(retained[key], `Missing fine detail proof: ${key}`); docs[key] = evidence(retained[key]);
  }
  const { initializationReport: init, nonzeroConversionProof: proof, independentStaticAudit: staticAudit,
    independentPreflightAudit: audit, independentGuardAudit: guard, protocol } = docs;
  for (const doc of [init, proof, audit, guard]) assert.equal(doc.status, 'passed', 'Fine detail proof failed');
  assert.equal(staticAudit.status, 'static_config_and_arithmetic_passed', 'Fine detail static audit failed');
  const base = retained.baseParentCheckpointSha256, initial = retained.initialParentCheckpointSha256, diagnostic = retained.forbiddenDiagnosticCheckpointSha256;
  for (const digest of [base, initial, diagnostic, init.oracle_sha256]) assert.match(digest ?? '', /^[a-f0-9]{64}$/);
  assert.equal(new Set([base, initial, diagnostic]).size, 3, 'Fine detail checkpoint identities must differ');
  assert.equal(config.resume_sha256, initial, 'Fine detail must resume the exact initial parent');
  for (const doc of [branch, init]) {
    assert.equal(doc.initial_checkpoint_sha256, initial); assert.equal(doc.nonzero_diagnostic_sha256, diagnostic);
  }
  assert.equal(branch.base_parent_checkpoint_sha256, base); assert.equal(init.parent_checkpoint_sha256, base);
  assert.equal(branch.nonzero_diagnostic_forbidden, true); assert.equal(init.nonzero_diagnostic_must_never_be_resumed, true);
  assert.ok(![initial, diagnostic].includes(validation.oracle.checkpoint_sha256), 'Fine detail probes cannot be selected');
  for (const ancestor of provenance.trainingRuns) assert.notEqual(evidence(ancestor.configuration).resume_sha256, diagnostic, 'Fine detail diagnostic cannot enter training ancestry');
  for (const doc of [branch, proof, audit, guard]) assert.equal(doc.initialization_report_sha256, retained.initializationReport.sha256, 'Fine detail initialization proof pair changed');
  for (const doc of [branch, audit, guard]) assert.equal(doc.conversion_report_sha256, retained.nonzeroConversionProof.sha256, 'Fine detail conversion proof pair changed');
  assert.equal(init.static_audit_sha256, retained.independentStaticAudit.sha256);
  assert.equal(init.static_audit_loaded, true);
  for (const doc of [proof, audit]) assert.equal(doc.oracle_sha256, init.oracle_sha256, 'Fine detail saved oracle changed');
  assert.equal(proof.selected_validation_sha256, protocol.parent_selected_validation_sha256);
  const requiredSource = (name, digest) => {
    assert.match(digest ?? '', /^[a-f0-9]{64}$/, `Missing fine detail source hash: ${name}`);
    const entry = retained.sources?.find((row) => basename(row.file) === name && row.sha256 === digest);
    assert.ok(entry, `Missing exact fine detail source: ${name}`); verifiedBytes(entry);
    if (name in config.source_sha256) assert.equal(config.source_sha256[name], digest, `Fine detail source differs from training: ${name}`);
  };
  for (const [doc, names] of [[init, ['word_fine_branch.py', 'word_fine_init.py', 'word_model.py', 'word_data.py']],
    [proof, ['word_fine_parity.py', 'word_bn_affine_parity.py', 'word_decode.py', 'word_benchmark.py', 'benchmark_angelina.py', 'word_data.py']]]) {
    assert.deepEqual(Object.keys(doc.source_sha256 ?? {}).sort(), names.sort(), 'Fine detail executed source dependencies changed');
    for (const [name, digest] of Object.entries(doc.source_sha256)) requiredSource(name, digest);
  }
  for (const name of ['word_train.py', 'word_fine_validation.py']) {
    const entry = run.sources.find((row) => basename(row.file) === name && row.sha256 === config.source_sha256[name]);
    assert.ok(entry, `Missing fine detail trainer guard source: ${name}`); verifiedBytes(entry);
  }
  for (const doc of [staticAudit, audit, guard]) if (doc.script_sha256) {
    const entry = retained.sources?.find((row) => row.sha256 === doc.script_sha256);
    assert.ok(entry, 'Missing original fine detail audit script'); verifiedBytes(entry);
  }
  assert.equal(Object.keys(staticAudit.source_sha256 ?? {}).length, 5, 'Fine detail static audit must retain its five executed dependencies');
  for (const digest of Object.values(staticAudit.source_sha256)) {
    const entry = retained.sources?.find((row) => row.sha256 === digest);
    assert.ok(entry, 'Missing exact fine detail static audit dependency'); verifiedBytes(entry);
  }
  assert.deepEqual(branch.spec, init.fine_branch, 'Fine detail spec changed');
  const spec = branch.spec;
  for (const [key, expected] of Object.entries({ tap: 'expanded_conv_depthwise_relu', tap_stride: 2, tap_pixel_origin: 1, tap_local_support: 7,
    output_stride: 4, output_pixel_origin: 3, output_local_support: 11, added_parameters: 1232, added_convolution_macs: 6727680, extra_batchnorm_layers: 0 })) assert.equal(spec[key], expected, `Fine detail coordinate/parameter contract changed: ${key}`);
  assert.deepEqual(spec.tap_shape, [null, 96, 240, 16]); assert.deepEqual(spec.branch_output_shape, [null, 48, 120, 64]);
  assert.deepEqual(spec.depthwise, { kernel: 3, stride: 2, padding: 'same', bias: false });
  assert.deepEqual(spec.lateral, { kernel: 1, channels: 64, bias: true });
  assert.equal(spec.addition, 'Original add_1 result plus branch, before original activation');
  assert.equal(spec.support_scope, 'Convolution support relative to the normalized tensor; raw RGB normalization is global.');
  assert.equal(config.parameters, 830539); assert.equal(init.parameters, 830539);
  assert.deepEqual(config.input_shape, [1, 192, 480, 3]); assert.deepEqual(config.output_shape, [1, 48, 120, 11]);
  assert.deepEqual(config.auxiliary_feature_strides, [2]); assert.deepEqual(config.feature_pyramid_strides, [32, 16, 8, 4]);
  assert.equal(config.image_normalization, 'channel_gray_std05', 'Fine detail changed inherited normalization');
  assert.deepEqual(config.imageNormalization, imageNormalization); assert.deepEqual(init.imageNormalization, config.imageNormalization, 'Fine detail initializer normalization changed');
  assert.equal(config.batch_normalization?.layers, 52); assert.equal(config.batch_normalization.trainable_affine_layers, 0, 'Fine detail requires 52 frozen BN');
  assert.deepEqual(config.batch_normalization.backbone_call_training_keywords, [false]);
  assert.equal(init.ordinary_frozen_batchnorm_layers, 52); assert.equal(init.inherited_array_count, 280); assert.equal(init.array_count, 283);
  for (const key of ['all_inherited_arrays_and_trainability_equal', 'initial_train_eval_parent_outputs_equal', 'initial_save_reload_values_and_outputs_equal',
    'diagnostic_save_reload_outputs_equal', 'diagnostic_only_new_lateral_changed', 'saved_parent_and_initial_unchanged']) assert.equal(init[key], true, `Fine detail initialization check failed: ${key}`);
  const first = init.zero_branch_gradients, second = init.after_isolated_lateral_step_gradients;
  assert.ok([first?.loss, second?.loss, first?.lateral_l1, second?.lateral_l1, second?.depthwise_l1].every((v) => Number.isFinite(v) && v > 0) && first.depthwise_l1 === 0, 'Fine detail gradient reachability failed');
  for (const value of [init.nonzero_max_output_delta, proof.nonzero_tf_output_delta, proof.nonzero_tflite_output_delta]) assert.ok(Number.isFinite(value) && value > 1e-3 && value <= 1, 'Fine detail nonzero probe must exceed conversion tolerance');
  assert.equal(audit.nonzero_max_output_delta, init.nonzero_max_output_delta);
  assert.equal(proof.absolute_tolerance, 1e-4); assert.equal(proof.relative_tolerance, 1e-4);
  assert.equal(proof.conversion_probes_must_not_be_activated_or_resumed, true); assert.equal(proof.trained_candidate_requires_full405_parity, true, 'Fine detail preflight cannot replace trained parity');
  for (const [name, checkpoint] of [['initial', initial], ['nonzero', diagnostic]]) {
    const probe = proof.probes?.[name]; assert.equal(probe?.checkpoint_sha256, checkpoint); assert.equal(probe.samples, 6);
    assert.match(probe.artifact_sha256 ?? '', /^[a-f0-9]{64}$/);
    for (const key of ['builtins_only', 'all_finite', 'eager_train_eval_exact', 'all_moments_and_weights_unchanged', 'raw_and_quality_effective_decisions_equal']) assert.equal(probe[key], true, `Fine detail conversion check failed: ${key}`);
    assert.deepEqual(probe.input, { count: 1, shape: [1, 192, 480, 3], dtype: 'float32' });
    assert.deepEqual(probe.output, { count: 1, shape: [1, 48, 120, 11], dtype: 'float32' });
    assert.ok(probe.operators?.length > 0 && probe.operators.every((op) => typeof op === 'string' && !/CUSTOM|Flex/i.test(op)), 'Fine detail conversion requires builtins only');
    assert.equal(probe.decisions?.length, 6);
    for (const row of probe.decisions) assert.deepEqual(row.effective, row.quality_reason ? { text: '', reason: row.quality_reason, cell_count: 0 } : row.raw, 'Fine detail quality-effective decisions contradict raw decisions');
    for (const key of ['torch_vs_tf_max_abs', 'tf_vs_tflite_max_abs', 'torch_vs_tflite_max_abs', 'compiled_vs_eager_max_abs']) assert.ok(Number.isFinite(probe[key]) && probe[key] >= 0 && probe[key] <= 2e-4, 'Fine detail numeric parity contradicts tolerance');
  }
  assert.notEqual(proof.probes.initial.artifact_sha256, proof.probes.nonzero.artifact_sha256, 'Fine detail nonzero artifact must differ');
  assert.deepEqual(proof.probes.initial.operators, proof.probes.nonzero.operators);
  assert.deepEqual(proof.probes.initial.decisions, proof.torch_parent_decisions, 'Fine detail zero parent decisions changed');
  assert.equal(staticAudit.parent.sha256, base); assert.equal(staticAudit.parent.plain_batchnorm_layers, 52); assert.equal(staticAudit.parent.all_batchnorm_frozen, true);
  assert.deepEqual(staticAudit.actual_parent_states?.expanded_conv_depthwise_relu, { shape: [96, 240, 16], jump: 2, origin: 1, rf: 7 });
  for (const doc of [audit, guard]) {
    assert.deepEqual(doc.defects, []); assert.equal(doc.base_parent_checkpoint_sha256, base); assert.equal(doc.initial_checkpoint_sha256, initial); assert.equal(doc.forbidden_diagnostic_sha256, diagnostic);
  }
  for (const key of ['zero_lateral_all_zero', 'random_depthwise_nonzero_and_diagnostic_unchanged', 'inherited_fulltail_lateral_nonzero_preserved',
    'backbone_and_layer_trainability_unchanged', 'inherited_normalization_unchanged', 'serialized_branch_has_no_relu_between_zero_and_add',
    'exact_saved_input_provenance_passed', 'raw_and_quality_decisions_recomputed_equal']) assert.equal(audit[key], true, `Fine detail independent preflight failed: ${key}`);
  assert.equal(audit.inherited_arrays_byte_identical, 280); assert.equal(audit.total_arrays, 283); assert.equal(audit.ordinary_frozen_BN_layers, 52); assert.equal(audit.initial_outputs_byte_identical_parent, 6);
  assert.deepEqual(audit.changed_diagnostic_arrays, ['fpn_2_zero/kernel', 'fpn_2_zero/bias']);
  assert.deepEqual(audit.same_padding_axes, [192, 480].map((input) => ({ input, new_depthwise_leading_trailing_padding: [0, 1], output: input / 4, origin: 3, stride: 4, local_support: 11 })), 'Fine detail SAME border semantics changed');
  for (const key of ['actual_report_guard_passed', 'default_off_returns_disabled', 'default_Patches_AST_equal_run24']) assert.equal(guard[key], true, `Fine detail guard audit failed: ${key}`);
  for (const key of ['exact_52_plain_frozen_BN', 'initial_zero_lateral_nonzero_depthwise', 'linear_depthwise_lateral_Add_before_original_activation', 'exclusive_other_initializations', 'guard_before_compile_fit']) assert.equal(guard.model_guard_source_review?.[key], true, `Fine detail model guard review failed: ${key}`);
  assert.deepEqual(guard.source_sha256, config.source_sha256, 'Fine detail guard sources differ');
  assert.equal(protocol.reviewed_plan_sha256, guard.plan_sha256); assert.deepEqual(protocol.source_sha256, config.source_sha256, 'Fine detail protocol sources differ');
  for (const key of ['independentPreflightAudit', 'independentGuardAudit']) assert.ok(protocol.preflight_audits.some((row) => row.sha256 === retained[key].sha256), `Fine detail prelaunch protocol omits ${key}`);
  assert.ok(Date.parse(protocol.created_at_utc) < Date.parse(config.started_at_utc), 'Fine detail protocol must precede training');
  if (run['fine-branch-protocol.json']) assert.equal(run['fine-branch-protocol.json'].sha256, retained.protocol.sha256);
  assert.equal(protocol.parent_checkpoint_sha256, initial); assert.equal(protocol.base_parent_checkpoint_sha256, base);
  assert.equal(protocol.fine_initialization_report_sha256, retained.initializationReport.sha256); assert.equal(protocol.fine_conversion_report_sha256, retained.nonzeroConversionProof.sha256);
  assert.equal(protocol.nonzero_diagnostic_sha256, diagnostic); assert.equal(protocol.nonzero_diagnostic_forbidden, true);
  assert.deepEqual(protocol.adaptation?.fine_branch, spec); assert.equal(protocol.adaptation.parameters, 830539);
  assert.equal(protocol.training_manifest_sha256, config.data_sha256?.[config.train_manifest]); assert.equal(protocol.point_metadata_sha256, run.dsbiPointMetadata?.sha256);
  for (const [key, expected] of Object.entries({ illumination_augmentation: true, horizontal_mirror_augmentation: false, vertical_stretch_augmentation: false, half_turn_augmentation: false, polarity_augmentation: false })) {
    assert.equal(config[key], expected, `Fine detail training flag changed: ${key}`); assert.equal(protocol[key], expected);
  }
  for (const [key, expected] of Object.entries({ fine_detail_branch: true, batchnorm_affine: false, geometry_conditioned_residual: false, dsbi_visible_partial_supervision: true, partial_geometry_supervision: false })) {
    assert.equal(config[key]?.enabled, expected); assert.equal(protocol[key], expected);
  }
  for (const [key, value] of Object.entries({ planned_epochs: config.epochs, batch_size: config.batch_size, learning_rate: config.learning_rate, seed: config.seed,
    geometry_loss: config.geometry_loss, negative_repeat: config.negative_expansion?.base_repeat, supplemental_negative_repeat: config.negative_expansion?.supplemental_repeat })) assert.equal(protocol[key], value);
  assert.deepEqual(protocol.planned_eligible_epochs, [8, 9, 10]); assert.equal(protocol.no_activation_before_original_gates, true);
  const ancestor = provenance.trainingRuns.find((row) => row !== run && row.configuration.sha256 === protocol.parent_training_config_sha256 && row['sampling-lineage-final.json']?.sha256 === protocol.parent_sampling_lineage_sha256);
  assert.ok(ancestor, 'Fine detail base has no matching retained ancestor');
  const lineage = evidence(ancestor['sampling-lineage-final.json']); assert.equal(lineage.status, 'passed');
  assert.equal(lineage.actual.config_sha256, ancestor.configuration.sha256); assert.equal(lineage.actual.sampling_history_sha256, ancestor['sampling-history.json']?.sha256);
  assert.ok(lineage.actual.observed_epochs.some((row) => row.checkpoint_sha256 === base && row.eligible_by_sampling_lineage === true && lineage.eligible_retained_epochs.includes(row.epoch)), 'Fine detail base checkpoint is not eligible');
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
  verifyHorizontalAspectPreflight(run, config);
  verifyBatchnormAffineInitialization(run, config);
  verifyFineDetailInitialization(run, config);
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
