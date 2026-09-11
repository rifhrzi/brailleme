// Fresh Indonesian decoder evidence. Runs actual TypeScript; never runs a model or camera.
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { Buffer } = require('node:buffer');
const { load, root, loadedSourcePaths } = require('../word-native-parity/geometry-reference.cjs');

const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
function fileSha(file) {
  const digest = createHash('sha256'), buffer = Buffer.alloc(1024 * 1024), fd = fs.openSync(file, 'r');
  try { let count; while ((count = fs.readSync(fd, buffer)) > 0) digest.update(buffer.subarray(0, count)); }
  finally { fs.closeSync(fd); }
  return digest.digest('hex');
}
function sources(extra = {}) {
  const files = [...loadedSourcePaths().map((file) => path.relative(root, file).replaceAll('\\', '/')),
    'scripts/word-native-parity/geometry-reference.cjs', 'scripts/word-decoder-parity/replay-id.cjs'];
  return { ...extra, ...Object.fromEntries(files.map((file) => [file, fileSha(path.join(root, file))])) };
}
function saveNew(destination, report) {
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.writeFileSync(destination, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
  const { rows: _rows, source_sha256: _sources, ...summary } = report;
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  if (!report.parity_passed) process.exitCode = 1;
}
const contract = load(path.join(root, 'src/ml/word-model-contract.ts'));
const { decodeWordOutput, detectWordProposals } = load(path.join(root, 'src/ml/word-decoder.ts'));
const { parseWordMasks } = load(path.join(root, 'src/ml/word-symbols.ts'));
const { isValidWordScanEvent } = load(path.join(root, 'src/types/word-scan.ts'));

function syntheticTensor(masks, weakFirst = false) {
  const output = new Float32Array(contract.WORD_MODEL_OUTPUT_LENGTH);
  const pitch = Math.min(.065, .88 / Math.max(1, masks.length));
  const left = .5 - masks.length * pitch / 2;
  masks.forEach((mask, index) => {
    const x = left + (index + .5) * pitch, column = Math.floor(x * 120);
    const offset = (24 * 120 + column) * 11;
    output.set([index === 0 && weakFirst ? .4 : .98, x * 120 - column, 0, pitch * .7, .25,
      ...Array.from({ length: 6 }, (_, dot) => mask & (1 << dot) ? .99 : .01)], offset);
  });
  return output;
}
function semanticEvidence(destination) {
  const fixturePath = path.join(root, 'training/fixtures/word-symbols-id-v1.json');
  const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  assert.equal(fixture.protocol, contract.WORD_SYMBOL_PROTOCOL);
  assert.equal(fixture.decoderVersion, contract.WORD_DECODER_VERSION);
  const cases = [...fixture.cases];
  for (const [letter, mask] of fixture.alphabet) {
    cases.push({ name: `lowercase-${letter}`, masks: [mask], text: letter });
    cases.push({ name: `capital-${letter}`, masks: [32, mask], text: letter.toUpperCase() });
  }
  cases.push({ name: '12-capital-individual', masks: Array.from({ length: 12 }, () => [32, 1]).flat(), text: 'A'.repeat(12) },
    { name: '12-capital-word', masks: [32, 32, ...Array(12).fill(1)], text: 'A'.repeat(12) },
    { name: '12-digit-integer', masks: [60, ...Array(12).fill(26)], text: '0'.repeat(12) },
    { name: '13-characters', masks: Array(13).fill(1), reason: 'too_many_characters' },
    { name: '25-physical-cells', masks: Array(25).fill(1), reason: 'too_many_cells' });
  const rows = cases.map((item, index) => {
    const parsed = parseWordMasks(item.masks), tensor = syntheticTensor(item.masks);
    const decoded = decodeWordOutput(tensor, contract.DEFAULT_WORD_MODEL_THRESHOLDS);
    const expectedReason = item.reason || '', actualReason = decoded.status === 'success' ? '' : decoded.reason;
    const parserMatch = item.reason ? parsed.status === 'rejected' && parsed.reason === item.reason :
      parsed.status === 'success' && parsed.text === item.text && (!item.outputs || JSON.stringify(parsed.symbols.map((cell) => cell.output)) === JSON.stringify(item.outputs)) &&
      (!item.roles || JSON.stringify(parsed.symbols.map((cell) => cell.role)) === JSON.stringify(item.roles));
    const decoderMatch = item.reason ? decoded.status === 'rejected' && actualReason === (item.masks.length ? expectedReason : 'no_grid') :
      decoded.status === 'success' && decoded.text === item.text && decoded.cells.length === item.masks.length;
    const eventValid = decoded.status !== 'success' || isValidWordScanEvent({ ...decoded, schemaVersion: 4, sessionId: 'semantic-replay', captureId: index,
      timestampMs: index, decoderVersion: contract.WORD_DECODER_VERSION, modelVersion: 'synthetic-no-model' });
    return { index, name: item.name, masks: item.masks, expected: item.text || '', expected_reason: expectedReason,
      parser: parsed, decoder_text: decoded.status === 'success' ? decoded.text : '', decoder_reason: actualReason,
      proposal_count: detectWordProposals(tensor, contract.DEFAULT_WORD_MODEL_THRESHOLDS).length,
      parser_match: parserMatch, decoder_match: decoderMatch, event_valid: eventValid,
      tensor_sha256: sha(Buffer.from(tensor.buffer)) };
  });
  for (const mask of [32, 60, 63]) {
    const tensor = syntheticTensor([mask, 1], true), proposals = detectWordProposals(tensor, contract.DEFAULT_WORD_MODEL_THRESHOLDS);
    const decoded = decodeWordOutput(tensor, contract.DEFAULT_WORD_MODEL_THRESHOLDS);
    rows.push({ index: rows.length, name: `weak-physical-${mask}`, masks: [mask, 1], expected: '', expected_reason: 'low_confidence',
      decoder_text: decoded.status === 'success' ? decoded.text : '', decoder_reason: decoded.status === 'success' ? '' : decoded.reason,
      proposal_count: proposals.length, parser_match: true, decoder_match: decoded.status === 'rejected' && decoded.reason === 'low_confidence' && proposals.length === 2,
      event_valid: true, tensor_sha256: sha(Buffer.from(tensor.buffer)) });
  }
  const mismatches = rows.filter((row) => !row.parser_match || !row.decoder_match || !row.event_valid);
  const fixtureFile = 'training/fixtures/word-symbols-id-v1.json';
  saveNew(destination, { schema_version: 1, created_at_utc: new Date().toISOString(), decoder_version: contract.WORD_DECODER_VERSION,
    symbol_protocol: contract.WORD_SYMBOL_PROTOCOL, scope: 'Actual TypeScript parser, decoder, proposal extraction and event validation on authored synthetic tensors. No model or camera inference.',
    source_sha256: sources({ [fixtureFile]: fileSha(fixturePath) }), fixture_sha256: fileSha(fixturePath), total: rows.length,
    parity_passed: !mismatches.length, mismatch_count: mismatches.length, first_mismatches: mismatches.slice(0, 10), rows });
}

function readExactly(fd, bytes) {
  let count = 0;
  while (count < bytes.length) {
    const read = fs.readSync(fd, bytes, count, bytes.length - count, null);
    assert.ok(read > 0, 'Truncated raw oracle data');
    count += read;
  }
}
function replay(directory, destination) {
  assert.equal(os.endianness(), 'LE');
  const referencePath = path.join(directory, 'python-reference.json');
  const reference = JSON.parse(fs.readFileSync(referencePath, 'utf8'));
  assert.equal(reference.decoderVersion ?? reference.decoder_version, contract.WORD_DECODER_VERSION);
  assert.equal(reference.symbolProtocol ?? reference.symbol_protocol, contract.WORD_SYMBOL_PROTOCOL);
  assert.match(reference.model_sha256, /^[a-f0-9]{64}$/);
  assert.ok(contract.isValidWordModelThresholds(reference.thresholds));
  assert.ok(Array.isArray(reference.rows) && reference.rows.length > 0);
  assert.ok(reference.source_sha256 && Object.keys(reference.source_sha256).length);
  assert.ok(reference.source_sha256['training/scripts/word_symbols.py'], 'Python symbol parser must be source-bound');
  for (const [file, expected] of Object.entries(reference.source_sha256)) {
    const absolute = path.resolve(root, file), relative = path.relative(root, absolute);
    assert.ok(relative && !relative.startsWith('..') && !path.isAbsolute(relative), 'Oracle source must stay inside repository');
    assert.equal(fileSha(absolute), expected, `Oracle source changed: ${file}`);
  }
  const { convertWordRawPixels, assessWordImageQuality } = load(path.join(root, 'src/ml/word-image-processing.ts'));
  const outputPath = path.join(directory, 'outputs.f32le'), inputPath = path.join(directory, 'inputs.rgb');
  assert.equal(fileSha(outputPath), reference.raw_outputs_sha256);
  assert.equal(fileSha(inputPath), reference.raw_inputs_sha256);
  const outputBytes = Buffer.alloc(contract.WORD_MODEL_OUTPUT_LENGTH * 4), inputBytes = new Uint8Array(contract.WORD_MODEL_INPUT_LENGTH);
  assert.equal(fs.statSync(outputPath).size, outputBytes.length * reference.rows.length, 'Output tensor count differs');
  assert.equal(fs.statSync(inputPath).size, inputBytes.length * reference.rows.length, 'Input image count differs');
  const outputs = fs.openSync(outputPath, 'r'), inputs = fs.openSync(inputPath, 'r'), rows = [];
  try {
    for (const [index, expected] of reference.rows.entries()) {
      assert.equal(expected.index, index, 'Oracle row order differs');
      readExactly(outputs, outputBytes); readExactly(inputs, inputBytes);
      const image = convertWordRawPixels({ width: 480, height: 192, pixelFormat: 'RGB', buffer: inputBytes.buffer });
      const quality = assessWordImageQuality(image) || '';
      const selection = expected.selection || [0, 0, 1, 1];
      assert.ok(Array.isArray(selection) && selection.length === 4 && selection.every(Number.isFinite));
      const [x, y, right, bottom] = selection;
      const decoded = decodeWordOutput(new Float32Array(outputBytes.buffer, outputBytes.byteOffset, contract.WORD_MODEL_OUTPUT_LENGTH), reference.thresholds,
        { x, y, width: right - x, height: bottom - y });
      const decoderText = decoded.status === 'success' ? decoded.text : '', decoderReason = decoded.status === 'success' ? '' : decoded.reason;
      const text = quality ? '' : decoderText, reason = quality || decoderReason;
      let cellsMatch = null;
      if (expected.python_cells && decoded.status === 'success') {
        cellsMatch = expected.python_cells.length === decoded.cells.length && decoded.cells.every((cell, i) => {
          const other = expected.python_cells[i];
          return cell.mask === other.mask && cell.role === other.role && cell.output === other.output &&
            [cell.box.x, cell.box.y, cell.box.x + cell.box.width, cell.box.y + cell.box.height].every((coordinate, c) => Math.abs(coordinate - other.box[c]) <= 1e-6);
        });
      }
      rows.push({ index, image: expected.image, expected: expected.expected, expected_reject: expected.expected_reject === true,
        typescript_text: text, python_text: expected.python_text, typescript_reason: reason, python_reason: expected.python_reason,
        decoder_text: decoderText, decoder_reason: decoderReason, quality_reason: quality,
        quality_match: quality === expected.python_quality_reason, decoder_text_match: decoderText === expected.python_decoder_text,
        decoder_reason_match: decoderReason === expected.python_decoder_reason, pipeline_text_match: text === expected.python_text,
        pipeline_reason_match: reason === expected.python_reason, cell_semantics_and_boxes_match: cellsMatch,
        correct: expected.expected_reject === true ? !text && !!reason : text === expected.expected });
    }
  } finally { fs.closeSync(outputs); fs.closeSync(inputs); }
  const mismatches = rows.filter((row) => !row.quality_match || !row.decoder_text_match || !row.decoder_reason_match || !row.pipeline_text_match || !row.pipeline_reason_match || row.cell_semantics_and_boxes_match === false);
  saveNew(destination, { schema_version: 1, created_at_utc: new Date().toISOString(), decoder_version: contract.WORD_DECODER_VERSION,
    symbol_protocol: contract.WORD_SYMBOL_PROTOCOL, scope: 'Actual TypeScript decoder/quality on Python-prepared model output tensors and RGB bytes. No model rerun, native execution, or camera accuracy claim.',
    model_sha256: reference.model_sha256, thresholds: reference.thresholds, source_sha256: sources(reference.source_sha256),
    python_reference_sha256: fileSha(referencePath), raw_outputs_sha256: reference.raw_outputs_sha256, raw_inputs_sha256: reference.raw_inputs_sha256,
    ...(reference.fixture_manifest_sha256 ? { fixture_manifest_sha256: reference.fixture_manifest_sha256 } : {}),
    ...(reference.cached_npz_sha256 ? { cached_npz_sha256: reference.cached_npz_sha256 } : {}),
    ...(reference.benchmark_sha256 ? { benchmark_sha256: reference.benchmark_sha256 } : {}),
    total: rows.length, accepted: rows.filter((row) => row.typescript_text).length, correct: rows.filter((row) => row.correct).length,
    compared_cell_semantics_and_boxes_cases: rows.filter((row) => row.cell_semantics_and_boxes_match !== null).length,
    parity_passed: !mismatches.length, mismatch_count: mismatches.length, first_mismatches: mismatches.slice(0, 20), rows });
}

const [input, destination] = process.argv.slice(2);
if (!input || !destination) throw new Error('Usage: node replay-id.cjs <prepared-directory|--semantics> <new-report.json>');
if (input === '--semantics') semanticEvidence(destination);
else replay(input, destination);
