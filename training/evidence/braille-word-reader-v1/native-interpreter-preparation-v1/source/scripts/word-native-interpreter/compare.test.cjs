// Synthetic bookkeeping tests only; these temporary files are never native evidence.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { test } = require('node:test');
const { load, root } = require('../word-native-parity/geometry-reference.cjs');
const { DEFAULT_WORD_MODEL_THRESHOLDS } = load(path.join(root, 'src/ml/word-model-contract.ts'));
const sha = (p) => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const cleanup = (directory) => {
  const resolved = path.resolve(directory);
  assert.equal(path.dirname(resolved), path.resolve(os.tmpdir()));
  assert.ok(path.basename(resolved).startsWith('word-interpreter-unit-test-'));
  fs.rmSync(resolved, { recursive: true, force: true });
};

function fixture(change) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'word-interpreter-unit-test-'));
  const bundle = path.join(dir, 'bundle'), native = path.join(dir, 'native');
  for (const p of ['bundle/inputs', 'bundle/desktop', 'native/outputs']) fs.mkdirSync(path.join(dir, p), { recursive: true });
  const write = (base, name, bytes) => fs.writeFileSync(path.join(base, name), bytes);
  const row = (base, name) => ({ path: name, sha256: sha(path.join(base, name)), bytes: fs.statSync(path.join(base, name)).size });
  const save = (base, name, value) => write(base, name, JSON.stringify(value));
  write(bundle, 'model.tflite', 'UNIT TEST ONLY: not a model and never executed');
  const cases = [], events = [{ kind: 'runtime', options: 'c_api_defaults_no_setters', input_shape: [1,192,480,3], output_shape: [1,48,120,11] }];
  const transfers = [], bundleFiles = [row(bundle, 'model.tflite')], nativeFiles = [];
  for (let index = 0; index < 6; index++) {
    const name = `${String(index).padStart(4, '0')}.f32`;
    const input = `inputs/${name}`, desktop = `desktop/${name}`;
    write(bundle, input, Buffer.alloc(192*480*3*4));
    const expected = new Float32Array(48*120*11), actual = new Float32Array(expected.length);
    if (index === 0) change?.(expected, actual);
    write(bundle, desktop, Buffer.from(expected.buffer));
    write(native, `outputs/${name}`, Buffer.from(actual.buffer));
    cases.push({ index, input, input_sha256: sha(path.join(bundle, input)), desktop_output: desktop,
      desktop_output_sha256: sha(path.join(bundle, desktop)), identity: 'UNIT TEST ONLY', selection: { x:0,y:0,width:1,height:1 } });
    transfers.push(row(bundle, input)); bundleFiles.push(row(bundle, input), row(bundle, desktop));
    nativeFiles.push(row(native, `outputs/${name}`));
    events.push({ kind:'case', index, file:name, invoke_status:0, input_bytes:192*480*3*4,
      output_bytes:48*120*11*4, all_finite:true, invoke_ms:0 });
  }
  events.push({ kind:'complete', cases:6, status:'passed_inference_only', peak_rss_kib:0 });
  save(bundle, 'manifest.json', { status:'trained_candidate_ready_native_execution_pending', cases,
    files:bundleFiles, model_sha256:sha(path.join(bundle, 'model.tflite')), absolute_tolerance:1e-4,
    relative_tolerance:1e-4, thresholds:DEFAULT_WORD_MODEL_THRESHOLDS });
  write(native, 'native-stdout.jsonl', events.map((e) => JSON.stringify(e)).join('\n'));
  nativeFiles.push(row(native, 'native-stdout.jsonl'));
  save(native, 'execution.json', { status:'native_process_completed_comparison_pending', returncode:0,
    bundle_manifest_sha256:sha(path.join(bundle,'manifest.json')), model_sha256:sha(path.join(bundle,'model.tflite')),
    files:nativeFiles, transferred_files_verified_on_device:transfers });
  const output = path.join(dir, 'comparison.json');
  return { dir, native, output, run:() => spawnSync(process.execPath, [path.join(__dirname,'compare.cjs'), bundle, native, output], { encoding:'utf8' }) };
}

test('six identical synthetic tensors and decisions pass comparator bookkeeping', () => {
  const f = fixture();
  try { const result=f.run(); assert.equal(result.status,0,result.stderr); assert.equal(JSON.parse(fs.readFileSync(f.output)).decoder_decisions_matched,6); }
  finally { cleanup(f.dir); }
});

test('missing native output fails closed', () => {
  const f = fixture();
  try { fs.unlinkSync(path.join(f.native,'outputs/0005.f32')); assert.notEqual(f.run().status,0); }
  finally { cleanup(f.dir); }
});

test('extra native output fails exact file inventory', () => {
  const f = fixture();
  try { fs.writeFileSync(path.join(f.native,'outputs/0006.f32'),'extra'); assert.notEqual(f.run().status,0); }
  finally { cleanup(f.dir); }
});

test('numeric mismatch fails even when both decoders reject the image', () => {
  const f = fixture((expected,actual) => { actual[0]=0.01; });
  try { assert.notEqual(f.run().status,0); const r=JSON.parse(fs.readFileSync(f.output)); assert.equal(r.decoder_decisions_matched,6); assert.equal(r.tensor_parity_passed,5); }
  finally { cleanup(f.dir); }
});

test('decoder decision mismatch fails even within tensor tolerance', () => {
  const f = fixture((expected,actual) => {
    const offset=(24*120+60)*11;
    expected.set([DEFAULT_WORD_MODEL_THRESHOLDS.proposalScore-0.000001,0.5,0.5,0.08,0.5,1,0,0,0,0,0],offset);
    actual.set(expected); actual[offset]=DEFAULT_WORD_MODEL_THRESHOLDS.proposalScore+0.000001;
  });
  try { assert.notEqual(f.run().status,0); const r=JSON.parse(fs.readFileSync(f.output)); assert.equal(r.tensor_parity_passed,6); assert.equal(r.decoder_decisions_matched,5); }
  finally { cleanup(f.dir); }
});
