"""Freeze private extensions of the existing backend-free cached parity harness."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
WORK = ROOT / '.braille-training-work/word-scan-v1-20260905'
OUT = WORK / 'selected-v26-typescript-parity-v2'
assert not OUT.exists()
OUT.mkdir()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


prepare = (ROOT / 'scripts/word-decoder-parity/prepare.py').read_text(encoding='utf-8')
prepare = replace_once(prepare, 'ROOT = Path(__file__).resolve().parents[2]', 'ROOT = Path.cwd()')
prepare = replace_once(prepare, '    tensor_path = args.run / "public-word-outputs.npz"', '''    protocol_path = Path(__file__).with_name('protocol.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    for path, digest in protocol['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in protocol['bound_artifacts'].items():
        assert sha(ROOT / path) == digest, path
    tensor_path = ROOT / protocol['cache_file']
    replay = json.loads((ROOT / protocol['replay_file']).read_text(encoding='utf-8'))''')
prepare = replace_once(prepare, '    with np.load(tensor_path, allow_pickle=False) as cached:', '''    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
    def memory():
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters
    budget = 350 * 1024 * 1024
    assert memory().WorkingSetSize + len(records) * 48 * 120 * 11 * 4 + 40 * 1024 * 1024 < budget, 'Insufficient process memory headroom for bounded cache replay'
    with np.load(tensor_path, allow_pickle=False) as cached:''')
prepare = replace_once(prepare, '            with Image.open(args.data / record["image"]) as source:', '''            image_path = args.data / record['image']
            assert sha(image_path) == replay['input_image_sha256'][record['image']], ('Changed existing PNG', index)
            with Image.open(image_path) as source:''')
prepare = replace_once(prepare, '"python_decoder_reason": decoder["reason"],', '"python_decoder_reason": decoder["reason"], "python_decoder_cells": decoder["cells"],')
prepare = replace_once(prepare, '    (args.output / "python-reference.json").write_text', '''    for path, digest in protocol['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    assert not any(name in sys.modules for name in ['torch', 'tensorflow', 'keras'])
    peak = memory().PeakWorkingSetSize
    assert peak < budget, ('Process exceeded 350 MiB budget', peak)
    report.update(protocol_file=str(protocol_path.relative_to(ROOT)), protocol_sha256=sha(protocol_path),
        decoder_version=protocol['decoder_version'], peak_working_set_bytes=peak,
        existing_png_hashes_verified=len(rows), no_model_backend_imported=True)
    report['source_sha256'].update(protocol['source_sha256'])
    (args.output / "python-reference.json").write_text''')
prepare_path = OUT / 'prepare.py'
prepare_path.write_text(prepare, encoding='utf-8')

compare = (ROOT / 'scripts/word-decoder-parity/compare.cjs').read_text(encoding='utf-8')
compare = replace_once(compare, "require('../word-native-parity/geometry-reference.cjs')", "require(path.join(process.cwd(), 'scripts/word-native-parity/geometry-reference.cjs'))")
compare = replace_once(compare, 'const reference = JSON.parse(fs.readFileSync(referencePath, \'utf8\'));', '''const reference = JSON.parse(fs.readFileSync(referencePath, 'utf8'));
const protocolPath = path.join(root, reference.protocol_file);
assert.equal(fileSha(protocolPath), reference.protocol_sha256);
const protocol = JSON.parse(fs.readFileSync(protocolPath, 'utf8'));
for (const [file, expected] of Object.entries(protocol.source_sha256)) assert.equal(fileSha(path.join(root, file)), expected);
const boxTolerance = 1e-12;
let maxBoxError = 0, maxConfidenceError = 0;''')
compare = replace_once(compare, '    results.push({ index:', '''    const pythonBoxes = row.python_decoder_cells.map((cell) => {
      const [left, top, right, bottom] = cell.box;
      const x = row.python_decoder_text ? left : Math.max(0, Math.min(1, left));
      const y = row.python_decoder_text ? top : Math.max(0, Math.min(1, top));
      return { x, y, width: (row.python_decoder_text ? right : Math.min(1, right)) - x,
        height: (row.python_decoder_text ? bottom : Math.min(1, bottom)) - y };
    }).filter((box) => box.width > 0 && box.height > 0);
    const boxes = decoded.boxes || [];
    let rowBoxError = 0;
    const boxesMatch = boxes.length === pythonBoxes.length && boxes.every((box, index) =>
      ['x', 'y', 'width', 'height'].every((key) => {
        const error = Math.abs(box[key] - pythonBoxes[index][key]);
        rowBoxError = Math.max(rowBoxError, error);
        return Number.isFinite(error) && error <= boxTolerance;
      }));
    maxBoxError = Math.max(maxBoxError, rowBoxError);
    const cells = decoded.cells || [];
    let rowConfidenceError = 0;
    const cellsMatch = row.python_decoder_text ? cells.length === row.python_decoder_cells.length && cells.every((cell, index) => {
      const expected = row.python_decoder_cells[index];
      const error = Math.abs(cell.confidence - Math.min(expected.score, expected.dot_confidence));
      rowConfidenceError = Math.max(rowConfidenceError, error);
      return cell.letter === row.python_decoder_text[index] && cell.mask === expected.mask && Number.isFinite(error) && error <= boxTolerance;
    }) : cells.length === 0;
    maxConfidenceError = Math.max(maxConfidenceError, rowConfidenceError);
    results.push({ decoder_boxes_match: boxesMatch, decoder_cell_order_match: cellsMatch,
      box_max_abs_error: rowBoxError, confidence_max_abs_error: rowConfidenceError,
      typescript_boxes: boxes, typescript_cells: cells, index:''')
compare = replace_once(compare, 'const mismatches = results.filter((row) => !row.quality_match', 'const mismatches = results.filter((row) => !row.decoder_boxes_match || !row.decoder_cell_order_match || !row.quality_match')
compare = replace_once(compare, 'fs.mkdirSync(path.dirname(destination), { recursive: true });', '''for (const [file, expected] of Object.entries(protocol.source_sha256)) assert.equal(fileSha(path.join(root, file)), expected);
const peakBytes = process.resourceUsage().maxRSS * 1024;
assert.ok(peakBytes < 350 * 1024 * 1024, `Node exceeded memory budget: ${peakBytes}`);
Object.assign(report, { decoder_version: protocol.decoder_version, protocol_sha256: reference.protocol_sha256,
  decoder_box_matches: results.filter((row) => row.decoder_boxes_match).length,
  decoder_cell_order_matches: results.filter((row) => row.decoder_cell_order_match).length,
  box_absolute_tolerance: boxTolerance, box_max_abs_error: maxBoxError,
  confidence_max_abs_error: maxConfidenceError, peak_working_set_bytes: peakBytes,
  python_peak_working_set_bytes: reference.peak_working_set_bytes,
  source_hashes_verified_before_after: true });
fs.mkdirSync(path.dirname(destination), { recursive: true });''')
compare_path = OUT / 'compare.cjs'
compare_path.write_text(compare, encoding='utf-8')

historical_path = WORK / 'selected-v15-independent-diagnosis/protocol.json'
assert sha(historical_path) == '771320895163a7b2188d7add60873c4e151a9e0476f519a77d138c2a955178f0'
historical = json.loads(historical_path.read_text(encoding='utf-8'))
for name, digest in historical['hashes'].items():
    assert sha(WORK / name) == digest
run = WORK / 'selected-v26'
benchmark = json.loads((run / 'public-benchmark.json').read_text(encoding='utf-8'))
replay = json.loads((run / 'cached-word-replay.json').read_text(encoding='utf-8'))
selection = json.loads((run / 'selected-validation-fullprecision.json').read_text(encoding='utf-8'))
assert benchmark['words'] == replay['words']
assert benchmark['model_sha256'] == replay['model_sha256'] == sha(run / 'word_reader.tflite') == sha(WORK / 'selected-v15/word_reader.tflite')
assert benchmark['thresholds'] == replay['thresholds'] == selection['thresholds']
assert benchmark['fixture_manifest_sha256'] == replay['fixture_manifest_sha256'] == sha(WORK / 'data/external-words.json')
assert selection['decoder_version'] == selection['oracle']['decoder_version'] == replay['decoder_version'] == 'six-dot-word-v2'
assert replay['cache_sha256'] == sha(WORK / 'selected-v15/public-word-outputs.npz')
assert sha(run / 'best-word.keras') == sha(WORK / 'selected-v15/best-word.keras') == selection['checkpoint_sha256']
sources = ['src/ml/word-decoder.ts', 'src/ml/word-model-contract.ts', 'src/ml/word-image-processing.ts', 'src/types/word-scan.ts',
    'training/scripts/word_decode.py', 'training/scripts/word_data.py', 'training/scripts/word_benchmark.py',
    'training/scripts/benchmark_angelina.py', 'training/scripts/word_anchor_zoom.py',
    'scripts/word-decoder-parity/prepare.py', 'scripts/word-decoder-parity/compare.cjs',
    'scripts/word-native-parity/geometry-reference.cjs', str(prepare_path.relative_to(ROOT)).replace('\\', '/'),
    str(compare_path.relative_to(ROOT)).replace('\\', '/'), str(Path(__file__).relative_to(ROOT)).replace('\\', '/')]
artifacts = [historical_path, run / 'public-benchmark.json', run / 'cached-word-replay.json',
    run / 'selected-validation-fullprecision.json', WORK / 'data/external-words.json',
    WORK / 'selected-v15/public-word-outputs.npz', run / 'word_reader.tflite', WORK / 'selected-v15/word_reader.tflite']
protocol = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Frozen607 cached public regression parity. No model/backend, native camera, fixture changes, new pixel corpus, calibration, or accuracy promotion.',
    'decoder_version': 'six-dot-word-v2', 'case_count': 607, 'box_absolute_tolerance': 1e-12,
    'cache_file': str((WORK / 'selected-v15/public-word-outputs.npz').relative_to(ROOT)).replace('\\', '/'),
    'replay_file': str((run / 'cached-word-replay.json').relative_to(ROOT)).replace('\\', '/'),
    'source_sha256': {name: sha(ROOT / name) for name in sources},
    'bound_artifacts': {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in artifacts},
    'per_process_memory_budget_bytes': 350 * 1024 * 1024,
    'large_outputs_are_binary_and_untracked': ['outputs.f32le', 'inputs.rgb'],
}
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'directory': str(OUT), 'protocol_sha256': sha(OUT / 'protocol.json')}, indent=2))
