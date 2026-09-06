"""Reuse bound raw607 outputs, then refresh Lite405 and all smaller v2 regressions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

WORK = Path(__file__).resolve().parent
SOURCE = Path('training/scripts').resolve()
sys.path.insert(0, str(SOURCE))
from word_decode import decode, DECODER_VERSION
from word_benchmark import quality, summarize_public_words
from benchmark_angelina import read_split

RUN = WORK/'selected-v26'
OLD = WORK/'selected-v15'
DATA = WORK/'data'
DATASET = Path('C:/bm-datasets/AngelinaDataset')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2)+'\n')


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize((480, 192), Image.Resampling.BILINEAR), np.float32)/255.


def establish():
    selected = json.loads((RUN/'selected-validation-fullprecision.json').read_text())
    old_export = json.loads((OLD/'export-report.json').read_text())
    assert sha(OLD/'export-report.json') == 'b709a6cc4bf2b45a2e37c7633ee3925235af2083fbf8f501195143b06b1bfd47'
    assert selected['decoder_version'] == selected['oracle']['decoder_version'] == DECODER_VERSION == 'six-dot-word-v2'
    assert sha(RUN/'best-word.keras') == sha(OLD/'best-word.keras') == old_export['source_checkpoint_sha256'] == selected['checkpoint_sha256']
    assert selected['imageNormalization'] == old_export['imageNormalization']
    assert sha(SOURCE/'word_model.py') == sha(OLD/'source/word_model.py') == selected['oracle']['source_sha256']['word_model.py']
    assert old_export['parity_pass'] and old_export['parity_samples'] == 405
    assert old_export['parity_fixture_sha256'] == selected['oracle']['fixture_sha256'] == sha(DATA/'validation-stress-selection.json')
    for name, digest in selected['oracle']['source_sha256'].items():
        assert sha(SOURCE/name) == sha(RUN/'source'/name) == digest
    historical = OLD.parent/'selected-v15-independent-diagnosis/protocol.json'
    assert sha(historical) == '771320895163a7b2188d7add60873c4e151a9e0476f519a77d138c2a955178f0'
    for name, digest in json.loads(historical.read_text())['hashes'].items():
        assert sha(WORK/name) == digest
    assert sha(OLD/'word_reader.tflite') == old_export['artifact_sha256'] == '406db0e55e8a5ba9e0d00fb22aadc24f19698219f375e4dcead9057cf3bf2230'
    if not (RUN/'word_reader.tflite').exists():
        shutil.copyfile(OLD/'word_reader.tflite', RUN/'word_reader.tflite')
    assert sha(RUN/'word_reader.tflite') == old_export['artifact_sha256']
    return selected, old_export, historical


def cached_words(selected, historical):
    fixture_path = DATA/'external-words.json'
    fixture = json.loads(fixture_path.read_text())
    records = fixture['records']
    original = json.loads((OLD/'public-benchmark.json').read_text())
    assert len(records) == 607 and original['fixture_manifest_sha256'] == sha(fixture_path)
    assert original['model_sha256'] == sha(RUN/'word_reader.tflite')
    cache_path = OLD/'public-word-outputs.npz'
    assert sha(cache_path) == 'c262510395aba57e717a8ea3232d3a18cba18a8f36e9c116fa75694e2dc1b49e'
    cached = np.load(cache_path)
    outputs = cached['outputs']; reasons = cached['quality_reasons']
    assert outputs.shape == (607, 48, 120, 11) and outputs.dtype == np.float32 and len(reasons) == 607
    assert np.isfinite(outputs).all() and outputs.min() >= 0 and outputs.max() <= 1
    rows, masked, image_hashes = [], outputs.copy(), {}
    for index, (record, output, reason) in enumerate(zip(records, outputs, reasons, strict=True)):
        image_path = DATA/record['image']
        assert str(reason) == quality(pixels(image_path)), ('Quality input changed', index)
        image_hashes[record['image']] = sha(image_path)
        result = decode(output, selected['thresholds'], record.get('selection')) if not reason else {'text': '', 'reason': str(reason), 'cells': []}
        if reason:
            masked[index] = 0
        rows.append(dict(record, predicted=result['text'], rejection=result['reason'], detected_cells=result['cells']))
    metric = summarize_public_words(masked, records, selected['thresholds'], rows)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'decoder_version': DECODER_VERSION,
              'scope': 'Fresh decoder decisions on all607 historically bound raw TFLite tensors; zero new model invocations for this stage.',
              'model_sha256': sha(RUN/'word_reader.tflite'), 'cache_sha256': sha(cache_path),
              'historical_binding': str(historical), 'historical_binding_sha256': sha(historical),
              'fixture_manifest_sha256': sha(fixture_path), 'selection_sha256': sha(RUN/'selected-validation-fullprecision.json'),
              'thresholds': selected['thresholds'], 'whole_words': metric, 'quality_rejected_before_decode': sum(bool(x) for x in reasons),
              'input_image_sha256': image_hashes, 'words': rows, 'script_sha256': sha(__file__),
              'source_sha256': {name: sha(SOURCE/name) for name in ('word_decode.py', 'word_benchmark.py', 'word_data.py')},
              'new_model_invocations': 0, 'external_outputs_used_for_selection': False}
    save(RUN/'cached-word-replay.json', report)
    print(json.dumps({'cached607': metric, 'report_sha256': sha(RUN/'cached-word-replay.json')}), flush=True)


def lite_regressions(selected, old_export):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = tf.lite.Interpreter(model_path=str(RUN/'word_reader.tflite'), num_threads=4)
    model.allocate_tensors()
    ins, outs = model.get_input_details(), model.get_output_details()
    assert len(ins) == len(outs) == 1
    inp, out = ins[0], outs[0]
    assert inp['shape'].tolist() == [1, 192, 480, 3] and out['shape'].tolist() == [1, 48, 120, 11]
    assert inp['dtype'] == out['dtype'] == np.float32
    calls = 0
    def invoke(image):
        nonlocal calls
        model.set_tensor(inp['index'], image[None]); model.invoke(); calls += 1
        result = model.get_tensor(out['index'])[0].copy()
        assert result.shape == (48, 120, 11) and np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
        return result
    records = json.loads((DATA/'validation-stress-selection.json').read_text())['records']
    with np.load(RUN/'backend-fullprecision.npz') as current_oracle:
        expected_outputs = current_oracle['outputs']
        quality_rejected = current_oracle['quality_rejected']
    with np.load(OLD/'backend-fullprecision.npz') as old_oracle:
        assert np.array_equal(expected_outputs, old_oracle['outputs'])
        assert np.array_equal(quality_rejected, old_oracle['quality_rejected'])
    outputs, raw_decisions, effective_decisions, image_hashes = [], [], [], {}
    for index, record in enumerate(records):
        path = DATA/record['image']; image = pixels(path); actual = invoke(image); expected = expected_outputs[index]
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=0)
        a, b = [decode(value, selected['thresholds'], record.get('selection')) for value in (expected, actual)]
        assert (a['text'], a['reason']) == (b['text'], b['reason']), ('Raw decoder parity', index)
        reason = quality(image)
        assert bool(reason) == bool(quality_rejected[index])
        raw_decisions.append({'text': b['text'], 'reason': b['reason']})
        effective_decisions.append({'text': '' if reason else b['text'], 'reason': reason or b['reason']})
        outputs.append(actual); image_hashes[record['image']] = sha(path)
        if (index + 1) % 50 == 0:
            print(json.dumps({'lite405_completed': index + 1}), flush=True)
    outputs = np.stack(outputs)
    max_error = float(np.max(np.abs(expected_outputs-outputs)))
    temporary_npz = RUN/'backend-tflite-v2.pending.npz'
    assert not temporary_npz.exists() and not (RUN/'backend-tflite-v2.npz').exists()
    np.savez_compressed(temporary_npz, outputs=outputs, quality_rejected=quality_rejected)
    with np.load(temporary_npz) as verify:
        assert np.array_equal(verify['outputs'], outputs)
        assert np.array_equal(verify['quality_rejected'], quality_rejected)
    temporary_npz.replace(RUN/'backend-tflite-v2.npz')
    inherited = {'report': str(OLD/'export-report.json'), 'report_sha256': sha(OLD/'export-report.json'),
                 'checkpoint_bytes_identical': True, 'serialized_graph_and_normalization_identical': True,
                 'word_model_source_sha256': sha(SOURCE/'word_model.py'), 'fixture_manifest_identical': True,
                 'old_and_current_torch405_outputs_exact': True, 'old_torch_oracle_sha256': sha(OLD/'backend-fullprecision.npz'),
                 'current_torch_oracle_sha256': sha(RUN/'backend-fullprecision.npz'),
                 'scope': 'Only prior TensorFlow numeric bounds are inherited for the exact same model/source/fixture and byte-identical Torch405 outputs. All v2 Lite405 outputs and raw/effective decisions were newly evaluated.'}
    conversion = dict(old_export, decoder_version=DECODER_VERSION, thresholds=selected['thresholds'],
                      decoded_text_parity=True, decoded_rejection_parity=True, parity_pass=True,
                      training_backend_vs_tflite_max_abs=max_error, inherited_tensorflow_numeric_proof=inherited,
                      actual_tflite_v2_oracle_sha256=sha(RUN/'backend-tflite-v2.npz'), actual_execution_script_sha256=sha(__file__),
                      artifact_reused_without_reconversion=True, actual_lite_interpreter_version=tf.__version__)
    save(RUN/'export-report.json', conversion)
    save(RUN/'decoder-parity-v2.json', {'status': 'passed', 'decoder_version': DECODER_VERSION, 'samples': 405,
         'model_sha256': sha(RUN/'word_reader.tflite'), 'numeric_atol': 5e-4, 'numeric_rtol': 0, 'max_abs': max_error,
         'raw_decisions': raw_decisions, 'effective_decisions': effective_decisions, 'input_image_sha256': image_hashes,
         'inherited_tensorflow_numeric_proof': inherited, 'selection_sha256': sha(RUN/'selected-validation-fullprecision.json'),
         'source_sha256': {name: sha(SOURCE/name) for name in ('word_model.py', 'word_decode.py', 'word_benchmark.py', 'word_data.py')},
         'script_sha256': sha(__file__)})
    def infer(image, selection=None):
        rgb = np.asarray(image.convert('RGB').resize((480, 192), Image.Resampling.BILINEAR), np.float32)/255.
        output = invoke(rgb); reason = quality(rgb)
        result = decode(output, selected['thresholds'], selection) if not reason else {'text': '', 'reason': reason, 'cells': []}
        return result
    negatives = []
    for path in read_split(DATASET, 'not_braille/train.txt'):
        with Image.open(path) as image:
            result = infer(image, [0., .1875, 1., .8125])
        negatives.append({'source': path.relative_to(DATASET).as_posix(), 'predicted': result['text'], 'rejection': result['reason']})
    legacy_path = WORK.parent/'word-detector-20260905/public-evaluation/report.json'
    legacy = []
    for row in json.loads(legacy_path.read_text())['blocks']:
        with Image.open(DATASET/row['source']) as image:
            result = infer(image.crop(row['crop']))
        legacy.append({'source': row['source'], 'crop': row['crop'], 'expected': row['expected'], 'predicted': result['text'], 'rejection': result['reason']})
    assert len(negatives) == 44 and len(legacy) == 66 and calls == 405+44+66
    cached = json.loads((RUN/'cached-word-replay.json').read_text())
    old_report = json.loads((OLD/'public-benchmark.json').read_text())
    metric = cached['whole_words']; negative_accepted = sum(bool(x['predicted']) for x in negatives)
    report = dict(old_report, decoder_version=DECODER_VERSION, thresholds=selected['thresholds'], whole_words=metric,
                  words=cached['words'], negative_rows=negatives, legacy_rows=legacy,
                  negative_regression={'total': 44, 'accepted': negative_accepted, 'independent_holdout': False, 'reason': old_report['negative_regression']['reason']},
                  legacy_arbitrary_block_regression={'total': 66, 'accepted': sum(bool(x['predicted']) for x in legacy), 'correct': sum(x['predicted']==x['expected'] for x in legacy), 'is_complete_word_benchmark': False},
                  numerical_internal_gates_pass=metric['exact_word_accuracy']>=.75 and (metric['accepted_accuracy'] or 0)>=.95 and negative_accepted==0,
                  release_ready=False, cached_word_replay_sha256=sha(RUN/'cached-word-replay.json'),
                  raw_word_outputs_reused_from={'file': str(OLD/'public-word-outputs.npz'), 'sha256': sha(OLD/'public-word-outputs.npz')},
                  actual_execution_script_sha256=sha(__file__), actual_new_invocations={'word607': 0, 'negative44': 44, 'legacy66': 66, 'parity405': 405})
    assert report['model_sha256'] == sha(RUN/'word_reader.tflite')
    save(RUN/'public-benchmark.json', report)
    print(json.dumps({'lite405_max_abs': max_error, 'whole_words': metric, 'negatives': report['negative_regression'], 'legacy': report['legacy_arbitrary_block_regression']}), flush=True)
    # Use the unchanged diagnostic implementation for the separate frozen12/14 sets.
    import word_diagnostics
    original_args = sys.argv
    try:
        for fixture, name in ((WORK/'repetition-fixtures/manifest.json', 'repetition-diagnostics.json'), (WORK/'rejection-fixtures-v3/manifest.json', 'rejection-diagnostics.json')):
            sys.argv = ['word_diagnostics.py', '--run', str(RUN), '--fixtures', str(fixture), '--report-name', name]
            word_diagnostics.main()
    finally:
        sys.argv = original_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('cached', 'lite'), required=True)
    args = parser.parse_args()
    selected, old_export, historical = establish()
    if args.phase == 'cached':
        cached_words(selected, historical)
    else:
        assert (RUN/'cached-word-replay.json').exists() and not (RUN/'export-report.json').exists()
        lite_regressions(selected, old_export)


if __name__ == '__main__':
    main()
