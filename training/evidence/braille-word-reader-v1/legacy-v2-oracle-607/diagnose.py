"""Frozen legacy classifier oracle-segmentation diagnosis; never trains/selects."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import PIL
from PIL import Image

OUT = Path(__file__).resolve().parent
WORK = OUT.parent
REPO = WORK.parent.parent
ROOT = Path('C:/bm-datasets/AngelinaDataset')
MODEL = REPO / 'assets/models/braille-cell-classifier-v2'
EVIDENCE = REPO / 'training/evidence/braille-cell-classifier-v2'
FIXTURE = WORK / 'data/external-words.json'
ARCHIVED_PREPROCESS = EVIDENCE / 'training-source/benchmark_angelina.py'
ARCHIVED_TRAIN = EVIDENCE / 'training-source/train_angelina_candidate.py'
WORD_REPORT = WORK / 'selected-v13/public-benchmark.json'
EXPECTED_MODEL = '7a41942cf330c9558884385221deb69f282adfd8d39be00a257c75ec9572c318'
EXPECTED_FIXTURE = 'e69d7c0d9582aef7d51a2c15706f6bf62bc3cdfe35d22ad28ad007301d56b41b'
EXPECTED_PREPROCESS = '8debc0aaea846306daf14eae15b56565e0ca287fa919a5bb2de05c9cbd57199a'
EXPECTED_TRAIN = 'a1f0773fae363069d80df1620f4547540a19bae4bc6efe14a95c8dcbf9a4e35e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def preprocessing():
    assert sha(ARCHIVED_PREPROCESS) == EXPECTED_PREPROCESS
    assert sha(ARCHIVED_TRAIN) == EXPECTED_TRAIN
    # Execute only the trusted, hash-verified historical pure preprocessing AST.
    # This avoids unrelated sklearn/TensorFlow imports and any modified live script.
    names = {'CELL_CROP_SCALE', 'MODEL_INPUT_SIZE', 'LETTER_BY_MASK', 'CONFIDENCE_THRESHOLD'}
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in ast.parse(ARCHIVED_PREPROCESS.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {'square_crop_box', 'preprocess_cell'}:
            nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {'np': np, 'Image': Image}
    exec(compile(module, str(ARCHIVED_PREPROCESS), 'exec'), namespace)
    assert namespace['CELL_CROP_SCALE'] == 1.25
    assert namespace['MODEL_INPUT_SIZE'] == (28, 28)
    assert namespace['CONFIDENCE_THRESHOLD'] == .9
    return namespace


def frozen_inputs():
    assert sha(MODEL / 'braille_model.tflite') == EXPECTED_MODEL
    assert sha(FIXTURE) == EXPECTED_FIXTURE
    manifest = read(MODEL / 'manifest.json')
    assert manifest['artifact']['sha256'] == EXPECTED_MODEL
    labels = read(MODEL / 'label.json')
    assert labels == list('abcdefghijklmnopqrstuvwxyz')
    fixtures = read(FIXTURE)['records']
    assert len(fixtures) == 607 and sum(len(r['boxes']) for r in fixtures) == 2242
    return fixtures, labels


def source_groups(fixtures):
    groups = defaultdict(list)
    for index, record in enumerate(fixtures):
        groups[record['source']].append((index, record))
    return groups


def cell_tensors(source, group, prep):
    path = ROOT / source
    csv_path = path.with_suffix('.csv')
    annotations = np.loadtxt(csv_path, delimiter=';', ndmin=2)
    assert np.isfinite(annotations).all()
    with Image.open(path) as source_image:
        rgb = source_image.convert('RGB')
    try:
        source_boxes = (annotations[:, :4] * np.asarray([rgb.width, rgb.height] * 2)).astype(np.float32)
        for fixture_index, record in group:
            indices = np.asarray(record['annotation_rows']) - 1
            crop = record['crop']
            context_boxes = (source_boxes[indices] - np.asarray(crop[:2] * 2)) * np.asarray(
                [480 / (crop[2] - crop[0]), 192 / (crop[3] - crop[1])] * 2)
            order = np.argsort(context_boxes[:, 0] + context_boxes[:, 2])
            assert np.max(np.abs(context_boxes[order] - np.asarray(record['boxes']))) < 1e-9
            assert annotations[indices[order], 4].astype(int).tolist() == record['masks']
            letters = ''.join(prep['LETTER_BY_MASK'][mask].upper() for mask in record['masks'])
            assert record.get('text', record.get('expected')) == letters
            for cell_index, annotation_index in enumerate(indices[order]):
                normalized = annotations[annotation_index, :4].tolist()
                cell = SimpleNamespace(**dict(zip(('left', 'top', 'right', 'bottom'), normalized)))
                tensor = prep['preprocess_cell'](rgb, cell)
                assert tensor.shape == (28, 28, 3) and tensor.dtype == np.float32
                crop_box = list(prep['square_crop_box'](rgb.width, rgb.height, tuple(normalized)))
                yield {'fixture_index': fixture_index, 'image': record['image'], 'source': source,
                       'cell_index': cell_index, 'annotation_row': int(annotation_index + 1),
                       'source_size': [rgb.width, rgb.height], 'normalized_box': normalized,
                       'crop_box': crop_box, 'mask': int(record['masks'][cell_index]),
                       'expected': letters[cell_index],
                       'input_float32_sha256': hashlib.sha256(tensor.tobytes()).hexdigest()}, tensor
    finally:
        rgb.close()


def split_sources(relative):
    path = ROOT / relative
    return {((path.parent / line.strip()).resolve().relative_to(ROOT.resolve())).as_posix()
            for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()}


def prepare():
    fixtures, _ = frozen_inputs()
    prep = preprocessing()
    assert not (OUT / 'protocol.json').exists(), 'Do not replace the frozen pre-inference protocol.'
    cells, sources = [], []
    for source, group in source_groups(fixtures).items():
        for cell, _ in cell_tensors(source, group, prep):
            cells.append(cell)
        path = ROOT / source
        sources.append({'source': source, 'image_sha256': sha(path), 'csv_sha256': sha(path.with_suffix('.csv'))})
    cells.sort(key=lambda row: (row['fixture_index'], row['cell_index']))
    assert len(cells) == 2242
    save(OUT / 'frozen-crops.json', cells)
    split_names = ['books/train.txt', 'handwritten/train.txt', 'books/val.txt', 'handwritten/val.txt', 'uploaded/test.txt']
    groups = {key: split_sources(key) for key in split_names}
    selected = {row['source'] for row in cells}
    word_split = read(WORK / 'data/source-split.json')
    source_audit = {
        'legacy_upstream_split_overlap': {name: len(selected & members) for name, members in groups.items()},
        'word_split_overlap': {name: len(selected & set(members)) for name, members in word_split['images'].items()},
        'legacy_train_pages': len(groups['books/train.txt'] | groups['handwritten/train.txt']),
        'legacy_validation_pages': len(groups['books/val.txt'] | groups['handwritten/val.txt']),
        'legacy_archived_population': read(EVIDENCE / 'split_independence_audit.json')['population'],
        'word_validation_families': word_split['validation_families'],
        'note': 'Legacy upstream page splits and Kaggle training differ from the word source-family split. Uploaded/test already informed historical v2 promotion; this is not a pristine holdout.'}
    assert source_audit['legacy_upstream_split_overlap']['uploaded/test.txt'] == len(selected) == 46
    assert all(source_audit['legacy_upstream_split_overlap'][name] == 0 for name in split_names[:-1])
    protocol = {
        'schemaVersion': 1, 'created_utc_before_inference': datetime.now(timezone.utc).isoformat(),
        'scope': 'Oracle annotation segmentation diagnostic ONLY; never actual scanner accuracy or a fallback/model selection experiment.',
        'fixture_count': 607, 'cell_occurrences': 2242, 'unique_source_cells': len({(r['source'], r['annotation_row']) for r in cells}),
        'model_sha256': EXPECTED_MODEL, 'model_manifest_sha256': sha(MODEL / 'manifest.json'),
        'label_sha256': sha(MODEL / 'label.json'), 'fixture_sha256': EXPECTED_FIXTURE,
        'archived_preprocessing_sha256': EXPECTED_PREPROCESS, 'archived_training_sha256': EXPECTED_TRAIN,
        'script_sha256': sha(__file__), 'crops_sha256': sha(OUT / 'frozen-crops.json'),
        'comparison_saved_v13_benchmark_sha256': sha(WORD_REPORT),
        'comparison_saved_v13_model_sha256': read(WORD_REPORT)['model_sha256'],
        'word_split_sha256': sha(WORK / 'data/source-split.json'),
        'split_files': {name: sha(ROOT / name) for name in split_names}, 'source_exposure': source_audit,
        'pillow_version': PIL.__version__, 'numpy_version': np.__version__,
        'preprocessing': 'Original JPEG decoded RGB (no EXIF transpose, matching archive); normalized source CSV cell; square side=max(W,H)*1.25 limited to page, translate inside bounds, Python round edges; PIL bilinear resize28x28; float32 RGB /255 NHWC.',
        'endpoints_fixed_before_inference': ['All-cell top1 accuracy', 'Oracle whole-word exactness from all GT cells in frozen left-to-right order', 'Same metrics by length', 'Fixed historical .90 confidence: accept full oracle word iff every cell >=.90; never drop a cell', 'C/H/J confusion and all26 confusion matrix', 'Approximate nearest unused-center matched v13 dot errors from saved benchmark only'],
        'v13_match_rule': 'Greedy nearest unused selected prediction center within .45 GT cell height, same rule as word_failure_analysis.py; approximate diagnostic, not detector AP.',
        'resources': {'cpu_threads': 1, 'gpu': False, 'source_images_in_memory': 1, 'memory_limit_bytes': 1000000000},
        'negative_scope': 'No new negative inference. Historical44 center-crop classifier negative result reported separately from archived evaluation; word model44 were training/regression.',
        'sources': sources}
    save(OUT / 'protocol.json', protocol)
    print(json.dumps({k: protocol[k] for k in ('fixture_count', 'cell_occurrences', 'unique_source_cells', 'source_exposure', 'pillow_version')}, indent=2))


def infer():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    for key in ('TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import psutil
    process = psutil.Process()
    protocol = read(OUT / 'protocol.json')
    assert sha(__file__) == protocol['script_sha256']
    assert sha(OUT / 'frozen-crops.json') == protocol['crops_sha256']
    assert PIL.__version__ == protocol['pillow_version'] and np.__version__ == protocol['numpy_version']
    assert not (OUT / 'report.json').exists(), 'Do not overwrite an evaluated diagnostic.'
    fixtures, labels = frozen_inputs()
    prep = preprocessing()
    assert sha(WORD_REPORT) == protocol['comparison_saved_v13_benchmark_sha256']
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    interpreter = tf.lite.Interpreter(model_path=str(MODEL / 'braille_model.tflite'), num_threads=1)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    assert len(inputs) == len(outputs) == 1
    assert inputs[0]['shape'].tolist() == [1, 28, 28, 3] and inputs[0]['dtype'] == np.float32
    assert outputs[0]['shape'].tolist() == [1, 26] and outputs[0]['dtype'] == np.float32
    frozen = read(OUT / 'frozen-crops.json')
    cells = {(c['fixture_index'], c['cell_index']): c for c in frozen}
    sources = {row['source']: row for row in protocol['sources']}
    peak_rss = process.memory_info().rss
    assert peak_rss < protocol['resources']['memory_limit_bytes'], 'CPU process exceeds diagnostic memory ceiling.'
    raw = np.empty((len(frozen), 26), np.float32)
    positions = {key: index for index, key in enumerate(cells)}
    for source, group in source_groups(fixtures).items():
        path = ROOT / source
        assert sha(path) == sources[source]['image_sha256'] and sha(path.with_suffix('.csv')) == sources[source]['csv_sha256']
        for current, tensor in cell_tensors(source, group, prep):
            key = current['fixture_index'], current['cell_index']
            assert current == cells[key], 'Crop/GT provenance changed after protocol freeze.'
            interpreter.set_tensor(inputs[0]['index'], tensor[None])
            interpreter.invoke()
            probabilities = interpreter.get_tensor(outputs[0]['index'])[0]
            assert np.isfinite(probabilities).all() and abs(float(probabilities.sum()) - 1) < 1e-4
            raw[positions[key]] = probabilities
        peak_rss = max(peak_rss, process.memory_info().rss)
        assert peak_rss < protocol['resources']['memory_limit_bytes']
    np.savez_compressed(OUT / 'probabilities.npz', probabilities=raw)
    for cell, probabilities in zip(frozen, raw):
        index = int(np.argmax(probabilities))
        cell.update(predicted=labels[index].upper(), confidence=float(probabilities[index]))
    report = summarize(fixtures, frozen, prep)
    report.update(protocol_sha256=sha(OUT / 'protocol.json'), probabilities_sha256=sha(OUT / 'probabilities.npz'),
                  model_sha256=EXPECTED_MODEL, fixture_sha256=EXPECTED_FIXTURE,
                  runtime={'tensorflow': tf.__version__, 'pillow': PIL.__version__, 'numpy': np.__version__, 'python': sys.version,
                           'tflite_threads': 1, 'sampled_peak_rss_bytes': peak_rss, 'windows_peak_working_set_bytes': getattr(process.memory_info(), 'peak_wset', None),
                           'gpu_devices': [str(v) for v in tf.config.get_visible_devices('GPU')]})
    save(OUT / 'cell-results.json', frozen)
    report['cell_results_sha256'] = sha(OUT / 'cell-results.json')
    save(OUT / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('words', 'by_source', 'confusion_matrix', 'per_class')}, indent=2))


def summarize(fixtures, cells, prep):
    by_word = defaultdict(list)
    matrix = np.zeros((26, 26), int)
    for cell in cells:
        by_word[cell['fixture_index']].append(cell)
        matrix[ord(cell['expected']) - 65, ord(cell['predicted']) - 65] += 1
    def metrics(indices):
        groups = [by_word[index] for index in indices]
        total = sum(len(group) for group in groups)
        correct = sum(c['expected'] == c['predicted'] for group in groups for c in group)
        exact = sum(all(c['expected'] == c['predicted'] for c in group) for group in groups)
        accepted = [group for group in groups if all(c['confidence'] >= .9 for c in group)]
        accepted_correct = sum(all(c['expected'] == c['predicted'] for c in group) for group in accepted)
        return {'letters': total, 'letters_correct': correct, 'letter_top1_accuracy': correct / total,
                'words': len(groups), 'oracle_exact_words': exact, 'oracle_word_top1_exactness': exact / len(groups),
                'fixed90_oracle_accepted_words': len(accepted), 'fixed90_oracle_correct_words': accepted_correct,
                'fixed90_oracle_exactness_rejections_fail': accepted_correct / len(groups),
                'fixed90_oracle_accepted_accuracy': accepted_correct / len(accepted) if accepted else None}
    v13 = read(WORD_REPORT)
    assert len(v13['words']) == len(fixtures)
    comparison = Counter()
    bit_errors = Counter()
    words = []
    for index, (fixture, old_word) in enumerate(zip(fixtures, v13['words'])):
        group = by_word[index]
        assert fixture['image'] == old_word['image']
        expected = ''.join(c['expected'] for c in group)
        assert expected == old_word['expected']
        gt = np.asarray(fixture['boxes'])
        heights = gt[:, 3] - gt[:, 1]
        centers = (gt[:, :2] + gt[:, 2:]) / 2
        predictions = old_word['detected_cells']
        pred_centers = np.asarray([[(p['box'][0] + p['box'][2]) * 240, (p['box'][1] + p['box'][3]) * 96] for p in predictions]).reshape(-1, 2)
        used = set()
        for cell_index, (center, cell) in enumerate(zip(centers, group)):
            candidates = [(float(np.linalg.norm(pc - center) / heights[cell_index]), pi) for pi, pc in enumerate(pred_centers) if pi not in used]
            distance, match = min(candidates) if candidates else (float('inf'), -1)
            legacy_correct = cell['expected'] == cell['predicted']
            if distance > .45:
                status = 'unmatched'
            else:
                used.add(match)
                mask = sum(int(dot >= .5) << bit for bit, dot in enumerate(predictions[match]['dots']))
                status = 'matched_correct_mask' if mask == cell['mask'] else 'matched_wrong_mask'
                for bit in range(6):
                    bit_errors[str(bit + 1)] += bool((mask ^ cell['mask']) & (1 << bit))
                cell['v13_matched_mask'] = mask
            cell['v13_match_status'] = status
            comparison[status] += 1
            comparison[status + ('_legacy_correct' if legacy_correct else '_legacy_wrong')] += 1
        comparison['unmatched_selected_predictions'] += len(predictions) - len(used)
        words.append({'image': fixture['image'], 'source': fixture['source'], 'expected': expected,
                      'oracle_top1': ''.join(c['predicted'] for c in group),
                      'oracle_fixed90_accept': all(c['confidence'] >= .9 for c in group),
                      'v13_actual_selected_word_result': old_word['predicted'], 'v13_rejection': old_word['rejection']})
    per_class = {}
    for index, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        support, predicted, correct = int(matrix[index].sum()), int(matrix[:, index].sum()), int(matrix[index, index])
        per_class[letter] = {'support': support, 'predicted_count': predicted, 'correct': correct,
                             'recall': correct / support if support else None, 'precision': correct / predicted if predicted else None,
                             'wrong_predictions_into_class': predicted - correct,
                             'wrong_sources': {chr(65 + i): int(matrix[i, index]) for i in range(26) if i != index and matrix[i, index]}}
    confusions = sorted([{'expected': chr(65 + i), 'predicted': chr(65 + j), 'count': int(matrix[i, j])}
                         for i in range(26) for j in range(26) if i != j and matrix[i, j]], key=lambda row: -row['count'])
    legacy_eval = read(EVIDENCE / 'evaluation_report.json')
    return {'scope': 'Oracle annotation segmentation ONLY: every GT cell supplied independently from original source RGB. NOT scanner accuracy, detector accuracy, fair architecture comparison, or deployment evidence.',
            'overall': metrics(range(len(fixtures))),
            'by_length': {str(length): metrics([i for i, r in enumerate(fixtures) if len(r['boxes']) == length]) for length in range(1, 13)},
            'by_source': {source: metrics([i for i, r in enumerate(fixtures) if r['source'] == source]) for source in sorted({r['source'] for r in fixtures})},
            'confusion_matrix_labels': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'confusion_matrix': matrix.tolist(),
            'per_class': per_class, 'C_H_J': {c: per_class[c] for c in 'CHJ'}, 'top_confusions': confusions[:20],
            'v13_saved_comparison': {'actual_word_metrics': v13['whole_words'], 'matched_counts': dict(comparison),
                                     'matched_bit_errors': dict(bit_errors), 'matching': 'Nearest unused selected center <=.45 GT height; approximate matching, not AP.',
                                     'limitation': 'V13 sees resized480x192 whole context and must detect/count/order/reject; v2 receives ideal original RGB1.25square crops. Training data and preprocessing also differ.'},
            'historical_negative_scope_separate': {'recomputed': False, 'archived_evaluation_sha256': sha(EVIDENCE / 'evaluation_report.json'),
                                                 'legacy_classifier_44': legacy_eval['public_negative_guardrail'],
                                                 'v13_word_regression_44': v13['negative_regression'],
                                                 'note': 'Different input/crop/acceptance contracts; exclude from607 word and2242 cell denominators. Neither establishes physical negative behavior.'},
            'limitations': ['Annotation crops are an oracle unavailable to the camera app.', 'The frozen607 boundaries are geometric and not human-reviewed lexical ground truth.',
                            'V2 used upstream train/validation pages plus Kaggle; word model uses family-separated data. Uploaded/test informed historical v2 promotion.',
                            'No physical user camera image was tested; a C/H/J cause on the user phone remains unproven.',
                            'Every letter is classified; no dictionary, omission, box tuning, threshold tuning, model selection or fallback integration.'],
            'words': words}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'infer'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else infer()
