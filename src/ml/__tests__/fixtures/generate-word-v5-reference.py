"""Build small CPU-only parity fixtures from the frozen OLD validation oracle.

No inference, TEST data, or historical evidence is modified. This file is only a
fixture producer; JavaScript tests execute the actual mobile runtime separately.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / 'training/evidence/braille-word-reader-id-v5-20260911'
EXPERIMENT = Path('C:/bm-symbols-20260908/overall80-20260911')
sys.path.insert(0, str(EVIDENCE / 'source'))
import word_overall_v5_zoom as zoom
import word_overall_v5_tta as flip
from word_decode_id_v4 import decode
from word_symbols_id_v4 import parse_word_masks, encode_text, DECODER_VERSION, SYMBOL_PROTOCOL


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def signature(value):
    return [value['text'], value['reason'], [
        [cell.get('mask'), sum(int(dot >= .5) << bit for bit, dot in enumerate(cell['dots'])),
         cell.get('role'), cell.get('output')] for cell in value['cells']]]


def main():
    folder = Path(__file__).parent
    metadata_path = EXPERIMENT / 'run-pipeline-selection/export/view-oracle.json'
    oracle_path = metadata_path.with_suffix('.npz')
    metadata = read(metadata_path)
    method_path = EXPERIMENT / 'final-method-selection.json'
    method = read(method_path)
    protocol = read(EXPERIMENT / 'protocol.json')
    manifest_path = Path(protocol['development_files']['validation-symbols.json'])
    rows = read(manifest_path)['records']
    assert len(rows) == 540 and metadata['samples'] == 540
    assert metadata['checkpoint_sha256'] == method['checkpoint_sha256']
    assert sha(oracle_path) == metadata['oracle_sha256']
    assert sha(manifest_path) == metadata['validation_sha256']
    assert metadata['scope'] == 'development_validation_only' and metadata['test_opened'] is False
    thresholds = method['thresholds']
    assert thresholds == metadata['thresholds']
    views = list(zoom.VIEWS)
    fixtures = [dict(source=row['source'], expected=row['expected'], group=row['symbol_group'],
                     selection=row.get('selection'), outputs={}) for row in rows]
    with np.load(oracle_path) as data:
        for name in views:
            values = data[name]
            for item, value in zip(fixtures, values):
                indices = np.flatnonzero(value.reshape(-1, 11)[:, 0] >= thresholds['proposalScore'])
                item['outputs'][name] = [[int(index) * 11, *value.reshape(-1, 11)[index].astype(float).tolist()]
                                         for index in indices]
    quality_count = 0
    for index, (item, row) in enumerate(zip(fixtures, rows)):
        image = np.asarray(Image.open(row['image']).convert('RGB'), np.float32) / 255
        quality = bool(flip.quality_rejections(image[None])[0])
        quality_count += quality
        outputs = {}
        for name in views:
            output = np.zeros((48 * 120 * 11,), np.float32)
            for entry in item['outputs'][name]:
                output[entry[0]:entry[0] + 11] = entry[1:]
            outputs[name] = output.reshape(48, 120, 11)
        value = zoom.runtime_decisions(outputs, thresholds, item['selection'], quality_rejected=quality,
                                       decoder=decode, policies=('cascade_zoom_fallback',))
        item['qualityRejected'] = quality
        item['signature'] = signature(value['decisions']['cascade_zoom_fallback'])
        item['route'] = value['routes']['cascade_zoom_fallback']
        if index % 100 == 0:
            print('reference', index, flush=True)
    # Verify sparsification itself against the already frozen full-oracle report.
    report = read(EXPERIMENT / 'run-pipeline-selection/zoom-validation.json')
    recorded = report['variants']['cascade_zoom_fallback']['decisions']
    assert all(item['signature'][0] == old['predicted'] and item['signature'][1] == old['reason']
               and item['route'] == old['route'] for item, old in zip(fixtures, recorded))
    probe = (np.arange(192 * 480 * 3, dtype=np.int64) * 37 % 251).astype(np.float32).reshape(1, 192, 480, 3) / np.float32(250)
    transformed = {name: hashlib.sha256(zoom.transformed_images(probe, name).tobytes()).hexdigest() for name in views}
    content = dict(version=1, scope='VALIDATION', runtimeModule=zoom.RUNTIME_MODULE,
                   modelSha256=method['tflite_sha256'], methodSha256=sha(method_path),
                   selectedPolicy=method['selected_policy'], decoderVersion=DECODER_VERSION,
                   thresholds=thresholds, sampleCount=len(fixtures), rawViews=6, views=views,
                   oracleSha256=metadata['oracle_sha256'], oracleMetadataSha256=sha(metadata_path),
                   validationSha256=sha(manifest_path), referenceSourceSha256={
                       name: sha(EVIDENCE / 'source' / name) for name in
                       ('word_overall_v5_zoom.py', 'word_overall_v5_tta.py', 'word_decode_id_v4.py', 'word_symbols_id_v4.py')},
                   transformProbe=dict(formula='float32((flat_index*37)%251)/float32(250)', sha256=transformed),
                   qualityRejectedCount=quality_count, cases=fixtures)
    output = folder / 'word-v5-runtime-reference.json'
    output.write_text(json.dumps(content, separators=(',', ':')), encoding='utf-8')
    base = read(ROOT / 'training/fixtures/word-symbols-id-v1.json')
    grammar = [{'name': row['name'], 'masks': row['masks']} for row in base['cases']]
    grammar.extend(dict(name=f'single-mask-{mask}', masks=[mask]) for mask in range(64))
    for word in ['a', 'Aku', 'AKU', '123', 'a-b', 'A-B']:
        for text in [word, *[word + mark for mark in '.,?!:;'], '(' + word + ')', '(' + word + ').', '(' + word + '?)']:
            grammar.append(dict(name=text, masks=encode_text(text)))
    for masks in [[54, 54], [54, 1], [1, 54], [54, 54, 1, 54, 54], [54, 1, 50, 54, 50],
                  [60, 1, 36, 60, 3], [36, 1], [1, 36], [1, 36, 36, 3], [60, 1, 50, 3],
                  [32, 50], [32, 32, 50], [54, 32, 1, 54], [50, 1], [1, 50, 50]]:
        grammar.append(dict(name=str(masks), masks=masks))
    for row in grammar:
        row['result'] = parse_word_masks(row['masks'])
    (folder / 'word-symbols-id-v4-reference.json').write_text(json.dumps(dict(
        protocol=SYMBOL_PROTOCOL, decoderVersion=DECODER_VERSION,
        referenceSourceSha256=sha(EVIDENCE / 'source/word_symbols_id_v4.py'), cases=grammar), indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(output), sha256=sha(output), bytes=output.stat().st_size,
                          samples=len(fixtures), qualityRejected=quality_count)))


if __name__ == '__main__':
    main()
