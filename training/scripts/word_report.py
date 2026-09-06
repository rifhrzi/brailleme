"""Create a row-verified Markdown candidate report without importing or running ML."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'training/evidence/braille-word-reader-v1'
REPORT_NAMES = ('public-benchmark.json', 'export-report.json', 'checkpoint-selection.json',
                'selected-validation-fullprecision.json', 'repetition-diagnostics.json', 'rejection-diagnostics.json')
REASONS = {'dark', 'glare', 'blur', 'too_far', 'no_grid', 'low_confidence', 'clipped',
           'multiple_rows', 'gap', 'overlapping_cells', 'too_many_cells', 'unsupported_pattern',
           'processing_error', 'capture_error'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    if Path(path).suffix == '.gz':
        raw = gzip.decompress(raw)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def counts(rows, expected=True):
    total = len(rows)
    accepted = sum(bool(row['predicted']) for row in rows)
    correct = sum(row['predicted'] == row['expected'] for row in rows) if expected else 0
    return {'total': total, 'accepted': accepted, 'correct': correct,
            'exact_word_accuracy': correct/total if total else 0.,
            'accepted_accuracy': correct/accepted if accepted else None}


def check_metrics(actual, expected, context, keys=None):
    for key in keys or expected:
        a, e = actual.get(key), expected[key]
        if key in ('total', 'accepted', 'correct'):
            require(type(a) is int and a == e, f'{context}: inconsistent {key}: {a} != {e}')
        elif e is None:
            require(a is None, f'{context}: {key} must be null with zero accepted cases')
        else:
            require(type(a) in (int, float) and math.isfinite(a) and math.isclose(a, e, rel_tol=1e-12, abs_tol=1e-12),
                    f'{context}: inconsistent {key}: {a} != {e}')


def check_rows(rows, context, *, expected=False, max_expected=12):
    for index, row in enumerate(rows):
        predicted, reason = row.get('predicted'), row.get('rejection')
        require(isinstance(predicted, str) and (not predicted or re.fullmatch('[A-Z]{1,12}', predicted)),
                f'{context} row {index}: invalid predicted word')
        require(isinstance(reason, str) and ((bool(predicted) and reason == '') or
                (not predicted and reason in REASONS)), f'{context} row {index}: contradictory acceptance/rejection')
        if expected:
            target = row.get('expected')
            require(isinstance(target, str) and re.fullmatch('[A-Z]+', target) and 1 <= len(target) <= max_expected,
                    f'{context} row {index}: invalid expected word')


def check_aggregate_from_rows(rows, aggregate, context):
    metric = counts(rows)
    check_metrics(aggregate, metric, context)
    by_length = {str(length): counts([r for r in rows if len(r['expected']) == length])
                 for length in range(1, 13) if any(len(r['expected']) == length for r in rows)}
    require(set(aggregate['by_length']) == set(by_length), f'{context}: length groups differ')
    for length, values in by_length.items():
        check_metrics(aggregate['by_length'][length], values, f'{context} length {length}')
    reasons = dict(Counter(row['rejection'] for row in rows if not row['predicted']))
    require(aggregate['rejection_reasons'] == reasons, f'{context}: aggregate rejection reasons differ from final rows')
    return metric, by_length, reasons


def check_validation_metric(metric, records, context):
    require(type(metric['accepted']) is int and type(metric['correct']) is int and
            0 <= metric['correct'] <= metric['accepted'] <= len(records), f'{context}: invalid counts')
    expected = {'total': len(records), 'accepted': metric['accepted'], 'correct': metric['correct'],
                'exact_word_accuracy': metric['correct']/len(records),
                'accepted_accuracy': metric['correct']/metric['accepted'] if metric['accepted'] else None}
    check_metrics(metric, expected, context)
    lengths = Counter(str(len(r['expected'])) for r in records)
    require(set(metric['by_length']) == set(lengths), f'{context}: validation length groups differ')
    for length, total in lengths.items():
        group = metric['by_length'][length]
        require(type(group['accepted']) is int and type(group['correct']) is int and
                0 <= group['correct'] <= group['accepted'] <= total, f'{context}: invalid length counts')
        check_metrics(group, {'total': total, 'accepted': group['accepted'], 'correct': group['correct'],
                              'exact_word_accuracy': group['correct']/total,
                              'accepted_accuracy': group['correct']/group['accepted'] if group['accepted'] else None}, context)
    for key in ('total', 'accepted', 'correct'):
        require(sum(g[key] for g in metric['by_length'].values()) == metric[key], f'{context}: length {key} sum differs')
    require(all(reason in REASONS and type(n) is int and n > 0 for reason, n in metric['rejection_reasons'].items()),
            f'{context}: invalid rejection counts')
    require(sum(metric['rejection_reasons'].values()) == metric['total']-metric['accepted'], f'{context}: rejection total differs')


def validate(reports, fixtures, model_hash, checkpoint_hash):
    benchmark, export, selection, validation, repetition, rejection = [reports[n] for n in REPORT_NAMES]
    require(re.fullmatch('[a-f0-9]{64}', model_hash) and re.fullmatch('[a-f0-9]{64}', checkpoint_hash), 'Invalid artifact hash')
    require(benchmark['model_sha256'] == export['artifact_sha256'] == repetition['model_sha256'] == rejection['model_sha256'] == model_hash,
            'Model hash differs across reports or actual binary')
    require(export['source_checkpoint_sha256'] == selection['selected_checkpoint_sha256'] == validation['checkpoint_sha256'] ==
            validation['oracle']['checkpoint_sha256'] == checkpoint_hash, 'Checkpoint hash differs across export/selection/oracle/binary')
    require(validation['source_checkpoint'] == selection['selected_checkpoint'], 'Selected checkpoint path differs')
    threshold = benchmark['thresholds']
    require(all(type(v) in (int, float) and math.isfinite(v) for v in threshold.values()), 'Threshold contains invalid values')
    require(all(report['thresholds'] == threshold for report in (export, validation, repetition, rejection)), 'Thresholds differ across reports')
    require(export['imageNormalization'] == validation['imageNormalization'], 'Export normalization differs')
    require(export['input_shape'] == [1,192,480,3] and export['output_shape'] == [1,48,120,11], 'Export tensor shape differs')
    require(export['dtype'] == "<class 'numpy.float32'>", 'Export dtype differs')
    require(selection['external_outputs_used'] is False, 'Checkpoint selection used external outputs')
    selected = [c for c in selection['candidates'] if c['checkpoint_sha256'] == checkpoint_hash and
                c['source_checkpoint'] == selection['selected_checkpoint']]
    require(len(selected) == 1, 'Selected candidate is absent or ambiguous')
    for key in ('thresholds', 'words', 'imageNormalization', 'validation_groups', 'equal_group_mean_exact', 'both_group_precision_gates_pass'):
        require(selected[0][key] == validation[key], f'Selected candidate differs from fullprecision validation: {key}')
    external, ext_hash, _ = fixtures['external']
    val_fixture, val_hash, _ = fixtures['validation']
    require(ext_hash == benchmark['fixture_manifest_sha256'], 'External fixture hash differs')
    require(val_hash == selection['fixture_sha256'] == export['parity_fixture_sha256'] == validation['oracle']['fixture_sha256'],
            'Validation/export fixture hash differs')
    require(len(external['records']) == 607 and len(val_fixture['records']) == 405, 'Frozen607/405 fixture counts differ')
    require(selection['validation_words'] == export['parity_samples'] == 405, 'Selection/export parity count differs')
    require(benchmark['fixture_boundary_definition'] == external['scope'] and
            benchmark['fixture_human_boundary_reviewed'] == external['human_boundary_reviewed'], 'Fixture boundary metadata differs')
    require(benchmark['excluded_by_ground_truth_geometry'] == external['excluded_by_ground_truth_geometry'], 'Fixture exclusions differ')
    require(benchmark['external_evaluation_status'] == 'observed_development_regression' and
            benchmark['external_independent_holdout'] is False, 'External regression scope differs')
    words = benchmark['words']
    require(len(words) == 607 and len({row['image'] for row in words}) == 607, 'Benchmark rows missing or duplicated')
    for row, fixture in zip(words, external['records']):
        observed = {k:v for k,v in row.items() if k not in ('predicted', 'rejection', 'detected_cells')}
        require(observed == fixture, 'Benchmark row differs from frozen fixture or its order')
    check_rows(words, 'whole words', expected=True)
    metric, by_length, reasons = check_aggregate_from_rows(words, benchmark['whole_words'], 'whole words')
    check_validation_metric(validation['words'], val_fixture['records'], 'validation405')
    require(set(validation['validation_groups']) == {'natural', 'stress'}, 'Validation group names differ')
    for name, group in validation['validation_groups'].items():
        check_validation_metric(group, [r for r in val_fixture['records'] if r['validation_group'] == name], 'validation '+name)
    mean = sum(g['exact_word_accuracy'] for g in validation['validation_groups'].values())/2
    require(math.isclose(validation['equal_group_mean_exact'], mean, abs_tol=1e-12), 'Validation equal-group mean differs')
    precision_gate = all((g['accepted_accuracy'] or 0) >= .95 for g in validation['validation_groups'].values())
    require(validation['both_group_precision_gates_pass'] is precision_gate, 'Validation precision gate differs')
    negatives, legacy = benchmark['negative_rows'], benchmark['legacy_rows']
    require(len(negatives) == 44 and len({r['source'].replace('\\','/') for r in negatives}) == 44, '44 negatives missing or duplicated')
    require(all(r['source'].replace('\\','/').startswith('not_braille/') for r in negatives), 'Unexpected negative source')
    require(benchmark['negative_regression']['independent_holdout'] is False, 'Trained negatives must not be called independent holdout')
    require(len(legacy) == 66 and len({(r['source'].replace('\\','/'), tuple(r['crop'])) for r in legacy}) == 66, '66 legacy blocks missing or duplicated')
    require(benchmark['legacy_arbitrary_block_regression']['is_complete_word_benchmark'] is False, 'Legacy blocks must be separate')
    check_rows(negatives, 'negatives')
    check_rows(legacy, 'legacy', expected=True, max_expected=1000)
    neg_metric, legacy_metric = counts(negatives, expected=False), counts(legacy)
    check_metrics(benchmark['negative_regression'], neg_metric, 'negatives', ('total','accepted'))
    check_metrics(benchmark['legacy_arbitrary_block_regression'], legacy_metric, 'legacy', ('total','accepted','correct'))
    diagnostics = {}
    for name, report, expected_count, must_reject in (('repetition', repetition, 12, False), ('rejection', rejection, 14, True)):
        fixture, fixture_hash, fixture_path = fixtures[name]
        require(report['fixture_manifest_sha256'] == fixture_hash, name+' fixture hash differs')
        require(len(report['rows']) == len(fixture['records']) == expected_count, name+' diagnostic count differs')
        require(len({r['image'] for r in report['rows']}) == expected_count, name+' rows duplicated')
        require(report['used_for_external_checkpoint_or_threshold_selection'] is False and
                report['is_natural_whole_word_benchmark'] is False, name+' diagnostic scope differs')
        check_rows(report['rows'], name)
        correct = 0
        for row, original in zip(report['rows'], fixture['records']):
            require(row['image'] == original['image'] and row['expected'] == original['expected'] and
                    row['image_sha256'] == original['image_sha256'] and row['expected_reject'] is must_reject,
                    name+' row identity differs from frozen fixture')
            require(digest(fixture_path.parent/original['image']) == original['image_sha256'], name+' fixture PNG changed')
            if must_reject:
                require(original['expected_reject'] is True and row['category'] == original['category'], 'Rejection fixture category differs')
            right = not row['predicted'] if must_reject else row['predicted'] == row['expected']
            require(row['correct'] is right, name+' per-row correctness flag differs')
            correct += right
        result = {'total': expected_count, 'accepted': sum(bool(r['predicted']) for r in report['rows']), 'correct': correct}
        check_metrics(report, result, name)
        diagnostics[name] = result
    numerical = metric['correct']/metric['total'] >= .75 and (metric['accepted_accuracy'] or 0) >= .95 and neg_metric['accepted'] == 0
    require(benchmark['numerical_internal_gates_pass'] is numerical, 'Reported numerical gate disagrees with actual rows')
    parity = all(export.get(key) is True for key in ('parity_pass','decoded_text_parity','decoded_rejection_parity'))
    return {'words': metric, 'by_length': by_length, 'reasons': reasons, 'negatives': neg_metric,
            'negative_reasons': dict(Counter(r['rejection'] for r in negatives if not r['predicted'])),
            'legacy': legacy_metric, 'legacy_reasons': dict(Counter(r['rejection'] for r in legacy if not r['predicted'])),
            'diagnostics': diagnostics, 'numerical_gate': numerical, 'export_parity': parity,
            'offline_gates': numerical and parity and diagnostics['rejection']['accepted'] == 0,
            'validation': validation['words']}


def percentage(value):
    return 'N/A' if value is None else f'{100*value:.2f}%'


def markdown_link(label, target, output):
    relative = Path(os.path.relpath(Path(target).resolve(), output.parent.resolve())).as_posix()
    return f'[{label}](<{relative}>)'


def render(summary, reports, fixture_hashes, links, output, name, attachments):
    benchmark, export, _, validation, repetition, rejection = [reports[n] for n in REPORT_NAMES]
    word = summary['words']
    lines = [f'# Ringkasan kandidat {name}', '',
             f"**Gerbang angka offline: {'LULUS' if summary['offline_gates'] else 'GAGAL'}.** "
             f"Kata utuh benar {word['correct']}/{word['total']} ({percentage(word['exact_word_accuracy'])}); "
             f"hasil diterima benar {word['correct']}/{word['accepted']} ({percentage(word['accepted_accuracy'])}).", '',
             'Regresi pengembangan publik yang telah diamati; bukan holdout fisik independen. '
             'Seluruh penolakan tetap masuk penyebut keberhasilan kata utuh. Presisi hanya memakai hasil yang diterima; N/A berarti tidak ada hasil diterima.', '',
             '| Panjang | Total | Diterima | Benar utuh | Kata utuh % | Presisi diterima % | Ditolak |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for length in range(1,13):
        group = summary['by_length'].get(str(length), counts([]))
        lines.append(f"| {length} | {group['total']} | {group['accepted']} | {group['correct']} | "
                     f"{percentage(group['exact_word_accuracy']) if group['total'] else 'N/A'} | {percentage(group['accepted_accuracy'])} | {group['total']-group['accepted']} |")
    lines += [f"| **Total** | **{word['total']}** | **{word['accepted']}** | **{word['correct']}** | **{percentage(word['exact_word_accuracy'])}** | **{percentage(word['accepted_accuracy'])}** | **{word['total']-word['accepted']}** |", '',
              '| Pemeriksaan | Hasil | Syarat | Status |', '| --- | --- | --- | --- |',
              f"| Kata utuh | {percentage(word['exact_word_accuracy'])} | >=75% | {'Lulus' if word['exact_word_accuracy'] >= .75 else 'Gagal'} |",
              f"| Presisi hasil diterima | {percentage(word['accepted_accuracy'])} | >=95% | {'Lulus' if (word['accepted_accuracy'] or 0) >= .95 else 'Gagal'} |",
              f"| Negatif publik diterima | {summary['negatives']['accepted']}/44 | 0/44 | {'Lulus' if not summary['negatives']['accepted'] else 'Gagal'} |",
              f"| Kasus penolakan diterima | {summary['diagnostics']['rejection']['accepted']}/14 | 0/14 | {'Lulus' if not summary['diagnostics']['rejection']['accepted'] else 'Gagal'} |",
              f"| Paritas ekspor | {export['parity_samples']} input | Tensor dan keputusan sesuai | {'Lulus' if summary['export_parity'] else 'Gagal'} |", '',
              'Gerbang angka tidak menyatakan fitur selesai atau APK siap diserahkan. Model/config, native runtime pada APK yang sama, signing, smoke test, dan pengujian perangkat tetap memerlukan bukti masing-masing.', '',
              '| Alasan penolakan 607 kata | Jumlah |', '| --- | --- |']
    lines += [f'| {reason} | {total} |' for reason,total in sorted(summary['reasons'].items())]
    negative, legacy = summary['negatives'], summary['legacy']
    lines += ['', f"44 gambar negatif publik: {negative['total']-negative['accepted']} ditolak, {negative['accepted']} diterima. "
              'Gambar ini juga dipakai training/regresi; tidak dianggap holdout independen.',
              'Alasan negatif: '+', '.join(f'{reason}={count}' for reason,count in sorted(summary['negative_reasons'].items()))+'.', '',
              f"66 blok eksperimen lama: {legacy['correct']} benar, {legacy['accepted']} diterima, {legacy['total']-legacy['accepted']} ditolak. "
              'Blok ini tidak otomatis merupakan kata utuh dan tidak digabung ke penyebut 607 kata.',
              'Alasan legacy: '+', '.join(f'{reason}={count}' for reason,count in sorted(summary['legacy_reasons'].items()))+'.', '',
              f"Komposisi berulang: {repetition['correct']}/12 benar, {repetition['accepted']} diterima. "
              f"Kasus penolakan: {rejection['correct']}/14 ditolak dengan benar, {rejection['accepted']} masih diterima. "
              'Kedua kelompok diagnostik terpisah dari benchmark kata natural dan tidak ditambahkan ke skor 607.', '',
              '| Diagnostik yang gagal | Target | Hasil | Alasan |', '| --- | --- | --- | --- |']
    failures = [r for report in (repetition,rejection) for r in report['rows'] if not r['correct']]
    for row in failures:
        lines.append(f"| {row['image']} | {'Tolak' if row['expected_reject'] else row['expected']} | {row['predicted'] or 'Ditolak'} | {row['rejection'] or '-'} |")
    if not failures:
        lines.append('| Tidak ada | - | - | - |')
    v = summary['validation']
    lines += ['', f"Validation: {v['correct']}/{v['total']} benar ({percentage(v['exact_word_accuracy'])}), "
              f"{v['correct']}/{v['accepted']} hasil diterima benar ({percentage(v['accepted_accuracy'])}). "
              'Jumlah validation diperiksa terhadap manifest dan agregat; generator ini tidak menjalankan ulang tensor atau mengklaim audit 405 baris keluaran validation.', '',
              f"Status artifact pada ekspor: `{export['status']}`. "
              f"`fixture_human_boundary_reviewed`: `{str(benchmark['fixture_human_boundary_reviewed']).lower()}`. "
              'Flag tersebut disalin apa adanya dari bukti beku; generator tidak mengesahkan pemeriksaan batas manusia.', '',
              'APK internal: '+('belum disertakan; tidak ada klaim APK utama selesai.' if 'apk' not in attachments else
                                'file disertakan dan model terbundel cocok; signing/smoke/aksesibilitas belum disahkan generator.'),
              'Perangkat fisik: '+('pending; bukti pengujian belum disertakan.' if 'physical' not in attachments else
                                   'laporan terkait model disertakan; isi dan pelaksanaan fisiknya tidak disahkan generator.'), '',
              '| Ambang kandidat | Nilai |', '| --- | --- |']
    lines += [f'| {key} | {value} |' for key,value in benchmark['thresholds'].items()]
    lines += ['', '| Identitas | SHA-256 |', '| --- | --- |',
              f"| Model TFLite aktual | `{benchmark['model_sha256']}` |",
              f"| Checkpoint aktual | `{validation['checkpoint_sha256']}` |"]
    lines += [f'| Fixture {label} (byte JSON setelah dekompresi jika gzip) | `{value}` |' for label,value in fixture_hashes.items()]
    lines += [f'| {markdown_link(label, p, output)} | `{digest(p)}` |' for label,p in links.items()]
    lines += [f'| Lampiran {label}: {markdown_link(p.name, p, output)} | `{digest(p)}` |' for label,p in attachments.items()]
    lines += [f'| {markdown_link("Generator", Path(__file__), output)} | `{digest(__file__)}` |', '',
              'Ringkasan dihitung dari baris final yang telah diterima/ditolak, lalu dicocokkan dengan agregat, fixture, model/checkpoint aktual, selection, export, dan diagnostik. Tidak ada inferensi, perubahan threshold, atau penulisan ulang laporan sumber.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name')
    parser.add_argument('--evidence-dir', type=Path, help='Use byte-identical retained reports for Markdown links')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--external-fixture', type=Path, default=EVIDENCE/'data-manifests/data__external-words.json.gz')
    parser.add_argument('--validation-fixture', type=Path, default=EVIDENCE/'data-manifests/data__validation-stress-selection.json.gz')
    parser.add_argument('--repetition-fixture', type=Path, default=EVIDENCE/'repetition-fixtures.json')
    parser.add_argument('--rejection-fixture', type=Path, default=EVIDENCE/'rejection-fixtures.json')
    parser.add_argument('--apk', type=Path, help='Optional actual APK; hash/model presence do not establish signing or smoke')
    parser.add_argument('--physical-evidence', type=Path, help='Optional JSON bound by model_sha256; recorded without automatic approval')
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite a generated report; choose a new output path')
    loaded = {name: read_json(args.run/name) for name in REPORT_NAMES}
    reports = {name: item[0] for name,item in loaded.items()}
    fixtures = {name: (*read_json(path), path) for name,path in
                [('external',args.external_fixture), ('validation',args.validation_fixture),
                 ('repetition',args.repetition_fixture), ('rejection',args.rejection_fixture)]}
    model, checkpoint = args.model or args.run/'word_reader.tflite', args.checkpoint or args.run/'best-word.keras'
    model_hash, checkpoint_hash = digest(model), digest(checkpoint)
    require(model.stat().st_size == reports['export-report.json']['artifact_bytes'], 'Export model byte count differs')
    summary = validate(reports, fixtures, model_hash, checkpoint_hash)
    links = {name: (args.evidence_dir or args.run)/name for name in REPORT_NAMES}
    for name,path in links.items():
        require(digest(path) == loaded[name][1] == digest(args.run/name), 'Retained or current report changed: '+name)
    attachments = {}
    if args.apk:
        with zipfile.ZipFile(args.apk) as apk:
            require(args.apk.suffix.lower() == '.apk' and 'AndroidManifest.xml' in apk.namelist(), 'APK container/manifest missing')
            matches = [item.filename for item in apk.infolist() if item.file_size == model.stat().st_size and
                       hashlib.sha256(apk.read(item)).hexdigest() == model_hash]
        require(len(matches) == 1, 'APK must contain exactly one byte-identical candidate model')
        attachments['apk'] = args.apk
    if args.physical_evidence:
        physical = read_json(args.physical_evidence)[0]
        require(physical.get('model_sha256') == model_hash, 'Physical evidence refers to a different model')
        attachments['physical'] = args.physical_evidence
    markdown = render(summary, reports, {name:item[1] for name,item in fixtures.items()}, links,
                      args.output, args.name or args.run.name, attachments)
    require(digest(model) == model_hash and digest(checkpoint) == checkpoint_hash, 'Artifact changed during report generation')
    for name,(_, original_hash, path) in fixtures.items():
        require(read_json(path)[1] == original_hash, 'Fixture changed during report generation: '+name)
    for name,(_, original_hash) in loaded.items():
        require(digest(args.run/name) == original_hash, 'Source report changed during generation: '+name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding='utf-8', newline='\n')
    print(json.dumps({'output':str(args.output), 'sha256':digest(args.output), 'whole_words':summary['words'],
                      'offline_gates_passed':summary['offline_gates'], 'model_executed':False}, indent=2))


if __name__ == '__main__':
    main()
