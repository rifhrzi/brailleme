"""Additive final retention after actual TFLite overall TEST reaches 80 percent.

Do not run against incomplete results. Without --apply this validates final
evidence and prints the concrete destination plan. No data/cache or Android
build directory is copied, and no deployed app asset is updated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

REPO = Path('C:/Users/rifqi/OneDrive/Documents/Github/brailleme')
EXPERIMENT = Path('C:/bm-symbols-20260908/overall80-20260911')
sys.path.insert(0, str(REPO/'training/scripts'))
from word_overall_v5_protocol import read, sha, verify_hashes, write_new
from word_overall_v5_runtime import DEFAULT_RUNTIME
from word_overall_v5_evaluate import (canonical, verify_method, report_for, load_cached_regression,
    summarize_decisions, decision_parity, OLD_REGRESSION_SHA256, OLD_BASELINE_SHA256)


def read_bound(path, bindings):
    """Keep the bytes actually validated fixed through the eventual copy."""
    path = Path(path).resolve()
    digest = sha(path)
    if str(path) in bindings and bindings[str(path)] != digest:
        raise ValueError('Validated evidence changed: '+str(path))
    value = read(path)
    verify_hashes({path: digest})
    bindings[str(path)] = digest
    return value


def validate_stage(report, method, experiment, stage, engine, bindings):
    if engine not in ('torch', 'tflite') or stage not in ('primary', 'confirm', 'regression'):
        raise ValueError('Unknown frozen evaluation stage or engine')
    method_path = experiment/'final-method-selection.json'
    identity = dict(method_sha256=sha(method_path), checkpoint_sha256=method['checkpoint_sha256'],
        tflite_sha256=method['tflite_sha256'], primary_engine=method['primary_engine'],
        selected_policy=method['selected_policy'])
    reservation_path = experiment/'test-evaluation-reservation.json'
    reservation = read_bound(reservation_path, bindings)
    if any(reservation.get(key) != value for key,value in identity.items()):
        raise ValueError('TEST reservation differs from the frozen method')
    if stage == 'primary' and engine != method['primary_engine']:
        raise ValueError('Primary report uses a different engine from its reservation')
    if stage == 'confirm' and engine == method['primary_engine']:
        raise ValueError('Confirmation must use the other frozen backend')
    if stage == 'regression' and engine != 'tflite':
        raise ValueError('Historical candidate must use the actual TFLite artifact')
    expected_runtime = {'torch':'native_torch_keras_archive', 'tflite':'tensorflow_lite'}[engine]
    if report.get('runtime', {}).get('engine') != expected_runtime:
        raise ValueError('Report runtime differs from its recorded stage engine')
    if report.get('method_sha256') != identity['method_sha256']:
        raise ValueError('Stage report belongs to a different frozen method')
    if stage != 'regression' and report.get('reservation_sha256') != bindings[str(reservation_path.resolve())]:
        raise ValueError('Report reservation hash differs from the actual reservation')
    started_path = experiment/f'{stage}-{engine}-started.json'
    started = read_bound(started_path, bindings)
    expected_stage = dict(identity,stage=stage,engine=engine)
    if (any(started.get(key) != value for key,value in expected_stage.items())
            or report.get('stage_started_sha256') != bindings[str(started_path.resolve())]):
        raise ValueError('Stage start identity or report hash differs')


def validate_confirmation(primary, confirmation):
    recomputed = {name:decision_parity(primary['results'][name]['decisions'],
                                      confirmation['results'][name]['decisions'])
                  for name in ('primary','native_context')}
    expected_status = 'passed' if all(part['status']=='passed' for part in recomputed.values()) else 'failed'
    if (canonical(confirmation.get('parity')) != canonical(recomputed)
            or confirmation.get('status') != expected_status):
        raise ValueError('Confirmation status or parity differs from complete decision/route recount')
    # A truthfully reported confirmation failure remains a separate outcome.
    # Actual TFLite accuracy is still measured against the same 500 positives.
    return recomputed


def bind_export_review(export, bindings):
    """Retain original failure and reviewer bytes when using a reviewed gate."""
    if 'original_report_path' not in export:
        if 'qualification_policy' in export or 'original_numerical_gate' in export:
            raise ValueError('Reviewed export is missing its original evidence binding')
        return {}
    sources = {}
    for field, target in (('original_report', 'export-parity-original-failed.json'),
                          ('review_script', 'review_export_float32.py')):
        path_field = 'original_report_path' if field == 'original_report' else field
        path = Path(export[path_field]).resolve()
        digest = export[field+'_sha256']
        if export['input_sha256'].get(str(path)) != digest:
            raise ValueError('Export review reference differs from its frozen input binding')
        if str(path) in bindings and bindings[str(path)] != digest:
            raise ValueError('Conflicting validated export review reference')
        verify_hashes({path: digest})
        bindings[str(path)] = digest
        sources[target] = path
    original = read_bound(sources['export-parity-original-failed.json'], bindings)
    gate = {key:original[key] for key in ('status', 'allclose', 'atol', 'rtol')}
    if (export.get('original_numerical_gate_passed') is not False
            or original.get('status') != 'failed' or original.get('allclose') is not False
            or export.get('original_numerical_gate') != gate):
        raise ValueError('Reviewed export must preserve the actual original numerical failure')
    for field in ('checkpoint_sha256', 'tflite_sha256', 'validation_sha256', 'samples',
                  'views', 'inference_count', 'torch_tflite_max_abs', 'per_view_max_abs',
                  'pipeline_decision_mismatches', 'thresholds', 'runtime_module', 'method'):
        if canonical(export.get(field)) != canonical(original.get(field)):
            raise ValueError('Export review changed recorded inference evidence: '+field)
    return sources


def validate_regression_result(result, old_rows):
    baseline = [dict(expected=row['expected'],predicted=row['baseline']['text']) for row in old_rows]
    before = summarize_decisions(baseline)
    if canonical(before) != canonical(result['baseline']):
        raise ValueError('Historical baseline summary differs from its cached evidence')
    decisions = result['decisions']
    if len(decisions) != len(old_rows) or any(old['source'] != new['source'] or old['expected'] != new['expected']
            or old['image_sha256'] != new['image_sha256'] for old,new in zip(old_rows,decisions)):
        raise ValueError('Historical candidate denominator or fixture binding differs')
    after = summarize_decisions(decisions)
    if canonical(after) != canonical(result['candidate']):
        raise ValueError('Historical candidate arithmetic differs from recorded decisions')
    lost = sum(old['baseline']['text']==old['expected'] and new['predicted']!=old['expected']
               for old,new in zip(old_rows,decisions) if old['expected'])
    gained = sum(old['baseline']['text']!=old['expected'] and new['predicted']==old['expected']
                 for old,new in zip(old_rows,decisions) if old['expected'])
    flags = dict(exact_regressed=after['exact_correct'] < before['exact_correct'],
        precision_regressed=before['accepted_precision'] is not None and
            (after['accepted_precision'] is None or after['accepted_precision'] < before['accepted_precision']),
        invalid_false_accepts_regressed=after['invalid_false_accepts'] > before['invalid_false_accepts'])
    if (result.get('lost') != lost or result.get('gained') != gained or result.get('flags') != flags
            or result.get('regression_observed') != any(flags.values())):
        raise ValueError('Historical loss/gain or regression flags disagree with recorded decisions')


def descriptor(path, repo=None):
    path = Path(path).resolve()
    relative = repo is not None and path.is_relative_to(Path(repo).resolve())
    return dict(file=path.relative_to(repo).as_posix() if relative else str(path),
                pathScope='repository_relative' if relative else 'absolute',
                sizeBytes=path.stat().st_size, sha256=sha(path))


def copy_verified(source, destination, expected):
    source, destination = Path(source), Path(destination)
    verify_hashes({source: expected})
    if destination.exists():
        raise ValueError('Refuse to overwrite a retained artifact: '+str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    verify_hashes({destination: expected})


def validate_report(report, method, experiment):
    method_sha = sha(experiment/'final-method-selection.json')
    if (report['method_sha256'] != method_sha or report['checkpoint_sha256'] != method['checkpoint_sha256']
            or report['tflite_sha256'] != method['tflite_sha256'] or report['selected_policy'] != method['selected_policy']
            or report['thresholds'] != method['thresholds'] or report.get('test_used_for_selection') is not False):
        raise ValueError('Final report differs from the frozen inference method')
    for name, field in (('primary', 'primary_manifest'), ('native_context', 'native_manifest')):
        part = report['results'][name]
        rows = read(method[field])['records']
        if part['manifest_sha256'] != sha(method[field]) or len(rows) != len(part['decisions']):
            raise ValueError('Final report denominator differs from frozen corpus')
        for row, decision in zip(rows, part['decisions']):
            if (row['source'] != decision['source'] or row['expected'] != decision['expected']
                    or row['symbol_group'] != decision['group']
                    or method['image_sha256'][row['image']] != decision['image_sha256']):
                raise ValueError('Final report truth or pixel binding differs')
        if canonical(report_for(part['decisions'], primary=name == 'primary')) != canonical(part['summary']):
            raise ValueError('Final report arithmetic disagrees with recorded decisions')


def qualification(method, actual_tflite, regression, export, confirmation_reports):
    primary = actual_tflite['results']['primary']['summary']
    if primary['positive_total'] != 500 or primary['overall_exact_accuracy'] < .8:
        raise ValueError('Packaging waits for at least 80 percent actual TFLite overall exact text')
    return dict(status='experimental_retained_not_promoted', protocol_version=method['protocol_version'],
        model_version='braille-word-reader-id-v5', decoder_version='six-dot-id-v4-experimental',
        primary_overall_target_met=True, primary_overall_exact_accuracy=primary['overall_exact_accuracy'],
        primary_exact_correct=primary['exact_correct'], primary_positive_total=500,
        accepted_precision=primary['accepted_precision'], precision95_met=primary['precision95_met'],
        invalid_total=primary['invalid_total'], invalid_false_accepts=primary['invalid_false_accepts'],
        invalid_rejection_met=primary['invalid_rejection_met'], per_group=primary['groups'],
        native_context=actual_tflite['results']['native_context']['summary'],
        historical_regression={name:{key:item[key] for key in ('baseline','candidate','lost','gained','regression_observed','flags')}
                               for name,item in regression['results'].items()},
        validation_export_parity=dict(status=export['status'], samples=export['samples'], views=export['views'],
            inference_count=export['inference_count'], allclose=export['allclose'],
            atol=export.get('atol'), rtol=export.get('rtol'), maximum_absolute_error=export.get('torch_tflite_max_abs'),
            original_numerical_gate_passed=export.get('original_numerical_gate_passed'),
            original_numerical_gate=export.get('original_numerical_gate'), qualification=export.get('qualification')),
        final_backend_confirmation={name:report['status'] for name,report in confirmation_reports.items()},
        selected_policy=method['selected_policy'], runtime_module=method.get('runtime_module', DEFAULT_RUNTIME),
        tta_method=method['tta_method'], thresholds=method['thresholds'],
        checkpoint_sha256=method['checkpoint_sha256'], tflite_sha256=method['tflite_sha256'],
        scope=method['scope'], physical_camera_validated=False, native_runtime_validated=False,
        app_promoted=False, deployed_assets_changed=False, build_folders_created=0,
        interpretation='Overall80 is a measured benchmark outcome. Precision, invalid rejection, native-context coverage, historical regression, and physical camera readiness remain separate outcomes.',
        inference_requirement='Use this checkpoint plus the exact selected TTA policy, v4 parser and thresholds; model weights alone do not reproduce the reported method')


def validated_final(experiment):
    bindings = {str((experiment/'final-method-selection.json').resolve()):sha(experiment/'final-method-selection.json')}
    method = verify_method(experiment)
    verify_hashes(bindings)
    bindings.update(method['input_sha256'])
    bindings.update(method['source_sha256'])
    bindings.update(method['source_snapshot_sha256'])
    bindings.update(method['image_sha256'])
    primary = read_bound(experiment/'final-test-evaluation.json', bindings)
    validate_report(primary, method, experiment)
    validate_stage(primary, method, experiment, 'primary', method['primary_engine'], bindings)
    if primary.get('target80_met') != primary['results']['primary']['summary']['target80_met']:
        raise ValueError('Primary overall target flag disagrees with its counted results')
    confirmations = {path.name:read_bound(path, bindings) for path in experiment.glob('final-test-*-confirmation.json')}
    for name,report in confirmations.items():
        engines = {f'final-test-{engine}-confirmation.json':engine for engine in ('torch','tflite')}
        if name not in engines:
            raise ValueError('Unknown confirmation report name')
        validate_report(report, method, experiment)
        validate_stage(report, method, experiment, 'confirm', engines[name], bindings)
        validate_confirmation(primary, report)
    if primary['runtime']['engine'] == 'tensorflow_lite':
        actual = primary
    else:
        actual = confirmations.get('final-test-tflite-confirmation.json')
        if not actual or actual['runtime']['engine'] != 'tensorflow_lite':
            raise ValueError('An actual TFLite final report is required')
    export = read_bound(method['export_parity_report'], bindings)
    if export['status'] != 'passed' or not export['allclose'] or export['tflite_sha256'] != method['tflite_sha256']:
        raise ValueError('Bound TFLite export parity has not passed')
    bind_export_review(export, bindings)
    regression = read_bound(experiment/'historical-regression.json', bindings)
    validate_stage(regression, method, experiment, 'regression', 'tflite', bindings)
    if (regression['method_sha256'] != sha(experiment/'final-method-selection.json')
            or regression['runtime']['engine'] != 'tensorflow_lite' or regression['baseline_inference_performed']
            or regression['cached_baseline_report_sha256'] != OLD_REGRESSION_SHA256
            or regression.get('baseline_checkpoint_sha256') != OLD_BASELINE_SHA256):
        raise ValueError('Historical regression must use actual selected TFLite and hash-fixed cached baseline')
    cached = load_cached_regression(method['regression_reference'], method['thresholds'])
    if set(regression['results']) != set(cached):
        raise ValueError('Historical regression must retain exactly the three separate datasets')
    for name, old_rows in cached.items():
        validate_regression_result(regression['results'][name], old_rows)
    verify_hashes(bindings)
    return method, actual, regression, export, confirmations, bindings


def report_text(value):
    primary = value['primary_overall_exact_accuracy']*100
    precision = value['accepted_precision']*100 if value['accepted_precision'] is not None else 0.
    text = [f'# Model eksperimen Braille ID-v5 — 11 September 2026', '',
        f'Hasil test utama TFLite: **{value["primary_exact_correct"]}/500 ({primary:.1f}%) teks utuh tepat**. Target keseluruhan 80% tercapai.', '',
        f'Presisi di antara teks positif yang diterima: {precision:.2f}%. Input tidak valid yang diterima: {value["invalid_false_accepts"]}/{value["invalid_total"]}. Kedua hasil ini dilaporkan terpisah dari target akurasi keseluruhan.', '',
        '| Kelompok | Tepat / total | Akurasi teks utuh | Presisi diterima |', '| --- | ---: | ---: | ---: |']
    for name,item in value['per_group'].items():
        if not item['positive_total']: continue
        precision_text = f'{item["accepted_precision"]*100:.2f}%' if item['accepted_precision'] is not None else '—'
        text.append(f'| {name} | {item["exact_correct"]}/{item["positive_total"]} | {item["overall_exact_accuracy"]*100:.1f}% | {precision_text} |')
    native = value['native_context']
    text.extend(['',f'Audit konteks asli terpisah: {native["exact_correct"]}/{native["positive_total"]} tepat. Audit ini terdiri atas potongan pendek dari sel asli yang berurutan; cakupannya tidak seimbang dan tidak memiliki kapital alami. Angkanya tidak dimasukkan ke denominator 500.', '',
        '| Regresi historis | Baseline tepat / total | Kandidat tepat / total |', '| --- | ---: | ---: |'])
    for name,item in value['historical_regression'].items():
        before, after = item['baseline'],item['candidate']
        text.append(f'| {name} | {before["exact_correct"]}/{before["positive_total"]} | {after["exact_correct"]}/{after["positive_total"]} |' if before['positive_total']
                    else f'| {name} | {before["invalid_false_accepts"]}/{before["invalid_total"]} invalid diterima | {after["invalid_false_accepts"]}/{after["invalid_total"]} invalid diterima |')
    text.extend(['', 'Model dan kebijakan inferensi disimpan sebagai kandidat eksperimen; aset model aplikasi tidak diganti dan tidak ada APK atau folder build baru.', '',
        f'Kebijakan inferensi terpilih: `{value["selected_policy"]}`. Gunakan kebijakan TTA, parser, dan ambang yang ikut dibekukan; berkas model saja tidak mewakili seluruh metode yang diuji.', '',
        'Test utama memakai 500 rangkaian seimbang dari potongan sel hasil pemindai DSBI yang disusun ulang, 96% multikarakter, dengan 40 urutan invalid terpisah. Halaman sumbernya baru dan terpisah dari sumber latihan serta test lama. Ini belum membuktikan akurasi kamera ponsel pada kertas Braille Indonesia nyata.', '',
        f'Checkpoint SHA-256: `{value["checkpoint_sha256"]}`', f'TFLite SHA-256: `{value["tflite_sha256"]}`', ''])
    parity = value['validation_export_parity']
    if parity.get('original_numerical_gate') is not None:
        original = parity['original_numerical_gate']
        count = parity['qualification']['strict_full_policy_decisions']
        text.extend([
            f'Pemeriksaan konversi awal **gagal** pada `atol={original["atol"]:g}, rtol={original["rtol"]:g}`. '
            f'Selisih absolut maksimum Torch/TFLite adalah `{parity["maximum_absolute_error"]:.10f}`. '
            f'Sebelum test dibuka, tinjauan terpisah menerima batas empiris `atol={parity["atol"]:g}, rtol={parity["rtol"]:g}` '
            f'untuk seluruh {parity["samples"]} citra validasi dalam {len(parity["views"])} orientasi/skala; '
            f'seluruh {count} keputusan kebijakan identik, termasuk teks, penolakan, urutan mask fisik, dan rute. '
            'Tinjauan ini tidak mengubah model atau ambang inferensi, dan bukan jaminan kesetaraan pada semua input baru. '
            'Akurasi test utama di atas diukur langsung dengan TFLite dan dilaporkan terpisah dari pemeriksaan konversi.', '',
            'Bukti asli tetap tersedia di [export-parity-original-failed.json](export-parity-original-failed.json); '
            'hasil tinjauan di [export-parity.json](export-parity.json) dan skripnya di '
            '[review_export_float32.py](review_export_float32.py).', ''])
    return '\n'.join(text)


def package(repo, experiment, apply=False):
    repo, experiment = Path(repo).resolve(), Path(experiment).resolve()
    if experiment != EXPERIMENT or repo != REPO:
        raise ValueError('This retention writer is bound to the named workspace and experiment')
    model_dir = repo/'training/checkpoints/braille-word-reader-id-v5'
    evidence_dir = repo/'training/evidence/braille-word-reader-id-v5-20260911'
    index_path = repo/'training/checkpoints/retained-models-20260911.json'
    if any(path.exists() for path in (model_dir, evidence_dir, index_path)):
        raise ValueError('A destination already exists; never overwrite retained results')
    writer_binding = {str(Path(__file__).resolve()):sha(__file__)}
    method, actual, regression, export, confirmations, validated_bindings = validated_final(experiment)
    validated_bindings.update(writer_binding)
    value = qualification(method, actual, regression, export, confirmations)
    prior_indexes = sorted((repo/'training/checkpoints').glob('retained-*.json'))
    deployed = [path for path in (repo/'assets/models').rglob('*') if path.is_file()]
    protected = {str(path):sha(path) for path in prior_indexes+deployed}
    plan = dict(model_directory=str(model_dir), evidence_directory=str(evidence_dir), retention_index=str(index_path),
        selected_checkpoint=method['checkpoint'], selected_tflite=method['tflite'],
        target_overall_accuracy=value['primary_overall_exact_accuracy'], app_promoted=False,
        training_images_copied=0, cache_files_copied=0, historical_indexes_preserved=len(prior_indexes))
    if not apply:
        print(json.dumps(plan,indent=2)); return
    model_dir.mkdir()
    evidence_dir.mkdir()
    copied = []
    for field,name in [('checkpoint','best-word.keras'),('tflite','word-reader.tflite')]:
        source, target = Path(method[field]),model_dir/name
        copy_verified(source,target,method[field+'_sha256'])
        copied.append(dict(source=descriptor(source),destination=descriptor(target,repo)))
    write_new(model_dir/'inference-policy.json',dict(selected_policy=method['selected_policy'],tta_method=method['tta_method'],
        runtime_module=method.get('runtime_module', DEFAULT_RUNTIME),
        thresholds=method['thresholds'], checkpoint_sha256=method['checkpoint_sha256'], tflite_sha256=method['tflite_sha256'],
        final_method_sha256=sha(experiment/'final-method-selection.json'), native_runtime_validated=False))
    write_new(model_dir/'qualification.json',value)
    write_new(model_dir/'RETENTION-COPY.json',dict(copies=copied, originals_preserved=True, app_promoted=False))
    sources = {name:experiment/name for name in ('protocol.json','test-source-reservation.json','corpus-audit.json',
        'source-separation-audit.json','final-method-selection.json','test-evaluation-reservation.json',
        'final-test-evaluation.json','historical-regression.json')}
    sources.update({'checkpoint-selection.json':Path(method['checkpoint_selection']),
        'training-config.json':Path(method['checkpoint_selection']).parent/'training-config.json',
        'tta-validation.json':Path(method['tta_validation_report']),
        'tta-validation.plan.json':Path(method['tta_validation_report']).with_suffix('.plan.json'),
        'export-parity.json':Path(method['export_parity_report']), 'package_final_v5.py':Path(__file__)})
    sources.update({name:experiment/name for name in confirmations})
    sources.update(bind_export_review(export, validated_bindings))
    sources.update({Path(path).name:Path(path) for path in validated_bindings
                    if Path(path).parent==experiment and Path(path).name.endswith('-started.json')})
    verify_hashes(validated_bindings)
    for name, source in sources.items():
        destination = evidence_dir/name
        expected = validated_bindings.get(str(source.resolve()))
        if expected is None:
            raise ValueError('Copy source was not bound when evidence was validated: '+str(source))
        copy_verified(source,destination,expected)
        copied.append(dict(source=descriptor(source),destination=descriptor(destination,repo)))
    for source,digest in method['source_snapshot_sha256'].items():
        destination = evidence_dir/'source'/Path(source).name
        copy_verified(source,destination,digest)
        copied.append(dict(source=descriptor(source),destination=descriptor(destination,repo)))
    with (evidence_dir/'REPORT.md').open('x',encoding='utf-8') as stream:
        stream.write(report_text(value))
    write_new(evidence_dir/'EVIDENCE-COPIES.json',dict(copies=copied, existing_sources_modified=False,
        validation_time_bindings=validated_bindings))
    verify_method(experiment)
    verify_hashes(validated_bindings)
    verify_hashes(protected)
    external_files = [descriptor(path) for path in sorted(experiment.rglob('*')) if path.is_file()]
    inventory = dict(experiment=str(experiment), preserveEntireDirectory=True, files=external_files,
        scope='All experiment models, source/configuration, datasets and caches remain at their existing absolute paths; no cleanup authorization',
        historical_index_union=[descriptor(path,repo) for path in prior_indexes],
        development_references=[descriptor(Path(path)) for path in read(experiment/'protocol.json')['development_files'].values()])
    write_new(evidence_dir/'EXPERIMENT-INVENTORY.json',inventory)
    retained = dict(schemaVersion=1,retainedAt='2026-09-11',createdAtUtc=datetime.now(timezone.utc).isoformat(),
        purpose='Additive retention of the overall80 ID-v5 experiment; no app promotion, cleanup or deletion authorization',
        relationship='Preserve union with every historical retention index and every referenced asset; historical evidence is unchanged',
        historicalIndexes=[descriptor(path,repo) for path in prior_indexes],
        candidate=dict(status=value['status'],targetOverallExactAccuracy=.8,targetMet=True,
            actualOverallExactAccuracy=value['primary_overall_exact_accuracy'],physicalCameraValidated=False,
            precision95Met=value['precision95_met'],invalidRejectionMet=value['invalid_rejection_met'],
            artifacts=[descriptor(path,repo) for path in sorted(model_dir.iterdir()) if path.is_file()]),
        evidence=dict(directory=evidence_dir.relative_to(repo).as_posix(),pathScope='repository_relative',preserveEntireDirectory=True,
            files=[descriptor(path,repo) for path in sorted(evidence_dir.rglob('*')) if path.is_file()]),
        experiment=dict(directory=str(experiment),pathScope='absolute',preserveEntireDirectory=True,
            inventory=descriptor(evidence_dir/'EXPERIMENT-INVENTORY.json',repo)),
        protectedBefore=protected, trainingImagesCopied=0,cacheFilesCopied=0,buildFoldersCreated=0,deployedAssetsChanged=False)
    write_new(index_path,retained)
    verify_hashes(validated_bindings)
    verify_hashes(protected)
    write_new(experiment/'retention-packaging-result.json',dict(status='passed',plan=plan,
        retention_index=descriptor(index_path,repo), model_artifacts=retained['candidate']['artifacts'],
        evidence_directory=str(evidence_dir), protected_historical_and_deployed_hashes_verified=len(protected),
        package_script_sha256=sha(__file__),no_deletions=True))
    print('RETAINED '+str(index_path),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='Write the reviewed additive package after final evidence passes')
    args = parser.parse_args()
    package(REPO,EXPERIMENT,args.apply)


if __name__ == '__main__': main()
