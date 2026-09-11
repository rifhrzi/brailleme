"""Freeze and evaluate one overall-v5 method, with TEST reserved before inference.

Commands are separate: freeze (metadata only), primary (TFLite by default),
confirm (the other bound backend), and regression (cached old baseline only).
No command trains, recalibrates thresholds, selects a TTA policy, or retries an
interrupted stage. All outputs are new immutable files in the new experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from word_overall_v5_protocol import VERSION, GROUPS, metrics, validation_rank, read, sha, verify_hashes, write_new
from word_overall_v5_runtime import DEFAULT_RUNTIME, get_runtime, get_views

METHOD_VERSION = 'overall-v5-final-method-1'
OLD_REGRESSION_SHA256 = 'f35716eec4e5dbbaf2ccfe68e76ef096f9a0a8ac00faf5bbd1cb544b9fdc5fb3'
OLD_BASELINE_SHA256 = 'cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_choices(selection, config, policy, exported, checkpoint_sha, tflite_sha, validation_sha, method_spec):
    runtime_name = policy.get('runtime_module', DEFAULT_RUNTIME)
    views = get_views(runtime_name)
    if selection.get('selection_scope') != 'validation_only' or selection.get('test_opened') is not False:
        raise ValueError('Checkpoint must be selected using validation only')
    if selection.get('sha256') != checkpoint_sha:
        raise ValueError('Checkpoint differs from validation selection')
    if (policy.get('selection_scope') != 'validation_only' or policy.get('test_opened') is not False
            or policy.get('old_or_fresh_test_used') is not False
            or policy.get('checkpoint_sha256') != checkpoint_sha or policy.get('validation_sha256') != validation_sha):
        raise ValueError('TTA policy must bind this checkpoint and development validation only')
    if (canonical(policy.get('method')) != canonical(method_spec)
            or policy.get('selected_validation_policy') not in method_spec['policies']):
        raise ValueError('TTA policy specification differs from the frozen implementation')
    if config.get('thresholds') != policy.get('thresholds') or exported.get('thresholds') != config.get('thresholds'):
        raise ValueError('Detector, policy, and export thresholds must remain identical')
    if (exported.get('status') != 'passed' or exported.get('allclose') is not True
            or exported.get('checkpoint_sha256') != checkpoint_sha or exported.get('tflite_sha256') != tflite_sha
            or exported.get('validation_sha256') != validation_sha or exported.get('scope') != 'development_validation_only'
            or exported.get('test_opened') is not False or exported.get('samples') != 540
            or exported.get('runtime_module', DEFAULT_RUNTIME) != runtime_name
            or exported.get('views') != list(views)
            or exported.get('inference_count') != 540*len(views) or exported.get('pipeline_decision_mismatches') != []
            or canonical(exported.get('method')) != canonical(method_spec)):
        raise ValueError('TFLite requires passed full-view development-validation export parity')


def validate_policy_results(policy, validation_rows):
    """Recount the recorded development decisions before trusting policy choice."""
    names = list(policy['method']['policies'])
    if set(policy['variants']) != set(names):
        raise ValueError('TTA validation must record every predefined policy')
    ranks = {}
    for name in names:
        variant = policy['variants'][name]
        decisions = variant['decisions']
        if len(decisions) != len(validation_rows):
            raise ValueError('TTA validation decisions changed denominator')
        for decision, row in zip(decisions, validation_rows):
            if (decision['source'] != row['source'] or decision['group'] != row['symbol_group']
                    or decision['expected'] != row['expected']):
                raise ValueError('TTA validation decision identity or truth differs')
        recomputed = metrics(decisions, require_primary=True)
        if canonical(recomputed) != canonical(variant['metrics']):
            raise ValueError('Recorded TTA validation metric disagrees with decisions')
        ranks[name] = validation_rank(decisions)
    if policy['selected_validation_policy'] != max(names, key=lambda name: ranks[name]):
        raise ValueError('TTA policy was not selected by frozen overall validation rank')


def load_cached_regression(path, thresholds):
    """Only previously recorded baseline decisions, never baseline inference."""
    verify_hashes({path: OLD_REGRESSION_SHA256})
    value = read(path)
    if value.get('baseline_sha256') != OLD_BASELINE_SHA256 or value.get('thresholds') != thresholds or value.get('decoder_version') != 'six-dot-id-v4-experimental':
        raise ValueError('Cached old baseline or decoder binding differs')
    verify_hashes(value['input_sha256'])
    result = {}
    for name, count in (('external607', 607), ('legacy405', 405), ('negative44', 44)):
        rows = value['results'][name]['rows']
        if len(rows) != count or len({(row['source'], row['image']) for row in rows}) != count:
            raise ValueError('Cached old regression denominator changed')
        for row in rows:
            if not isinstance(row['baseline'].get('text'), str) or not isinstance(row.get('expected'), str):
                raise ValueError('Cached baseline decision is incomplete')
        result[name] = rows
    return result


def freeze(args):
    experiment = args.experiment.resolve()
    if (experiment/'test-evaluation-reservation.json').exists() or (experiment/'final-method-selection.json').exists():
        raise ValueError('Final method already frozen or TEST reserved')
    checkpoint, tflite = args.checkpoint.resolve(), args.tflite.resolve()
    selection_path = (args.selection or checkpoint.parent/'selection.json').resolve()
    config_path = (args.training_config or checkpoint.parent/'training-config.json').resolve()
    policy_path, parity_path = args.policy_report.resolve(), args.export_parity.resolve()
    protocol, corpus = read(experiment/'protocol.json'), read(experiment/'corpus-audit.json')
    if protocol['version'] != VERSION:
        raise ValueError('Expected the frozen overall exact-text protocol')
    verify_hashes(protocol['input_sha256'])
    verify_hashes(protocol['source_sha256'])
    verify_hashes(corpus['artifact_sha256'])
    verify_hashes(corpus['source_sha256'])
    if read(experiment/'source-separation-audit.json')['status'] != 'passed':
        raise ValueError('Fresh source separation audit must pass')
    selection, config, policy, parity = map(read, (selection_path, config_path, policy_path, parity_path))
    runtime_name = policy.get('runtime_module', DEFAULT_RUNTIME)
    runtime = get_runtime(runtime_name)
    verify_hashes({config_path: selection['training_config_sha256']})
    verify_hashes(config['input_sha256'])
    verify_hashes(parity['input_sha256'])
    validation_manifest = Path(protocol['development_files']['validation-symbols.json'])
    validate_choices(selection, config, policy, parity, sha(checkpoint), sha(tflite), sha(validation_manifest), runtime.method_specification())
    validate_policy_results(policy, read(validation_manifest)['records'])
    tta_source = Path(__file__).with_name(runtime_name+'.py')
    if policy.get('source_sha256') != sha(tta_source):
        raise ValueError('TTA validation report source differs from final inference implementation')
    policy_plan_path = policy_path.with_suffix('.plan.json')
    verify_hashes({policy_plan_path: policy['plan_sha256']})
    policy_plan = read(policy_plan_path)
    verify_hashes(policy_plan['input_sha256'])
    if (policy_plan.get('runtime_module', DEFAULT_RUNTIME) != runtime_name
            or policy_plan['thresholds'] != policy['thresholds'] or canonical(policy_plan['method']) != canonical(policy['method'])):
        raise ValueError('TTA validation execution differs from its frozen plan')
    # Preserve validation policy provenance, including any plan/image bindings.
    for binding in ('input_sha256', 'source_sha256'):
        if isinstance(policy.get(binding), dict) and all(Path(path).is_absolute() for path in policy[binding]):
            verify_hashes(policy[binding])
    regression_path = args.regression_reference.resolve()
    regression = load_cached_regression(regression_path, config['thresholds'])
    files = [checkpoint, tflite, selection_path, config_path, policy_path, policy_plan_path, parity_path, regression_path,
        experiment/'protocol.json', experiment/'test-source-reservation.json', experiment/'test-download-manifest.json',
        experiment/'corpus-audit.json', experiment/'source-separation-audit.json',
        experiment/'test-symbols.json', experiment/'test-image-hashes.json',
        experiment/'native-context.json', experiment/'native-context-image-hashes.json']
    inputs = {str(path): sha(path) for path in files}
    # Bind all final test pixels before exposure without opening their content.
    image_bindings = {}
    for name in ('test-image-hashes.json', 'native-context-image-hashes.json'):
        image_bindings.update(read(experiment/name))
    for rows in regression.values():
        for row in rows:
            previous = image_bindings.setdefault(row['image'], row['image_sha256'])
            if previous != row['image_sha256']:
                raise ValueError('Conflicting frozen image hashes')
    scripts = Path(__file__).resolve().parent
    snapshot = experiment/'final-method-source'
    snapshot.mkdir(exist_ok=False)
    source_hashes = {}
    for path in scripts.glob('*.py'):
        shutil.copy2(path, snapshot/path.name)
        source_hashes[str(path)] = sha(path)
    frozen = dict(version=METHOD_VERSION, protocol_version=VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(), selection_scope='validation_only', test_opened=False,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), tflite=str(tflite), tflite_sha256=sha(tflite),
        primary_engine=args.primary_engine, selected_policy=policy['selected_validation_policy'],
        runtime_module=runtime_name, tta_method=runtime.method_specification(), thresholds=config['thresholds'],
        checkpoint_selection=str(selection_path), tta_validation_report=str(policy_path), export_parity_report=str(parity_path),
        primary_manifest=str(experiment/'test-symbols.json'), native_manifest=str(experiment/'native-context.json'),
        regression_reference=str(regression_path), regression_reference_sha256=OLD_REGRESSION_SHA256,
        input_sha256=inputs, image_sha256=image_bindings, source_sha256=source_hashes,
        source_snapshot_sha256={str(snapshot/Path(path).name): digest for path, digest in source_hashes.items()},
        primary_target='overall exact text >=0.80 on all500positive examples; rejected positives count wrong',
        quality_flags='Accepted precision95, invalid rejection, native context, and historical regression are reported separately',
        scope='fresh_source_disjoint_recomposed_public_flatbed_scans_not_physical_phone_camera',
        physical_camera_validated=False, native_runtime_validated=False,
        after_exposure='No checkpoint, TTA, thresholds, labels, or image choices may change; optional alternate backend only confirms the same bound method')
    write_new(experiment/'final-method-selection.json', frozen)
    print('METHOD_FROZEN '+str(experiment/'final-method-selection.json'), flush=True)


def verify_method(experiment):
    path = Path(experiment)/'final-method-selection.json'
    method = read(path)
    if method.get('version') != METHOD_VERSION or method.get('selection_scope') != 'validation_only' or method.get('test_opened') is not False:
        raise ValueError('Expected immutable validation-selected final method')
    verify_hashes(method['input_sha256'])
    verify_hashes(method['source_sha256'])
    verify_hashes(method['source_snapshot_sha256'])
    validate_method_bindings(method, experiment)
    runtime = get_runtime(method.get('runtime_module', DEFAULT_RUNTIME))
    if canonical(method['tta_method']) != canonical(runtime.method_specification()):
        raise ValueError('TTA implementation differs after method selection')
    return method


def validate_method_bindings(method, experiment):
    """Declared inference paths must refer to the files actually hash-bound."""
    experiment = Path(experiment).resolve()
    for field in ('checkpoint', 'tflite'):
        if method['input_sha256'].get(str(Path(method[field]).resolve())) != method[field+'_sha256']:
            raise ValueError('Declared artifact path/SHA is absent from frozen input bindings: '+field)
    for field, filename in (('primary_manifest', 'test-symbols.json'), ('native_manifest', 'native-context.json')):
        if Path(method[field]).resolve() != experiment/filename or str(experiment/filename) not in method['input_sha256']:
            raise ValueError('Final inference must use the canonical frozen holdout: '+field)
    policy_path = str(Path(method['tta_validation_report']).resolve())
    if policy_path not in method['input_sha256']:
        raise ValueError('Selected TTA validation report is not hash-bound')
    policy = read(policy_path)
    if (policy['checkpoint_sha256'] != method['checkpoint_sha256'] or policy['thresholds'] != method['thresholds']
            or policy['selected_validation_policy'] != method['selected_policy']
            or policy.get('runtime_module', DEFAULT_RUNTIME) != method.get('runtime_module', DEFAULT_RUNTIME)):
        raise ValueError('Declared method differs from its bound validation policy')
    for name in ('test-image-hashes.json', 'native-context-image-hashes.json'):
        path = experiment/name
        if str(path) not in method['input_sha256']:
            raise ValueError('Final image index is not hash-bound')
        for image, digest in read(path).items():
            if method['image_sha256'].get(image) != digest:
                raise ValueError('Declared image hash differs from frozen holdout index')


def reserve_stage(experiment, method, stage, engine):
    experiment = Path(experiment)
    method_path = experiment/'final-method-selection.json'
    reservation = experiment/'test-evaluation-reservation.json'
    identity = dict(method_sha256=sha(method_path), checkpoint_sha256=method['checkpoint_sha256'],
                    tflite_sha256=method['tflite_sha256'], primary_engine=method['primary_engine'], selected_policy=method['selected_policy'])
    if stage == 'primary':
        if engine != method['primary_engine']:
            raise ValueError('Primary evaluation engine was fixed before TEST exposure')
        write_new(reservation, dict(identity, created_utc=datetime.now(timezone.utc).isoformat(),
            policy='One final method; interruption remains recorded; no tuning or implicit retries'))
    else:
        if not reservation.is_file() or not (experiment/'final-test-evaluation.json').is_file():
            raise ValueError('Primary final evaluation must finish before confirmation/regression')
        old = read(reservation)
        if any(old.get(key) != value for key, value in identity.items()):
            raise ValueError('Stage differs from the original final TEST reservation')
        result = read(experiment/'final-test-evaluation.json')
        if result['method_sha256'] != identity['method_sha256']:
            raise ValueError('Completed primary evaluation belongs to a different method')
        if stage == 'confirm' and engine == method['primary_engine']:
            raise ValueError('Confirmation must use the other bound inference engine')
    started = experiment/f'{stage}-{engine}-started.json'
    write_new(started, dict(identity, stage=stage, engine=engine, created_utc=datetime.now(timezone.utc).isoformat()))
    return started


def make_predictor(engine, method, device='cpu'):
    if engine == 'tflite':
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=method['tflite'], num_threads=4)
        interpreter.allocate_tensors()
        incoming, outgoing = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        if (list(incoming['shape']) != [1,192,480,3] or list(outgoing['shape']) != [1,48,120,11]
                or incoming['dtype'] != np.float32 or outgoing['dtype'] != np.float32):
            raise ValueError('TFLite runtime tensor interface differs')
        def predict(images):
            outputs = []
            for image in images:
                interpreter.set_tensor(incoming['index'], image[None].astype(np.float32))
                interpreter.invoke()
                outputs.append(interpreter.get_tensor(outgoing['index'])[0].copy())
            return np.stack(outputs)
        return predict, dict(engine='tensorflow_lite', version=tf.__version__, threads=4, device='cpu')
    if engine != 'torch':
        raise ValueError('Only bound Torch and TFLite inference engines are allowed')
    import torch
    from word_torch_port import load_keras_checkpoint
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_keras_checkpoint(method['checkpoint']).to(device).eval()
    def predict(images):
        with torch.no_grad():
            return model(torch.from_numpy(images).to(device)).cpu().numpy()
    return predict, dict(engine='native_torch_keras_archive', version=torch.__version__, device=device, threads=4)


def pixels_for(rows, image_hashes, *, allow_resize=False):
    pixels = []
    for row in rows:
        path = row['image']
        if path not in image_hashes:
            raise ValueError('Image is absent from frozen final method')
        verify_hashes({path: image_hashes[path]})
        with Image.open(path) as source:
            image = source.convert('RGB')
            if image.size != (480,192):
                if not allow_resize:
                    raise ValueError('Prepared final TEST dimensions changed')
                image = image.resize((480,192), Image.Resampling.BILINEAR)
            pixels.append(np.asarray(image, np.float32)/255.)
    return np.stack(pixels)


def semantic_signature(decision):
    return [decision['text'], decision['reason'],
            [[None if cell.get('mask') is None else int(cell['mask']),
              sum(int(dot >= .5) << bit for bit, dot in enumerate(cell['dots'])) if 'dots' in cell else None,
              cell.get('role'), cell.get('output')]
             for cell in decision['cells']]]


def infer_rows(rows, predict, method, *, allow_resize=False):
    runtime = get_runtime(method.get('runtime_module', DEFAULT_RUNTIME))
    from word_decode_id_v4 import decode
    if not rows or len({(row['source'], row['image']) for row in rows}) != len(rows):
        raise ValueError('Evaluation requires unique frozen row identities')
    pixels = pixels_for(rows, method['image_sha256'], allow_resize=allow_resize)
    result = runtime.infer_decisions(predict, pixels, method['thresholds'],
        [row.get('selection') for row in rows], policy=method['selected_policy'], batch_size=8, decoder=decode)
    if len(result['decisions']) != len(rows) or len(result['routes']) != len(rows):
        raise ValueError('Inference must return exactly one decision per example')
    decisions = []
    for row, decision, route in zip(rows, result['decisions'], result['routes']):
        if not isinstance(decision.get('text'), str) or not isinstance(decision.get('reason'), str):
            raise ValueError('Malformed final text decision')
        if any(len(cell.get('dots', [])) != 6 for cell in decision['cells']):
            raise ValueError('Every selected cell must retain all six physical dot probabilities')
        decisions.append(dict(source=row['source'], group=row.get('symbol_group', 'rejection' if not row['expected'] else 'lowercase'),
            expected=row['expected'], predicted=decision['text'], reason=decision['reason'], route=route,
            expected_cells=len(row.get('masks', [])), detected_cells=len(decision['cells']),
            semantic_signature=semantic_signature(decision), image_sha256=method['image_sha256'][row['image']]))
    return decisions, result['latency']


def summarize_decisions(decisions):
    positives = [row for row in decisions if row['expected']]
    negatives = [row for row in decisions if not row['expected']]
    correct = sum(row['expected'] == row['predicted'] for row in positives)
    accepted = sum(bool(row['predicted']) for row in positives)
    return dict(total=len(decisions), positive_total=len(positives), exact_correct=correct,
        overall_exact_accuracy=correct/len(positives) if positives else None,
        accepted_positive_total=accepted, accepted_precision=correct/accepted if accepted else None,
        invalid_total=len(negatives), invalid_false_accepts=sum(bool(row['predicted']) for row in negatives))


def report_for(decisions, *, primary):
    summary = metrics(decisions, require_primary=True) if primary else summarize_decisions(decisions)
    summary['groups'] = {group: summarize_decisions([row for row in decisions if row['group'] == group])
                         for group in sorted({row['group'] for row in decisions})}
    summary['by_printed_length'] = {str(length): summarize_decisions([row for row in decisions if len(row['expected']) == length])
                                   for length in sorted({len(row['expected']) for row in decisions if row['expected']})}
    summary['by_punctuation'] = {mark: summarize_decisions([row for row in decisions if mark in row['expected']]) for mark in '.,?!:;-()'}
    summary['rejection_reasons'] = dict(Counter(row['reason'] for row in decisions if row['expected'] and not row['predicted']))
    summary['routing_counts'] = dict(Counter(row['route'] for row in decisions))
    return summary


def decision_parity(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError('Both inference engines must cover the same complete denominator')
    semantic, routes = [], []
    fields = ('source', 'group', 'expected', 'predicted', 'reason', 'expected_cells', 'detected_cells', 'semantic_signature', 'image_sha256')
    for index, (left, right) in enumerate(zip(reference, candidate)):
        changed = [field for field in fields if left.get(field) != right.get(field)]
        if changed:
            semantic.append(dict(index=index, source=left.get('source'), changed_fields=changed))
        if left.get('route') != right.get('route'):
            routes.append(dict(index=index, source=left.get('source'), primary_route=left.get('route'), confirmation_route=right.get('route')))
    return dict(samples=len(reference), status='passed' if not semantic and not routes else 'failed',
                semantic_mismatches=semantic, routing_differences=routes)


def evaluate(args):
    experiment = args.experiment.resolve()
    method = verify_method(experiment)
    engine = method['primary_engine'] if args.command == 'primary' else ('torch' if method['primary_engine'] == 'tflite' else 'tflite')
    started_path = reserve_stage(experiment, method, args.command, engine)
    # The reservation is written before reading even one evaluation image or
    # constructing an inference engine. Labels never enter TTA routing.
    predictor, runtime = make_predictor(engine, method, args.device)
    started = time.perf_counter()
    results = {}
    for name, manifest_path in (('primary', method['primary_manifest']), ('native_context', method['native_manifest'])):
        rows = read(manifest_path)['records']
        decisions, latency = infer_rows(rows, predictor, method)
        results[name] = dict(summary=report_for(decisions, primary=name == 'primary'), decisions=decisions, latency=latency,
            manifest_sha256=sha(manifest_path), scope='balanced_stress_primary_500' if name == 'primary' else 'separate_original_mask_native_context_diagnostic')
        print('EVALUATED '+name+' '+json.dumps(results[name]['summary']), flush=True)
    verify_method(experiment)
    value = dict(version=METHOD_VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        method_sha256=sha(experiment/'final-method-selection.json'), reservation_sha256=sha(experiment/'test-evaluation-reservation.json'),
        stage_started_sha256=sha(started_path), checkpoint_sha256=method['checkpoint_sha256'],
        tflite_sha256=method['tflite_sha256'], selected_policy=method['selected_policy'], thresholds=method['thresholds'],
        runtime=runtime, elapsed_seconds=time.perf_counter()-started, results=results,
        physical_camera_validated=False, native_runtime_validated=False, test_used_for_selection=False)
    if args.command == 'primary':
        value['target80_met'] = results['primary']['summary']['target80_met']
        write_new(experiment/'final-test-evaluation.json', value)
    else:
        reference = read(experiment/'final-test-evaluation.json')
        value['parity'] = {name: decision_parity(reference['results'][name]['decisions'], results[name]['decisions']) for name in results}
        value['status'] = 'passed' if all(item['status'] == 'passed' for item in value['parity'].values()) else 'failed'
        write_new(experiment/f'final-test-{engine}-confirmation.json', value)


def regression(args):
    experiment = args.experiment.resolve()
    method = verify_method(experiment)
    engine = 'tflite'  # Historical comparison must measure the actual exported candidate artifact.
    started_path = reserve_stage(experiment, method, 'regression', engine)
    cached = load_cached_regression(method['regression_reference'], method['thresholds'])
    predictor, runtime = make_predictor(engine, method, args.device)
    results = {}
    for name, archived in cached.items():
        rows = [dict(image=row['image'], source=row['source'], expected=row['expected'],
            selection=row.get('selection'), symbol_group='lowercase' if row['expected'] else 'rejection') for row in archived]
        predicted, latency = infer_rows(rows, predictor, method, allow_resize=True)
        baseline = [dict(source=row['source'], expected=row['expected'], predicted=row['baseline']['text']) for row in archived]
        before, after = summarize_decisions(baseline), summarize_decisions(predicted)
        lost = sum(left['predicted'] == left['expected'] and right['predicted'] != left['expected'] for left,right in zip(baseline,predicted) if left['expected'])
        gained = sum(left['predicted'] != left['expected'] and right['predicted'] == left['expected'] for left,right in zip(baseline,predicted) if left['expected'])
        flags = dict(exact_regressed=after['exact_correct'] < before['exact_correct'],
            precision_regressed=before['accepted_precision'] is not None and (after['accepted_precision'] is None or after['accepted_precision'] < before['accepted_precision']),
            invalid_false_accepts_regressed=after['invalid_false_accepts'] > before['invalid_false_accepts'])
        results[name] = dict(baseline=before, candidate=after, lost=lost, gained=gained,
            regression_observed=any(flags.values()), flags=flags, decisions=predicted, latency=latency)
        print('REGRESSION '+name+' '+json.dumps(dict(baseline=before, candidate=after, flags=flags)), flush=True)
    verify_method(experiment)
    write_new(experiment/'historical-regression.json', dict(version=METHOD_VERSION,
        method_sha256=sha(experiment/'final-method-selection.json'), stage_started_sha256=sha(started_path),
        cached_baseline_report_sha256=OLD_REGRESSION_SHA256, baseline_checkpoint_sha256=OLD_BASELINE_SHA256,
        baseline_inference_performed=False, runtime=runtime, results=results,
        scope='separate_previously_observed_public_and_legacy_regression; never changes final method or primary denominator'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    binding = commands.add_parser('freeze')
    binding.add_argument('--experiment', type=Path, required=True)
    binding.add_argument('--checkpoint', type=Path, required=True)
    binding.add_argument('--selection', type=Path)
    binding.add_argument('--training-config', type=Path)
    binding.add_argument('--policy-report', type=Path, required=True)
    binding.add_argument('--tflite', type=Path, required=True)
    binding.add_argument('--export-parity', type=Path, required=True)
    binding.add_argument('--primary-engine', choices=('tflite', 'torch'), default='tflite')
    binding.add_argument('--regression-reference', type=Path, default=Path('C:/bm-symbols-20260908/accuracy80-20260910/run-dots/public-regression-v4.json'))
    for command in ('primary', 'confirm', 'regression'):
        task = commands.add_parser(command)
        task.add_argument('--experiment', type=Path, required=True)
        task.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.command == 'freeze': freeze(args)
    elif args.command == 'regression': regression(args)
    else: evaluate(args)


if __name__ == '__main__': main()
