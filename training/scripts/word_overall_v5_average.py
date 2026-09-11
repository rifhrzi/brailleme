"""One fixed CPU candidate: equal-weight average of the top three saved epochs.

Epoch ranking uses identity development-validation exact text, then precision,
then fewer invalid accepts, with earlier epoch breaking ties. Baseline is
excluded. This command performs no inference and does not pick a cross-run
winner. Evaluate every fixed TTA policy on the new candidate afterwards.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil

from word_overall_v5_protocol import metrics, read, sha, validation_rank, verify_hashes, write_new
from word_overall_v5_tta import assert_development_open

METHOD = 'equal-float64-mean-top-three-identity-validation-epochs-v1'


def top_three(entries):
    candidates = [entry for entry in entries if entry['epoch'] > 0]
    if len(candidates) < 3 or len({entry['epoch'] for entry in candidates}) != len(candidates):
        raise ValueError('Need at least three distinct positive saved epochs; baseline is excluded')
    for entry in candidates:
        if (not isinstance(entry['epoch'], int) or len(entry['rank']) != 3
                or not all(math.isfinite(value) for value in entry['rank'])):
            raise ValueError('Epoch identity or three-component validation rank is malformed')
    return sorted(candidates, key=lambda entry: (*entry['rank'], -entry['epoch']), reverse=True)[:3]


def average_models(models):
    """Update the first CPU model in memory; never change parent files.

    Trainable parameters receive a float64 arithmetic mean cast back to their
    original dtype. Every frozen parameter and buffer must agree bit for bit.
    This includes all frozen BN affine values and statistics in the port.
    """
    import torch
    if len(models) != 3:
        raise ValueError('The fixed method requires exactly three models')
    if any(model.architecture != models[0].architecture for model in models[1:]):
        raise ValueError('Checkpoint source architectures differ')
    states = [model.state_dict() for model in models]
    parameters = [dict(model.named_parameters()) for model in models]
    keys = list(states[0])
    if any(list(state) != keys for state in states[1:]):
        raise ValueError('Checkpoint state keys or ordering differ')
    if any(set(value) != set(parameters[0]) for value in parameters[1:]):
        raise ValueError('Checkpoint parameter identities differ')
    trainable = {name for name, value in parameters[0].items() if value.requires_grad}
    if not trainable or any({name for name, value in group.items() if value.requires_grad} != trainable
                            for group in parameters[1:]):
        raise ValueError('Checkpoint trainability differs or has no trainable weights')
    # Validate every tensor before mutating the destination model.
    for name in keys:
        values = [state[name] for state in states]
        first = values[0]
        if any(value.shape != first.shape or value.dtype != first.dtype for value in values[1:]):
            raise ValueError('Checkpoint tensor shape/dtype differs: '+name)
        if any(value.device.type != 'cpu' or not torch.isfinite(value).all() for value in values):
            raise ValueError('Averaging accepts only finite CPU tensors: '+name)
        if name in trainable:
            if not first.is_floating_point():
                raise ValueError('Trainable parameter must be floating point: '+name)
        else:
            bits = first.reshape(-1).contiguous().view(torch.uint8)
            if any(not torch.equal(bits, value.reshape(-1).contiguous().view(torch.uint8)) for value in values[1:]):
                raise ValueError('Frozen parameter or buffer differs: '+name)
    with torch.no_grad():
        for name in keys:
            if name in trainable:
                mean = torch.stack([state[name].to(torch.float64) for state in states]).mean(0)
                result = mean.to(states[0][name].dtype)
                if not torch.isfinite(result).all():
                    raise ValueError('Averaged parameter is nonfinite: '+name)
                states[0][name].copy_(result)
    return models[0], dict(averaged_parameter_tensors=len(trainable),
        averaged_parameter_elements=sum(states[0][name].numel() for name in trainable),
        frozen_parameter_tensors=len(parameters[0])-len(trainable),
        buffer_tensors=len(keys)-len(parameters[0]), frozen_values_bitwise_equal=True,
        accumulator_dtype='float64', output_dtypes=sorted({str(states[0][name].dtype) for name in trainable}))


def inspect_epochs(source_run, validation_rows):
    history = read(source_run/'history.json')
    completed = read(source_run/'selection.json')['epochs_completed']
    if len(history) != completed or [row['epoch'] for row in history] != list(range(1, completed+1)):
        raise ValueError('Completed epoch history must be contiguous and agree with finished run')
    entries, bindings = [], {}
    for item in history:
        epoch = item['epoch']
        path = source_run/f'epoch{epoch:02d}.keras'
        report_path = source_run/f'epoch{epoch:02d}-validation.json'
        report = read(report_path)
        decisions = report['decisions']
        if len(decisions) != len(validation_rows):
            raise ValueError('Saved epoch validation denominator changed')
        for decision, row in zip(decisions, validation_rows):
            if (decision['source'] != row['source'] or decision['group'] != row['symbol_group']
                    or decision['expected'] != row['expected']):
                raise ValueError('Saved epoch validation identities/truth changed')
        value = metrics(decisions, require_primary=True)
        expected = dict(correct=value['exact_correct'], total=value['positive_total'],
            exact_accuracy=value['overall_exact_accuracy'], accepted=value['accepted_positive_total'],
            accepted_precision=value['accepted_precision'] or 0., invalid_false_accepts=value['invalid_false_accepts'],
            lowest_group_exact=min(group['correct']/group['total'] for group in value['groups'].values()))
        if report['epoch'] != epoch or item['overall'] != expected or report['overall'] != expected:
            raise ValueError('Saved identity-validation metric disagrees with its decisions')
        digest = sha(path)
        if digest != report['checkpoint_sha256'] or digest != item['checkpoint_sha256']:
            raise ValueError('Saved epoch checkpoint changed')
        bindings[str(path)], bindings[str(report_path)] = digest, sha(report_path)
        entries.append(dict(epoch=epoch, checkpoint=str(path), checkpoint_sha256=digest,
            validation_report=str(report_path), validation_report_sha256=sha(report_path),
            rank=validation_rank(decisions), identity_validation=value))
    return entries, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    args = parser.parse_args()
    experiment, source_run = args.experiment.resolve(), args.source_run.resolve()
    run = experiment/'run-average'
    if source_run.parent != experiment or run.exists():
        raise ValueError('Use a finished direct experiment child and a new run-average')
    assert_development_open(experiment)
    protocol_path = experiment/'protocol.json'
    config_path, selection_path = source_run/'training-config.json', source_run/'selection.json'
    protocol, config, selection = read(protocol_path), read(config_path), read(selection_path)
    if protocol['version'] != 'overall-exact-text-v5-20260911':
        raise ValueError('Expected the frozen overall-v5 protocol')
    if selection.get('selection_scope') != 'validation_only' or selection.get('test_opened') is not False:
        raise ValueError('Source run must have completed validation-only selection')
    verify_hashes({config_path: selection['training_config_sha256'], source_run/'best-word.keras': selection['sha256']})
    manifest = Path(protocol['development_files']['validation-symbols.json']).resolve()
    image_index = Path(protocol['development_files']['validation-image-hashes.json']).resolve()
    for path in (manifest, image_index):
        if sha(path) != protocol['input_sha256'][str(path)] or sha(path) != config['input_sha256'][str(path)]:
            raise ValueError('Source run and protocol validation bindings differ')
    document = read(manifest)
    if document['split'] != 'validation':
        raise ValueError('Only identity development-validation may rank saved epochs')
    entries, inputs = inspect_epochs(source_run, document['records'])
    chosen = top_three(entries)
    inputs.update({str(path): sha(path) for path in
                   (protocol_path, config_path, selection_path, source_run/'history.json', manifest, image_index)})
    inputs.update(config['input_sha256'])
    scripts = Path(__file__).resolve().parent
    source_bindings = {str(source_run/'source'/name): digest for name, digest in config['source_sha256'].items()}
    dependencies = ('word_torch_port.py', 'word_data.py', 'word_decode_id_v4.py', 'word_symbols_id_v4.py',
                    'word_symbols.py', 'word_anchor_zoom.py', 'benchmark_angelina.py', 'word_overall_v5_protocol.py')
    source_bindings.update({str(scripts/name): config['source_sha256'][name] for name in dependencies})
    source_bindings[str(Path(__file__).resolve())] = sha(__file__)
    verify_hashes(inputs); verify_hashes(source_bindings)
    import torch
    from word_torch_port import load_keras_checkpoint, save_keras_checkpoint
    torch.set_num_threads(2)
    models = [load_keras_checkpoint(entry['checkpoint']) for entry in chosen]
    averaged, tensor_audit = average_models(models)
    verify_hashes(inputs); verify_hashes(source_bindings); assert_development_open(experiment)
    # This is a training artifact directory, not an Android build/staging copy.
    run.mkdir()
    snapshot = run/'source'; snapshot.mkdir()
    for name in config['source_sha256']:
        shutil.copy2(source_run/'source'/name, snapshot/name)
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    destination = run/'best-word.keras'
    save_keras_checkpoint(averaged, chosen[0]['checkpoint'], destination)
    reloaded = load_keras_checkpoint(destination)
    if reloaded.architecture != averaged.architecture or any(
            not torch.equal(value, reloaded.state_dict()[name]) for name, value in averaged.state_dict().items()):
        raise ValueError('Saved averaged checkpoint roundtrip changed architecture or tensors')
    created = datetime.now(timezone.utc).isoformat()
    new_config = dict(created_utc=created, arguments=dict(experiment=str(experiment), source_run=str(source_run)),
        method=METHOD, thresholds=config['thresholds'], input_sha256=inputs,
        source_sha256={path.name: sha(path) for path in snapshot.glob('*.py')},
        averaging_source_sha256=source_bindings, target_overall_exact_accuracy=.8, primary_denominator=500,
        chosen_epochs=chosen, tensor_audit=tensor_audit, baseline_excluded=True,
        ranking='Overall identity exact text, accepted precision, fewer invalid false accepts; earlier epoch breaks ties',
        weights=[1/3]*3, candidate_only=True, cross_run_selection_performed=False,
        batchnorm_recalibration_performed=False, model_inference_performed=False,
        test_opened=False, physical_camera_validated=False, build_folders_created=0)
    write_new(run/'training-config.json', new_config)
    write_new(run/'selection.json', dict(epoch=None, path=str(destination), sha256=sha(destination), rank=None,
        selection_scope='validation_only', test_opened=False, training_config_sha256=sha(run/'training-config.json'),
        candidate_method=METHOD, source_epochs=[entry['epoch'] for entry in chosen],
        candidate_only=True, candidate_evaluated=False, cross_run_selection_performed=False,
        selection_meaning='Fixed average of validation-ranked saved epochs; no claim that this candidate beats any run',
        next_evaluation='Run word_overall_v5_tta.py for all three policies, saving run-average/candidate-validation.json regardless of identity result'))
    verify_hashes(inputs); verify_hashes(source_bindings); assert_development_open(experiment)
    print('AVERAGED_CANDIDATE '+str(destination)+' epochs='+str([entry['epoch'] for entry in chosen]), flush=True)


if __name__ == '__main__':
    main()
