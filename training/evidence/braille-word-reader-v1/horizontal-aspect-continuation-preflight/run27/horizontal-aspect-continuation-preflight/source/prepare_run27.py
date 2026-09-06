"""Freeze a weight-only continuation of the best new horizontal-aspect checkpoint."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

WORK = Path(__file__).resolve().parent
RUN = WORK / 'run27'
PARENT_SHA = '6113867b21316669169391fa7216f9d1b9bf147a13c153889ff6de1c26e9a4ea'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': digest(path)}


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--audit', action='append', default=[], type=Path)
    args = parser.parse_args()
    if (RUN / 'training-config.json').exists() or (RUN / 'horizontal-aspect-continuation-protocol.json').exists():
        raise SystemExit('Run27 is already frozen or started; preserve it.')
    RUN.mkdir(exist_ok=True)
    (RUN / 'source').mkdir(exist_ok=True)
    parent_config_path = WORK / 'run26/training-config.json'
    parent_lineage_path = WORK / 'run26/sampling-lineage-final.json'
    parent = WORK / 'run26/retained-epochs/epoch-10.keras'
    config = json.loads(parent_config_path.read_text())
    lineage = json.loads(parent_lineage_path.read_text())
    assert digest(parent) == PARENT_SHA
    assert lineage['status'] == 'passed' and lineage['eligible_retained_epochs'] == [8, 9, 10]
    row = lineage['actual']['observed_epochs'][9]
    assert row['epoch'] == 10 and row['checkpoint_sha256'] == PARENT_SHA
    assert row['all_positive_records_seen_in_this_run'] and row['completed_batches'] == 1037
    assert lineage['actual']['config_sha256'] == digest(parent_config_path)
    assert len(config['source_sha256']) == 22
    for name, expected in config['source_sha256'].items():
        original = Path('training/scripts') / name
        assert digest(original) == digest(WORK / 'run26/source' / name) == expected
        snapshot = RUN / 'source' / name
        if snapshot.exists():
            assert digest(snapshot) == expected
        else:
            snapshot.write_bytes(original.read_bytes())
    for name, expected in config['data_sha256'].items():
        assert digest(WORK / 'data' / name) == expected
    assert config['horizontal_aspect_augmentation'] is True
    assert config['parameters'] == 829307
    assert config['batch_normalization']['layers'] == 52
    assert config['batch_normalization']['trainable_affine_layers'] == 0
    for key in ('geometry_conditioned_residual', 'batchnorm_affine', 'fine_detail_branch'):
        assert config[key]['enabled'] is False
    control = WORK / 'selected-v22/best-word.keras'
    control_validation = WORK / 'selected-v22/selected-validation.json'
    assert digest(control) == 'cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
    assert json.loads(control_validation.read_text())['checkpoint_sha256'] == digest(control)
    previous_protocol = WORK / 'run26/horizontal-aspect-protocol.json'
    plan = json.loads(previous_protocol.read_text())
    plan.pop('reviewed_plan_sha256', None)
    plan.pop('parent_selected_validation_sha256', None)
    plan.update(
        created_at_utc=datetime.now(timezone.utc).isoformat(), run=str(RUN),
        parent_checkpoint=str(parent), parent_checkpoint_sha256=PARENT_SHA,
        parent_training_config_sha256=digest(parent_config_path),
        parent_sampling_lineage_sha256=digest(parent_lineage_path),
        control_checkpoint=str(control), control_checkpoint_sha256=digest(control),
        control_selected_validation_sha256=digest(control_validation),
        selection='Unchanged selected-v22/v15 control first plus actual run27 eligible epochs8-10 on frozen405; weak-peaks-v2/group gates/tie rule unchanged. No external outputs used.',
        rationale='Run26 epoch10 improves held matched cells1518->1524, wrong masks8->7 and paired mean IoU+.042902, while losing7/gaining5 words. Training and validation loss still decline at epoch10 with unchanged2e-5 learning rate. Continue optimizing this same horizontal-aspect objective; do not select or activate the failed parent as the control.',
        preflight_audits=[ref(path) for path in args.audit],
        continuation={
            'kind': 'weight_only_same_sample_realizations',
            'parent_retained_epoch': 10,
            'parent_is_selected_control': False,
            'parent_training_config': ref(parent_config_path),
            'parent_sampling_lineage': ref(parent_lineage_path),
            'parent_sampling_history': ref(WORK / 'run26/sampling-history.json'),
            'parent_selection': ref(WORK / 'selected-v21/checkpoint-selection.json'),
            'parent_held_diagnosis': ref(WORK / 'run26/held-diagnosis/horizontal-aspect-comparison/comparison.json'),
            'original_initial_checkpoint_sha256': config['resume_sha256'],
            'optimizer': 'Reset deliberately: load_model(compile=False), then fresh Adam(lr2e-5,clipnorm10). No stored optimizer variables are restored.',
            'scheduler': 'Fresh ReduceLROnPlateau(val_loss,patience4,factor.5,min_lr1e-5), identical policy to run26.',
            'sample_realizations': 'Seed20260905, sampler cursor and dataset epoch restart exactly as run26. Repeat the same10epoch source order and image/point/illumination/width-factor draws; this is additional optimization, not new augmentation exposure.',
            'sampling_eligibility': 'Require this run to complete its own23600-positive-record cycle; only local epochs8/9/10 eligible. Parent coverage does not make early local epochs eligible.',
            'source_policy': 'All22 training source files and all data manifests/point metadata are byte-identical to run26.',
            'inherited_horizontal_aspect_protocol': ref(previous_protocol),
            'inherited_transform_audit': ref(WORK / 'run26/horizontal-aspect-integration-audit.json'),
            'inherited_source_bound_tests': ref(WORK / 'run26/preflight-tests.json'),
            'inherited_scope': 'The b1d974 integration audit and141 tests establish identical transform/source semantics and originally bind the v15 initializer. They are not rerun or relabeled as an audit of the new epoch10 parent. New independent parent/continuation audits establish that binding.',
        },
    )
    if not args.freeze:
        save(RUN / 'run27-plan.json', plan)
        print(json.dumps({'plan': str(RUN / 'run27-plan.json'), 'sources': 22}, indent=2))
        return
    if len(args.audit) < 2:
        raise SystemExit('Require separate actual-parent and continuation/protocol audit reports.')
    reviewed_path = RUN / 'run27-plan.json'
    reviewed = json.loads(reviewed_path.read_text())
    for key, expected in reviewed.items():
        if key not in ('created_at_utc', 'preflight_audits'):
            assert plan[key] == expected, ('Reviewed plan changed', key)
    for path in args.audit:
        assert json.loads(path.read_text())['status'] == 'passed'
    plan['reviewed_plan_sha256'] = digest(reviewed_path)
    shared = {key: plan[key] for key in (
        'created_at_utc', 'source_sha256', 'parent_checkpoint', 'parent_checkpoint_sha256',
        'horizontal_mirror_augmentation', 'vertical_stretch_augmentation', 'horizontal_aspect_augmentation',
        'geometry_conditioned_residual', 'batchnorm_affine', 'fine_detail_branch', 'selection')}
    for name in ('dsbi-addition-protocol.json', 'dsbi-visible-partial-protocol.json'):
        value = json.loads((WORK / 'run26' / name).read_text())
        value.update(shared, scope='Weight-only horizontal-aspect continuation from eligible run26 epoch10; unchanged data/source/geometry/normalization and fresh same-seed optimization.',
                     rationale=plan['rationale'], continuation=plan['continuation'])
        value['preflight_audits'] = plan['preflight_audits']
        save(RUN / name, value)
    save(RUN / 'horizontal-aspect-continuation-protocol.json', plan)
    print(json.dumps({name: digest(RUN/name) for name in ('horizontal-aspect-continuation-protocol.json', 'dsbi-addition-protocol.json', 'dsbi-visible-partial-protocol.json')}, indent=2))


if __name__ == '__main__':
    main()
