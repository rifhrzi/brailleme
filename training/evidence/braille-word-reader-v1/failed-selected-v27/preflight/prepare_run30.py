"""Freeze one higher-learning-rate continuation from plain v15 with decoder v2."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

WORK = Path(__file__).resolve().parent
SOURCE = Path('training/scripts')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    run = WORK/'run30'
    run.mkdir(exist_ok=False)
    (run/'source').mkdir()
    tree = ast.parse((SOURCE/'word_train.py').read_text())
    names = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'TRAINING_SOURCES' for t in n.targets))
    sources = {name: sha(SOURCE/name) for name in names}
    assert len(sources) == 24
    assert sources['word_decode.py'] == 'fa2179b74459d84055b244ac60353d52d59cd12eda80b5cc279382c28f1be444'
    for name in names:
        (run/'source'/name).write_bytes((SOURCE/name).read_bytes())
    baseline = json.loads((WORK/'run20/training-config.json').read_text())
    for name, digest in baseline['data_sha256'].items():
        assert sha(WORK/'data'/name) == digest
    parent = WORK/'selected-v15/best-word.keras'
    control = WORK/'selected-v26/best-word.keras'
    assert sha(parent) == sha(control) == 'cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
    selection = WORK/'selected-v26/selected-validation-fullprecision.json'
    selected = json.loads(selection.read_text())
    assert selected['decoder_version'] == 'six-dot-word-v2'
    assert selected['imageNormalization'] == baseline['imageNormalization']
    plan = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run': str(run),
        'source_sha256': sources, 'parent_checkpoint': str(parent), 'parent_checkpoint_sha256': sha(parent),
        'control_checkpoint': str(control), 'control_checkpoint_sha256': sha(control),
        'control_selection': str(selection), 'control_selection_sha256': sha(selection),
        'parent_training_config_sha256': sha(WORK/'run20/training-config.json'),
        'data_sha256': baseline['data_sha256'],
        'training_manifest_sha256': sha(WORK/'data'/baseline['train_manifest']),
        'planned_epochs': 10, 'planned_eligible_epochs': [8, 9, 10], 'batch_size': 24,
        'learning_rate': 5e-5, 'baseline_learning_rate': 2e-5, 'seed': 20260905,
        'negative_repeat': 27, 'supplemental_negative_repeat': 2,
        'geometry_loss': 'relative_size_huber', 'illumination_augmentation': True,
        'anchor_zoom_cap': False, 'horizontal_aspect_augmentation': False,
        'vertical_stretch_augmentation': False, 'horizontal_mirror_augmentation': False,
        'camera_augmentation': False, 'half_turn_augmentation': False, 'polarity_augmentation': False,
        'partial_geometry_supervision': False, 'dsbi_visible_partial_supervision': True,
        'frozen_batchnorm': True, 'geometry_conditioned_residual': False,
        'batchnorm_affine': False, 'fine_detail_branch': False, 'normalization_transition': False,
        'imageNormalization': baseline['imageNormalization'],
        'point_metadata_sha256': sha(WORK/'data'/baseline['dsbi_point_metadata']),
        'supplemental_source_indexes': baseline['supplemental_source_indexes'],
        'decoder_version': 'six-dot-word-v2',
        'decoder_integration_audit': str(WORK/'position-nms-v2-integration/report.json'),
        'decoder_integration_audit_sha256': sha(WORK/'position-nms-v2-integration/report.json'),
        'validation_fixture': str(WORK/'data/validation-stress-selection.json'),
        'validation_fixture_sha256': sha(WORK/'data/validation-stress-selection.json'),
        'selector_source_sha256': {name: sha(SOURCE/name) for name in selected['oracle']['source_sha256']},
        'threshold_grid': 'weak-peaks-v2',
        'selection': 'Control plain v15 first, then own-cycle eligible epochs 8-10 on unchanged 405. Weak-peaks-v2, >=95% accepted precision separately for natural/stress, maximize equal-group whole-word mean; ties preserve control. Decoder v2 applies to all candidates. No external607/44/66/12/14 checkpoint or threshold selection.',
        'rationale': 'One optimizer-rate experiment, 5e-5 instead of the 2e-5 used by recent full-tail/DSBI continuations. This is a hypothesis; failed normalization, augmentation and architecture branches are not stacked.',
        'optimizer_and_sampling': 'Load plain v15 with compile=False; fresh Adam(5e-5, clipnorm=10), unchanged ReduceLROnPlateau val_loss patience4 factor.5 min_lr1e-5. Same seed and dataset phase restart, all data/augmentation/loss targets unchanged. Own23600-positive-record cycle required before selection; no inherited eligibility shortcut.',
        'decoder_scope': 'Position NMS correction is common to the v26 control and new candidates and affects evaluation, not training targets. It was integrated and tested before this protocol.',
        'no_activation_before_original_gates': True,
        'preparation_script_sha256': sha(__file__),
    }
    assert json.loads((WORK/'position-nms-v2-integration/report.json').read_text())['status'] == 'passed'
    shared = {key: plan[key] for key in ('created_at_utc', 'source_sha256', 'parent_checkpoint', 'parent_checkpoint_sha256',
              'planned_epochs', 'planned_eligible_epochs', 'batch_size', 'learning_rate', 'seed', 'selection')}
    for name in ('dsbi-addition-protocol.json', 'dsbi-visible-partial-protocol.json'):
        inherited = WORK/'run20'/name
        value = json.loads(inherited.read_text())
        value.update(shared)
        value['scope'] = 'Higher-learning-rate continuation from plain v15; original DSBI source, point, partial, negative and sampling policies unchanged.'
        value['inherited_protocol'] = {'path': str(inherited), 'sha256': sha(inherited), 'historical_preflight_audits_unchanged': True}
        save(run/name, value)
    save(run/'learning-rate-protocol.json', plan)
    print(json.dumps({'protocol_sha256': sha(run/'learning-rate-protocol.json'), 'sources': len(sources)}))


if __name__ == '__main__':
    main()
