"""Freeze six existing TRAIN fixture identities and source bytes; no ML import."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SOURCE = Path('training/scripts')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = ROOT / 'fixture-protocol.json'
    assert not out.exists()
    cohort_path = WORK / 'v15-train-batchnorm-activations/protocol.json'
    negative_cohort_path = WORK / 'geometry-head-parent-v1/geometry-initialization.json'
    cohort = json.loads(cohort_path.read_text())
    negative_cohort = json.loads(negative_cohort_path.read_text())
    manifest_path = WORK / 'combined-training-v5/combined-training-manifest.json'
    rows = json.loads(manifest_path.read_text())
    families = ['books/chudo_derevo_redmi', 'dsbi/Massage', 'dsbi/Math',
                'dsbi/Ordinary Printed Document', 'dsbi/Shaver Yang Fengting']
    samples = [next(row for row in cohort['samples'] if row['family'] == family).copy() for family in families]
    negative = negative_cohort['train_inputs'][2].copy()
    negative['manifest_index'] = next(i for i, row in enumerate(rows) if row['image'] == negative['image'])
    negative['family'] = 'not_braille/misc'
    samples.append(negative)
    for sample in samples:
        row = rows[sample['manifest_index']]
        assert row['source'] == sample['source'] and row['image'] == sample['image']
        sample['record_json_sha256'] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert [sample['manifest_index'] for sample in samples] == [253, 21577, 22437, 23134, 23435, 6326]
    config_path = WORK / 'run30/training-config.json'
    config = json.loads(config_path.read_text())
    source_dir = ROOT / 'source'
    source_dir.mkdir(exist_ok=False)
    names = list(config['source_sha256']) + ['benchmark_angelina.py']
    sources = {}
    for name in names:
        original = SOURCE / name
        if name in config['source_sha256']:
            assert sha(original) == config['source_sha256'][name]
        (source_dir / name).write_bytes(original.read_bytes())
        sources[name] = sha(original)
    prototype = WORK / 'dot-auxiliary-prototype-v1'
    protocol = {
        'status': 'prepared_not_executed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'preparer_sha256': sha(__file__),
        'parent_checkpoint': str(WORK / 'selected-v15/best-word.keras'),
        'parent_checkpoint_sha256': 'cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7',
        'prototype_index_sha256': sha(prototype / 'index.json'),
        'prototype_candidate_sha256': {path.name: sha(path) for path in sorted((prototype / 'candidate').glob('*.py'))},
        'source_sha256': sources, 'source_config_sha256': sha(config_path),
        'data_sha256': config['data_sha256'],
        'cohort_sha256': sha(cohort_path), 'negative_cohort_sha256': sha(negative_cohort_path),
        'sampling_rule': 'First frozen q=1 sample in each named existing TRAIN family cohort, then the already frozen negative train_inputs[2] from geometry initialization. IDs fixed before pixels or inference; no replacement on failure.',
        'samples': samples,
        'micro_batch': {'records': 6, 'batch_size': 6, 'seed': 20260905, 'dataset_epoch': 2,
                        'balanced_source_families': True, 'illumination_augmentation': True,
                        'dsbi_point_states': True, 'other_augmentations': False},
        'auxiliary_coefficient': .25, 'diagnostic_step_learning_rate': 2e-5,
        'normalization': {'targetStd': .05, 'stdFloor': .001},
        'scope': 'One CPU Torch preflight after explicit root scheduling. TRAIN inputs only, exact Patches comparison, gradient/accounting and one real optimizer step, reader/full-wrapper roundtrips. No selection, held inputs, TF/Lite conversion or accuracy claim.',
        'execution_limits': 'No backend executed during preparation. Result reader-after-one-step.keras and training-after-one-step.keras are diagnostic-only and must never initialize a training run or active asset.',
    }
    out.write_text(json.dumps(protocol, indent=2) + '\n')
    for path in [ROOT / 'runtime-preflight.py', Path(__file__)]:
        ast.parse(path.read_text())
    print(json.dumps({'protocol_sha256': sha(out), 'source_files': len(sources), 'sample_ids': [x['manifest_index'] for x in samples]}))


if __name__ == '__main__':
    main()
