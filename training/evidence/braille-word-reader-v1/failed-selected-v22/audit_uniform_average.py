"""Independent complete-array check; does not import the candidate builder or ML."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import h5py
import numpy as np

WORK = Path(__file__).resolve().parent
FOLDER = WORK / 'run20-uniform-average-preparation'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_model(path):
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == ['config.json', 'metadata.json', 'model.weights.h5']
        config = json.loads(z.read('config.json'))
        graph = {k: v for k, v in config.items() if k != 'compile_config'}
        bn_names = set()

        def layers(layer):
            if layer.get('class_name') == 'BatchNormalization':
                assert layer['config']['trainable'] is False
                bn_names.add(layer['config']['name'])
            for child in layer.get('config', {}).get('layers', []):
                layers(child)

        layers(graph)
        assert len(bn_names) == 52
        arrays, owners, optimizer = {}, {}, []
        with h5py.File(io.BytesIO(z.read('model.weights.h5')), 'r') as h:
            def visit(name, value):
                if isinstance(value, h5py.Dataset):
                    if name.startswith('optimizer/'):
                        optimizer.append(name)
                    else:
                        assert name.startswith('layers/') and '/vars/' in name
                        arrays[name] = value[()]
                        owners[name] = value.parent.attrs['name']
            h.visititems(visit)
        return config, graph, bn_names, arrays, owners, optimizer


def main():
    plan_path = FOLDER / 'plan.json'
    assert sha(plan_path) == '6520a4e0d41e7bbbb8e5e7ffd9aaed5b6a80d4d54ccb38b96f29341db76c1f98'
    plan = json.loads(plan_path.read_bytes())
    report_path = FOLDER / 'candidate-report.json'
    assert sha(report_path) == '0ce6817ef003e8fe86c03e1e183ebf8f5c3c602bb18eeaeb5d825559ed0ced1b'
    report = json.loads(report_path.read_bytes())
    candidate = Path(plan['candidate_path'])
    assert sha(candidate) == '7c831094852512ef79d791fa31570e6016d15e513004defdeadb091c401b9cda'
    assert plan['input_epoch_order'] == [8, 9, 10]
    bindings = {str(plan_path): sha(plan_path), str(report_path): sha(report_path), str(candidate): sha(candidate)}
    for entry in [*plan['inputs'], plan['source_prepare'], plan['source_average'], plan['inventory'],
                  plan['source_training_config'], plan['source_sampling_lineage'], plan['source_sampling_history'],
                  *plan['selection']['sources'].values()]:
        assert sha(entry['path']) == entry['sha256']
        bindings[entry['path']] = entry['sha256']
    lineage = json.loads(Path(plan['source_sampling_lineage']['path']).read_bytes())
    history = json.loads(Path(plan['source_sampling_history']['path']).read_bytes())
    assert lineage['status'] == 'passed' and lineage['eligible_retained_epochs'] == [8, 9, 10]
    assert lineage['actual']['config_sha256'] == plan['source_training_config']['sha256']
    assert lineage['actual']['sampling_history_sha256'] == plan['source_sampling_history']['sha256']
    for entry in plan['inputs']:
        assert entry['weight'] == {'numerator': 1, 'denominator': 3}
        observed = [r for r in lineage['actual']['observed_epochs'] if r['epoch'] == entry['epoch']]
        assert observed == [entry['sampling']]
        assert observed[0]['checkpoint_sha256'] == entry['sha256']
        assert observed[0]['eligible_by_sampling_lineage'] is True
        assert observed[0]['all_positive_records_seen_in_this_run'] is True
        assert observed[0]['actual_cumulative_positive_records'] == 23600
        seen = [r for r in history if r['epoch'] == entry['epoch']]
        assert len(seen) == 1 and seen[0]['complete_epoch'] is True
        assert seen[0]['completed_batches'] == seen[0]['expected_batches'] == 1037
        assert all(r['cumulative_unique_records'] == r['total_unique_records'] for r in seen[0]['families'].values())
    inputs = [read_model(r['path']) for r in plan['inputs']]
    output = read_model(candidate)
    assert output[0]['compile_config'] == {} and output[5] == []
    assert all(len(m[5]) == 146 for m in inputs)
    assert all(m[1] == output[1] and m[2] == output[2] and m[4] == output[4] for m in inputs)
    assert all(set(m[3]) == set(output[3]) for m in inputs)
    assert len(output[3]) == 280
    preserved, averaged, elements = 0, 0, 0
    max_delta = 0.0
    for name, result in output[3].items():
        values = [m[3][name] for m in inputs]
        assert all(v.shape == result.shape and v.dtype == result.dtype == np.dtype('float32') for v in values)
        assert all(np.isfinite(v).all() for v in [*values, result])
        if output[4][name] in output[2]:
            assert all(v.tobytes() == result.tobytes() for v in values)
            preserved += 1
        else:
            # Separate vectorized implementation, no builder helpers imported.
            expected = np.mean(np.stack(values).astype(np.float64), axis=0, dtype=np.float64).astype(np.float32)
            assert result.tobytes() == expected.tobytes(), name
            averaged += 1
            max_delta = max(max_delta, float(np.max(np.abs(result.astype(np.float64) - values[1]))))
        assert hashlib.sha256(result.tobytes()).hexdigest() == report['array_sha256'][name]
        elements += result.size
    assert (preserved, averaged, elements) == (208, 72, 829307)
    assert max_delta == report['maximum_weight_delta_vs_epoch09'] and max_delta > 0
    assert all(sha(path) == digest for path, digest in bindings.items())
    assert not any(name in sys.modules for name in ('keras', 'torch', 'tensorflow'))
    result = {
        'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(__file__), 'plan_sha256': sha(plan_path), 'candidate_report_sha256': sha(report_path),
        'candidate_sha256': sha(candidate), 'source_bindings': bindings,
        'all_three_parents_sampling_eligible': True, 'complete_epoch_batches': 1037,
        'positive_records_seen_per_parent': 23600, 'graph_equal_to_all_three': True,
        'model_arrays_checked': 280, 'model_values_checked': elements, 'bn_arrays_byte_exact': preserved,
        'other_arrays_independent_vectorized_mean_exact': averaged, 'optimizer_dataset_count': 0,
        'compile_config_empty': True, 'all_original_hashes_unchanged': True,
        'maximum_weight_delta_vs_epoch09': max_delta, 'model_execution': False,
        'corpus_pixels_read': False, 'evaluation_pending': True,
        'limits': 'This checks checkpoint construction and source eligibility. Actual Keras loading, recognition, conversion parity, native execution and all release gates remain pending.',
    }
    target = FOLDER / 'independent-audit.json'
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'passed', 'report_sha256': sha(target), 'model_values_checked': elements}))


if __name__ == '__main__':
    main()
