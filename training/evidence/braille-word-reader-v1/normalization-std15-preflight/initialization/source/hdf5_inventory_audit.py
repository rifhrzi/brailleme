"""Read-only run27 proposed-parent proof. No ML backend or tensor inference."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import sys
import zipfile
import h5py
import numpy as np

OUT = Path(__file__).resolve().parent
WORK = OUT.parent
ROOT = WORK.parent.parent
RUN = WORK / 'run26'
ARCHIVE = ROOT / 'training/evidence/braille-word-reader-v1/failed-selected-v21'
PARENT_SHA = '6113867b21316669169391fa7216f9d1b9bf147a13c153889ff6de1c26e9a4ea'
CONTROL_SHA = 'cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def inventory(path, expected):
    raw = path.read_bytes()
    assert sha(raw) == expected
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        config_raw = archive.read('config.json')
        config = json.loads(config_raw)
        weights_raw = archive.read('model.weights.h5')
    backbone = next(l for l in config['config']['layers'] if l['config']['name'] == 'mobilenetv2_half')
    bn = [l for l in backbone['config']['layers'] if l['class_name'] == 'BatchNormalization']
    assert len(bn) == 52 and all(l['config']['trainable'] is False for l in bn)
    bn_names = [l['config']['name'] for l in bn]
    arrays, meta, auxiliary = {}, {}, []
    with h5py.File(io.BytesIO(weights_raw), 'r') as handle:
        def visit(name, node):
            if not isinstance(node, h5py.Dataset):
                return
            if not name.startswith('layers/'):
                auxiliary.append(dict(path=name, shape=list(node.shape), dtype=str(node.dtype)))
                return
            value = np.asarray(node)
            assert value.dtype == np.float32 and np.isfinite(value).all()
            layer_name = node.parent.attrs['name']
            index = int(name.rsplit('/', 1)[1])
            if layer_name in bn_names:
                variable = ('gamma', 'beta', 'moving_mean', 'moving_variance')[index]
                category = 'bn_moving' if index >= 2 else 'bn_gamma_beta'
            else:
                variable = str(index)
                category = 'backbone_convolution' if name.startswith('layers/functional/') else 'fpn_and_output_head'
            arrays[name] = value
            meta[name] = dict(layer=layer_name, variable=variable, category=category,
                              shape=list(value.shape), dtype=str(value.dtype))
        handle.visititems(visit)
    assert len(arrays) == 280 and sum(v.size for v in arrays.values()) == 829307
    moving = hashlib.sha256()
    for layer_name in bn_names:
        for variable in ('moving_mean', 'moving_variance'):
            keys = [k for k, m in meta.items() if m['layer'] == layer_name and m['variable'] == variable]
            assert len(keys) == 1
            value = np.asarray(arrays[keys[0]], dtype='<f4')
            moving.update(json.dumps([layer_name, variable, list(value.shape)], separators=(',', ':')).encode())
            moving.update(value.tobytes(order='C'))
    return dict(config=config, config_bytes=config_raw, weights_sha256=sha(weights_raw),
                arrays=arrays, metadata=meta, auxiliary=auxiliary, moving_sha256=moving.hexdigest())

def main():
    started = datetime.now(timezone.utc).isoformat()
    assert not (OUT / 'report.json').exists()
    paths = {
        'config': RUN / 'training-config.json',
        'lineage': RUN / 'sampling-lineage-final.json',
        'sampling': RUN / 'sampling-history.json',
        'validation_history': RUN / 'validation-history.json',
        'selection': WORK / 'selected-v21/checkpoint-selection.json',
        'checkpoint_cache_metadata': RUN / 'held-diagnosis/cache-report.json',
        'archive_index': ARCHIVE / 'archive.json',
    }
    raw = {k: p.read_bytes() for k, p in paths.items()}
    obj = {k: json.loads(v) for k, v in raw.items()}
    config = obj['config']
    assert sha(raw['config']) == '5e667caadfe937cbecaa57210cee1039df0cfa9a9a0390143e65015138c48f7b'
    assert config['resume_sha256'] == CONTROL_SHA and config['parameters'] == 829307
    assert config['input_shape'] == [1, 192, 480, 3] and config['output_shape'] == [1, 48, 120, 11]
    assert all(config[k]['enabled'] is False for k in ('fine_detail_branch', 'batchnorm_affine', 'geometry_conditioned_residual'))
    sources = config['source_sha256']
    assert len(sources) == 22
    for name, expected in sources.items():
        assert sha((RUN / 'source' / name).read_bytes()) == expected
        assert sha((ROOT / 'training/scripts' / name).read_bytes()) == expected
    control_path = WORK / 'selected-v15/best-word.keras'
    parent_path = RUN / 'retained-epochs/epoch-10.keras'
    control, parent = inventory(control_path, CONTROL_SHA), inventory(parent_path, PARENT_SHA)
    # Includes trainable flags, dtype policies, nested layer wiring, normalization and compile metadata.
    assert control['config'] == parent['config']
    assert control['metadata'] == parent['metadata']
    assert control['arrays'].keys() == parent['arrays'].keys()
    rows = []
    for key in sorted(control['arrays']):
        before, after = control['arrays'][key], parent['arrays'][key]
        rows.append(dict(path=key, **parent['metadata'][key],
            control_sha256=sha(before.tobytes()), parent_sha256=sha(after.tobytes()),
            identical_bytes=before.tobytes() == after.tobytes(),
            changed_values=int(np.count_nonzero(before != after)),
            max_absolute_change=float(np.max(np.abs(after.astype(np.float64) - before.astype(np.float64))))))
    counts = {category: dict(arrays=sum(r['category'] == category for r in rows),
        changed_arrays=sum(r['category'] == category and not r['identical_bytes'] for r in rows))
        for category in ('bn_moving', 'bn_gamma_beta', 'backbone_convolution', 'fpn_and_output_head')}
    assert counts['bn_moving'] == counts['bn_gamma_beta'] == dict(arrays=104, changed_arrays=0)
    assert control['moving_sha256'] == parent['moving_sha256'] == '1f190a8fe5f4b3f7162147f7cf254a9fc437dfd20e0b34f3603003fffdc00ae7'
    lineage = obj['lineage']
    assert lineage['status'] == 'passed'
    assert lineage['actual']['config_sha256'] == sha(raw['config'])
    assert lineage['actual']['sampling_history_sha256'] == sha(raw['sampling'])
    manifest = WORK / 'combined-training-v5/combined-training-manifest.json'
    assert lineage['actual']['manifest_sha256'] == sha(manifest.read_bytes())
    row = next(r for r in lineage['actual']['observed_epochs'] if r['epoch'] == 10)
    assert row['checkpoint_sha256'] == PARENT_SHA
    assert row['actual_cumulative_positive_records'] == lineage['actual']['positive_records'] == 23600
    assert row['completed_batches'] == 1037 and row['eligible_by_sampling_lineage'] and row['all_positive_records_seen_in_this_run']
    assert 10 in lineage['eligible_retained_epochs']
    sampling = next(r for r in obj['sampling'] if r['epoch'] == 10)
    callback = next(r for r in obj['validation_history'] if r['epoch'] == 10)
    assert sampling['complete_epoch'] and sampling['completed_batches'] == sampling['expected_batches'] == 1037
    assert sum(r['cumulative_unique_records'] for r in sampling['families'].values()) == 23600
    assert all(r['cumulative_unique_records'] == r['total_unique_records'] for r in sampling['families'].values())
    assert callback['sampling'] == sampling
    selection = obj['selection']
    assert selection['selected_checkpoint_sha256'] == CONTROL_SHA and selection['external_outputs_used'] is False
    assert any(r['checkpoint_sha256'] == PARENT_SHA for r in selection['candidates'])
    cache = obj['checkpoint_cache_metadata']
    assert cache['checkpoint_sha256'] == PARENT_SHA
    # Bind the retained archive's exact selection/lineage/config/source payload, not an absent model copy.
    archive_rows = {r['file']: r for r in obj['archive_index']['files']}
    archive_verified = []
    for rel in ['run26/training-config.json', 'run26/sampling-lineage-final.json',
                'run26/sampling-history.json', 'selected-v21/checkpoint-selection.json'] + ['run26/source/' + n for n in sources]:
        payload = (ARCHIVE / rel).read_bytes()
        expected = archive_rows[rel]
        assert sha(payload) == expected['sha256'] and len(payload) == expected['sizeBytes']
        assert payload == (WORK / rel).read_bytes()
        archive_verified.append(dict(file=rel, sha256=sha(payload)))
    for k, p in paths.items():
        assert p.read_bytes() == raw[k]
    assert sha(parent_path.read_bytes()) == PARENT_SHA and sha(control_path.read_bytes()) == CONTROL_SHA
    assert not any(m in sys.modules for m in ('keras', 'tensorflow', 'torch'))
    for key, payload in raw.items():
        (OUT / (key + '-snapshot.json')).write_bytes(payload)
    (OUT / 'control-model-config.json').write_bytes(control['config_bytes'])
    (OUT / 'parent-model-config.json').write_bytes(parent['config_bytes'])
    report = dict(schema_version=1, status='passed', started_at_utc=started,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read-only serialized HDF5/config/source/lineage audit of proposed run27 weight parent; no backend loading, model inference, re-evaluation or training.',
        parent_checkpoint=str(parent_path), parent_checkpoint_sha256=PARENT_SHA,
        control_checkpoint=str(control_path), control_checkpoint_sha256=CONTROL_SHA,
        source_sha256=sources, bindings={k: dict(path=str(paths[k]), sha256=sha(raw[k])) for k in paths},
        identical_complete_serialized_model_config=True,
        canonical_model_config_sha256=sha(canonical(parent['config'])),
        control_config_bytes_sha256=sha(control['config_bytes']), parent_config_bytes_sha256=sha(parent['config_bytes']),
        camera_normalization=next(l for l in parent['config']['config']['layers'] if l['class_name'] == 'CameraNormalization'),
        model_arrays=280, model_parameters=829307, all_model_array_names_shapes_dtypes_identical=True,
        frozen_batchnorm_arrays_byte_identical=208, moving_moments_sha256=parent['moving_sha256'],
        moving_digest_method='Serialized BN layer order; JSON[layer,variable,shape] then little-endian float32 C bytes; excludes optimizer.',
        counts=counts, arrays=rows, sampling_epoch10=row, sampling_callback_epoch10=sampling,
        selection_status='Epoch10 is eligible and best-new diagnostic checkpoint, but v21 selection retained v15 control; no active candidate promotion.',
        archive_verified_payloads=archive_verified,
        archive_checkpoint_scope='Archive stores hash-bound selection and lineage naming actual epoch10; no model binary copy is asserted or added.',
        auxiliary_nonmodel_arrays={'control': control['auxiliary'], 'parent': parent['auxiliary']},
        runtime_versions={'h5py': h5py.__version__, 'numpy': np.__version__}, backend_modules_imported=[],
        script_sha256=sha(Path(__file__).read_bytes()),
        limitations=['This proves stored parent identity, not useful predictions, conversion parity or release readiness.',
                    'Sampling coverage counts record consumption, not survival of every cell under augmentation.',
                    'Proposed run27 is a weight-only continuation with fresh Adam and reset RNG/order/epochs, intentionally repeating prior realizations; this audit does not claim new data or verify run27 execution.'])
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    files = [dict(file=p.name, sha256=sha(p.read_bytes()), size_bytes=p.stat().st_size) for p in sorted(OUT.iterdir()) if p.is_file()]
    (OUT / 'index.json').write_text(json.dumps(dict(files=files), indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='passed', report_sha256=sha((OUT/'report.json').read_bytes()), index_sha256=sha((OUT/'index.json').read_bytes()), counts=counts, parent_sha256=PARENT_SHA)))

if __name__ == '__main__':
    main()
