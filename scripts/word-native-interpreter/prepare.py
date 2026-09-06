"""Freeze six original TRAIN/analytic tensors, optionally binding a trained model oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ORACLE_SHA256 = '9330946dafe06c7705b3f2c85cce26f8bbfeca11763ba7465d62e809180d5669'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--initialization-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--desktop-report', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory; do not overwrite frozen inputs')
    init = json.loads(args.initialization_report.read_text())
    if sha(args.oracle) != ORACLE_SHA256 or init['oracle_sha256'] != ORACLE_SHA256:
        raise ValueError('Original six-input oracle changed')
    with np.load(args.oracle, allow_pickle=False) as oracle:
        images = oracle['images']
    if images.dtype != np.float32 or images.shape != (6, 192, 480, 3):
        raise ValueError('Unexpected six-input oracle contract')
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError('Inputs must be finite normalized RGB')
    desktop = None
    expected = None
    if args.desktop_report:
        desktop = json.loads(args.desktop_report.read_text())
        if desktop['status'] != 'desktop_reference_ready_native_comparison_pending':
            raise ValueError('Expected actual trained-candidate desktop inference report')
        if desktop['oracle_npz_sha256'] != ORACLE_SHA256 or sha(desktop['model']) != desktop['model_sha256']:
            raise ValueError('Desktop model/oracle binding differs')
        if sha(desktop['outputs']) != desktop['outputs_npy_sha256']:
            raise ValueError('Desktop output array changed')
        expected = np.load(desktop['outputs'], allow_pickle=False)
        if expected.dtype != np.float32 or expected.shape != (6, 48, 120, 11) or not np.isfinite(expected).all():
            raise ValueError('Unexpected actual desktop output contract')
        run = args.desktop_report.parent
        exported = json.loads((run/'export-report.json').read_text())
        selected = json.loads((run/'selected-validation-fullprecision.json').read_text())
        if (not exported['parity_pass'] or exported['parity_samples'] != 405 or
                exported['artifact_sha256'] != desktop['model_sha256'] or
                exported['source_checkpoint_sha256'] != selected['checkpoint_sha256'] or
                selected['checkpoint_sha256'] != desktop['checkpoint_sha256'] or
                sha(run/'best-word.keras') != desktop['checkpoint_sha256'] or
                exported['thresholds'] != selected['thresholds'] or selected['thresholds'] != desktop['thresholds']):
            raise ValueError('Candidate/export/405 parity/checkpoint/threshold binding differs')
    args.output.mkdir(parents=True)
    (args.output/'inputs').mkdir()
    if desktop:
        (args.output/'desktop').mkdir()
        shutil.copyfile(desktop['model'], args.output/'model.tflite')
        for source in (args.desktop_report, args.desktop_report.parent/'export-report.json',
                       args.desktop_report.parent/'selected-validation-fullprecision.json'):
            shutil.copyfile(source, args.output/source.name)
    rows = []
    labels = init['train_inputs'] + [{'analytic': name} for name in init['analytic_inputs']]
    for index, image in enumerate(images):
        path = args.output/'inputs'/f'{index:04d}.f32'
        path.write_bytes(np.ascontiguousarray(image, dtype='<f4').tobytes())
        row = {'index': index, 'identity': labels[index], 'input': path.relative_to(args.output).as_posix(),
               'input_sha256': sha(path), 'input_bytes': path.stat().st_size,
               'selection': {'x': 0, 'y': 0, 'width': 1, 'height': 1}}
        if expected is not None:
            path = args.output/'desktop'/f'{index:04d}.f32'
            path.write_bytes(np.ascontiguousarray(expected[index], dtype='<f4').tobytes())
            row.update(desktop_output=path.relative_to(args.output).as_posix(), desktop_output_sha256=sha(path))
        rows.append(row)
    shutil.copyfile(args.initialization_report, args.output/'geometry-initialization.json')
    report = {'schema_version': 1,
              'status': 'trained_candidate_ready_native_execution_pending' if desktop else 'inputs_only_model_and_native_execution_pending',
              'scope': 'Six original TRAIN/analytic inputs; whole-image decoder comparison, no accuracy/quality/camera claim.',
              'oracle_npz_sha256': ORACLE_SHA256, 'oracle_path': str(args.oracle.resolve()),
              'initialization_report_sha256': sha(args.initialization_report),
              'input_shape': [1, 192, 480, 3], 'output_shape': [1, 48, 120, 11],
              'dtype': 'float32_little_endian', 'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-4,
              'source_sha256': sha(__file__), 'cases': rows,
              'model_sha256': desktop['model_sha256'] if desktop else None,
              'observed_model_operators': desktop['observed_operator_names'] if desktop else None,
              'thresholds': desktop['thresholds'] if desktop else None,
              'desktop_report_sha256': sha(args.desktop_report) if desktop else None,
              'limits': ['No threshold changes are permitted for native parity.',
                         'These are TRAIN/analytic inputs, not six held-out word accuracy examples.',
                         'Synthetic random float inputs are not rounded to RGB8; quality preprocessing is not replayed.']}
    report['files'] = [{'path': p.relative_to(args.output).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                       for p in sorted(args.output.rglob('*')) if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'manifest_sha256': sha(args.output/'manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
