"""Export the frozen detector and verify every validation view and text decision.

The model tensor ABI stays unchanged. The accompanying fixed NumPy pipeline
owns conditional flip inference. This verifies TFLite, not an Android camera.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from word_symbol_v4_export import (decision_signature, pixels_for, runtime_quality_rejected,
    tensorflow_function, validate_outputs, validate_tensor_interface)
from word_overall_v5_protocol import read, sha, verify_hashes, write_new
from word_overall_v5_tta import (transformed_images, runtime_decisions, method_specification)
from word_overall_v5_runtime import DEFAULT_RUNTIME, RUNTIME_VIEWS, get_runtime, get_views

VIEWS = ('identity', 'horizontal', 'vertical', 'half_turn')
SOURCE_NAMES = ('word_overall_v5_export.py', 'word_overall_v5_runtime.py', 'word_overall_v5_tta.py', 'word_overall_v5_protocol.py',
    'word_symbol_v4_export.py', 'word_torch_port.py', 'word_decode_id_v4.py',
    'word_symbols_id_v4.py', 'word_symbols.py', 'word_data.py', 'word_anchor_zoom.py', 'benchmark_angelina.py')


def bindings(scripts, runtime_name=DEFAULT_RUNTIME):
    get_views(runtime_name)
    names = tuple(dict.fromkeys((*SOURCE_NAMES, runtime_name+'.py')))
    return {str(scripts/name): sha(scripts/name) for name in names}


def validate_view_bundle(outputs, count, views=VIEWS):
    if set(outputs) != set(views):
        raise ValueError('Every fixed raw view is required')
    for values in outputs.values():
        if values.dtype != np.float32:
            raise ValueError('View oracle must preserve float32 model output')
        validate_outputs(values, count)


def pipeline_signatures(values, thresholds, selection, quality, runtime=None):
    invoke = runtime_decisions if runtime is None else runtime.runtime_decisions
    result = invoke(values, thresholds, selection, quality_rejected=quality)
    # Early confidence/geometry rejection may leave later cells without a
    # decoder mask. Include every physical six-bit mask, even on those cells.
    return {policy: (decision_signature(decision),
                     tuple(sum(int(dot >= .5) << bit for bit, dot in enumerate(cell['dots']))
                           for cell in decision['cells']), result['routes'][policy])
            for policy, decision in result['decisions'].items()}


def selected_run_bindings(args, scripts):
    run = args.checkpoint.parent
    if (run.parent != args.experiment or args.checkpoint.name != 'best-word.keras'
            or args.thresholds != run/'training-config.json'):
        raise ValueError('Export requires the selected run checkpoint and its exact training configuration')
    selection_path = run/'selection.json'
    selection, config = read(selection_path), read(args.thresholds)
    if selection.get('selection_scope') != 'validation_only' or selection.get('test_opened') is not False:
        raise ValueError('Checkpoint must have been selected using validation only')
    verified = {str(args.checkpoint): selection['sha256'],
                str(args.thresholds): selection['training_config_sha256'],
                str(selection_path): sha(selection_path)}
    # New TTA/export utilities are separately bound; the detector/decoder must
    # still be exactly the implementation used by the selected training run.
    for name in ('word_torch_port.py', 'word_decode_id_v4.py', 'word_symbols_id_v4.py',
                 'word_symbols.py', 'word_data.py', 'word_anchor_zoom.py', 'benchmark_angelina.py'):
        expected = config['source_sha256'][name]
        verified[str(run/'source'/name)] = expected
        verified[str(scripts/name)] = expected
    verify_hashes(verified)
    return verified


def validate_oracle_metadata(metadata, args, frozen, count, archive_hash):
    runtime_name = getattr(args, 'runtime_module', DEFAULT_RUNTIME)
    runtime = get_runtime(runtime_name)
    expected = dict(checkpoint_sha256=frozen[str(args.checkpoint)],
        validation_sha256=frozen[str(args.validation)], samples=count,
        views=list(get_views(runtime_name)), orientation='raw_model_output_for_each_transformed_image',
        method=runtime.method_specification(), thresholds=read(args.thresholds)['thresholds'],
        scope='development_validation_only', test_opened=False,
        physical_camera_validated=False, oracle_sha256=archive_hash)
    if (metadata.get('runtime_module', DEFAULT_RUNTIME) != runtime_name
            or metadata.get('input_sha256') != frozen or any(metadata.get(key) != value for key,value in expected.items())):
        raise ValueError('Frozen oracle contract changed')
    verify_hashes(frozen)


def reference(args, rows, frozen):
    import torch
    from word_torch_port import load_keras_checkpoint
    runtime_name = getattr(args, 'runtime_module', DEFAULT_RUNTIME)
    runtime, views = get_runtime(runtime_name), get_views(runtime_name)
    if any((args.output/name).exists() for name in ('view-oracle.npz', 'view-oracle.json')):
        raise ValueError('Preserve existing oracle; output must be new')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_keras_checkpoint(args.checkpoint).cuda().eval()
    outputs, started = {}, time.perf_counter()
    with torch.no_grad():
        for name in views:
            predictions = []
            for start in range(0, len(rows), 8):
                pixels = np.stack([pixels_for(row) for row in rows[start:start+8]])
                values = runtime.transformed_images(pixels, name)
                predictions.append(model(torch.from_numpy(values).cuda()).cpu().numpy())
            outputs[name] = np.concatenate(predictions)
            print('REFERENCE_VIEW '+name, flush=True)
    validate_view_bundle(outputs, len(rows), views)
    del model
    torch.cuda.empty_cache()
    archive = args.output/'view-oracle.npz'
    with archive.open('xb') as stream:
        np.savez_compressed(stream, **outputs)
    verify_hashes(frozen)
    write_new(args.output/'view-oracle.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256=frozen, oracle_sha256=sha(archive), checkpoint_sha256=sha(args.checkpoint),
        validation_sha256=sha(args.validation), thresholds=read(args.thresholds)['thresholds'],
        samples=len(rows), views=views, orientation='raw_model_output_for_each_transformed_image',
        runtime_module=runtime_name, method=runtime.method_specification(), torch=torch.__version__, elapsed_seconds=time.perf_counter()-started,
        scope='development_validation_only', test_opened=False, physical_camera_validated=False))


def convert(args, rows, frozen):
    import tensorflow as tf
    runtime_name = getattr(args, 'runtime_module', DEFAULT_RUNTIME)
    runtime, views = get_runtime(runtime_name), get_views(runtime_name)
    metadata_path = args.output/'view-oracle.json'
    metadata = read(metadata_path)
    archive = args.output/'view-oracle.npz'
    oracle_bindings = {str(metadata_path): sha(metadata_path), str(archive): sha(archive)}
    validate_oracle_metadata(metadata, args, frozen, len(rows), oracle_bindings[str(archive)])
    target, report_path = args.output/'word-reader.tflite', args.output/'export-parity.json'
    if target.exists() or report_path.exists():
        raise ValueError('Preserve existing export evidence')
    function = tensorflow_function(args.checkpoint)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([function.get_concrete_function()])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converted = converter.convert()
    with target.open('xb') as stream:
        stream.write(converted)
    interpreter = tf.lite.Interpreter(model_path=str(target), num_threads=4)
    interpreter.allocate_tensors()
    inp, out = validate_tensor_interface(interpreter.get_input_details(), interpreter.get_output_details())
    with np.load(archive, allow_pickle=False) as stream:
        expected = {name: stream[name].copy() for name in views}
    validate_view_bundle(expected, len(rows), views)
    thresholds = metadata['thresholds']
    numerical, maximum, mismatches, started = True, 0., [], time.perf_counter()
    per_view_max = dict.fromkeys(views, 0.)
    for index, row in enumerate(rows):
        pixels = pixels_for(row)[None]
        actual = {}
        for name in views:
            interpreter.set_tensor(inp['index'], runtime.transformed_images(pixels, name))
            interpreter.invoke()
            actual[name] = interpreter.get_tensor(out['index'])[0]
            validate_outputs(actual[name][None], 1)
            difference = float(np.max(np.abs(expected[name][index]-actual[name])))
            per_view_max[name] = max(per_view_max[name], difference)
            maximum = max(maximum, difference)
            numerical &= bool(np.allclose(expected[name][index], actual[name], atol=1e-4, rtol=1e-4))
        quality = runtime_quality_rejected(pixels)
        original = {name: expected[name][index] for name in views}
        before = pipeline_signatures(original, thresholds, row.get('selection'), quality, runtime)
        after = pipeline_signatures(actual, thresholds, row.get('selection'), quality, runtime)
        if before != after:
            mismatches.append(dict(index=index, source=row['source'], torch=before, tflite=after))
        if (index+1) % 100 == 0:
            print('VIEW_PARITY '+str(index+1)+'/'+str(len(rows)), flush=True)
    operators = sorted({item['op_name'] for item in interpreter._get_ops_details()})
    passed = numerical and not mismatches and not any('Flex' in op or 'CUSTOM' in op for op in operators)
    verify_hashes(frozen)
    verify_hashes(oracle_bindings)
    write_new(report_path, dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='passed' if passed else 'failed', checkpoint_sha256=sha(args.checkpoint), tflite_sha256=sha(target),
        bytes=target.stat().st_size, input_sha256=frozen, oracle_metadata_sha256=oracle_bindings[str(metadata_path)],
        oracle_sha256=oracle_bindings[str(archive)], validation_sha256=sha(args.validation), samples=len(rows),
        views=views, inference_count=len(rows)*len(views), allclose=numerical,
        torch_tflite_max_abs=maximum, per_view_max_abs=per_view_max, atol=1e-4, rtol=1e-4,
        pipeline_decision_mismatches=mismatches, semantic_contract='all policies: text, rejection, full ordered masks/roles/output, chosen route',
        thresholds=thresholds, runtime_module=runtime_name, method=runtime.method_specification(), input_shape=inp['shape'].tolist(),
        output_shape=out['shape'].tolist(), dtype='float32', operators=operators, tensorflow=tf.__version__,
        elapsed_seconds=time.perf_counter()-started, scope='development_validation_only',
        test_opened=False, physical_camera_validated=False, native_runtime_validated=False))
    print('EXPORT_PARITY '+json.dumps(dict(passed=passed, maximum_absolute_error=maximum, mismatches=len(mismatches))), flush=True)
    if not passed:
        raise ValueError('Full-view TFLite parity failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--reference', action='store_true')
    modes.add_argument('--convert', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--thresholds', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-module', choices=tuple(RUNTIME_VIEWS), default=DEFAULT_RUNTIME)
    args = parser.parse_args()
    for key in ('checkpoint','validation','thresholds','experiment','output'):
        setattr(args,key,getattr(args,key).resolve())
    if not args.output.is_relative_to(args.experiment):
        raise ValueError('Write export evidence inside this experiment only')
    for name in ('test-evaluation-reservation.json', 'final-test-evaluation.json'):
        if (args.experiment/name).exists():
            raise ValueError('Final TEST already opened; validation export checks must be frozen first')
    protocol_path = args.experiment/'protocol.json'
    protocol = read(protocol_path)
    if args.validation != Path(protocol['development_files']['validation-symbols.json']).resolve():
        raise ValueError('Only frozen development validation is allowed for export checks')
    content = read(args.validation)
    if content.get('split') != 'validation':
        raise ValueError('Never use TEST for conversion calibration')
    rows = content['records']
    if len(rows) != 540 or len({row['source'] for row in rows}) != len(rows):
        raise ValueError('All 540 validation examples are required')
    index_path = Path(protocol['development_files']['validation-image-hashes.json'])
    images = read(index_path)
    if set(images) != {row['image'] for row in rows}:
        raise ValueError('Validation image hash coverage changed')
    verify_hashes({args.validation:protocol['input_sha256'][str(args.validation)],
                   index_path:protocol['input_sha256'][str(index_path)]})
    frozen = {str(path):sha(path) for path in (args.checkpoint,args.validation,args.thresholds,protocol_path,index_path)}
    frozen.update(images)
    frozen.update(bindings(Path(__file__).resolve().parent, args.runtime_module))
    frozen.update(selected_run_bindings(args, Path(__file__).resolve().parent))
    verify_hashes(frozen)
    args.output.mkdir(parents=True,exist_ok=True)
    (reference if args.reference else convert)(args,rows,frozen)


if __name__ == '__main__':
    main()
