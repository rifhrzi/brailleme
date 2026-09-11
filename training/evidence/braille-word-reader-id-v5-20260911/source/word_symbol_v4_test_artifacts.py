"""Confirm final TEST decisions for the already selected TFLite and baseline.

Run only after native Torch TEST evaluation has completed. This cannot select a
checkpoint or tune thresholds. TensorFlow is imported only inside main; neither
Torch nor Keras is imported by this module or its metrics/authorization helpers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from word_symbol_v4_export import (decision_signature, pixels_for,
    runtime_quality_rejected, sha, validate_outputs, validate_tensor_interface)

POSITIVE_GROUPS = ('lowercase', 'capital_single', 'capital_word', 'numbers', 'punctuation')
REQUIRED_GROUPS = set(POSITIVE_GROUPS) | {'rejection'}
NATIVE_PARITY_FIELDS = ('source', 'group', 'expected', 'predicted', 'reason', 'expected_cells', 'detected_cells')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def verify(path, digest):
    if not Path(path).is_file() or sha(path) != digest:
        raise ValueError('Frozen artifact changed or missing: '+str(path))


def summarize(items):
    positive=[row for row in items if row['expected']]
    negative=[row for row in items if not row['expected']]
    accepted=sum(bool(row['predicted']) for row in positive)
    correct=sum(row['predicted']==row['expected'] for row in positive)
    return dict(total=len(items), positive_total=len(positive), correct=correct,
        accepted=accepted, exact_accuracy=correct/len(positive) if positive else None,
        accepted_precision=correct/accepted if accepted else 0., rejection_total=len(negative),
        false_accepts=sum(bool(row['predicted']) for row in negative))


def group_metrics(rows, decisions):
    if len(rows)!=len(decisions) or not rows:
        raise ValueError('Exactly one decision is required for every frozen row')
    if len({row['source'] for row in rows})!=len(rows):
        raise ValueError('Duplicate frozen row identity')
    grouped={}
    for row, decision in zip(rows, decisions):
        if (row['source']!=decision['source'] or row['expected']!=decision['expected']
                or row['symbol_group']!=decision['group']):
            raise ValueError('Decision identity/order differs from frozen row')
        if bool(row['expected']) == (row['symbol_group']=='rejection'):
            raise ValueError('Positive/invalid semantic denominator mismatch')
        grouped.setdefault(row['symbol_group'],[]).append(decision)
    result={group:summarize(items) for group,items in sorted(grouped.items())}
    for group,items in grouped.items():
        multi=[row for row in items if len(row['expected'])>=2]
        if multi:
            result[group]['multi_character']=summarize(multi)
        if group=='punctuation':
            result[group]['by_punctuation']={mark:summarize([r for r in items if mark in r['expected']]) for mark in '.,?!:;-()'}
    return result


def acceptance_gates(groups):
    complete=REQUIRED_GROUPS <= set(groups)
    slices={}
    for group in POSITIVE_GROUPS:
        metrics=groups.get(group,{})
        if not metrics.get('positive_total') or not metrics.get('multi_character',{}).get('positive_total'):
            complete=False
            continue
        slices[group]=metrics
        slices[group+'/multi_character']=metrics['multi_character']
    exact=complete and all(row['exact_accuracy']>=.80 for row in slices.values())
    precision=complete and all(row['accepted_precision']>=.95 for row in slices.values())
    rejects=bool(groups.get('rejection',{}).get('rejection_total')) and groups['rejection']['false_accepts']==0
    return dict(coverage_complete=complete, exact80_each_positive_group_and_multichar=bool(exact),
        accepted_precision95_each_positive_group_and_multichar=bool(precision),
        mandatory_invalid_sequences_all_rejected=rejects,
        combined_acceptance_passed=bool(exact and precision and rejects),
        evaluated_slices=list(slices), per_punctuation_metrics='Reported separately; not pooled into another denominator')


def native_parity(candidate, native):
    if len(candidate)!=len(native):
        raise ValueError('Native TEST decisions must cover every candidate row')
    mismatches=[]
    for index,(actual,expected) in enumerate(zip(candidate,native)):
        missing=[key for key in NATIVE_PARITY_FIELDS if key not in actual or key not in expected]
        if missing:
            raise ValueError('Native/candidate parity fields missing: '+str(missing))
        differences={key:dict(native=expected[key],tflite=actual[key]) for key in NATIVE_PARITY_FIELDS if actual[key]!=expected[key]}
        if differences:
            mismatches.append(dict(index=index,source=actual['source'],differences=differences))
    return dict(status='passed' if not mismatches else 'failed', samples=len(candidate),
        compared_fields=list(NATIVE_PARITY_FIELDS), mismatches=mismatches,
        full_semantic_cell_sequence_compared_on_test=False,
        limitation='Native TEST report stores summary fields only; full ordered mask/role/output parity was checked on export validation')


def authorize_artifacts(experiment, run, export, baseline):
    """Check completed native reservation and immutable conversion before TEST reads."""
    experiment,run,export,baseline=map(lambda p:Path(p).resolve(),(experiment,run,export,baseline))
    if run.parent!=experiment:
        raise ValueError('Selected run must be a direct experiment child')
    reservation_path=experiment/'test-evaluation-reservation.json'
    native_path=run/'test-evaluation.json'
    # Deliberately precedes any access to TEST manifests, image hashes or pixels.
    if not reservation_path.is_file() or not native_path.is_file():
        raise ValueError('Completed native TEST evaluation and existing reservation are required')
    if (run/'test-artifact-comparison.json').exists() or (run/'test-artifact-comparison-started.json').exists():
        raise ValueError('Artifact comparison already exists or was started; never silently repeat')
    reservation,native=read(reservation_path),read(native_path)
    completion_path=experiment/'final-test-evaluation.json'
    if not completion_path.is_file():
        raise ValueError('Native TEST final completion index is required')
    completion=read(completion_path)
    selection_path,config_path=run/'selection.json',run/'training-config.json'
    selection,config=read(selection_path),read(config_path)
    checkpoint=run/'best-word.keras'
    if (Path(reservation['run']).resolve()!=run or reservation['checkpoint_sha256']!=selection['sha256']
            or native['checkpoint_sha256']!=selection['sha256'] or native.get('reservation')!=reservation):
        raise ValueError('Native TEST reservation/checkpoint binding differs from selected run')
    if Path(completion['run']).resolve()!=run or completion['checkpoint_sha256']!=selection['sha256']:
        raise ValueError('Native TEST completion index names another run/checkpoint')
    verify(native_path,completion['report_sha256'])
    if selection.get('selection_scope')!='validation_only' or selection.get('test_opened') is not False:
        raise ValueError('Checkpoint must have been frozen before TEST')
    verify(selection_path,reservation['selection_sha256'])
    verify(config_path,selection['training_config_sha256'])
    verify(checkpoint,selection['sha256'])
    if native['thresholds']!=config['thresholds']:
        raise ValueError('Native TEST thresholds differ from frozen training thresholds')
    source_dir=Path(__file__).resolve().parent
    for name in ('word_decode_id_v4.py','word_symbols_id_v4.py','word_symbols.py','word_data.py'):
        verify(source_dir/name,config['source_sha256'][name])
    parity_path,model_path=export/'export-parity.json',export/'word-reader.tflite'
    parity=read(parity_path)
    if (parity.get('status')!='passed' or parity.get('allclose') is not True
            or parity.get('all_finite') is not True or parity.get('decision_mismatches')
            or parity.get('quality_effective_decision_mismatches')
            or parity['checkpoint_sha256']!=selection['sha256'] or parity['thresholds']!=config['thresholds']):
        raise ValueError('Candidate export parity did not pass for the frozen checkpoint/thresholds')
    verify(model_path,parity['tflite_sha256'])
    if model_path.stat().st_size!=parity['bytes']:
        raise ValueError('Candidate export byte count differs')
    # Bind validation confirmation to the same immutable experiment inputs.
    validation_path=experiment/'validation-symbols.json'
    verify(validation_path,parity['validation_sha256'])
    oracle_path=export/'torch-oracle.json'
    oracle=read(oracle_path)
    if (oracle['checkpoint_sha256']!=selection['sha256'] or oracle['thresholds']!=config['thresholds']
            or oracle['validation_sha256']!=parity['validation_sha256']
            or oracle['oracle_sha256']!=parity['oracle_sha256']):
        raise ValueError('Export oracle binding differs')
    for name in ('word_decode_id_v4.py','word_symbols_id_v4.py','word_symbols.py','word_data.py'):
        if (parity['source_sha256'][name]!=config['source_sha256'][name]
                or oracle['source_sha256'][name]!=config['source_sha256'][name]):
            raise ValueError('Export/native decoder source versions differ: '+name)
    verify(export/'torch-oracle.npz',parity['oracle_sha256'])
    protected_path=run/'protected-before.json'
    protected=read(protected_path)
    expected_baseline=next((digest for path,digest in protected.items() if Path(path).resolve()==baseline),None)
    if expected_baseline is None:
        raise ValueError('Baseline is not the deployed model frozen before training')
    verify(baseline,expected_baseline)
    test_path,index_path=experiment/'test-symbols.json',experiment/'test-image-hashes.json'
    for path in (test_path,index_path,validation_path):
        digest=next((value for name,value in config['input_sha256'].items() if Path(name).resolve()==path),None)
        if digest is None:
            raise ValueError('TEST/validation input was not frozen by training config')
        verify(path,digest)
    if native['test_manifest_sha256']!=sha(test_path):
        raise ValueError('Native TEST manifest binding differs')
    bound=[reservation_path,native_path,completion_path,selection_path,config_path,checkpoint,parity_path,
           model_path,oracle_path,export/'torch-oracle.npz',protected_path,baseline,test_path,index_path,validation_path]
    return dict(experiment=experiment,run=run,export=export,baseline=baseline,model=model_path,
        native=native,selection=selection,config=config,export_parity=parity,
        input_sha256={str(path):sha(path) for path in bound})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=Path('assets/models/braille-word-reader-v1/word-reader.tflite'))
    args=parser.parse_args()
    started=time.perf_counter()
    bound=authorize_artifacts(args.experiment,args.run,args.export,args.baseline)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    from word_decode_id_v4 import decode
    interpreters={}
    for name,path in (('candidate',bound['model']),('baseline',bound['baseline'])):
        engine=tf.lite.Interpreter(model_path=str(path),num_threads=2)
        engine.allocate_tensors()
        inp,out=validate_tensor_interface(engine.get_input_details(),engine.get_output_details())
        interpreters[name]=(engine,inp,out)
    manifest=read(bound['experiment']/'test-symbols.json')
    if manifest.get('split')!='test':
        raise ValueError('Expected the frozen TEST split')
    rows=manifest['records'];hashes=read(bound['experiment']/'test-image-hashes.json')
    if len(rows)!=540 or {row['symbol_group'] for row in rows}!=REQUIRED_GROUPS:
        raise ValueError('Expected all 540 frozen TEST rows and six groups')
    if len({row['source'] for row in rows})!=540 or set(hashes)!={row['image'] for row in rows}:
        raise ValueError('TEST identity/image coverage differs')
    for row in rows:
        verify(row['image'],hashes[row['image']])
    # Complete identity/denominator validation before committing to inference.
    group_metrics(rows,bound['native']['decisions'])
    marker=dict(created_utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=bound['selection']['sha256'],
        native_test_sha256=bound['input_sha256'][str(bound['run']/'test-evaluation.json')],
        purpose='Same frozen checkpoint conversion confirmation and unchanged baseline comparison; never tuning')
    write_new(bound['run']/'test-artifact-comparison-started.json',marker)
    decisions={'candidate':[],'baseline':[]}
    for index,row in enumerate(rows):
        pixels=pixels_for(row)
        rejected=runtime_quality_rejected(pixels)
        for name,(engine,inp,out) in interpreters.items():
            engine.set_tensor(inp['index'],pixels[None])
            engine.invoke()
            output=engine.get_tensor(out['index'])
            validate_outputs(output,1)
            decision=decode(np.zeros_like(output[0]) if rejected else output[0],bound['config']['thresholds'],row.get('selection'))
            decisions[name].append(dict(source=row['source'],group=row['symbol_group'],expected=row['expected'],
                predicted=decision['text'],reason=decision['reason'],expected_cells=len(row['masks']),
                detected_cells=len(decision['cells']),quality_rejected=rejected,semantic_signature=decision_signature(decision)))
        if (index+1)%100==0:
            print(f'TEST_ARTIFACT_CONFIRMATION {index+1}/{len(rows)}',flush=True)
    comparison={}
    for name,path in (('candidate',bound['model']),('baseline',bound['baseline'])):
        metrics=group_metrics(rows,decisions[name])
        comparison[name]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,
            groups=metrics,gates=acceptance_gates(metrics),decisions=decisions[name])
    parity=native_parity(decisions['candidate'],bound['native']['decisions'])
    for path,digest in bound['input_sha256'].items():
        verify(path,digest)
    for row in rows:
        verify(row['image'],hashes[row['image']])
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='passed' if parity['status']=='passed' else 'parity_failed',
        samples=len(rows),checkpoint_sha256=bound['selection']['sha256'],thresholds=bound['config']['thresholds'],
        threshold_policy='Identical frozen run thresholds for both artifacts; no calibration',
        input_sha256=bound['input_sha256'],script_sha256=sha(__file__),tensorflow=tf.__version__,
        source_sha256={p.name:sha(p) for p in [Path(__file__).with_name(name) for name in ('word_symbol_v4_export.py','word_decode_id_v4.py','word_symbols_id_v4.py','word_symbols.py','word_data.py')]},
        scope='Source-disjoint recomposed public photograph TEST; same chosen checkpoint conversion confirmation plus deployed-weight baseline with experimental v4 parser',
        physical_camera_validated=False,native_runtime_validated=False,checkpoint_selection_performed=False,
        comparison_replaces_native_test=False,native_torch_test_parity=parity,artifacts=comparison,
        elapsed_seconds=time.perf_counter()-started)
    write_new(bound['run']/'test-artifact-comparison.json',report)
    print(json.dumps({name:dict(gates=value['gates'],groups={group:g['exact_accuracy'] for group,g in value['groups'].items()}) for name,value in comparison.items()}),flush=True)
    if parity['status']!='passed':
        raise ValueError('TEST artifact decisions differ from native Torch; report saved, do not qualify deployment')


if __name__=='__main__':
    main()
