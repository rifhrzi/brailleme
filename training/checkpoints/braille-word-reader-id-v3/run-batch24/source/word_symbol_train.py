"""Evaluate then run at most one ten-epoch continuation for Indonesian symbols.

Frozen baseline thresholds are shared by all candidates. Public observed external
examples never enter training or checkpoint selection. All original files survive.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
os.environ['KERAS_BACKEND']='torch'
import numpy as np
from PIL import Image
import keras
import torch
from word_data import save_json
from word_decode import decode,summarize,DECODER_VERSION
from word_symbols import parse_word_masks
from word_model import word_relative_loss,image_normalization_spec
from word_train import Patches
from word_sources import audit_supplemental_sources,expand_negative_records
from word_point_metadata import load_point_states


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(outputs,rows,thresholds):
    decisions=[decode(output,thresholds,row.get('selection')) for output,row in zip(outputs,rows)]
    groups={}
    for group in sorted({row['symbol_group'] for row in rows}):
        pairs=[(row,decision) for row,decision in zip(rows,decisions) if row['symbol_group']==group]
        accepted=sum(bool(decision['text']) for _,decision in pairs)
        correct=sum(bool(row['expected']) and row['expected']==decision['text'] for row,decision in pairs)
        if group=='rejection':
            groups[group]=dict(total=len(pairs),false_acceptances=accepted,all_rejected=accepted==0)
        else:
            groups[group]=dict(total=len(pairs),accepted=accepted,correct=correct,exact_word_accuracy=correct/len(pairs),accepted_accuracy=correct/accepted if accepted else 0.)
    rows_out=[dict(source=row['source'],group=row['symbol_group'],expected=row['expected'],predicted=result['text'],reason=result['reason']) for row,result in zip(rows,decisions)]
    return groups,rows_out


def ranking(groups,baseline):
    regressions=[name for name in ('legacy_natural','legacy_stress') if groups[name]['exact_word_accuracy']<baseline[name]['exact_word_accuracy'] or groups[name]['accepted_accuracy']<baseline[name]['accepted_accuracy']]
    positives=[value for name,value in groups.items() if name!='rejection']
    precision=min(value['accepted_accuracy'] for value in positives)
    exact=float(np.mean([value['exact_word_accuracy'] for value in positives]))
    eligible=precision>=.95 and groups['rejection']['all_rejected']
    return (not regressions,eligible,exact if eligible else precision,exact),regressions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--epochs',type=int,default=10,choices=range(1,11))
    parser.add_argument('--batch-size',type=int,default=24,choices=[24,12])
    args=parser.parse_args(); start=time.perf_counter()
    experiment=args.experiment.resolve();repo=args.repo.resolve()
    run=experiment/f'run-batch{args.batch_size}'
    if (run/'training-config.json').exists():
        raise SystemExit('Use a new run; an existing bounded experiment must not be silently repeated')
    if args.batch_size==12 and not (experiment/'run-batch24/oom.json').exists():
        raise SystemExit('Batch12 is permitted only after the original attempt ran out of GPU memory')
    run.mkdir(exist_ok=True)
    freeze=json.loads((experiment/'baseline-freeze.json').read_text())
    checkpoint=experiment/'baseline.keras'
    expected={name.replace('\\','/'):value for name,value in freeze['files'].items()}['training/checkpoints/braille-word-reader-v1/best-word.keras']
    if sha(checkpoint)!=expected:
        raise SystemExit('Frozen baseline checkpoint changed')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required for this authorized GPU continuation')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')
    keras.utils.set_random_seed(20260908)
    data=repo/'.braille-training-work/word-scan-v1-20260905/data'
    symbols=json.loads((experiment/'validation-symbols.json').read_text())['records']
    for row in symbols:
        parsed=parse_word_masks(row['masks'])
        if parsed.get('text','')!=row['expected'] or bool(row['expected'])!=(parsed['status']=='success'):
            raise ValueError('Symbol corpus disagrees with the audited Indonesian grammar')
    old_validation=json.loads((data/'validation-stress-selection.json').read_text())['records']
    # Old records contain only A-Z cells; text case changes only in this new copy.
    legacy=[dict(row,image=str(data/row['image']),expected=row['expected'].lower(),symbol_group='legacy_'+row['validation_group']) for row in old_validation]
    records=legacy+symbols
    fixture=experiment/'selection-fixtures.json'
    save_json(fixture,dict(scope='legacy_public_validation_and_separately_reported_recomposed_symbol_validation',records=records))
    legacy_fixture=experiment/'legacy-selection-fixtures.json'
    save_json(legacy_fixture,dict(scope='unchanged405_geometry_with_lowercase_expected_text_for_new_decoder',records=legacy))
    thresholds=json.loads((repo/'assets/models/braille-word-reader-v1/manifest.json').read_text())['thresholds']
    pixels=np.stack([np.asarray(Image.open(row['image']).convert('RGB'),np.float32)/255. for row in records])
    luminance=pixels@np.asarray([.2126,.7152,.0722],np.float32)
    quality=(luminance.mean((1,2))<.06)|(luminance.mean((1,2))>.985)|(luminance.std((1,2))<.01)
    del luminance
    sources={name:sha(Path(__file__).parent/name) for name in ['word_symbol_train.py','word_symbol_prepare.py','word_symbols.py','word_decode.py','word_model.py','word_data.py','word_geometry_head.py','word_bn_affine.py','word_anchor_zoom.py','word_train.py','word_point_metadata.py','word_sources.py']}
    for name in sources:
        target=run/'source'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(Path(__file__).parent/name,target)
    model=keras.models.load_model(checkpoint,compile=False)
    def evaluate(model):
        outputs=model.predict(pixels,batch_size=4,verbose=0)
        if not np.isfinite(outputs).all():
            raise ValueError('Non-finite validation output')
        scoring=outputs.copy();scoring[quality]=0
        groups,decisions=metrics(scoring,records,thresholds)
        return outputs,groups,decisions
    outputs,baseline,decisions=evaluate(model)
    save_json(run/'baseline-evaluation.json',dict(checkpoint_sha256=expected,decoder_version=DECODER_VERSION,thresholds=thresholds,groups=baseline,decisions=decisions,physical_camera_validated=False))
    print('BASELINE '+json.dumps(baseline),flush=True)
    best_rank,_=ranking(baseline,baseline)
    best=dict(path=checkpoint,sha=expected,rank=best_rank,groups=baseline,outputs=outputs,epoch=0)
    needs_training=not baseline['rejection']['all_rejected'] or any(baseline[group]['exact_word_accuracy']<.75 or baseline[group]['accepted_accuracy']<.95 for group in ['lowercase','capital_single','capital_word','numbers'])
    config=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),epochs_limit=args.epochs,batch_size=args.batch_size,learning_rate=.00002,baseline_checkpoint_sha256=expected,training_triggered=needs_training,decoder_version=DECODER_VERSION,source_sha256=sources,fixture_sha256=sha(fixture),thresholds=thresholds,threshold_policy='Frozen pre-update thresholds, identical for baseline and every epoch; no external tuning',selection_rule='Disallow decline of natural or stress legacy exact/accepted accuracy; prefer all-group95% precision and zero mandatory false accepts then mean exactness; else weakest-group precision then mean exactness',physical_camera_validated=False)
    save_json(run/'training-config.json',config)
    save_json(run/'environment.json',dict(python=subprocess.check_output([os.sys.executable,'--version'],text=True).strip(),keras=keras.__version__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),packages=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines()))
    epoch_reports=[]
    if needs_training:
        train_rows=json.loads((data/'../combined-training-v5/combined-training-manifest.json').read_text())
        indexes=[Path('../dsbi-training-contexts-v2/source-index.json')]
        audit=audit_supplemental_sources(train_rows,data,Path('../complete-source-training-v3/combined-training-manifest.json'),indexes)
        points,point_audit=load_point_states(data/'../dsbi-point-metadata-v1/points.json',data,indexes,train_rows)
        train_rows,expansion=expand_negative_records(train_rows,27,audit['supplemental_sources'],2)
        symbol_rows=json.loads((experiment/'train-symbols.json').read_text())['records']
        train_rows+=symbol_rows
        save_json(run/'training-data-audit.json',dict(source_audit=audit,point_audit=point_audit,negative_expansion=expansion,synthetic_count=len(symbol_rows),epoch_record_count=len(train_rows),symbol_manifest_sha256=sha(experiment/'train-symbols.json')))
        training=Patches(data,train_rows,args.batch_size,True,20260908,illumination_augmentation=True,balance_source_families=True,point_states=points)
        validation=Patches(data,json.loads((data/'validation.json').read_text()),args.batch_size,False,20260909)
        model.compile(optimizer=keras.optimizers.Adam(.00002,clipnorm=10.),loss=word_relative_loss,jit_compile=False)
        class CaptureEpoch(keras.callbacks.Callback):
            def on_epoch_end(self,epoch,logs=None):
                nonlocal best
                path=run/'retained-epochs'/f'epoch{epoch+1:02d}.keras';path.parent.mkdir(exist_ok=True)
                model.save(path)
                fresh,groups,decisions=evaluate(model)
                rank,regressions=ranking(groups,baseline)
                report=dict(epoch=epoch+1,checkpoint_sha256=sha(path),groups=groups,regressions=regressions,rank=rank,logs={k:float(v) for k,v in (logs or {}).items()})
                save_json(run/f'epoch{epoch+1:02d}-evaluation.json',dict(report,decisions=decisions))
                epoch_reports.append(report);save_json(run/'epoch-history.json',epoch_reports)
                if rank>best['rank']:
                    best=dict(path=path,sha=sha(path),rank=rank,groups=groups,outputs=fresh,epoch=epoch+1)
                print('EPOCH_EVALUATION '+json.dumps(report),flush=True)
        try:
            history=model.fit(training,validation_data=validation,epochs=args.epochs,shuffle=False,verbose=2,callbacks=[CaptureEpoch(),keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=4,factor=.5,min_lr=1e-5),keras.callbacks.TerminateOnNaN()])
        except torch.OutOfMemoryError as error:
            save_json(run/'oom.json',dict(error=str(error),completed_epochs=len(epoch_reports),fallback='New run-batch12 starts from frozen original baseline; no previous checkpoints removed'))
            raise
        save_json(run/'history.json',dict(elapsed_seconds=time.perf_counter()-start,history=history.history))
    selection=experiment/'selection';selection.mkdir(exist_ok=False)
    shutil.copy2(best['path'],selection/'best-word.keras')
    fixture_hash=sha(legacy_fixture)
    oracle=dict(backend='torch',tf32_enabled=False,fixture_file=str(legacy_fixture),fixture_sha256=fixture_hash,checkpoint_sha256=best['sha'],source_sha256=sources,decoder_version=DECODER_VERSION)
    selected=dict(source_checkpoint=str(best['path']),checkpoint_sha256=best['sha'],decoder_version=DECODER_VERSION,thresholds=thresholds,oracle=oracle,groups=best['groups'],epoch=best['epoch'],imageNormalization=image_normalization_spec(model),quality_rejected_before_decode=int(quality.sum()),status='unverified_candidate',physical_camera_validated=False)
    legacy_scoring=best['outputs'][:len(legacy)].copy();legacy_scoring[quality[:len(legacy)]]=0
    selected['words']=summarize(legacy_scoring,legacy,thresholds)
    selected['validation_groups']={name:best['groups']['legacy_'+name] for name in ['natural','stress']}
    selected['threshold_origin']='Frozen baseline thresholds; selection checked both legacy groups and separately labeled synthetic symbol groups'
    selected['quality_rejected_before_decode']=int(quality[:len(legacy)].sum())
    for name in ['selected-validation.json','selected-validation-fullprecision.json']:
        save_json(selection/name,selected)
    np.savez_compressed(selection/'backend-fullprecision.npz',outputs=best['outputs'][:len(legacy)],quality_rejected=quality[:len(legacy)],fixture_sha256=fixture_hash,checkpoint_sha256=best['sha'])
    np.savez_compressed(selection/'symbol-backend-fullprecision.npz',outputs=best['outputs'][len(legacy):],quality_rejected=quality[len(legacy):],fixture_sha256=sha(experiment/'validation-symbols.json'),checkpoint_sha256=best['sha'])
    save_json(selection/'symbol-evaluation.json',dict(checkpoint_sha256=best['sha'],decoder_version=DECODER_VERSION,groups={name:value for name,value in best['groups'].items() if not name.startswith('legacy_')},scope='synthetic_recomposed_public_cell_photographs_not_natural_indonesian_or_physical_camera_accuracy',fixture_sha256=sha(experiment/'validation-symbols.json'),baseline_groups=baseline,training_epochs_completed=len(epoch_reports),selected_epoch=best['epoch']))
    save_json(selection/'checkpoint-selection.json',dict(selected_checkpoint=str(best['path']),selected_sha256=best['sha'],baseline_retained=best['epoch']==0,candidates=epoch_reports,baseline=baseline,configuration=config))
    print('SELECTED '+json.dumps({k:v for k,v in selected.items() if k!='oracle'}),flush=True)


if __name__=='__main__':
    main()
