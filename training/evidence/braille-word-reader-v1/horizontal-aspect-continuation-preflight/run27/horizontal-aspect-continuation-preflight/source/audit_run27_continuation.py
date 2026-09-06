"""Independent CPU source/command/sampler continuation guard; no ML imports."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

WORK=Path(__file__).resolve().parent
RUN=WORK/'run27'
ROOT=WORK.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    output=RUN/'continuation-guard-audit.json'
    if output.exists():
        raise RuntimeError('Append-only audit already exists')
    plan_path=RUN/'run27-plan.json'
    plan=read(plan_path)
    assert sha(plan_path)=='f3b9f59cf8acaf7fa39fe3fe36c75279c0423d3b5bc50fd1c552469bc0030dbb'
    parent=read(WORK/'run26/training-config.json')
    inherited=plan['continuation']
    for key in ('parent_training_config','parent_sampling_lineage','parent_sampling_history',
                'inherited_horizontal_aspect_protocol','inherited_transform_audit','inherited_source_bound_tests'):
        assert sha(inherited[key]['path'])==inherited[key]['sha256']
    assert parent['source_sha256']==plan['source_sha256'] and len(plan['source_sha256'])==22
    for name,digest in plan['source_sha256'].items():
        assert sha(ROOT/'training/scripts'/name)==sha(RUN/'source'/name)==sha(WORK/'run26/source'/name)==digest
    for path,digest in parent['data_sha256'].items():
        assert sha(WORK/'data'/path)==digest
    old_audit=read(inherited['inherited_transform_audit']['path'])
    assert old_audit['parent_checkpoint_sha256']==parent['resume_sha256']==inherited['original_initial_checkpoint_sha256']
    assert plan['parent_checkpoint_sha256']!=old_audit['parent_checkpoint_sha256']
    for name in ('learning_rate','batch_size','seed','geometry_loss','negative_repeat','supplemental_negative_repeat',
                 'illumination_augmentation','horizontal_aspect_augmentation','horizontal_mirror_augmentation',
                 'vertical_stretch_augmentation','camera_augmentation','half_turn_augmentation','polarity_augmentation'):
        assert plan[name]==parent[name]
    assert plan['planned_epochs']==10 and plan['planned_eligible_epochs']==[8,9,10]
    for name in ('dsbi_visible_partial_supervision','partial_geometry_supervision','batchnorm_affine','fine_detail_branch','geometry_conditioned_residual'):
        assert plan[name]==parent[name]['enabled']
    lineage=read(inherited['parent_sampling_lineage']['path'])
    history=read(inherited['parent_sampling_history']['path'])
    e10=lineage['actual']['observed_epochs'][9]
    assert lineage['status']=='passed' and lineage['actual']['config_sha256']==sha(WORK/'run26/training-config.json')
    assert lineage['actual']['sampling_history_sha256']==sha(inherited['parent_sampling_history']['path'])
    assert e10['epoch']==10 and e10['checkpoint_sha256']==plan['parent_checkpoint_sha256']
    assert e10['eligible_by_sampling_lineage'] and e10['all_positive_records_seen_in_this_run']
    assert lineage['eligible_retained_epochs']==[8,9,10]

    tree=ast.parse((RUN/'source/word_train.py').read_text())
    main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    model_assignment=next(n for n in main_node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='model' for t in n.targets))
    compile_node=next(n for n in main_node.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='model.compile')
    fit_call=next(n for n in ast.walk(main_node) if isinstance(n,ast.Call) and ast.unparse(n.func)=='model.fit')
    assert 'initial_epoch' not in {k.arg for k in fit_call.keywords}
    scheduler=next(n for n in ast.walk(fit_call) if isinstance(n,ast.Call) and ast.unparse(n.func)=='keras.callbacks.ReduceLROnPlateau')
    assert {k.arg:ast.literal_eval(k.value) for k in scheduler.keywords}==dict(monitor='val_loss',patience=4,factor=.5,min_lr=1e-5)
    assert any(isinstance(n,ast.Call) and ast.unparse(n.func)=='keras.utils.set_random_seed' and ast.unparse(n.args[0])=='args.seed' for n in ast.walk(main_node))
    calls=[]
    class Model:
        optimizer=object()
        def compile(self,**kwargs):
            calls.append(('compile',kwargs))
    model=Model()
    old_optimizer=model.optimizer
    def load_model(path,**kwargs):
        calls.append(('load',path,kwargs))
        return model
    def adam(lr,**kwargs):
        obj=SimpleNamespace(lr=lr,**kwargs)
        calls.append(('Adam',obj))
        return obj
    args=SimpleNamespace(resume=plan['parent_checkpoint'],learning_rate=plan['learning_rate'],geometry_loss=plan['geometry_loss'])
    env={'args':args,'keras':SimpleNamespace(models=SimpleNamespace(load_model=load_model),optimizers=SimpleNamespace(Adam=adam)),
         'word_relative_loss':object(),'word_loss':object()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[model_assignment,compile_node],type_ignores=[])),str(RUN/'source/word_train.py'),'exec'),env)
    assert calls[0]==('load',plan['parent_checkpoint'],{'compile':False})
    assert calls[1][0]=='Adam' and calls[1][1].lr==2e-5 and calls[1][1].clipnorm==10.
    assert calls[2][1]['optimizer'] is calls[1][1] and calls[2][1]['optimizer'] is not old_optimizer
    assert calls[2][1]['loss'] is env['word_relative_loss'] and calls[2][1]['jit_compile'] is False

    sampler=module(RUN/'source/word_sampling.py','audit_sampling')
    sources=module(RUN/'source/word_sources.py','audit_sources')
    width=module(RUN/'source/word_horizontal_aspect.py','audit_width')
    training=read(WORK/'data'/parent['train_manifest'])
    supplemental=[]
    for path in parent['supplemental_source_indexes']:
        supplemental.extend(row['source'] for row in read(WORK/'data'/path)['sources'])
    records,expansion=sources.expand_negative_records(training,27,supplemental,2)
    assert expansion==parent['negative_expansion']
    class Dataset:
        def __init__(self,**kwargs):
            pass
    patches=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Patches')
    patch_env={'keras':SimpleNamespace(utils=SimpleNamespace(PyDataset=Dataset)),'np':np,'math':math,'BalancedFamilyOrder':sampler.BalancedFamilyOrder}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[patches],type_ignores=[])),str(RUN/'source/word_train.py'),'exec'),patch_env)
    datasets=[patch_env['Patches'](WORK/'data',records,24,True,20260905,illumination_augmentation=True,balance_source_families=True,horizontal_aspect_augmentation=True) for _ in range(2)]
    assert all(d.epoch==1 for d in datasets)
    # The already-observed same Keras environment reset before epoch1 is phase2.
    # This is an expected replay preflight, not an observation of future run27.
    for d in datasets:
        d.on_epoch_end()
    orders=[]
    for number,old in enumerate(history):
        a,b=datasets
        assert a.epoch==b.epoch==old['dataset_epoch']==number+2
        assert np.array_equal(a.order,b.order)
        digest=hashlib.sha256(np.asarray(a.order,dtype='<i8').tobytes()).hexdigest()
        assert digest==old['order_int64_sha256']
        assert len(a)==len(b)==old['completed_batches']==1037
        width_checks=[]
        for batch in (0,len(a)//2,len(a)-1):
            indices=a.order[batch*24:(batch+1)*24]
            factors=width.horizontal_aspect_factors(20260905,a.epoch,indices)
            np.testing.assert_array_equal(factors,width.horizontal_aspect_factors(20260905,b.epoch,b.order[batch*24:(batch+1)*24]))
            width_checks.append(hashlib.sha256(factors.tobytes()).hexdigest())
        orders.append({'local_epoch':number+1,'expected_dataset_epoch':a.epoch,'order_int64_sha256':digest,'sample_width_factor_sha256':width_checks})
        for d in datasets:
            d.on_epoch_end()
    assert len(orders)==10
    assert 'self.seed + self.epoch * 100000 + index' in ast.unparse(patches)

    launcher=module(WORK/'launch_run27.py','run27_launcher_guard')
    actual_command=launcher.command_for(plan)
    old_command=read(WORK/'run26/launch-command.json')['command']
    changed=[i for i,(a,b) in enumerate(zip(old_command,actual_command,strict=True)) if a!=b]
    assert changed==sorted([old_command.index('--output')+1,old_command.index('--resume')+1])
    assert actual_command[actual_command.index('--resume')+1]==plan['parent_checkpoint']
    assert actual_command[actual_command.index('--output')+1]==str(RUN)
    # Before freeze only, satisfy the report-list gate using independent already
    # frozen historical PASS documents. This validates source/command checks;
    # these stand-ins are not written into a final protocol or called run27 audits.
    probe=copy.deepcopy(plan)
    probe['reviewed_plan_sha256']=sha(plan_path)
    probe['preflight_audits']=[inherited['inherited_transform_audit'],inherited['inherited_source_bound_tests']]
    launcher.validate(probe)
    negative=[]
    for name,change in (
        ('parent_sha',lambda p:p.update(parent_checkpoint_sha256='0'*64)),
        ('learning_rate',lambda p:p.update(learning_rate=5e-5)),
        ('partial_geometry_metadata',lambda p:p.update(partial_geometry_supervision=True)),
        ('point_metadata_hash',lambda p:p.update(point_metadata_sha256='0'*64)),
        ('early_eligibility',lambda p:p.update(planned_eligible_epochs=[1,8,9,10])),
        ('optimizer_description',lambda p:p['continuation'].update(optimizer='restore optimizer slots')),
        ('historical_parent_relabel',lambda p:p['continuation'].update(original_initial_checkpoint_sha256=p['parent_checkpoint_sha256'])),
    ):
        bad=copy.deepcopy(probe);change(bad)
        try:
            launcher.validate(bad)
        except (AssertionError,ValueError):
            negative.append(name)
        else:
            raise AssertionError(f'Launcher allowed contradictory protocol: {name}')
    # The command file is never mutated. Intercept its digest only to verify the
    # inherited-command binding rejects a different file before launch.
    original_sha=launcher.sha
    launcher.sha=lambda p:'0'*64 if Path(p).resolve()==(WORK/'run26/launch-command.json').resolve() else original_sha(p)
    try:
        launcher.command_for(probe)
    except AssertionError:
        negative.append('parent_launch_hash')
    else:
        raise AssertionError('Launcher did not pin inherited command')
    finally:
        launcher.sha=original_sha
    imported=[name for name in ('tensorflow','torch','keras') if name in sys.modules]
    assert not imported
    report={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Independent CPU continuation/source/command/reset preflight; native separately audits actual checkpoint HDF5 graph/BN.',
            'plan_sha256':sha(plan_path),'script_sha256':sha(__file__),'launcher_sha256':sha(WORK/'launch_run27.py'),
            'preparation_script_sha256':sha(WORK/'prepare_run27.py'),'parent_launch_sha256':sha(WORK/'run26/launch-command.json'),
            'parent_checkpoint_sha256':plan['parent_checkpoint_sha256'],'parent_training_config_sha256':sha(WORK/'run26/training-config.json'),
            'parent_sampling_lineage_sha256':sha(inherited['parent_sampling_lineage']['path']),
            'inherited_horizontal_aspect_protocol_sha256':inherited['inherited_horizontal_aspect_protocol']['sha256'],
            'inherited_transform_audit_sha256':inherited['inherited_transform_audit']['sha256'],
            'historical_transform_audit_original_parent_preserved':True,'source_sha256':plan['source_sha256'],
            'data_sha256':parent['data_sha256'],'sources_equal_run26_current_and_snapshots':True,
            'data_hashes_equal_run26':True,'profile_equal_run26':True,'new_parent_epoch10_sampling_eligible':True,
            'actual_source_mock_load_compile_false_fresh_adam':True,'fresh_scheduler_same_policy':True,
            'no_initial_epoch_or_optimizer_restore':True,'fresh_dataset_starts_epoch1':True,
            'expected_pre_fit_reset_to_epoch2_matches_parent_history':True,
            'expected_ten_epoch_order_replay':orders,'sample_width_factors_replay_exact':True,
            'legacy_pixel_rng_expression_unchanged':True,'negative_expansion':expansion,
            'own_cycle_required_local_eligible_epochs':[8,9,10],'parent_coverage_does_not_qualify_early_local_epochs':True,
            'command_changed_arguments_only':['--output','--resume'],'command':actual_command,
            'negative_guard_cases':negative,'backend_modules_imported':[],
            'new_training_or_model_execution':False,'benchmark_pixels_or_predictions_read':False,
            'future_observation_required':'Actual run27 callback phase/order/full batches must still be checked; GPU optimization is not claimed byte-deterministic.',
            'pre_freeze_validation_probe':'Historical PASS refs used only as in-memory stand-ins for audit list; never written into final protocol. Final freeze binds actual two new reports.'}
    with output.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'status':'passed','report':str(output),'sha256':sha(output),'negative_cases':negative},indent=2))


if __name__=='__main__':
    main()
