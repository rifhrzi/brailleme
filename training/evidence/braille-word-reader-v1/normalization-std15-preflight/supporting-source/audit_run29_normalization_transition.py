"""Bounded source/guard and saved-six-output audit; no model/backend execution."""
import argparse
import ast
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
from unittest import mock
import numpy as np

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent.parent
RUN=WORK/'run29'
RUNTIME=WORK/'normalization-std15-runtime-v1'
INITIAL=WORK/'normalization-std15-parent-v1/initialization-report.json'
PARITY=RUNTIME/'parity-report.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def main():
    output=RUN/'normalization-transition-guard-audit.json'
    assert not output.exists()
    trainer=RUN/'source/word_train.py'
    guard=RUN/'source/word_normalization_transition.py'
    tests=read(RUN/'preflight-tests.json')
    for name,digest in tests['source_sha256'].items():
        assert sha(ROOT/'training/scripts'/name)==digest
        if (RUN/'source'/name).exists():assert sha(RUN/'source'/name)==digest
    for name in ('word_data.py','word_model.py','word_decode.py','word_sampling.py','word_point_geometry.py','word_point_metadata.py'):
        assert sha(ROOT/'training/scripts'/name)==sha(WORK/'run28/source'/name)
    old_tree=ast.parse((WORK/'run28/source/word_train.py').read_text())
    tree=ast.parse(trainer.read_text())
    pick=lambda t,name:next(x for x in t.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name==name)
    assert ast.dump(pick(tree,'Patches'))==ast.dump(pick(old_tree,'Patches'))
    for name in ('word_train.py','word_normalization_transition.py'):
        assert sha(RUN/'source'/name)==sha(ROOT/'training/scripts'/name)
    spec=importlib.util.spec_from_file_location('reviewed_transition',ROOT/'training/scripts/word_normalization_transition.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=read(PARITY);t=read(RUNTIME/'torch-report.json');initial=read(INITIAL)
    assert sha(PARITY)=='0ae7507e53b84780b44ad71cc58963a2f69ac57e1b17fd32b7f16e336292c28c'
    assert sha(INITIAL)==p['initialization_report_sha256']==t['initialization_report_sha256']
    assert sha(RUNTIME/'probe.py')==p['script_sha256']==t['script_sha256']
    assert sha(RUNTIME/'torch-report.json')==p['torch_report_sha256']
    assert sha(RUNTIME/'torch-oracle.npz')==p['torch_oracle_sha256']==t['torch_oracle_sha256']
    assert sha(initial['initialized_checkpoint'])==p['initializer_sha256']==initial['initialized_checkpoint_sha256']
    assert sha(initial['parent_checkpoint'])==p['base_checkpoint_sha256']==initial['parent_checkpoint_sha256']
    for name,digest in p['source_sha256'].items():
        assert sha(RUNTIME/'source'/name)==sha(ROOT/'training/scripts'/name)==digest
    assert initial['all_model_arrays_byte_identical'] and initial['frozen_bn_arrays_byte_identical']==208
    assert initial['full_graph_equal_except_target_std_and_compile_reset'] and initial['gradient_updates']==0
    current_spec=p['imageNormalization']
    legacy_spec=dict(current_spec,targetStd=.05)
    expected_text='RGB float32 /255; internal per-image per-channel mean and population std (ddof=0, std floor .001), mean standardized channels, .5+.05*gray clipped [0,1], repeat RGB, then Rescaling(2,-1)'
    assert module.training_normalization_metadata(legacy_spec)==('channel_gray_std05',expected_text)
    assert module.training_normalization_metadata(current_spec)==('channel_gray_std15',expected_text.replace('.05*gray','.15*gray'))
    assert module.training_normalization_metadata({'kind':'none'})==('raw','RGB float32 /255; internal Rescaling(2,-1)')
    for value in (0,-.1,float('nan'),float('inf')):
        try:module.training_normalization_metadata(dict(current_spec,targetStd=value))
        except ValueError:pass
        else:raise AssertionError('Invalid normalization metadata accepted')
    assert module.validated_normalization_transition(None,None,None,None,legacy_spec)=={'enabled':False}
    assert module.validated_normalization_transition(None,None,None,None,{'kind':'none'})=={'enabled':False}
    try:module.validated_normalization_transition(None,None,None,None,current_spec)
    except ValueError:pass
    else:raise AssertionError('Unproven std15 accepted')

    # Source-level guard branches use an explicit lightweight stand-in. Actual model
    # weights/runtime were exercised by the bound runtime probe, not by this audit.
    BN=type('BatchNormalization',(),{})
    bn=[BN() for _ in range(52)]
    for b in bn:b.trainable=False
    backbone=types.SimpleNamespace(layers=bn,outputs=[types.SimpleNamespace(shape=[None,192//s,480//s,16]) for s in [4,8,16,32]])
    model=types.SimpleNamespace(count_params=lambda:829307,get_weights=lambda:[np.zeros(1,np.float32)]*280,get_layer=lambda name:backbone)
    call=lambda:module.validated_normalization_transition(model,Path(initial['initialized_checkpoint']),INITIAL,PARITY,current_spec)
    with mock.patch.object(module,'weight_digest',return_value=p['weights_sha256']):
        assert call()['enabled']
        rejected=[]
        for name,target,field,value in [
            ('numeric_nan','parity','torch_tf_max_abs',float('nan')),
            ('numeric_negative','parity','tf_tflite_max_abs',-1e-5),
            ('numeric_too_large','parity','torch_tflite_max_abs',.001),
            ('custom_operator','parity','operators',['CUSTOM']),
            ('float64_interface','parity','input',dict(p['input'],dtype='float64')),
            ('moments_changed','torch','all_weights_and_moments_unchanged',False),
            ('topology_changed','torch','runtime_graph_change_only_target_std',False),
            ('missing_output_change','torch','parent_output_max_abs_delta_by_input',[0]*6),
            ('wrong_transition','initial','only_model_configuration_change',dict(initial['only_model_configuration_change'],to=.1)),
            ('raw_decoder_mismatch','parity','decisions',list(reversed(p['decisions'])))]:
            original_json_loads=json.loads
            texts={Path(INITIAL).read_text():'initial',PARITY.read_text():'parity',(RUNTIME/'torch-report.json').read_text():'torch'}
            def semantic_loads(text,*args,**kwargs):
                obj=original_json_loads(text,*args,**kwargs)
                if texts.get(text)==target:obj[field]=copy.deepcopy(value)
                return obj
            with mock.patch.object(module.json,'loads',side_effect=semantic_loads):
                try:call()
                except ValueError:rejected.append(name)
                else:raise AssertionError(name)
        bn[0].trainable=True
        try:call()
        except ValueError:rejected.append('trainable_BN')
        else:raise AssertionError('Trainable BN accepted')
        bn[0].trainable=False
    ns={'argparse':argparse,'Path':Path}
    exec(compile(ast.Module(body=[pick(tree,'parse_args')],type_ignores=[]),str(trainer),'exec'),ns)
    base=['--data','data','--output','out','--resume','initial.keras','--geometry-loss','relative_size_huber',
          '--illumination-augmentation','--dsbi-visible-partial-supervision','--dsbi-point-metadata','points.json',
          '--base-train-manifest','base.json','--supplemental-source-index','source-index.json','--supplemental-negative-repeat','2',
          '--normalization-initialization','initial.json','--normalization-runtime-proof','parity.json']
    assert ns['parse_args'](base).normalization_initialization==Path('initial.json')
    cli=[]
    for extra in [['--anchor-zoom-cap'],['--horizontal-aspect-augmentation'],['--vertical-stretch-augmentation'],
                  ['--horizontal-mirror-augmentation'],['--camera-augmentation'],['--half-turn-augmentation'],
                  ['--polarity-augmentation'],['--partial-geometry-supervision'],['--image-normalization','channel_gray_std05']]:
        with contextlib.redirect_stderr(io.StringIO()):
            try:ns['parse_args'](base+extra)
            except SystemExit as e:assert e.code==2;cli.append(extra[0])
            else:raise AssertionError(extra)
    source=trainer.read_text()
    assert 'load_model(args.resume, compile=False)' in source
    assert source.index('normalization_transition = validated_normalization_transition')<source.index('model.compile(')
    assert 'keras.optimizers.Adam(args.learning_rate, clipnorm=10.)' in source
    # Replay already saved six outputs only, with the exact frozen decoder source.
    sys.path.insert(0,str(RUNTIME/'source'))
    from word_bn_affine_parity import check_outputs,decision_rows
    oracle=np.load(RUNTIME/'torch-oracle.npz',allow_pickle=False)
    images=oracle['images'];outputs=oracle['outputs'];old=oracle['parent_outputs']
    lite=np.load(RUNTIME/'tflite-six-outputs.npy',allow_pickle=False)
    check_outputs(outputs);check_outputs(lite);check_outputs(old)
    np.testing.assert_allclose(lite,outputs,atol=1e-4,rtol=1e-4)
    assert decision_rows(outputs,images,p['thresholds'])==p['decisions']==decision_rows(lite,images,p['thresholds'])
    observed_delta=np.max(np.abs(old-outputs),axis=(1,2,3))
    np.testing.assert_array_equal(observed_delta,np.asarray(t['parent_output_max_abs_delta_by_input'],np.float32))
    max_abs=float(np.max(np.abs(lite-outputs)))
    assert max_abs==p['torch_tflite_max_abs']
    assert not any(x in sys.modules for x in ('torch','tensorflow','keras'))
    report={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':sha(__file__),'source_sha256':tests['source_sha256'],
        'focused_test_report_sha256':sha(RUN/'preflight-tests.json'),
        'initialization_report_sha256':sha(INITIAL),'runtime_proof_sha256':sha(PARITY),
        'initializer_sha256':p['initializer_sha256'],'parent_sha256':p['base_checkpoint_sha256'],
        'Patches_AST_exact_run28':True,'model_loss_data_decoder_point_sources_unchanged':True,
        'legacy05_identity_formula_exact':True,'raw_identity_formula_exact':True,'actual15_identity_formula_truthful':True,
        'explicit_initialization_proof_required':True,'semantic_negative_guard_cases':rejected,'rejected_isolation_flags':cli,
        'six_saved_Torch_Lite_decisions_replayed_exact':True,'saved_Torch_Lite_max_abs':max_abs,
        'loaded_with_compile_false_before_guard_and_fresh_Adam':True,'model_execution':False,'backend_modules_imported':[],
        'limits':['Semantic guard branches use a lightweight model stand-in and a mocked weight digest; this is not a second actual loaded-model check.',
                  'Bound runtime source/proof already executes the actual initializer and shows all280 arrays,52 frozen BN layers, full tail, save/reload and output parity. This audit reviews and binds that evidence.',
                  'Six TRAIN/analytic probes establish runtime behavior only; none is whole-word accuracy, sample eligibility or production evidence.',
                  'Default .05 Patches/loss/model paths are unchanged. Future std15 checkpoint continuation needs explicit lineage, not reuse of this initializer-only guard.']}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','file':str(output),'sha256':sha(output),'negative_guard_cases':len(rejected),'isolation_flags':len(cli)},indent=2))


if __name__=='__main__':main()
