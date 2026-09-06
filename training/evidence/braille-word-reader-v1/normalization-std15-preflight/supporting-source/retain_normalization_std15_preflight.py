"""Immutable compact run29 evidence retention; stdlib only, no model execution."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent.parent
EVIDENCE=ROOT/'training/evidence/braille-word-reader-v1'
OUT=EVIDENCE/'normalization-std15-preflight'
RUN=WORK/'run29'
PARENT=WORK/'normalization-std15-parent-v1'
RUNTIME=WORK/'normalization-std15-runtime-v1'
EPOCH=RUN/'epoch01-independent-hdf5-audit'
sha=lambda data:hashlib.sha256(data).hexdigest()
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
relative=lambda path:Path(path).resolve().relative_to(ROOT).as_posix()


def checked(path,expected=None):
    data=Path(path).read_bytes()
    if expected is not None:assert sha(data)==expected,path
    return data


def main():
    assert not OUT.exists(),'Archive is immutable; use a new version instead.'
    cfg=read(RUN/'training-config.json');protocol=read(RUN/'normalization-std15-protocol.json')
    tests=read(RUN/'preflight-tests.json');review=read(RUN/'normalization-transition-guard-audit.json')
    initial=read(PARENT/'initialization-report.json');torch=read(RUNTIME/'torch-report.json');parity=read(RUNTIME/'parity-report.json')
    epoch=read(EPOCH/'report.json');launch=read(RUN/'launch-command.json')
    checked(RUN/'training-config.json','c20b26876b042015fdd580d6acfaf3509d34edbfb7f1fbcc1b8709d719ac5dfb')
    checked(RUN/'normalization-std15-protocol.json','78fcad0078192e567687e6600a0841e3e982ffac83e507c1c5ffec06e9a515a6')
    checked(RUN/'preflight-tests.json','909e76bd87ba5177a120b532c25563583eb4b824a805d4f597069c6c60bad55e')
    checked(RUN/'normalization-transition-guard-audit.json','c9cf75b0da2c0e98e6d982674a11c30f1d6dd062b2866745ef37430ec1d066a9')
    checked(EPOCH/'report.json','4af1a19ad6d04dd90b029188e29318376a969869bfc2b89c87c7b95c6aa8efb1')
    assert cfg['source_sha256']==protocol['source_sha256']==tests['source_sha256']==review['source_sha256']==epoch['training_source_sha256']
    assert len(cfg['source_sha256'])==24
    assert cfg['data_sha256']==protocol['data_sha256']
    assert cfg['normalization_transition']==tests['actual_loaded_initializer_guard']
    assert cfg['image_normalization']=='channel_gray_std15' and '.5+.15*gray' in cfg['normalization']
    assert cfg['imageNormalization']==parity['imageNormalization']==torch['imageNormalization']
    assert cfg['imageNormalization']['targetStd']==.15
    assert cfg['illumination_augmentation'] and cfg['dsbi_visible_partial_supervision']['enabled']
    for k in ('anchor_zoom_cap','horizontal_aspect_augmentation','vertical_stretch_augmentation',
              'horizontal_mirror_augmentation','camera_augmentation','half_turn_augmentation','polarity_augmentation'):
        assert not cfg[k] and not protocol[k]
    for k in ('partial_geometry_supervision','geometry_conditioned_residual','batchnorm_affine','fine_detail_branch'):
        assert not cfg[k]['enabled'] and not protocol[k]
    assert tests['status']==review['status']==parity['status']==torch['status']==epoch['status']=='passed'
    assert tests['tests_run']==21 and tests['failures']==tests['errors']==0
    start=datetime.fromisoformat(cfg['started_at_utc'])
    assert datetime.fromisoformat(protocol['created_at_utc'])<start
    assert datetime.fromisoformat(tests['completed_at_utc'])<start<datetime.fromisoformat(epoch['completed_at_utc'])
    for ref in protocol['preflight_audits']:checked(ref['path'],ref['sha256'])
    checked(PARENT/'initialization-report.json',protocol['initialization_report_sha256'])
    checked(RUNTIME/'parity-report.json',protocol['runtime_proof_sha256'])
    assert protocol['initialization_report_sha256']==parity['initialization_report_sha256']==torch['initialization_report_sha256']
    checked(RUNTIME/'torch-report.json',parity['torch_report_sha256'])
    checked(RUNTIME/'probe.py',parity['script_sha256'])
    assert parity['script_sha256']==torch['script_sha256']
    assert initial['all_model_arrays_byte_identical'] and initial['frozen_bn_arrays_byte_identical']==208
    assert initial['only_model_configuration_change']=={'path':'config.layers[1].config.target_std','from':.05,'to':.15}
    assert initial['full_graph_equal_except_target_std_and_compile_reset'] and initial['gradient_updates']==0
    assert epoch['model_arrays']==280 and epoch['model_parameters']==829307
    assert epoch['frozen_batchnorm_arrays_byte_identical']==208 and epoch['serialized_topology_equal']
    assert epoch['target_std']==.15 and not epoch['epoch_sampling_eligible']
    assert epoch['initial_checkpoint_sha256']==cfg['resume_sha256']==protocol['parent_checkpoint_sha256']==parity['initializer_sha256']
    checked(EPOCH/'training-config-snapshot.json',epoch['training_configuration_sha256'])
    checked(EPOCH/'validation-history-snapshot.json',epoch['callback_history_snapshot_sha256'])
    checked(EPOCH/'audit.py',epoch['audit_source_sha256'])
    checked(EPOCH/'hdf5_inventory_source.py',epoch['inventory_source_sha256'])
    checked(RUN/'preflight-tests.log',tests['log_sha256'])
    checked(WORK/'test_run29.py',tests['runner_sha256'])
    checked(WORK/'audit_run29_normalization_transition.py',review['script_sha256'])
    checked(WORK/'launch_run29.py',launch['launcher_sha256'])
    for name,digest in cfg['source_sha256'].items():checked(RUN/'source'/name,digest)
    for name,digest in tests['tests_sha256'].items():checked(RUN/'test-source'/name,digest)
    for name,digest in parity['source_sha256'].items():checked(RUNTIME/'source'/name,digest)
    for name,digest in initial['source_sha256'].items():
        checked(PARENT/name if name=='prepare.py' else PARENT/'source'/name,digest)

    OUT.mkdir(parents=True)
    payloads=[];deduplicated={}
    def retain(path,name,expected=None,compress=False):
        raw=checked(path,expected);rawsha=sha(raw)
        if (rawsha,compress) in deduplicated:return dict(deduplicated[(rawsha,compress)])
        data=gzip.compress(raw,compresslevel=9,mtime=0) if compress else raw
        dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();dest.write_bytes(data)
        ref={'file':relative(dest),'sha256':sha(data)}
        if compress:
            assert gzip.decompress(data)==raw
            ref.update(encoding='gzip',rawSha256=rawsha,rawSizeBytes=len(raw))
        payloads.append(dict(ref,sizeBytes=len(data),originalFile=relative(path)))
        deduplicated[(rawsha,compress)]=ref
        return dict(ref)
    def link(path,expected=None):return {'file':relative(path),'sha256':sha(checked(path,expected))}
    def omit(path,expected,role):
        raw=checked(path,expected)
        return {'originalWorkPath':relative(path),'sha256':sha(raw),'sizeBytes':len(raw),
                'role':role,'retainedInThisArchive':False,'reason':'Model binary or raw tensor omitted by design; exact bytes were hashed during retention.'}
    sources={n:retain(RUN/'source'/n,'run29/source/'+n,d) for n,d in cfg['source_sha256'].items()}
    test_sources={n:retain(RUN/'test-source'/n,'run29/test-source/'+n,d) for n,d in tests['tests_sha256'].items()}
    run_files={n:retain(RUN/n,'run29/'+n) for n in ('training-config.json','normalization-std15-protocol.json',
        'launch-command.json','dsbi-addition-protocol.json','dsbi-visible-partial-protocol.json',
        'preflight-tests.json','preflight-tests.log','normalization-transition-guard-audit.json')}
    helper_sources={n:retain(WORK/n,'supporting-source/'+n) for n in ('prepare_run29.py','launch_run29.py',
        'test_run29.py','audit_run29_normalization_transition.py')}
    helper_sources['retainer']=retain(__file__,'supporting-source/retain_normalization_std15_preflight.py')
    initialization={n:retain(PARENT/n,'initialization/'+n+('.gz' if n.endswith('.json') else ''),compress=n.endswith('.json'))
                    for n in ('initialization-report.json','parent-config.json','initialized-config.json','prepare.py')}
    initialization['sources']={n:retain(PARENT/'source'/n,'initialization/source/'+n,d)
        for n,d in initial['source_sha256'].items() if n!='prepare.py'}
    runtime={n:retain(RUNTIME/n,'runtime/'+n) for n in ('parity-report.json','torch-report.json','probe.py')}
    runtime['sources']={n:retain(RUNTIME/'source'/n,'runtime/source/'+n,d) for n,d in parity['source_sha256'].items()}
    runtime['inputIdentityReport']=retain(WORK/'geometry-head-parent-v1/geometry-initialization.json',
        'runtime/six-input-identity-original-report.json',parity['input_identity_report_sha256'])
    epoch_files={n:retain(EPOCH/n,'postlaunch-epoch01/'+n+('.gz' if n=='report.json' else ''),compress=n=='report.json')
                 for n in ('report.json','audit.py','hdf5_inventory_source.py','training-config-snapshot.json','validation-history-snapshot.json')}
    omitted=[omit(Path(initial['parent_checkpoint']),initial['parent_checkpoint_sha256'],'Base std05 v15 checkpoint'),
        omit(Path(initial['initialized_checkpoint']),initial['initialized_checkpoint_sha256'],'Actual std15 initialization'),
        omit(RUNTIME/'reload-check.keras',torch['roundtrip_checkpoint_sha256'],'Actual save/load roundtrip'),
        omit(RUNTIME/'std15-probe.tflite',parity['artifact_sha256'],'Actual builtins-only six-input conversion'),
        omit(RUNTIME/'torch-oracle.npz',parity['torch_oracle_sha256'],'Six fixed RGB inputs and Torch/base outputs'),
        omit(WORK/'geometry-head-parent-v1/initialization-oracle.npz',parity['input_oracle_sha256'],'Original fixed six-input oracle'),
        omit(RUNTIME/'tflite-six-outputs.npy',None,'Saved six Lite outputs; hash measured now, original parity report did not bind this separate NPY'),
        omit(Path(epoch['retained_checkpoint']),epoch['retained_checkpoint_sha256'],'Epoch01 trained checkpoint, not sampling eligible')]
    prior=read(EVIDENCE/'anchor-zoom-preflight/preflight.json')
    data_refs=[]
    for key,digest in cfg['data_sha256'].items():
        matches=[x for x in prior['linkedDataArchives'] if x['rawSha256']==digest]
        assert len(matches)==1,key
        ref=matches[0];data=checked(ROOT/ref['file'],ref['sha256'])
        assert sha(gzip.decompress(data))==digest
        data_refs.append(dict(ref,configurationKey=key))
    existing={name:link(EVIDENCE/name) for name in ('anchor-zoom-preflight/preflight.json',
        'data-manifests/index.json','dsbi-training-preparation/archive.json','dsbi-training-preparation/source-index.json',
        'dsbi-training-preparation/independent-context-audit.json','dsbi-visible-partial-preflight/archive.json',
        'dsbi-visible-partial-preflight/real-dsbi-point-run20-verifier-check.json')}
    parent_proofs={name:retain(WORK/name,'parent/'+name.replace('/','__'),digest) for name,digest in (
        ('run20/training-config.json',protocol['parent_training_config_sha256']),
        ('selected-v15/selected-validation.json',parity['selected_validation_sha256']))}
    report={'schemaVersion':1,'createdAtUtc':datetime.now(timezone.utc).isoformat(),'run':'run29',
        'scope':'Source/runtime/prelaunch evidence for the single embedded targetStd .05 to .15 transition, plus a separately identified postlaunch epoch01 HDF5 observation. Not final selection, recognition accuracy, native-camera proof or model activation.',
        'configuration':run_files['training-config.json'],'protocol':run_files['normalization-std15-protocol.json'],
        'sourceSnapshots':sources,'testSourceSnapshots':test_sources,'runFiles':run_files,'supportingSources':helper_sources,
        'initialization':initialization,'runtimeProof':runtime,'postlaunchEpoch01Audit':epoch_files,
        'parentProofs':parent_proofs,'omittedLargeArtifacts':omitted,'linkedDataArchives':data_refs,'linkedExistingProofs':existing,
        'verification':{'actualSources':24,'prelaunchTests':21,'testSourceFiles':len(test_sources),
            'actualLoadedInitializerGuard':True,'independentGuardSemanticNegativeCases':11,'independentCliIsolationCases':9,
            'sixInputRuntimeMaxAbs':max(parity[k] for k in ('torch_tf_max_abs','tf_tflite_max_abs','torch_tflite_max_abs')),
            'sixRawAndEffectiveDecisionsExact':True,'initialModelArraysUnchanged':280,'epoch01FrozenBNArraysUnchanged':208,
            'epoch01NotSamplingEligible':True,'newBackendExecutionByRetainer':False,'newTestsByRetainer':False},
        'limitations':['Initializer changes outputs for nonconstant inputs; parity is between runtimes on that initializer, never equality against std05.',
            'The initialization report was written before actual model loading; the separate later Torch/TF/Lite and loaded-guard reports close that specific pending preflight step.',
            'Six original probes comprise two positive TRAIN crops, one TRAIN negative, black, white and seeded random RGB. They are not a physical holdout or accuracy benchmark.',
            'The immutable independent guard report uses a model stand-in for negative branch tests; actual loaded initializer is separately covered by the21-test report.',
            'Epoch01 array changes prove stored updates only. Its source cycle is incomplete; no selected checkpoint is asserted.',
            'Original reports retain original paths. Repository-relative reference maps resolve retained bytes; model/tensor omissions are explicit and cannot be replayed from this compact archive alone.',
            'The index of24 source files includes dormant utility modules. Actual configuration leaves anchor cap, horizontal/mirror/vertical, geometry, BN-affine and fine branches off.']}
    dest=OUT/'preflight.json';dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');payloads.append(dict(link(dest),sizeBytes=dest.stat().st_size))
    readme=OUT/'README.md';readme.write_text('# Run29 embedded normalization preflight\n\n'
        'The actual initializer changes only CameraNormalization.target_std from0.05 to0.15, retaining all280 model arrays and the full-tail graph. '
        'Source24, actual configuration/protocol,21 source-bound prelaunch tests, independent guard review, and six fixed-input Torch/TF/Lite runtime evidence are retained. '
        'Nonconstant outputs change by design; runtime parity compares that same std15 initializer across runtimes.\n\n'
        'The postlaunch-epoch01 material is explicitly later evidence:208 BatchNorm arrays remain exact while convolution/head arrays changed. '
        'Epoch1 is not sampling eligible and does not establish recognition improvement. No active model contract or release gate was changed by this retainer.\n\n'
        'preflight.json maps each role to exact bytes. Large JSON inventories/configs use lossless gzip with raw hashes. '
        'Model binaries and raw tensors are omitted with original paths, measured sizes and verified hashes. Existing DSBI data/point archives are linked rather than copied. '
        'All retained references resolve within the repository; original reports preserve historical paths and wording. '
        'archive-index.json hashes every new payload. Full model replay additionally requires the explicitly omitted artifacts.\n',encoding='utf-8')
    payloads.append(dict(link(readme),sizeBytes=readme.stat().st_size))
    for ref in payloads:checked(ROOT/ref['file'],ref['sha256'])
    index={'scope':'All new immutable payloads; original model/tensor omissions and existing proof references are listed in preflight.json.',
        'files':sorted(payloads,key=lambda x:x['file']),'payloadCount':len(payloads),'payloadBytes':sum(x['sizeBytes'] for x in payloads)}
    ix=OUT/'archive-index.json';ix.write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','preflight':link(dest),'index':link(ix),
                      'payloadCount':index['payloadCount'],'payloadBytes':index['payloadBytes'],'omittedLargeArtifacts':len(omitted)},indent=2))


if __name__=='__main__':main()
