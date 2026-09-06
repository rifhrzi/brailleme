"""Freeze the single embedded std.15 adaptation from the unchanged v15 weights."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

WORK=Path(__file__).resolve().parent;SOURCE=Path('training/scripts')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
    assert not path.exists(),path;path.write_text(json.dumps(value,indent=2)+'\n')
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',required=True,type=Path);args=parser.parse_args()
    run=WORK/'run29';assert not (run/'training-config.json').exists()
    tests=run/'preflight-tests.json'
    for p in (args.audit,tests):assert json.loads(p.read_text())['status']=='passed'
    tree=ast.parse((SOURCE/'word_train.py').read_text());names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAINING_SOURCES' for t in n.targets))
    sources={n:sha(SOURCE/n) for n in names};assert len(sources)==24
    for n in names:(run/'source'/n).write_bytes((SOURCE/n).read_bytes())
    baseline=json.loads((WORK/'run20/training-config.json').read_text())
    for n,h in baseline['data_sha256'].items():assert sha(WORK/'data'/n)==h
    initializer=WORK/'normalization-std15-parent-v1/std15-parent.keras';initial=WORK/'normalization-std15-parent-v1/initialization-report.json';proof=WORK/'normalization-std15-runtime-v1/parity-report.json'
    control=WORK/'selected-v24/best-word.keras';base=WORK/'selected-v15/best-word.keras'
    assert sha(initializer)=='918d148a43399642f44c5d18693eb2733eecca338f0332fed8782cbe79beea00'
    assert sha(control)==sha(base)=='cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
    assert json.loads(proof.read_text())['status']=='passed'
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'source_sha256':sources,
      'parent_checkpoint':str(initializer),'parent_checkpoint_sha256':sha(initializer),'base_parent_checkpoint':str(base),'base_parent_checkpoint_sha256':sha(base),
      'control_checkpoint':str(control),'control_checkpoint_sha256':sha(control),'parent_training_config_sha256':sha(WORK/'run20/training-config.json'),
      'initialization_report':str(initial),'initialization_report_sha256':sha(initial),'runtime_proof':str(proof),'runtime_proof_sha256':sha(proof),
      'normalization_transition':True,'target_std_before':.05,'target_std_after':.15,
      'data_sha256':baseline['data_sha256'],'training_manifest_sha256':sha(WORK/'data'/baseline['train_manifest']),
      'planned_epochs':10,'planned_eligible_epochs':[8,9,10],'batch_size':24,'learning_rate':2e-5,'seed':20260905,
      'negative_repeat':27,'supplemental_negative_repeat':2,'geometry_loss':'relative_size_huber','illumination_augmentation':True,
      'anchor_zoom_cap':False,'horizontal_aspect_augmentation':False,'vertical_stretch_augmentation':False,'horizontal_mirror_augmentation':False,
      'camera_augmentation':False,'half_turn_augmentation':False,'polarity_augmentation':False,'partial_geometry_supervision':False,
      'dsbi_visible_partial_supervision':True,'frozen_batchnorm':True,'geometry_conditioned_residual':False,'batchnorm_affine':False,'fine_detail_branch':False,
      'point_metadata_sha256':sha(WORK/'data'/baseline['dsbi_point_metadata']),'supplemental_source_indexes':baseline['supplemental_source_indexes'],
      'selection':'Control std.05 first, then own-cycle eligible8-10 std.15 checkpoints on unchanged405. Weak-peaks-v2, >=95% accepted precision per natural/stress, maximize equal-group whole-word mean, ties preserve control. No external607/44/66/12/14 threshold or checkpoint selection.',
      'rationale':'A single embedded input-amplitude adaptation. TRAIN first-BN variance differs from frozen moments, but this is only a hypothesis, not a causal accuracy finding. Existing later full-tail/DSBI training used std.05; changing to fixed.15 is trained here without stacking failed augmentation branches.',
      'initializer_contract':'Exactly one serialized CameraNormalization.target_std .05→.15; all280 model arrays/full-tail829307params/52ordinaryfrozenBN preserved. Actual std.15 initializer itself passed runtime parity; no zero branch or second perturbed diagnostic.',
      'optimizer_and_sampling':'Fresh Adam/scheduler and same v15 seed/dataset phase restart, original RGB/augmentation/targets unchanged; only embedded normalization differs. Own23600 positive-record cycle required before selection, no inherited eligibility shortcut.',
      'preflight_audits':[{'path':str(p.resolve()),'sha256':sha(p)} for p in (args.audit,tests)],'no_activation_before_original_gates':True}
    shared={k:plan[k] for k in ('created_at_utc','source_sha256','parent_checkpoint','parent_checkpoint_sha256','base_parent_checkpoint_sha256','normalization_transition','target_std_after','anchor_zoom_cap','horizontal_aspect_augmentation','vertical_stretch_augmentation','horizontal_mirror_augmentation','selection')}
    for name in ('dsbi-addition-protocol.json','dsbi-visible-partial-protocol.json'):
        value=json.loads((WORK/'run20'/name).read_text());value.update(shared);value['scope']='One embedded normalization scale adaptation from unchanged v15 weights, same DSBI sources/points/partial labels/sampling/negative exposure, no runtime pixel or gate change.';value['preflight_audits'].extend(plan['preflight_audits']);save(run/name,value)
    save(run/'normalization-std15-protocol.json',plan);print(json.dumps({'protocol_sha256':sha(run/'normalization-std15-protocol.json'),'sources':len(sources)}))
if __name__=='__main__':main()
