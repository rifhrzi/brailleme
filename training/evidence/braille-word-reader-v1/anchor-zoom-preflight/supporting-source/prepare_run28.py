"""Freeze one isolated anchor-preserving zoom ablation from v15."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

WORK=Path(__file__).resolve().parent;SOURCE=Path('training/scripts')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',required=True,type=Path)
    parser.add_argument('--tests',required=True,type=Path)
    args=parser.parse_args()
    for p in (args.audit,args.tests):assert json.loads(p.read_text())['status']=='passed'
    run=WORK/'run28';run.mkdir(exist_ok=True);(run/'source').mkdir(exist_ok=True)
    assert not (run/'training-config.json').exists()
    tree=ast.parse((SOURCE/'word_train.py').read_text())
    names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAINING_SOURCES' for t in n.targets))
    sources={name:sha(SOURCE/name) for name in names};assert len(sources)==23
    for name in names:(run/'source'/name).write_bytes((SOURCE/name).read_bytes())
    parent=WORK/'selected-v15/best-word.keras';control=WORK/'selected-v23/best-word.keras'
    expected='cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
    assert sha(parent)==sha(control)==expected
    baseline=json.loads((WORK/'run20/training-config.json').read_text())
    for name,digest in baseline['data_sha256'].items():assert sha(WORK/'data'/name)==digest
    private=WORK/'anchor-preserving-zoom-prototype-v2'
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'source_sha256':sources,
        'parent_checkpoint':str(parent),'parent_checkpoint_sha256':expected,
        'control_checkpoint':str(control),'control_checkpoint_sha256':expected,
        'parent_training_config_sha256':sha(WORK/'run20/training-config.json'),
        'parent_selected_validation_sha256':sha(WORK/'selected-v15/selected-validation.json'),
        'data_sha256':baseline['data_sha256'],'training_manifest_sha256':sha(WORK/'data'/baseline['train_manifest']),
        'planned_epochs':10,'planned_eligible_epochs':[8,9,10],'batch_size':24,'learning_rate':2e-5,'seed':20260905,
        'negative_repeat':27,'supplemental_negative_repeat':2,'geometry_loss':'relative_size_huber',
        'illumination_augmentation':True,'anchor_zoom_cap':True,'horizontal_aspect_augmentation':False,
        'vertical_stretch_augmentation':False,'horizontal_mirror_augmentation':False,'camera_augmentation':False,
        'half_turn_augmentation':False,'polarity_augmentation':False,'partial_geometry_supervision':False,
        'dsbi_visible_partial_supervision':True,'frozen_batchnorm':True,'geometry_conditioned_residual':False,
        'batchnorm_affine':False,'fine_detail_branch':False,'point_metadata_sha256':sha(WORK/'data'/baseline['dsbi_point_metadata']),
        'supplemental_source_indexes':baseline['supplemental_source_indexes'],
        'selection':'Control first, own actual full-cycle eligible epochs8-10 on unchanged405; weak-peaks-v2, >=95% accepted precision separately per natural/stress, maximize equal-group whole-word mean, control wins ties. No external607/44/66/12/14 selection.',
        'rationale':'TRAIN-only paired target audit shows oversized short contexts lose six-bit supervision under the existing zoom. Preserve one original full known cell during the existing affine when feasible, without changing labels or runtime/decoder.',
        'anchor_zoom_policy':json.loads((private/'prototype-spec.json').read_text())['policy'],
        'optimizer_and_sampling':'Fresh Adam/scheduler, same seed and source-balanced dataset epoch restart as v15/run20. Old RNG draws deliberately repeat; only capped affine geometry changes. Own complete23600-positive-record cycle required, no inherited eligibility shortcut.',
        'private_prototype':{'directory':str(private),'spec_sha256':sha(private/'prototype-spec.json'),'proof_sha256':sha(private/'proof-report.json'),'integration_sha256':sha(private/'integration-record.json')},
        'preflight_audits':[{'path':str(p.resolve()),'sha256':sha(p)} for p in (args.audit,args.tests)],
        'no_activation_before_original_gates':True}
    shared={k:plan[k] for k in ('created_at_utc','source_sha256','parent_checkpoint','parent_checkpoint_sha256','anchor_zoom_cap','horizontal_aspect_augmentation','vertical_stretch_augmentation','horizontal_mirror_augmentation','selection')}
    for name in ('dsbi-addition-protocol.json','dsbi-visible-partial-protocol.json'):
        value=json.loads((WORK/'run20'/name).read_text());value.update(shared)
        value['scope']='Isolated anchor zoom-cap ablation from unchanged plain v15; same DSBI sources, exact points, partial semantics, architecture, loss, BN and gates.'
        value['preflight_audits'].extend(plan['preflight_audits']);save(run/name,value)
    save(run/'anchor-zoom-protocol.json',plan)
    print(json.dumps({'protocol_sha256':sha(run/'anchor-zoom-protocol.json'),'sources':len(sources)}))

if __name__=='__main__':main()
