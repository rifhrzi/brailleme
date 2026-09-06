"""One focused CPU proof: boundary algebra, actual Patches, final target survival."""
import ast
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from unittest import mock
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

OUT=Path(__file__).resolve().parent;WORK=OUT.parent;DATA=WORK/'data'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
START=datetime.now(timezone.utc).isoformat()
assert not (OUT/'proof-report.json').exists()
sys.path.insert(0,str(OUT/'source-private'))
from word_anchor_zoom import anchor_preserving_zoom
from word_point_geometry import visible_partial_candidates
from word_point_metadata import load_point_states
from word_sampling import BalancedFamilyOrder
from word_sources import expand_negative_records
from word_horizontal_aspect import horizontal_aspect_factors
from word_vertical_stretch import vertical_stretch_factors

def load_data(folder):
    names={'affine_patch','augment_patch','make_target','mirror_patch_horizontal','rotate_patch_180'}
    tree=ast.parse((folder/'word_data.py').read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(tree.body)==len(names)
    ns=dict(np=np,Image=Image,ImageEnhance=ImageEnhance,ImageFilter=ImageFilter,WIDTH=480,HEIGHT=192,STRIDE=4,GRID_W=120,GRID_H=48,anchor_preserving_zoom=anchor_preserving_zoom)
    exec(compile(tree,str(folder/'word_data.py'),'exec'),ns)
    return ns

class DatasetStub:
    def __init__(self,**kwargs):pass

def load_patches(folder,data):
    node=next(n for n in ast.parse((folder/'word_train.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Patches')
    ns=dict(data,keras=types.SimpleNamespace(utils=types.SimpleNamespace(PyDataset=DatasetStub)),math=math,
        BalancedFamilyOrder=BalancedFamilyOrder,visible_partial_candidates=visible_partial_candidates,
        horizontal_aspect_factors=horizontal_aspect_factors,vertical_stretch_factors=vertical_stretch_factors)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(folder/'word_train.py'),'exec'),ns)
    return ns

OLD=load_data(OUT/'source-original');OFF=load_data(OUT/'source-private');ON=load_data(OUT/'source-private')
PROOFS={}
def inside(boxes):
    return (boxes[:,0]>=0)&(boxes[:,1]>=0)&(boxes[:,2]<=480)&(boxes[:,3]<=192)

# Every side, translated one-sided and center-crossing boxes; no labels involved.
geometry_checks=0
for box in ([20,10,90,180],[390,5,475,185],[200,1,280,191],[1,70,479,130],[2,2,70,80],[410,110,478,190],[210,50,260,130]):
    b=np.asarray([box],float)
    for tx in (-19.2,0,19.2):
        for ty in (-7.68,0,7.68):
            z=anchor_preserving_zoom(b,[True],1.7,tx,ty)
            assert .8<=z<=1.7
            if z<1.7:
                _,transformed=ON['affine_patch'](Image.new('RGB',(480,192)),b,z,z,tx,ty)
                assert inside(transformed)[0]
                # A cap is maximal up to numerical rounding: meaningful growth
                # crosses at least one boundary; no arbitrary shrink margin.
                q=z+1e-7
                _,too_big=ON['affine_patch'](Image.new('RGB',(480,192)),b,q,q,tx,ty)
                assert not inside(too_big)[0]
            geometry_checks+=1
assert anchor_preserving_zoom(np.empty((0,4)),[],1.7,0,0)==1.7
assert anchor_preserving_zoom([[0,0,480,192]],[False],1.7,0,0)==1.7
assert anchor_preserving_zoom([[-10,0,40,190]],[True],1.7,0,0)==1.7
assert .8 < anchor_preserving_zoom([[0,0,50,192]],[True],1.7,-19.2,0) < .93
# Equal-center-distance anchors use the first input index.
assert anchor_preserving_zoom([[210,30,230,170],[250,70,270,130]],[True,True],1.7,0,0) < anchor_preserving_zoom([[250,70,270,130],[210,30,230,170]],[True,True],1.7,0,0)
PROOFS['boundary_algebra_cases']=geometry_checks+5

config=json.loads((WORK/'run20/training-config.json').read_text())
assert config['illumination_augmentation'] and not config['horizontal_mirror_augmentation']
unique=json.loads((DATA/config['train_manifest']).read_text())
assert sha(DATA/config['train_manifest'])==config['data_sha256'][config['train_manifest']]
train,negative_expansion=expand_negative_records(unique,config['negative_repeat'],config['supplemental_source_audit']['supplemental_sources'],config['supplemental_negative_repeat'])
point_states,point_proof=load_point_states(config['dsbi_point_metadata'],DATA,config['supplemental_source_indexes'],unique)
cohort=json.loads((WORK/'train-target-survivorship-audit/cohort.json').read_text())['records']
ids=[r['index'] for r in cohort];info={r['index']:r for r in cohort}
for r in cohort:
    assert train[r['index']]['image']==r['image'] and train[r['index']]['source']==r['source']

yy,xx=np.mgrid[:192,:480]
texture=Image.fromarray(np.stack(((xx*3+yy*5)%256,(xx*7+yy)%256,(xx+yy*3)%256),-1).astype(np.uint8))
constant=Image.new('RGB',(480,192),(127,127,127))
CURRENT={'record':None,'image':texture}
by_path={str((DATA/train[i]['image']).resolve()):i for i in ids}
def fake_open(path,*args,**kwargs):
    CURRENT['record']=by_path[str(Path(path).resolve())]
    return CURRENT['image'].copy()

def instrument(data):
    ns=load_patches(OUT/'source-original' if data is OLD else OUT/'source-private',data)
    traces=[];active={};real_affine=data['affine_patch'];real_aug=data['augment_patch'];real_target=data['make_target']
    def affine(image,boxes,sx,sy,tx=0.,ty=0.):
        result=real_affine(image,boxes,sx,sy,tx,ty)
        active['affine']={'sx':sx,'sy':sy,'tx':tx,'ty':ty,'boxes':result[1].copy()}
        return result
    data['affine_patch']=affine
    def augment(image,boxes,rng,*args,**kwargs):
        active.clear();active.update(record=CURRENT['record'],before=boxes.copy(),original=kwargs['originally_complete'].copy(),affine=None,point_before=kwargs.get('point_state'),rng_before=copy.deepcopy(rng.bit_generator.state))
        value=real_aug(image,boxes,rng,*args,**kwargs)
        active.update(after=value[1].copy(),eligible=value[2].copy(),point_after=value[3] if len(value)>3 else None,rng_after=copy.deepcopy(rng.bit_generator.state))
        return value
    def target(boxes,masks,eligible,*args,**kwargs):
        result=real_target(boxes,masks,eligible,*args,**kwargs)
        known=(result[...,0]==1)&np.all(result[...,5:]>=0,axis=-1)
        partial=(result[...,0]==1)&np.all(result[...,5:]==-1,axis=-1)
        full_indices=np.flatnonzero(eligible)
        centers={tuple(np.minimum([119,47],((boxes[i,:2]+boxes[i,2:])/8).astype(int))):i for i in full_indices}
        assert int(known.sum())==len(centers)
        for (gx,gy),i in centers.items():
            np.testing.assert_array_equal(result[gy,gx,5:],[(int(masks[i])>>j)&1 for j in range(6)])
        n=len(boxes)
        assert not np.any(active['eligible'] & ~active['original'])
        assert not np.any(active['eligible'][n:])
        if active['point_after'] is not None:
            assert not np.any(active['point_after'].survivors & ~active['point_before'].survivors)
        counts=Counter(records=1,original_full=int(active['original'][:n].sum()),full_six_bit_targets=int(known.sum()),geometry_only_targets=int(partial.sum()),ignored_grid_positions=int((result[...,0]<0).sum()),zero_full_target_records=int(not known.any()),unknown_boxes=len(active['after'])-n)
        if active['affine'] is not None:
            af=active['affine'];counts['affine_draws']=1
            counts['full_lost_at_affine']=int((active['original'][:n]&~inside(af['boxes'][:n])).sum())
        active['counts']=counts;traces.append(copy.deepcopy(active))
        return result
    ns.update(augment_patch=augment,make_target=target)
    opts=dict(root=DATA,records=train,batch_size=24,training=True,seed=config['seed'],illumination_augmentation=True,point_states=point_states)
    if data is ON:opts['anchor_zoom_cap']=True
    patches=ns['Patches'](**opts);patches.order=np.asarray(ids,np.int64)
    return patches,traces

old,oldt=instrument(OLD);off,offt=instrument(OFF);on,ont=instrument(ON)
def run(patches,traces,epoch,image):
    traces.clear();patches.epoch=epoch;CURRENT['image']=image
    pixels=[];targets=[]
    with mock.patch.object(Image,'open',fake_open):
        for batch in range(math.ceil(len(ids)/24)):
            x,y=patches[batch];pixels.append(x);targets.append(y)
    return np.concatenate(pixels),np.concatenate(targets),copy.deepcopy(traces)

for epoch in (0,7):
    a=run(old,oldt,epoch,texture);b=run(off,offt,epoch,texture)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    for x,y in zip(a[2],b[2]):
        assert x['record']==y['record'] and x['rng_before']==y['rng_before'] and x['rng_after']==y['rng_after']
        np.testing.assert_array_equal(x['after'],y['after']);np.testing.assert_array_equal(x['eligible'],y['eligible'])
PROOFS['default_off_paired_actual_patches_records']=80

totals={k:Counter() for k in ('baseline','capped')};groups=defaultdict(lambda:{k:Counter() for k in ('baseline','capped')});details=[];capped_draws=0
for epoch in range(32):
    a=run(off,offt,epoch,constant);b=run(on,ont,epoch,constant)
    for index,(x,y) in enumerate(zip(a[2],b[2])):
        assert x['record']==y['record'] and x['rng_before']==y['rng_before'] and x['rng_after']==y['rng_after']
        af,bf=x['affine'],y['affine'];assert (af is None)==(bf is None)
        changed=False
        if af:
            assert af['tx']==bf['tx'] and af['ty']==bf['ty'] and bf['sx']==bf['sy']<=af['sx']==af['sy']
            changed=bf['sx']!=af['sx'];capped_draws+=changed
        if not changed:
            np.testing.assert_array_equal(a[0][index],b[0][index]);np.testing.assert_array_equal(a[1][index],b[1][index])
        for label,trace in (('baseline',x),('capped',y)):
            totals[label].update(trace['counts']);groups[str(info[x['record']]['original_complete_cells'])][label].update(trace['counts'])
        details.append({'epoch':epoch,'record':x['record'],'row_length':info[x['record']]['original_complete_cells'],'capped':changed,'original_zoom':af['sx'] if af else None,'capped_zoom':bf['sx'] if bf else None,'baseline':dict(x['counts']),'capped_counts':dict(y['counts']),'rng_after_sha256':hashlib.sha256(json.dumps(x['rng_after'],sort_keys=True).encode()).hexdigest()})
    if epoch%8==7:print('Completed paired diagnostic epoch',epoch+1,flush=True)

PROOFS.update(paired_main_rng_records=1280,unchanged_draw_pixels_and_targets_exact=True,irreversible_complete_and_point_visibility=True,unknowns_never_full=True,capped_affine_draws=capped_draws)
assert totals['capped']['full_six_bit_targets']>totals['baseline']['full_six_bit_targets']
assert not any(k.split('.')[0] in {'torch','tensorflow','keras'} for k in sys.modules)
spec=json.loads((OUT/'prototype-spec.json').read_text())
for folder,key in (('source-original','source_original_sha256'),('source-private','source_private_sha256')):
    for name,digest in spec[key].items():assert sha(OUT/folder/name)==digest
report={'status':'passed','created_at_utc':START,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'prototype_spec_sha256':sha(OUT/'prototype-spec.json'),'script_sha256':sha(__file__),
 'scope':'PRIVATE actual AST-loaded Patches/augment/target execution, frozen TRAIN metadata and exact DSBI point states, synthetic Pillow rasters. No ML/model/validation/external inference or active-source mutation.',
 'baseline':'V15 isotropic illumination and exact-point partial supervision; horizontal/vertical/mirror/half-turn/camera/polarity OFF.',
 'cohort_sha256':sha(WORK/'train-target-survivorship-audit/cohort.json'),'training_manifest_sha256':sha(DATA/config['train_manifest']),'point_metadata_proof':point_proof,
 'proofs':PROOFS,'totals':{k:dict(v) for k,v in totals.items()},'by_original_row_length':{k:{a:dict(b) for a,b in v.items()} for k,v in groups.items()},
 'limitations':['40 fixed metadata-stratified TRAIN records over32 diagnostic epochs; not actual source-balanced epoch exposure or recognition accuracy.','Only one full known anchor is protected at the affine stage when feasible at zoom>=.8; later rotation can cut it.','Other known cells and fragments can gain or lose visibility with changed scale; immutable source annotations, unknown slots and irreversible rules remain unchanged.','No candidate or release claim; one ablation still needs unchanged frozen405 selection and original external gates.']}
(OUT/'paired-record-results.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
report['paired_results_sha256']=sha(OUT/'paired-record-results.json')
(OUT/'proof-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report_sha256':sha(OUT/'proof-report.json'),'proofs':PROOFS,'totals':report['totals'],'by_length':report['by_original_row_length']}))
