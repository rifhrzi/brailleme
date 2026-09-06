"""Independent CPU-only vertical-stretch audit; pass a frozen source directory.

Executes actual Patches/function AST with a neutral PyDataset base, without any
model/backend import. Report is written only to a new requested output path.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

cli=argparse.ArgumentParser()
cli.add_argument('--source',type=Path,required=True)
cli.add_argument('--output',type=Path,required=True)
args=cli.parse_args()
assert not args.output.exists(), 'Never overwrite a published audit'
WORK=Path('.braille-training-work/word-scan-v1-20260905').resolve()
SOURCE=args.source.resolve()
sys.path.insert(0,str(SOURCE))
from word_point_geometry import OccupiedPointState, visible_partial_candidates
from word_sampling import BalancedFamilyOrder
from word_vertical_stretch import vertical_stretch_factors

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan_path=SOURCE.parent/'run23-plan.json'
plan=json.loads(plan_path.read_text())
assert all(sha(SOURCE/name)==expected for name,expected in plan['source_sha256'].items())
assert plan['illumination_augmentation'] and plan['vertical_stretch_augmentation']
assert not any(plan[k] for k in ['horizontal_mirror_augmentation','half_turn_augmentation','polarity_augmentation','partial_geometry_supervision','geometry_conditioned_residual'])
assert plan['vertical_stretch']['range']==[.9,1.1] and plan['vertical_stretch']['stream_id']==0x56535452
def nodes(path,names):
    module=ast.parse(path.read_text(encoding='utf-8'))
    selected=[n for n in module.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(selected)==len(names)
    return ast.Module(body=selected,type_ignores=[])
NAMES=['make_target','affine_patch','mirror_patch_horizontal','rotate_patch_180','augment_patch']
def load_data(path):
    ns=dict(np=np,Image=Image,ImageEnhance=ImageEnhance,ImageFilter=ImageFilter,
            WIDTH=480,HEIGHT=192,STRIDE=4,GRID_W=120,GRID_H=48)
    exec(compile(nodes(path,NAMES),str(path),'exec'),ns)
    return ns
old=load_data(WORK/'run20/source/word_data.py')
current=load_data(SOURCE/'word_data.py')

# Compare isolated stream to independently constructed PCG64 seed identity,
# checking batch prefixes and changes in each identity component.
legacy_global_before=copy.deepcopy(np.random.get_state())
for identity in [(20260905,2,0),(1,8,17),(0,0,0)]:
    expected=np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity,0x56535452]))).uniform(.9,1.1,24)
    actual=vertical_stretch_factors(*identity,24)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual[:6],vertical_stretch_factors(*identity,6))
    assert np.all((actual>=.9)&(actual<=1.1))
    for field in range(3):
        other=list(identity);other[field]+=1
        assert not np.array_equal(actual,vertical_stretch_factors(*other,24))
after=np.random.get_state()
assert legacy_global_before[0]==after[0]
np.testing.assert_array_equal(legacy_global_before[1],after[1])
assert legacy_global_before[2:]==after[2:]

# Actual Pillow intensity centroids validate corner/index semantics separately
# from analytical boxes. No annotation-generated raster reaches the model.
marker_errors=[]
yy,xx=np.indices((192,480))
coords=np.array([[100.,41.],[235.,97.],[371.,149.]])
rgb=np.stack([np.rint(255*np.exp(-((xx-x)**2+(yy-y)**2)/18)) for x,y in coords],axis=-1).astype(np.uint8)
source_box=np.array([[88.,25.,389.,173.]])
for factor in [.9,1.,1.1]:
    sx,sy,tx,ty=1.03,1.03*factor,3.,-2.
    image,box=current['affine_patch'](Image.fromarray(rgb),source_box,sx,sy,tx,ty)
    expected_box=(source_box-[240,96,240,96])*[sx,sy,sx,sy]+[240+tx,96+ty,240+tx,96+ty]
    np.testing.assert_allclose(box,expected_box,atol=1e-12,rtol=0)
    expected=(coords+.5-[240,96])*[sx,sy]+[240+tx,96+ty]-.5
    state=OccupiedPointState.from_source_crop(np.repeat(coords[:,None],6,axis=1),[1]*3,[0,0,480,192]).affine(sx,sy,tx,ty)
    np.testing.assert_allclose(state.points[:,0],expected,atol=1e-12,rtol=0)
    raster=np.asarray(image,float)
    centroids=np.array([[(raster[...,i]*xx).sum(),(raster[...,i]*yy).sum()] for i in range(3)])/raster.sum(axis=(0,1))[:,None]
    error=float(np.max(np.abs(centroids-expected)));assert error<.04
    marker_errors.append({'factor':factor,'max_centroid_error_pixels':error})

# All63 occupied patterns: exact q+.5 affine formula, per-stage hull AND,
# unchanged slot labels, border cuts and subsequent coordinate returns.
point_checks=0
q=np.array([[[2.,0.],[2.,95.],[2.,191.],[27.,0.],[27.,96.],[27.,191.]]])
for mask in range(1,64):
    state=OccupiedPointState.from_source_crop(q,[mask],[0,0,480,192])
    expected=q.copy();surv=state.survivors.copy()
    for sx,sy,tx,ty in [(1,1.1,0,0),(1,1/1.1,0,0),(1,.9,0,-30),(1,1/.9,0,30/.9)]:
        expected=(expected+.5-[240,96])*[sx,sy]+[240+tx,96+ty]-.5
        surv &= (expected[...,0]>=0)&(expected[...,0]<=479)&(expected[...,1]>=0)&(expected[...,1]<=191)
        state=state.affine(sx,sy,tx,ty)
        np.testing.assert_allclose(state.points,expected,rtol=0,atol=1e-11)
        np.testing.assert_array_equal(state.survivors,surv)
        point_checks+=1
    # Top/bottom slots were lost in the first enlargement and stay lost.
    assert not state.survivors[0,[0,2,3,5]].any()

# Exercise the actual combined augmentation: enlargement clips the top, then
# positive rotation brings its known box/point coordinates inside again.
class ForcedDraws:
    def __init__(self):self.randoms=iter([0.,0.,1.,1.]);self.uniforms=iter([1.,0.,0.,4.,1.,.5])
    def random(self):return next(self.randoms)
    def uniform(self,*a):return next(self.uniforms)
recover_box=np.array([[20.,0.,40.,38.]])
recover_points=np.array([[[23.,0.],[23.,17.],[23.,35.],[37.,0.],[37.,17.],[37.,35.]]])
recover_state=OccupiedPointState.from_source_crop(recover_points,[1],[0,0,480,192])
aug=current['augment_patch'](Image.fromarray(rgb),recover_box,ForcedDraws(),illumination_augmentation=True,
                             originally_complete=np.array([True]),point_state=recover_state,vertical_stretch=1.1)
assert np.all(aug[1][:,:2]>=0) and np.all(aug[1][:,2:]<=[480,192]), aug[1]
assert not aug[2][0], 'Complete eligibility was restored after raster clipping'
assert np.all(aug[3].points[0,0]>=0) and np.all(aug[3].points[0,0]<=[479,191])
assert not aug[3].survivors[0,0], 'Previously cut occupied point was restored'
recovered_target=current['make_target'](aug[1],np.array([1]),aug[2],visible_partial_boxes=visible_partial_candidates(aug[1],aug[3])[1])
assert not np.any(recovered_target[...,0]==1), 'Restored annotation supplied a label or fragment'
class NoAffineDraws:
    def __init__(self):self.randoms=iter([1.,1.,1.,1.]);self.uniforms=iter([1.,.5])
    def random(self):return next(self.randoms)
    def uniform(self,*a):return next(self.uniforms)
no_affine=[]
for factor in [.9,1.,1.1]:
    no_affine.append(current['augment_patch'](Image.fromarray(rgb),recover_box,NoAffineDraws(),illumination_augmentation=True,
                     originally_complete=np.array([True]),point_state=recover_state,vertical_stretch=factor))
for result in [no_affine[0],no_affine[2]]:
    for x,y in zip(result[:3],no_affine[1][:3]):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(result[3].points,recover_state.points)
    np.testing.assert_array_equal(result[3].survivors,recover_state.survivors)

class Base:
    def __init__(self,**kwargs):pass
class RandomTrace:
    def __init__(self,seed):self.inner=np.random.default_rng(seed);self.calls=[]
    def __getattr__(self,name):
        method=getattr(self.inner,name)
        def call(*a,**kw):
            value=method(*a,**kw);self.calls.append((name,a,kw,np.asarray(value).tolist()));return value
        return call
class NumpyProxy:
    def __init__(self,traces):
        def rng(seed):
            trace=RandomTrace(seed);traces.append(trace);return trace
        self.random=SimpleNamespace(default_rng=rng)
    def __getattr__(self,name):return getattr(np,name)
target_checks=[]
def factory(path,data):
    traces=[]
    ns=dict(data,np=NumpyProxy(traces),math=math,keras=SimpleNamespace(utils=SimpleNamespace(PyDataset=Base)),
            BalancedFamilyOrder=BalancedFamilyOrder,visible_partial_candidates=visible_partial_candidates,
            vertical_stretch_factors=vertical_stretch_factors)
    augmentation_factors=[]
    original_augmentation=data['augment_patch']
    def augmentation(*a,**kw):
        augmentation_factors.append(kw.get('vertical_stretch',1.0))
        return original_augmentation(*a,**kw)
    ns['augment_patch']=augmentation
    original=data['make_target']
    def target(boxes,masks,eligible,*a,**kw):
        t=original(boxes,masks,eligible,*a,**kw);known={}
        for box,mask,full in zip(boxes,masks,eligible):
            if full and np.all(box[:2]>=0) and np.all(box[2:]<=[480,192]):
                gx,gy=((box[:2]+box[2:])/8).astype(int);known[(gy,gx)]=int(mask)
        for (gy,gx),mask in known.items():
            assert t[gy,gx,0]==1
            np.testing.assert_array_equal(t[gy,gx,5:],[(mask>>bit)&1 for bit in range(6)])
        peaks=np.argwhere(t[...,0]==1)
        for gy,gx in peaks:
            if (gy,gx) not in known:assert np.all(t[gy,gx,5:]==-1)
        target_checks.append(dict(complete=len(known),geometry_only=len(peaks)-len(known),unknown_boxes=len(kw.get('ignore_boxes',[]))))
        return t
    ns['make_target']=target
    exec(compile(nodes(path,['Patches']),str(path),'exec'),ns)
    return ns['Patches'],traces,augmentation_factors

config=json.loads((WORK/'run20/training-config.json').read_text())
data_root=WORK/'data'
manifest=data_root/config['train_manifest']
records=json.loads(manifest.read_text())
index=json.loads((data_root/config['supplemental_source_indexes'][0]).read_text())
extra={r['source'] for r in index['sources']}
base_neg=[r for r in records if not r['boxes'] and r['source'] not in extra]
extra_neg=[r for r in records if not r['boxes'] and r['source'] in extra]
assert (len(base_neg),len(extra_neg))==(44,47)
records+=base_neg*26+extra_neg
assert len(records)==24882
metadata_path=WORK/'dsbi-point-metadata-v1/points.json'
metadata=json.loads(metadata_path.read_text())
states={image:OccupiedPointState.from_source_crop(np.array(row['source_points']).reshape(-1,6,2),row['masks'],row['crop']) for image,row in metadata['records'].items()}
state_digest=lambda:hashlib.sha256(b''.join(s.points.tobytes()+s.survivors.tobytes() for s in states.values())).hexdigest()
before=state_digest()
Old,ta,fa=factory(WORK/'run20/source/word_train.py',old)
Off,tb,fb=factory(SOURCE/'word_train.py',current)
On,tc,fc=factory(SOURCE/'word_train.py',current)
opts=dict(batch_size=6,training=True,seed=config['seed'],illumination_augmentation=True,balance_source_families=True,point_states=states)
a,b,c=Old(data_root,records,**opts),Off(data_root,records,**opts),On(data_root,records,**opts,vertical_stretch_augmentation=True)
for dataset in [a,b,c]:dataset.on_epoch_end()
np.testing.assert_array_equal(a.order,b.order);np.testing.assert_array_equal(a.order,c.order)
families=set();chosen=[]
for batch in range(len(c)):
    seen={str(Path(records[i]['source']).parent).replace('\\','/') for i in c.order[batch*6:(batch+1)*6] if records[i]['boxes']}
    if not seen.issubset(families):chosen.append(batch);families |= seen
    if len(families)==12:break
for predicate in [lambda r:not r['boxes'] and r['source'] in extra,lambda r:r['source'] in extra and r.get('ignore_boxes') and not all(r.get('originally_complete',[]))]:
    chosen.append(next(i for i,r in enumerate(c.order) if predicate(records[r]))//6)
rows=[]
for batch in sorted(set(chosen)):
    aa,bb,cc=a[batch],b[batch],c[batch]
    expected_factors=vertical_stretch_factors(config['seed'],c.epoch,batch,len(cc[0]))
    np.testing.assert_array_equal(fc[-len(cc[0]):],expected_factors)
    assert fa[-len(aa[0]):]==fb[-len(bb[0]):]==[1.]*len(bb[0])
    for x,y in zip(aa,bb):np.testing.assert_array_equal(x,y)
    assert ta[-1].calls==tb[-1].calls==tc[-1].calls
    assert ta[-1].inner.bit_generator.state==tb[-1].inner.bit_generator.state==tc[-1].inner.bit_generator.state
    repeated=c[batch]
    for x,y in zip(cc,repeated):np.testing.assert_array_equal(x,y)
    rows.append({'batch':batch,'indices':c.order[batch*6:(batch+1)*6].tolist(),
                 'off_pixels_sha256':hashlib.sha256(bb[0].tobytes()).hexdigest(),'off_targets_sha256':hashlib.sha256(bb[1].tobytes()).hexdigest(),
                 'on_pixels_sha256':hashlib.sha256(cc[0].tobytes()).hexdigest(),'on_targets_sha256':hashlib.sha256(cc[1].tobytes()).hexdigest(),
                 'main_rng_draws':len(tb[-1].calls)})
assert any(r['off_pixels_sha256']!=r['on_pixels_sha256'] for r in rows)
assert before==state_digest()
assert any(r['geometry_only'] for r in target_checks) and any(r['unknown_boxes'] for r in target_checks)
# Evaluation never activates stretch even if its constructor flag is true.
eval_options=dict(opts,training=False)
eval_a=Old(data_root,records,**eval_options)
eval_c=On(data_root,records,**eval_options,vertical_stretch_augmentation=True)
for x,y in zip(eval_a[0],eval_c[0]):np.testing.assert_array_equal(x,y)
assert all(sha(SOURCE/name)==expected for name,expected in plan['source_sha256'].items())
for module in ['tensorflow','torch','keras']:assert module not in sys.modules
report={'status':'passed','scope':'Independent transform/target CPU proof, not recognition accuracy or Keras scheduling evidence.',
        'script_sha256':sha(__file__),'source_sha256':{p.name:sha(p) for p in SOURCE.glob('*.py')},
        'plan_sha256':sha(plan_path),'plan_source_flags_and_range_verified':True,
        'training_manifest_sha256':sha(manifest),'point_metadata_sha256':sha(metadata_path),
        'stream_identity_and_prefixes_passed':True,'global_numpy_rng_unchanged':True,'all63_stage_point_visibility_checks':point_checks,
        'pillow_marker_checks':marker_errors,'run20_off_pixels_targets_rng_byte_identical':True,'on_off_main_rng_calls_and_final_state_equal':True,
        'actual_augment_cut_then_rotation_return_never_recovers_complete_or_occupied_point':True,
        'actual_patches_factor_order_matches_independent_stream':True,
        'factor_does_nothing_when_existing_affine_is_skipped':True,'evaluation_mode_ignores_flag_and_matches_run20':True,
        'repeated_batch_order_independent':True,'full_balanced_order_identical':True,'source_families':sorted(families),'real_batches':rows,
        'target_checks':{k:sum(r[k] for r in target_checks) for k in ['complete','geometry_only','unknown_boxes']},
        'cached_point_states_unchanged':True,'defects':[],
        'limitations':['Actual Patches AST uses a neutral PyDataset base: Keras workers/GPU are not executed.','Known complete-cell targets still win one-grid-center collisions; unknown slots can overlap known-cell Gaussian soft regions but do not supply labels.','Fixed small batch sample proves exercised transform paths, not whole-corpus performance or every stochastic draw.']}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(args.output),'sha256':sha(args.output),'batches':len(rows),'checks':report['target_checks']}))
