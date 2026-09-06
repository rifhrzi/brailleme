"""Bounded source/NumPy probes of complete-cell target/loss/decoder contracts."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def sha(data):return hashlib.sha256(data).hexdigest()
sources={}
for name in ('word_data.py','word_model.py','word_decode.py','benchmark_angelina.py'):
    path=ROOT/'training/scripts'/name;data=path.read_bytes();sources[name]=data
ts_path=ROOT/'src/ml/word-decoder.ts';ts_bytes=ts_path.read_bytes()
mapping_node=next(n for n in ast.parse(sources['benchmark_angelina.py']).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LETTER_BY_MASK' for t in n.targets))
mapping=ast.literal_eval(mapping_node.value);assert len(mapping)==26
D=dict(np=np,WIDTH=480,HEIGHT=192,STRIDE=4,GRID_W=120,GRID_H=48,LETTER_BY_MASK=mapping)
for name,functions in [('word_data.py',{'make_target'}),('word_decode.py',{'decode','iou'})]:
    nodes=[n for n in ast.parse(sources[name]).body if isinstance(n,ast.FunctionDef) and n.name in functions]
    assert len(nodes)==len(functions)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),name,'exec'),D)
thresholds=dict(proposalScore=.25,acceptScore=.25,dotConfidence=.7,nmsIou=.25,minCellWidth=.008,minCellHeight=.06,edgeMargin=.008,maxRowDeviation=.55,maxGapRatio=1.2,maxOverlapRatio=.5)
cases=[]
for label,boxes in [('adjacent_grid_centers',[[98,90,102,102],[102.5,90,106.5,102]]),
                    ('two_grid_steps',[[98,90,102,102],[106,90,110,102]])]:
    target=D['make_target'](np.asarray(boxes,np.float64),np.asarray([1,3]))
    result=D['decode'](target,thresholds)
    centers=np.argwhere(target[...,0]==1)
    assert len(centers)==2
    cases.append(dict(case=label,boxes_pixels=boxes,target_centers_yx=centers.tolist(),
        target_center_values=[target[tuple(p)].tolist() for p in centers],actual_decoder=result,
        expected_word='AB',reproduces_shortened_acceptance=result['text']!='AB' and not result['reason']))
assert cases[0]['actual_decoder']['text']=='A' and cases[0]['actual_decoder']['reason']==''
assert cases[1]['actual_decoder']['text']=='AB'
# Same sites with unequal objectness still cannot retain both: no tie-only explanation.
target=D['make_target'](np.asarray(cases[0]['boxes_pixels'],np.float64),np.asarray([1,3]))
target[24,25,0]=.999;target[24,26,0]=.998
unequal=D['decode'](target,thresholds)
assert unequal['text']=='A' and not unequal['reason']

max_box_error=0.;roundtrips=0
for mask,letter in mapping.items():
    for shift in (0.,.125,3.875):
        box=np.asarray([[160+shift,64+shift,180+shift,104+shift]])
        target=D['make_target'](box,np.asarray([mask]))
        result=D['decode'](target,thresholds)
        assert result['text']==letter.upper() and result['reason']==''
        predicted=np.asarray(result['cells'][0]['box'])*[480,192,480,192]
        max_box_error=max(max_box_error,float(np.max(np.abs(predicted-box[0]))))
        center=target[...,0]>=.999
        assert int(center.sum())==1 and float(target[...,0][~center].max())<.999
        np.testing.assert_array_equal(target[center][0,5:],[(mask>>j)&1 for j in range(6)])
        roundtrips+=1
assert max_box_error<1e-5

# Execute the source loss algebra with NumPy-compatible ops; this is not autograd proof.
ops=SimpleNamespace(clip=np.clip,cast=lambda x,dtype:x.astype(dtype),maximum=np.maximum,sum=np.sum,
    square=np.square,log=np.log,power=np.power,concatenate=np.concatenate,abs=np.abs,where=np.where,
    convert_to_tensor=np.asarray)
loss_env={'ops':ops}
loss_node=next(n for n in ast.parse(sources['word_model.py']).body if isinstance(n,ast.FunctionDef) and n.name=='joint_word_loss')
exec(compile(ast.Module(body=[loss_node],type_ignores=[]),'actual joint_word_loss NumPy algebra','exec'),loss_env)
truth=D['make_target'](np.asarray([[160.,64.,180.,104.]]),np.asarray([1]))[None]
prediction=np.full(truth.shape,.5,np.float32);prediction[...,1:5]=truth[...,1:5]
loss=lambda p:float(loss_env['joint_word_loss'](truth,p,True))
baseline=loss(prediction);gy,gx=np.argwhere(truth[0,...,0]==1)[0]
positive=prediction.copy();positive[0,gy,gx,5]=.9
negative=prediction.copy();negative[0,gy,gx,6]=.9
positive_delta=loss(positive)-baseline;negative_delta=loss(negative)-baseline
# Global float32 sum adds a small rounding error because the background loss is large.
assert abs(positive_delta-.5*np.log(.5/.9))<1e-3
assert abs(negative_delta-.5*np.log(.5/.1))<1e-3
noncenter=prediction.copy();noncenter[...,5:]=.9;noncenter[0,gy,gx,5:]=prediction[0,gy,gx,5:]
assert loss(noncenter)==baseline
assert not any(m in sys.modules for m in ('keras','tensorflow','torch'))
for name,data in sources.items():assert (ROOT/'training/scripts'/name).read_bytes()==data
assert ts_path.read_bytes()==ts_bytes
assert not (OUT/'report.json').exists()
(OUT/'source').mkdir(exist_ok=True)
for name,data in sources.items():(OUT/'source'/name).write_bytes(data)
(OUT/'source/word-decoder.ts').write_bytes(ts_bytes)
report=dict(status='one_bounded_contract_mismatch_reproduced',created_at_utc=datetime.now(timezone.utc).isoformat(),
    scope='Source review and small analytic NumPy target/decoder/loss-algebra probes. No ML backend, gradients, training, image inputs, dataset/output corpus, threshold tuning or active edits.',
    source_sha256={**{f'training/scripts/{n}':sha(v) for n,v in sources.items()},'src/ml/word-decoder.ts':sha(ts_bytes)},
    script_sha256=sha(Path(__file__).read_bytes()),thresholds=thresholds,
    finding=dict(title='Adjacent valid center sites are removed before positional box NMS, allowing a shortened word',
        cause='make_target/loss supervise both adjacent sites as positive centers; 3x3 local-maximum selection in Python and TS permits only the stronger site or first tied site, irrespective of disjoint predicted boxes.',
        cases=cases,unequal_scores_result=unequal,
        location=['training/scripts/word_data.py:214-224','training/scripts/word_model.py:54-59','training/scripts/word_decode.py:34-39','src/ml/word-decoder.ts:98-113'],
        scale_limit='Reproducer width4px/height12px passes current minimum3.84px/11.52px. Center separation4.5px maps to adjacentstride4 sites. A center pitch at least8px avoids immediate-grid adjacency on the same row; this is not a measured corpus distribution.',
        classification='Demonstrable small-scale target/decoder representability mismatch. Not evidence of the cause or frequency of errors in current real fixtures.'),
    consistent_contracts=dict(single_cell_AZ_roundtrip_cases=roundtrips,max_reconstructed_box_error_pixels=max_box_error,
        offsets='Target fractional center=(box_center/4)-floor(box_center/4); decoder (grid+offset)/(WIDTH/4,HEIGHT/4) exactly reverses it apart from float32 storage.',
        box_sizes='Target width/480,height/192; same normalized sizes used in decoder. Relative-size log loss compares the same quantities, and offset loss uses fractional grid units.',
        gaussian='Only exp(0)=1 centers pass>=.999; nearest noncenterGaussianexp(-.5)<.999. Shoulders do not receive center geometry/bit supervision.',
        bit_order='Target and decoder both use bit0dot1..bit5dot6; all26 ordinary A-Z roundtrips pass.',
        bit_loss_algebra={'positive_bit_delta':positive_delta,'negative_bit_delta':negative_delta,'expected_positive_delta':float(.5*np.log(.5/.9)),
                          'expected_negative_delta':float(.5*np.log(.5/.1)),'noncenter_bits_do_not_change_loss':True,'autograd_checked':False},
        edge_rejection='Training includes fully visible cells at frame edges; decoder edgeMargin conservatively rejects them by design, not an x/y scaling mismatch.'),
    recommended_followup='Assess tiny-cell adjacency explicitly before changing decoder behavior; any fix should retain separate valid positions or reject ambiguity. This audit authorizes no decoder/source/threshold changes.',
    limitations=['No actual TRAIN/validation/external prevalence measured.','No predicted-model tensors or physical pixels read.','No claim that a decoder intervention improves held/public accuracy.','Existing partial-point/augmentation-survival audits were not repeated.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
files=[dict(file=p.relative_to(OUT).as_posix(),sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()]
(OUT/'index.json').write_text(json.dumps(dict(files=files),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(report_sha256=sha((OUT/'report.json').read_bytes()),index_sha256=sha((OUT/'index.json').read_bytes()),roundtrips=roundtrips,max_box_error_pixels=max_box_error,positive_delta=positive_delta,negative_delta=negative_delta)))
