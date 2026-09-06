"""Synthetic contract/cache tests; never load dataset outputs or model backends."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=json.loads((OUT/'prototype-spec.json').read_text())
sys.path.insert(0,str(OUT/'source-original/training/scripts'))
def load(side):
    p=OUT/side/'training/scripts/word_decode.py'
    s=importlib.util.spec_from_file_location(side,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('source-original');new=load('source-private')
t=dict(new.DEFAULT_THRESHOLDS,proposalScore=.25,acceptScore=.25,dotConfidence=.7)
cases=[]
def cell(cx=100,cy=96,width=4,height=12,mask=1,score=.99,dots=None):
    gx=int(cx//4);gy=int(cy//4)
    row=np.array([score,cx/4-gx,cy/4-gy,width/480,height/192]+([float(bool(mask&(1<<b))) for b in range(6)] if dots is None else dots),np.float32)
    return {'gy':gy,'gx':gx,'values':row.tolist()}
def add(name,rows,text='',reason='',thresholds=None,selection=None):
    cases.append({'name':name,'rows':rows,'expected_text':text,'expected_reason':reason,
                  'thresholds':thresholds or t,'selection':selection or [0,0,1,1]})
for word,masks in [('AB',[1,3]),('AA',[1,1])]:
    for label,scores in [('equal',[.99,.99]),('descending',[.99,.98]),('ascending',[.98,.99])]:
        add(f'adjacent_{word}_{label}',[cell(mask=masks[0],score=scores[0]),cell(cx=104.5,mask=masks[1],score=scores[1])],word)
add('separate_grid_AB',[cell(),cell(cx=108,mask=3)],'AB')
add('overlapping_duplicate',[cell(width=12),cell(cx=104,width=12,score=.9)],'A')
add('overlapping_duplicate_tie',[cell(width=12,score=.9),cell(cx=104,width=12,score=.9)],'A')
add('weak_adjacent_rejects',[cell(),cell(cx=104.5,mask=3,score=.3)],reason='low_confidence',thresholds=dict(t,acceptScore=.4))
add('uncertain_dot_adjacent_rejects',[cell(),cell(cx=104.5,mask=3,score=.98,dots=[1,.5,0,0,0,0])],reason='low_confidence')
add('unsupported_adjacent_rejects',[cell(),cell(cx=104.5,mask=0,score=.98)],reason='unsupported_pattern')
add('NMS_chain_retains_AC',[cell(width=7,score=.99),cell(cx=104,width=7,mask=3,score=.98),cell(cx=108,width=7,mask=9,score=.97)],'AC')
add('outside_guide_stronger_neighbor',[cell(score=.9),cell(cx=104.5,mask=3,score=.99)],'A',selection=[.20,.45,.213,.55])
add('guide_cut_adjacent_rejects',[cell(),cell(cx=104.5,mask=3,score=.98)],reason='clipped',selection=[.20,.45,.22,.55])
# Above right=.22 cuts the second cell whose right is106.5/480=.221875.
for n in (1,12,13):
    add('length_'+str(n),[cell(cx=100+8*i,mask=1) for i in range(n)],'A'*n if n<=12 else '',reason='' if n<=12 else 'too_many_cells')
add('blank',[],reason='no_grid')
add('below_proposal_neighbor_is_not_a_cell',[cell(),cell(cx=104.5,mask=3,score=.1)],'A')

def tensor(case):
    result=np.zeros((48,120,11),np.float32)
    for row in case['rows']:
        assert not result[row['gy'],row['gx'],0],case['name']
        result[row['gy'],row['gx']]=row['values']
    return result
def comparable(result):
    return {'text':result['text'],'reason':result['reason'],'boxes':[c['box'] for c in result['cells']],
            'masks':[c['mask'] for c in result['cells']] if result['text'] else []}
rows=[];arrays=[]
for case in cases:
    out=tensor(case);arrays.append(out)
    actual=new.decode(out,case['thresholds'],case['selection'])
    assert actual['text']==case['expected_text'] and actual['reason']==case['expected_reason'],(case['name'],actual)
    rows.append({'name':case['name'],'baseline':comparable(old.decode(out,case['thresholds'],case['selection'])),
                 'private':comparable(actual)})
predictions=np.stack(arrays)
before=predictions.copy();prepared=new.prepare_calibration_outputs(predictions)
np.testing.assert_array_equal(predictions,before)
np.testing.assert_array_equal(prepared[...,1:],before[...,1:])
assert np.all(prepared[...,0][before[...,0]>=.15]==before[...,0][before[...,0]>=.15])
grid=ast.parse((OUT/'source-original/training/scripts/word_select.py').read_text())
grids=ast.literal_eval(next(n.value for n in grid.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='THRESHOLD_GRIDS' for x in n.targets)))
trial=grids['weak-peaks-v2'];checks=0
for proposal in trial['proposal']:
    for accept in trial['accept']:
        if accept<proposal:continue
        for dot in trial['dot']:
            thresholds=dict(t,proposalScore=proposal,acceptScore=accept,dotConfidence=dot)
            for i,case in enumerate(cases):
                assert new.decode(predictions[i],thresholds,case['selection'])==new.decode(prepared[i],thresholds,case['selection'])
                checks+=1
malformed=np.stack([predictions[0]]*3)
malformed[0,0,0,0]=np.nan;malformed[1,0,0,0]=-1;malformed[2,0,0,0]=2
np.testing.assert_array_equal(new.prepare_calibration_outputs(malformed),malformed)
for out in malformed:assert new.decode(out,t)['reason']=='malformed'
legacy=old.prepare_calibration_outputs(predictions)
assert new.decode(legacy[0],t)['text']=='A'
assert new.decode(prepared[0],t)['text']=='AB'
for side,key in [('source-original','source_original_sha256'),('source-private','source_private_sha256')]:
    for name,digest in spec[key].items():assert sha(OUT/side/name)==digest
assert not any(m in sys.modules for m in ['torch','tensorflow','keras'])
for name,data in [('cases.json',cases),('python-results.json',rows),('python-test-report.json',{'status':'passed',
    'cases':len(cases),'cache_threshold_equivalence_checks':checks,'malformed_rows_preserved':3,
    'old_peak_filtered_cache_cannot_be_reused':True,'original_inputs_unchanged':True,
    'script_sha256':sha(__file__),'prototype_spec_sha256':sha(OUT/'prototype-spec.json'),'model_execution':False})]:
    path=OUT/name;assert not path.exists();path.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'status':'passed','cases':len(cases),'cache_checks':checks}))
