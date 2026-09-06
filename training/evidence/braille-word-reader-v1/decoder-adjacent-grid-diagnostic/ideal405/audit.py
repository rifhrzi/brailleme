"""Frozen held405 annotation adjacency only; cached quality flags, no predictions."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

OUT=Path(__file__).resolve().parent;WORK=OUT.parents[1];ROOT=WORK.parent.parent
FIXTURE=WORK/'data/validation-stress-selection.json'
QUALITY=WORK/'selected-v15/backend-fullprecision.npz'
def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

raw=FIXTURE.read_bytes();assert sha(raw)=='e2ab968f7c4dcf7e774181fde687e1321f58d20567528af4b21fd03ff5a3e625'
fixture=json.loads(raw);records=fixture['records'];assert len(records)==405
members={}
with zipfile.ZipFile(QUALITY) as archive:
    for name in ('quality_rejected.npy','fixture_sha256.npy','checkpoint_sha256.npy'):
        members[name]=archive.read(name)
quality=np.load(io.BytesIO(members['quality_rejected.npy']),allow_pickle=False)
assert quality.shape==(405,) and quality.dtype==np.bool_
assert str(np.load(io.BytesIO(members['fixture_sha256.npy']),allow_pickle=False))==sha(raw)
checkpoint=str(np.load(io.BytesIO(members['checkpoint_sha256.npy']),allow_pickle=False))
assert checkpoint=='cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
selection_path=WORK/'selected-v15/selected-validation-fullprecision.json'
selection=json.loads(selection_path.read_bytes())
assert selection['oracle']['fixture_sha256']==sha(raw) and selection['oracle']['checkpoint_sha256']==checkpoint
assert int(quality.sum())==selection['quality_rejected_before_decode']
def inside(box):return box[0]>=0 and box[1]>=0 and box[2]<=480 and box[3]<=192
def intersects(a,b):return a[2]>b[0] and a[0]<b[2] and a[3]>b[1] and a[1]<b[3]
def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union>0 else 0.
def grid(box):
    # Actual targets accept float32 Patches boxes before arithmetic.
    x0,y0,x1,y1=box
    return min(119,int((x0+x1)/2/4)),min(47,int((y0+y1)/2/4))
groups={name:Counter() for name in ('all','quality_pass','natural','stress')}
rows=[];near=[];selected_cheb=[];context_cheb=[]
for index,record in enumerate(records):
    chosen=np.asarray(record['boxes'],np.float32).reshape(-1,4)
    context=np.asarray(record['context_boxes'],np.float32).reshape(-1,4)
    guide=np.asarray(record['selection'])*[480,192,480,192]
    assert len(chosen)==len(record['masks'])==len(record['annotation_rows'])==len(record['expected'])
    assert len(set(record['annotation_rows']))==len(chosen)
    assert all(inside(box) and intersects(box,guide) for box in chosen)
    selected_context=[]
    for box,mask in zip(chosen,record['masks']):
        matches=[j for j,(other,m) in enumerate(zip(context,record['context_masks'])) if m==mask and np.array_equal(box,other)]
        assert len(matches)==1,(index,matches)
        selected_context.append(matches[0])
    selected_set=set(selected_context)
    full_context=[j for j,box in enumerate(context) if inside(box)]
    annotation_by_context={j:record['annotation_rows'][k] for k,j in enumerate(selected_context)}
    stats=Counter(records=1,selected_cells=len(chosen),full_context_cells=len(full_context),quality_rejected=int(quality[index]))
    for pos,j in enumerate(full_context):
        for k in full_context[pos+1:]:
            if j not in selected_set and k not in selected_set:continue
            a,b=context[j],context[k];ga,gb=grid(a),grid(b)
            dx,dy=abs(ga[0]-gb[0]),abs(ga[1]-gb[1]);cheb=max(dx,dy)
            both_selected=j in selected_set and k in selected_set
            kind='selected_selected' if both_selected else 'selected_other_full_context'
            stats[kind+'_pairs']+=1
            (selected_cheb if both_selected else context_cheb).append(cheb)
            if cheb<=1:
                duplicate=np.array_equal(a,b) and record['context_masks'][j]==record['context_masks'][k]
                item=dict(index=index,image=record['image'],validation_group=record['validation_group'],quality_rejected=bool(quality[index]),
                    kind=kind,context_indices=[j,k],source=record['source'],annotation_rows=[annotation_by_context.get(j),annotation_by_context.get(k)],
                    boxes=[a.tolist(),b.tolist()],grid_indices=[ga,gb],grid_chebyshev=cheb,
                    exact_duplicate_geometry_mask=bool(duplicate),box_iou=iou(a,b),
                    context_boxes_intersect_guide=[intersects(a,guide),intersects(b,guide)])
                near.append(item);stats[kind+'_adjacent_or_same_site']+=1
    for name in ('all',record['validation_group'])+(() if quality[index] else ('quality_pass',)):groups[name].update(stats)
    rows.append(dict(index=index,image=record['image'],source=record['source'],validation_group=record['validation_group'],
        quality_rejected=bool(quality[index]),expected_length=len(record['expected']),selected_annotation_rows=record['annotation_rows'],
        selected_grid_indices=[grid(box) for box in chosen],counts=dict(stats)))
report=dict(status='completed',created_at_utc=datetime.now(timezone.utc).isoformat(),
    scope='Frozen405 ideal annotation centers and existing raw-RGB quality flags only. No model-output arrays, image pixels, inference, training or external607 read.',
    fixture_sha256=sha(raw),quality_npz_sha256=file_sha(QUALITY),quality_npz_path=str(QUALITY),
    npz_members_decoded=list(members),npz_member_sha256={name:sha(data) for name,data in members.items()},
    quality_source_selected_report_sha256=file_sha(selection_path),quality_source_checkpoint_sha256=checkpoint,
    quality_rejected=int(quality.sum()),quality_pass=int((~quality).sum()),
    groups={name:dict(value) for name,value in groups.items()},
    minimum_selected_pair_grid_chebyshev=min(selected_cheb) if selected_cheb else None,
    minimum_selected_vs_other_full_context_grid_chebyshev=min(context_cheb) if context_cheb else None,
    adjacent_or_same_site_pairs=near,rows=rows,
    source_sha256={name:file_sha(ROOT/name) for name in ('training/scripts/word_data.py','training/scripts/word_decode.py','training/scripts/word_select.py','src/ml/word-decoder.ts')},
    script_sha256=sha(Path(__file__).read_bytes()),
    qualifications=['Natural135 and their270 derived stress variants are correlated, not405 independent words.',
      'Only completely represented context cells can create the complete targets considered here; clipped context cells are outside this complete-cell audit.',
      'Pairs are distinct context entries; chosen annotation rows are unique and exact matching to context is verified. Near-pair records distinguish identical geometry/mask from separate source rows and retain IoU.',
      'Zero adjacent ideal centers would rule out this exact perfect-target failure geometry in this corpus, not suppressions caused by imperfect predicted peaks/geometry.',
      'Equal and unequal objectness were demonstrated on analytic tensors in the separate source probe; no scores are synthesized or tuned for the corpus here.'])
assert not (OUT/'report.json').exists()
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(report_sha256=sha((OUT/'report.json').read_bytes()),quality_pass=report['quality_pass'],groups=report['groups'],
    adjacent_pairs=len(near),min_selected_grid_chebyshev=report['minimum_selected_pair_grid_chebyshev'],min_other_grid_chebyshev=report['minimum_selected_vs_other_full_context_grid_chebyshev'])))
