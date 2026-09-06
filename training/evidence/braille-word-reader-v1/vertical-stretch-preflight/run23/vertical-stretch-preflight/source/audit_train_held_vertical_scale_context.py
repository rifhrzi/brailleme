"""Annotation-only scale preflight; no external records or predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('.braille-training-work/word-scan-v1-20260905')
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={'train':ROOT/'combined-training-v5/combined-training-manifest.json',
       'validation_patches':ROOT/'data/validation.json',
       'held405':ROOT/'data/validation-stress-selection.json'}
def quantiles(values):
    a=np.asarray(values,float)
    return dict(zip(['min','p05','median','p95','max'],[float(x) for x in np.quantile(a,[0,.05,.5,.95,1])])) if len(a) else None

result={}
for label,path in files.items():
    blob=json.loads(path.read_text());records=blob['records'] if isinstance(blob,dict) else blob
    grouped={}
    for rec in records:
        if label=='train':group='DSBI' if rec.get('source','').startswith('dsbi/') else 'Angelina'
        elif label=='held405':group=rec['stress_transform']['name'] if 'stress_transform' in rec else rec['validation_group']
        else:group='all'
        boxes=np.asarray(rec['boxes'],float).reshape(-1,4)
        if not len(boxes):continue
        full=(boxes[:,0]>=0)&(boxes[:,1]>=0)&(boxes[:,2]<=480)&(boxes[:,3]<=192)
        if 'originally_complete' in rec:full &= rec['originally_complete']
        bucket=grouped.setdefault(str(group),{'records':0,'all':0,'full':[]})
        bucket['records']+=1;bucket['all']+=len(boxes);bucket['full'].extend(boxes[full].tolist())
    measured={}
    for group,bucket in grouped.items():
        b=np.asarray(bucket.pop('full'),float);w=b[:,2]-b[:,0];h=b[:,3]-b[:,1]
        stretched=b.copy();stretched[:,[1,3]]=96+(stretched[:,[1,3]]-96)*1.1
        newly_clipped=(stretched[:,1]<0)|(stretched[:,3]>192)
        measured[group]={**bucket,'full_cell_occurrences':len(b),'width_pixels':quantiles(w),'height_pixels':quantiles(h),
                         'box_width_height_ratio':quantiles(w/h),'minimum_vertical_canvas_clearance_pixels':quantiles(np.minimum(b[:,1],192-b[:,3])),
                         'height_pixels_after_vertical_0_9':quantiles(h*.9),'height_pixels_after_vertical_1_1':quantiles(h*1.1),
                         'cells_crossing_vertical_canvas_at_centered_1_1':int(newly_clipped.sum()),
                         'fraction_crossing_vertical_canvas_at_centered_1_1':float(newly_clipped.mean())}
    result[label]={'file':str(path),'sha256':sha(path),'manifest_records':len(records),'groups':measured}

report={'status':'descriptive annotation-only preflight complete',
        'script_sha256':sha(__file__),'scope':'TRAIN cache and held-validation annotations only. No model outputs, external corpus, image relabeling, or threshold tuning.',
        'weighting':'Cached complete-cell occurrences, before stochastic zoom/translation/rotation and sampler reweighting; not unique physical cells and not expected training epoch exposure.',
        'hypothetical_transform':'Vertical factor1.1 around corner-coordinate image center y96, x unchanged; this isolates the added factor, not the full combined augmentation.',
        'datasets':result,
        'physical_padding_reference':{'file':'train-cell-extent-review/findings.json','sha256':sha(ROOT/'train-cell-extent-review/findings.json'),
          'conclusion':'Exact DSBI padding is .3 horizontal source pitch per edge. Four small TRAIN-only Angelina multi-dot optical estimates distinguish larger/manual margins from dot pitch but have +/-2px uncertainty. Held Angelina provides only boxes/labels, so physical point-to-box padding cannot be recovered exactly and is not inferred from bbox aspect.'},
        'interpretation':'The proposed .9-1.1 factor is the paper-prescribed ablation, not fitted to these sizes. It changes height10percent rather than imposing a new size band. Border cells can be cut and must retain the existing all-stage completeness/point-survivor logic. No new acceptance gates or cell rejection thresholds are proposed.'}
out=ROOT/'train-held-vertical-scale-context-audit.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(out),'sha256':sha(out),'datasets':result},indent=2))
