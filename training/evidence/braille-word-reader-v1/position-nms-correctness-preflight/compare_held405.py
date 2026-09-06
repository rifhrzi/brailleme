"""One authorized fixed-threshold cached405 comparison; no inference/calibration."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

OUT=Path(__file__).resolve().parent
WORK=OUT.parent
DEST=OUT/'held405-comparison'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def save(name,data):
    p=DEST/name;assert not p.exists();p.write_text(json.dumps(data,indent=2)+'\n')
def load(side):
    path=OUT/side/'training/scripts/word_decode.py';spec=importlib.util.spec_from_file_location(side,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def metric(rows,key):
    accepted=sum(bool(r[key]['text']) for r in rows);correct=sum(r[key]['text']==r['expected'] for r in rows)
    return {'total':len(rows),'accepted':accepted,'correct':correct,'exact_word_accuracy':correct/len(rows),
            'accepted_accuracy':correct/accepted if accepted else None,
            'rejection_reasons':dict(Counter(r[key]['reason'] for r in rows if not r[key]['text'])),
            'accepted_wrong_shorter':sum(bool(r[key]['text']) and len(r[key]['text'])<len(r['expected']) for r in rows)}
def proposals(output,t,selection):
    peaks=0;all_count=0;suppressed=[]
    for gy,gx in np.argwhere(output[...,0]>=t['proposalScore']):
        row=output[gy,gx].astype(np.float64)
        cx,cy=(gx+row[1])/120,(gy+row[2])/48
        b=[cx-row[3]/2,cy-row[4]/2,cx-row[3]/2+row[3],cy-row[4]/2+row[4]]
        if b[2]<=selection[0] or b[3]<=selection[1] or b[0]>=selection[2] or b[1]>=selection[3]:continue
        all_count+=1;nb=output[max(0,gy-1):gy+2,max(0,gx-1):gx+2,0]
        tied=np.argwhere(nb==row[0])+[max(0,gy-1),max(0,gx-1)]
        is_peak=row[0]>=nb.max() and not any(y*120+x<gy*120+gx for y,x in tied)
        if is_peak:peaks+=1
        else:suppressed.append({'grid_yx':[int(gy),int(gx)],'score':float(row[0]),'box':b,
            'width_pixels':float(row[3]*480),'height_pixels':float(row[4]*192)})
    return all_count,peaks,suppressed
def main():
    assert not DEST.exists();DEST.mkdir()
    prototype=read(OUT/'prototype-spec.json')
    pytests=read(OUT/'python-test-report.json');tstests=read(OUT/'typescript-test-report.json')
    assert pytests['status']==tstests['status']=='passed' and pytests['cases']==tstests['cases']==20
    for side,key in [('source-original','source_original_sha256'),('source-private','source_private_sha256')]:
        for name,digest in prototype[key].items():assert sha(OUT/side/name)==digest
    selected_path=WORK/'selected-v15/selected-validation.json';selected=read(selected_path)
    cache=WORK/'selected-v15/backend-fullprecision.npz';manifest=WORK/'data/validation-stress-selection.json'
    assert sha(cache)=='20f0a6c0224ec7f640ca0e46275914e912a2b1b440882f0b1e9e94e8bb72556b'
    assert sha(manifest)=='e2ab968f7c4dcf7e774181fde687e1321f58d20567528af4b21fd03ff5a3e625'
    assert selected['checkpoint_sha256']=='cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
    thresholds=selected['thresholds'];assert [thresholds[k] for k in ['proposalScore','acceptScore','dotConfidence']]==[.25,.25,.7]
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Single root-authorized cached405 comparison; no model execution, calibration, selection or external607 data.',
        'checkpoint_sha256':selected['checkpoint_sha256'],'cache_path':str(cache),'raw_cache_sha256':sha(cache),
        'fixture_sha256':sha(manifest),'selected_validation_sha256':sha(selected_path),'thresholds':thresholds,
        'prototype_spec_sha256':sha(OUT/'prototype-spec.json'),'python_test_report_sha256':sha(OUT/'python-test-report.json'),
        'typescript_test_report_sha256':sha(OUT/'typescript-test-report.json'),'script_sha256':sha(__file__),
        'raw_cache_provenance':'word_select saves original model.predict outputs; prepare_calibration_outputs copies and changes calibration-only tensors. Do not use peak-filtered cache.',
        'source_private_sha256':prototype['source_private_sha256'],'source_original_sha256':prototype['source_original_sha256']}
    save('comparison-protocol.json',protocol)
    raw=np.load(cache,allow_pickle=False);outputs=raw['outputs'];quality=raw['quality_rejected']
    assert outputs.shape==(405,48,120,11) and outputs.dtype==np.float32 and quality.shape==(405,) and quality.dtype==np.bool_
    assert str(raw['checkpoint_sha256'])==selected['checkpoint_sha256'] and str(raw['fixture_sha256'])==sha(manifest)
    assert int(quality.sum())==selected['quality_rejected_before_decode']==30
    records=read(manifest)['records'];assert len(records)==405
    sys.path.insert(0,str(OUT/'source-original/training/scripts'))
    old=load('source-original');new=load('source-private')
    rows=[];seconds=Counter();extras=[]
    for index,(output,record) in enumerate(zip(outputs,records,strict=True)):
        selection=record.get('selection') or [0.,0.,1.,1.]
        row={'index':index,'image':record['image'],'source':record['source'],'group':record['validation_group'],
             'expected':record['expected'],'quality_rejected':bool(quality[index])}
        if quality[index]:
            row.update(baseline={'text':'','reason':'quality_rejected_cached','cells':[]},private={'text':'','reason':'quality_rejected_cached','cells':[]})
        else:
            for key,module in [('baseline',old),('private',new)]:
                start=time.perf_counter();result=module.decode(output,thresholds,selection);seconds[key]+=time.perf_counter()-start
                row[key]={'text':result['text'],'reason':result['reason'],'cells':[{'box':c['box'],'score':c['score']} for c in result['cells']]}
            all_count,peaks,nonmax=proposals(output,thresholds,selection)
            row.update(intersecting_score_proposals=all_count,localmax_intersecting_proposals=peaks,
                       raw_nonmax_intersecting_proposals=len(nonmax),baseline_nms_cells=len(row['baseline']['cells']),private_nms_cells=len(row['private']['cells']))
            if row['baseline']['text']!=row['private']['text'] or row['baseline']['reason']!=row['private']['reason']:
                extras.append({'index':index,'expected':record['expected'],'baseline':row['baseline'],'private':row['private'],'newly_exposed_before_NMS':nonmax})
        rows.append(row)
        if (index+1)%100==0:print('decoded',index+1,flush=True)
    summary={key:metric(rows,key) for key in ('baseline','private')}
    for k in ('total','accepted','correct','exact_word_accuracy','accepted_accuracy'):
        assert summary['baseline'][k]==selected['words'][k],k
    groups={g:{key:metric([r for r in rows if r['group']==g],key) for key in ('baseline','private')} for g in ('natural','stress')}
    by_length={str(n):{key:metric([r for r in rows if len(r['expected'])==n],key) for key in ('baseline','private')}
               for n in sorted(set(len(r['expected']) for r in rows))}
    changed=[r['index'] for r in rows if r['baseline']['text']!=r['private']['text'] or r['baseline']['reason']!=r['private']['reason']]
    gained=[r['index'] for r in rows if r['baseline']['text']!=r['expected'] and r['private']['text']==r['expected']]
    lost=[r['index'] for r in rows if r['baseline']['text']==r['expected'] and r['private']['text']!=r['expected']]
    wrong={key:[{'index':r['index'],'expected':r['expected'],'predicted':r[key]['text'],'shorter':len(r[key]['text'])<len(r['expected'])}
                 for r in rows if r[key]['text'] and r[key]['text']!=r['expected']] for key in ('baseline','private')}
    assert sha(cache)==protocol['raw_cache_sha256'] and sha(manifest)==protocol['fixture_sha256']
    assert not any(x in sys.modules for x in ('torch','tensorflow','keras'))
    save('rows.json',rows);save('changed-proposals.json',extras)
    report={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),'comparison_protocol_sha256':sha(DEST/'comparison-protocol.json'),
        'summary':summary,'groups':groups,'by_length':by_length,'changed_indices':changed,'gained_exact_indices':gained,'lost_exact_indices':lost,
        'accepted_wrong':wrong,'quality_rejected_unchanged':int(quality.sum()),'quality_pass_proposal_counts':{
            key:sum(r.get(key,0) for r in rows) for key in ('intersecting_score_proposals','localmax_intersecting_proposals','raw_nonmax_intersecting_proposals','baseline_nms_cells','private_nms_cells')},
        'decoder_cpu_seconds_in_this_run':dict(seconds),'rows_sha256':sha(DEST/'rows.json'),'changed_proposals_sha256':sha(DEST/'changed-proposals.json'),
        'no_model_execution':True,'no_recalibration':True,'no_external607_read':True,
        'limitations':['Same fixed raw outputs, thresholds and cached quality flags; this comparison selects no checkpoint and does not promote the prototype.',
            'The raw cache records only a Boolean quality rejection, so30 such reasons are reported as quality_rejected_cached. Their original dark/glare/low-contrast subtype is not guessed from new pixel reads.',
            'CPU decoder timings are descriptive single-run timings, not Android performance. Additional score proposals can expose unsupervised box/dot channels at downweighted-negative sites.',
            'Ideal405 has no adjacent-grid chosen GT pair according to the separate native audit; this corpus cannot demonstrate recovery of the tiny-cell synthetic contract bug.']}
    save('report.json',report)
    print(json.dumps({'status':'passed','summary':summary,'changed':changed,'gained':gained,'lost':lost,'report_sha256':sha(DEST/'report.json')}))
if __name__=='__main__':main()
