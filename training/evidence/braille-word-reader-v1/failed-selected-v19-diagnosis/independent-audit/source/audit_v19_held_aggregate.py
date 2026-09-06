"""Verify existing native-agent held comparison hashes/aggregates, no inference."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
OUT = WORK/'v19-held-diagnosis'
OUT.mkdir(exist_ok=False)
(OUT/'source').mkdir()
def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()
def link(path):
    return {'file':str(Path(path).relative_to(ROOT)), 'sha256':sha(path)}
path = WORK/'run24/held-diagnosis/bn-affine-comparison/comparison.json'
assert sha(path) == '0d2b9dd0126f0a25d3deadfc3fbc0ae79de059f57fb44419267f2658f7b3fc48'
report = json.loads(path.read_bytes())
for name, digest in report['source_bindings'].items():
    assert sha(ROOT/name) == digest, name
fixture = json.loads((WORK/'data/validation-stress-selection.json').read_bytes())['records']
cache = json.loads((WORK/'run24/held-diagnosis/cache-report.json').read_bytes())
selection = json.loads((WORK/'selected-v19/checkpoint-selection.json').read_bytes())
baseline = report['summary']['baseline']; candidate=report['summary']['candidate']
assert baseline['checkpoint_sha256'] == 'cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7'
assert candidate['checkpoint_sha256'] == cache['checkpoint_sha256'] == '172d5d59d0f817e55df734af4359456fb3f32cac51e3a3ac3a0a4406583c942f'
assert baseline['thresholds'] == candidate['thresholds']
assert [candidate['thresholds'][key] for key in ('proposalScore','acceptScore','dotConfidence')] == [.25,.25,.7]
assert report['model_execution'] is False and report['external_outputs_used'] is False
summaries={}
for model, rows in report['rows'].items():
    assert len(rows)==len(fixture)==405
    for idx,(row,gt) in enumerate(zip(rows,fixture)):
        assert row['index']==idx and row['expected']==gt['expected'] and row['source']==gt['source'] and row['image']==gt['image']
        assert len(row['gt_cells'])==len(gt['boxes'])==len(gt['masks'])==len(gt['expected'])
        assert [c['gt_mask'] for c in row['gt_cells']]==gt['masks']
        assert row['correct'] == (row['text']==gt['expected'])
        assert sha(WORK/'data'/row['image']) == cache['image_sha256'][row['image']]
    metrics={'total':len(rows),'correct':sum(r['correct'] for r in rows),'accepted':sum(bool(r['text']) for r in rows),
             'rejected':sum(not r['text'] for r in rows),'post_nms_detections':sum(len(r['cells']) for r in rows),
             'unmatched_gt_cells':sum(len(r['unmatched_gt']) for r in rows),'unmatched_prediction_cells':sum(len(r['unmatched_predictions']) for r in rows),
             'rejection_reasons':dict(Counter(r['reason'] for r in rows if not r['text']))}
    assert all(report['summary'][model]['words'][key]==value for key,value in metrics.items())
    cells=[c for r in rows for c in r['gt_cells']]
    matched=[c for c in cells if c['matched']]
    assert len(cells)==1629 and len(matched)==1518
    assert sum(not c['mask_correct'] for c in matched)==report['summary'][model]['cells']['matched_wrong_mask']
    wrong=[r for r in rows if r['text'] and not r['correct']]
    summaries[model]={**metrics,'quality_rejected':sum(r['quality_rejected'] for r in rows),
                      'unmatched_GT_quality_placeholders':sum(len(r['unmatched_gt']) for r in rows if r['quality_rejected']),
                      'unmatched_GT_after_quality_pass':sum(len(r['unmatched_gt']) for r in rows if not r['quality_rejected']),
                      'matched_wrong_masks':sum(not c['mask_correct'] for c in matched),
                      'accepted_wrong':[{'index':r['index'],'expected':r['expected'],'text':r['text'],'unmatched_gt':r['unmatched_gt'],
                                         'wrong_confident_matched_masks':[{'truth':c['gt_mask'],'predicted':c['predicted_mask'],'dot_confidence':c['dot_confidence']} for c in r['gt_cells'] if c['matched'] and not c['mask_correct'] and c['dot_confidence_pass']]} for r in wrong]}
a,b=report['rows']['baseline'],report['rows']['candidate']
assert [r['quality_rejected'] for r in a]==[r['quality_rejected'] for r in b]
changed=[{'index':x['index'],'expected':x['expected'],'baseline':[x['text'],x['reason']],'candidate':[y['text'],y['reason']]} for x,y in zip(a,b) if (x['text'],x['reason'])!=(y['text'],y['reason'])]
assert [v['index'] for v in changed]==[23,46]
assert all(x['unmatched_gt']==y['unmatched_gt'] and x['unmatched_predictions']==y['unmatched_predictions'] for x,y in zip(a,b))
selected=json.loads((WORK/'selected-v19/selected-validation.json').read_bytes())
assert selected['checkpoint_sha256']==baseline['checkpoint_sha256']
sources=[Path(__file__),WORK/'compare_held_cache.py',WORK/'selected-v19/source/word_decode.py',WORK/'selected-v19/source/word_data.py',ROOT/'training/scripts/benchmark_angelina.py']
retained=[]
for source in sources:
    target=OUT/'source'/source.name;shutil.copyfile(source,target);assert sha(target)==sha(source);retained.append(link(target))
result={'status':'passed_hash_and_aggregate_review','scope':'Existing native-agent report and source/fixture/cache hashes plus row aggregation checked once. No repeated proposal trace, decoding, NPZ inference, external607 access, export, thresholds or tie-rule changes.',
        'inputs':[link(path),link(WORK/'run24/held-diagnosis/cache-report.json'),link(WORK/'selected-v19/checkpoint-selection.json'),link(WORK/'selected-v19/selected-validation.json')],
        'verified_source_and_input_bindings':report['source_bindings'],'sources':retained,'summaries':summaries,'changed_final_decisions':changed,
        'selected_control_first_tie_preserved':True,
        'important_limit':'The111 unmatched GT cells include103 cells in30 quality-rejected zero placeholders; only8 are unmatched after quality pass. Do not describe111 as detector misses. Greedy IoU>=.25 can also conflate severe geometry errors with misses; raw proposal detail was not repeated.',
        'interpretation':'Both models retain357/405 correct,362accepted and43rejected. One gained NA and one lost ZWUKOW cancel. All5 wrong accepted results stay shorter than GT; only PTICA->RC includes a confidently wrong matched T->R as well as missing cells. Four clipped S decisions and all matched/unmatched assignments stay unchanged. This does not establish physical accuracy or a causal failure diagnosis.'}
p=OUT/'independent-aggregate-audit.json';p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'report':str(p),'sha256':sha(p),'changed':changed,'status':result['status']}))
