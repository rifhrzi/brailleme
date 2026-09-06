"""Summarize exact decisions and approximate geometry from the saved comparison."""
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
r=json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
assert r['summary']['baseline']['thresholds']==r['summary']['candidate']['thresholds']
decisions=[]
patterns=[]
confidence=[]
for a,b in zip(r['rows']['baseline'],r['rows']['candidate']):
    if (a['text'],a['reason'])!=(b['text'],b['reason']):
        decisions.append(dict(index=a['index'],expected=a['expected'],baseline=dict(text=a['text'],reason=a['reason']),candidate=dict(text=b['text'],reason=b['reason'])))
    for x,y in zip(a['gt_cells'],b['gt_cells']):
        assert x['matched']==y['matched']
        if not x['matched']:continue
        item=dict(index=a['index'],word=a['expected'],gt_index=x['gt_index'],gt_letter=x['gt_letter'],
            baseline=dict(mask=x['predicted_mask'],letter=x['predicted_letter'],dot_confidence=x['dot_confidence'],confidence_pass=x['dot_confidence_pass']),
            candidate=dict(mask=y['predicted_mask'],letter=y['predicted_letter'],dot_confidence=y['dot_confidence'],confidence_pass=y['dot_confidence_pass']))
        if x['predicted_mask']!=y['predicted_mask']:patterns.append(item)
        if x['dot_confidence_pass']!=y['dot_confidence_pass']:confidence.append(item)
assert [v['index'] for v in decisions]==[23,46]
assert len(patterns)==3 and len(confidence)==3
data=dict(scope=r['scope'],comparison_sha256=sha(OUT/'comparison.json'),source_sha256=sha(__file__),
    thresholds=r['summary']['candidate']['thresholds'],changed_decisions=decisions,
    changed_matched_masks=patterns,changed_matched_dot_confidence_pass=confidence,
    geometry=r['transitions']['paired_geometry_and_confidence_deltas_candidate_minus_baseline'])
(OUT/'decision-transitions.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
lines=['# Fixed-moment BN affine epoch 9: cached held405 diagnosis','',
    'No model execution, external 607 access, decoder edits, threshold search, or accuracy improvement claim. The frozen 405 consists of 135 originals with two correlated stress variants. Geometry matching is approximate greedy IoU >=0.25; it is not AP.','',
    'Both models produce 357 correct / 405, 362 accepted, 43 rejected, and 98.6188% accepted precision. Both have 130/135 natural and 227/270 stress correct. Thresholds are identical: proposal 0.25, accept 0.25, dot confidence 0.7.','',
    'Only two final word decisions change. ZWUKOW at held 23 changes from correct to low_confidence: its W mask stays 58, while dot confidence falls 0.760993 -> 0.643575. NA at held 46 changes from low_confidence to correct: N mask stays 29, while confidence rises 0.675837 -> 0.704330. Thus one gained and one lost word cancel; all other final texts and rejection reasons are identical.','',
    'All 1,518 matched GT cells stay matched and all 111 unmatched GT cells stay unmatched. Post-NMS detection count stays 1,522. Among matched cells, wrong masks increase 8 -> 9 and wrong individual bits 12 -> 13. Three masks change:','',
    '| Held index | Word | GT index | Truth | Baseline | BN epoch 9 | Final word result |',
    '|---:|---|---:|---|---|---|---|']
for p in patterns:
    lines.append(f"| {p['index']} | {p['word']} | {p['gt_index']} | {p['gt_letter']} | {p['baseline']['letter']} ({p['baseline']['mask']}) | {p['candidate']['letter']} ({p['candidate']['mask']}) | low_confidence in both |")
lines+=['','The existing G -> H error in held 126 becomes more confident 0.635941 -> 0.728144; that word remains rejected because other cells are uncertain. This change is not a correct reading.','',
    'Geometry shifts are small: on the same 1,518 matches, mean IoU changes by +0.002617897 and mean center error by -0.047338901 px. Median paired deltas are +0.001603767 IoU and -0.029191799 px. Mean objectness increases 0.010992009 and mean dot confidence 0.001050740, without improving total whole-word success.','',
    'PRIMERA unsupported_pattern, the four S clipped rows, and the accepted-short PTICA/WRAQ/EGO failures retain their original final decisions. Learned arrays and numeric geometry changed; the tied word score is not evidence that training did nothing.','',
    'The selection rule retained v15 control after the tie. This diagnostic does not change that decision or establish performance on new/physical data.','',
    'Evidence: [comparison.json](comparison.json), [decision-transitions.json](decision-transitions.json), [summarize.py](summarize.py). Cache, checkpoint, all source and fixture hashes are bound in comparison.json.','']
(OUT/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(comparison_sha256=sha(OUT/'comparison.json'),transitions_sha256=sha(OUT/'decision-transitions.json'),findings_sha256=sha(OUT/'FINDINGS.md'))))
