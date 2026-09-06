"""Quantify column ambiguity from the frozen cached-output comparison only."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE/'mirror-comparison/comparison.json'
FIXTURE = HERE.parents[1]/'data/validation-stress-selection.json'
OUT = HERE/'mirror-comparison/pattern-followup.json'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def stats(values):
    return {'n':len(values), 'mean':float(np.mean(values)) if values else None,
            'median':float(np.median(values)) if values else None,
            'min':float(min(values)) if values else None, 'max':float(max(values)) if values else None}


def main():
    assert not OUT.exists()
    report=json.loads(BASE.read_text(encoding='utf-8'))
    fixture=json.loads(FIXTURE.read_text(encoding='utf-8'))['records']
    assert len(fixture)==405
    assert sha(FIXTURE) in report['source_bindings'].values()
    lost=set(report['transitions']['lost_indices'])
    cases=[]
    for row in report['rows']['mirror']:
        for gt in row['gt_cells']:
            if not gt['matched'] or gt['mask_correct']:
                continue
            mirrored=((gt['gt_mask']&7)<<3)|((gt['gt_mask']>>3)&7)
            if gt['predicted_mask']!=mirrored:
                continue
            box=fixture[row['index']]['boxes'][gt['gt_index']]
            control=report['rows']['control'][row['index']]['gt_cells'][gt['gt_index']]
            cell=row['cells'][gt['prediction_index']]
            assert control['matched']
            cases.append({'index':row['index'],'image':row['image'],'source':row['source'], 'expected':row['expected'],
                'gt_index':gt['gt_index'],'gt_letter':gt['gt_letter'],'gt_mask':gt['gt_mask'],
                'predicted_mask':gt['predicted_mask'],'lost_word':row['index'] in lost,'reason':row['reason'],
                'width_px':box[2]-box[0],'center_dx_px':gt['center_dx_px'],
                'center_dx_div_gt_width':gt['center_dx_px']/(box[2]-box[0]),
                'mirror_minus_control_dx_div_gt_width':(gt['center_dx_px']-control['center_dx_px'])/(box[2]-box[0]),
                'control_mask_correct':control['mask_correct'], 'score':gt['score'],
                'dot_confidence':gt['dot_confidence'], 'gt_box_px':box, 'predicted_box_px':cell['box_px'],
                'iou':gt['iou']})
    clipped=[]
    for row in report['rows']['mirror']:
        if row['index'] not in lost or row['reason']!='clipped':
            continue
        selection=np.asarray(fixture[row['index']]['selection'])*[480,192,480,192]
        violations=[]
        for pi,cell in enumerate(row['cells']):
            x0,y0,x1,y1=cell['box_px']
            sides=[]
            if x0<selection[0]: sides.append('guide_left')
            if y0<selection[1]: sides.append('guide_top')
            if x1>selection[2]: sides.append('guide_right')
            if y1>selection[3]: sides.append('guide_bottom')
            if x0<.008*480: sides.append('frame_left_margin')
            if y0<.008*192: sides.append('frame_top_margin')
            if x1>(1-.008)*480: sides.append('frame_right_margin')
            if y1>(1-.008)*192: sides.append('frame_bottom_margin')
            if sides:
                matched=[gt for gt in row['gt_cells'] if gt['matched'] and gt['prediction_index']==pi]
                violations.append({'prediction_index':pi,'mask':cell['mask'],'sides':sides,
                    'matched_gt':matched[0]['gt_letter'] if matched else None,
                    'exact_mirror_error':any(c['index']==row['index'] and c['gt_index']==matched[0]['gt_index'] for c in cases) if matched else False})
        assert violations
        clipped.append({'index':row['index'],'expected':row['expected'],'violations':violations})
    summary={'scope':'Post-hoc approximate diagnosis on held405 saved predictions; not evidence of training-code label bug or causal isolation.',
        'source_sha256':sha(__file__),'comparison_sha256':sha(BASE),'fixture_sha256':sha(FIXTURE),
        'exact_horizontal_mirror_mask_errors':len(cases),
        'by_pattern':dict(Counter(c['gt_letter'] for c in cases)),
        'on_gt_masks_using_only_left_column':sum((c['gt_mask']&56)==0 for c in cases),
        'control_correct_masks_among_cases':sum(c['control_mask_correct'] for c in cases),
        'unique_words':len(set(c['index'] for c in cases)),
        'unique_base_crops':len(set(fixture[c['index']].get('natural_image',fixture[c['index']]['image']) for c in cases)),
        'previously_correct_words_lost':len(set(c['index'] for c in cases if c['lost_word'])),
        'center_dx_px':stats([c['center_dx_px'] for c in cases]),
        'center_dx_div_gt_width':stats([c['center_dx_div_gt_width'] for c in cases]),
        'mirror_minus_control_dx_div_gt_width':stats([c['mirror_minus_control_dx_div_gt_width'] for c in cases]),
        'objectness':stats([c['score'] for c in cases]),'dot_confidence':stats([c['dot_confidence'] for c in cases]),
        'lost_clipped_words':len(clipped),
        'lost_clipped_words_with_exact_mirror_violator':sum(any(v['exact_mirror_error'] for v in row['violations']) for row in clipped),
        'lost_clipped_violation_sides':dict(Counter(side for row in clipped for v in row['violations'] for side in v['sides'])),
        'non_mirror_errors':dict(Counter((str(c['gt_mask'])+'->'+str(c['predicted_mask'])) for row in report['rows']['mirror'] for c in row['gt_cells'] if c['matched'] and not c['mask_correct'] and c['predicted_mask']!=(((c['gt_mask']&7)<<3)|((c['gt_mask']>>3)&7)))),
        'cases':cases,'lost_clipped_cases':clipped,
        'interpretation':'All exact mirror errors are left-column-only patterns with boxes shifted left. This is consistent with the detector explaining the same visible column as the right column of a differently centered logical cell. These cached outputs cannot establish exact dot positions in Angelina annotations or isolate the training intervention from all optimization effects. No threshold was changed.'}
    OUT.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('cases','lost_clipped_cases')},indent=2))
    print('REPORT_SHA256',sha(OUT))


if __name__=='__main__':
    main()
