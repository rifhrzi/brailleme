"""Diagnose a frozen failed benchmark without changing its fixtures or thresholds."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from word_data import save_json
from word_decode import decode


def quantiles(values):
    return dict(zip(('p05', 'p25', 'median', 'p75', 'p95'), np.quantile(values, [.05, .25, .5, .75, .95]).tolist())) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    if args.validation:
        fixtures=json.loads((args.data/'validation-words.json').read_text())['records']
        outputs=np.load(args.run/'backend-fullprecision.npz')['outputs']
        selected=json.loads((args.run/'selected-validation-fullprecision.json').read_text())
        oracle_records=json.loads((args.data/selected.get('oracle',{}).get('fixture_file','validation-words.json')).read_text())['records']
        if len(outputs)!=len(oracle_records):raise ValueError('Oracle count differs from its declared fixture manifest')
        indices={record['image']:index for index,record in enumerate(oracle_records)}
        outputs=outputs[[indices[record['image']] for record in fixtures]]
        thresholds=selected['thresholds']
        rows=[]
        for record,output in zip(fixtures,outputs):
            result=decode(output,thresholds,record['selection'])
            rows.append(dict(record,predicted=result['text'],rejection=result['reason'],detected_cells=result['cells']))
        report={'words':rows}
    else:
        report = json.loads((args.run/'public-benchmark.json').read_text())
    evaluated_group='natural_validation_words' if args.validation else 'external'
    stats = defaultdict(list)
    counts = Counter()
    by_height = defaultdict(Counter)
    bit_errors=Counter()
    by_mask=defaultdict(Counter)
    for record in report['words']:
        gt = np.asarray(record['boxes'])
        gh = gt[:, 3]-gt[:, 1]
        gc = (gt[:, :2]+gt[:, 2:])/2
        predictions = record['detected_cells']
        centers = np.asarray([[(v['box'][0]+v['box'][2])*240, (v['box'][1]+v['box'][3])*96] for v in predictions]).reshape(-1, 2)
        used = set()
        bucket = str(int(np.median(gh)//25)*25)
        by_height[bucket]['words'] += 1
        by_height[bucket]['correct'] += record['expected']==record['predicted']
        stats[evaluated_group+'_gt_height'].extend(gh.tolist())
        for index, center in enumerate(gc):
            counts['ground_truth_cells'] += 1
            by_mask[str(record['masks'][index])]['total']+=1
            candidates = [(float(np.linalg.norm(pc-center)/gh[index]), pi) for pi, pc in enumerate(centers) if pi not in used]
            distance, match = min(candidates) if candidates else (float('inf'), -1)
            if distance>.45:
                counts['unmatched_ground_truth_cells'] += 1
                continue
            used.add(match)
            p = predictions[match]
            pb = np.asarray(p['box'])*[480, 192, 480, 192]
            stats['matched_center_error_per_height'].append(distance)
            stats['matched_signed_x_center_per_height'].append(float((centers[match,0]-center[0])/gh[index]))
            stats['matched_signed_y_center_per_height'].append(float((centers[match,1]-center[1])/gh[index]))
            stats['matched_width_ratio'].append(float((pb[2]-pb[0])/(gt[index, 2]-gt[index, 0])))
            stats['matched_height_ratio'].append(float((pb[3]-pb[1])/gh[index]))
            stats['matched_objectness'].append(p['score'])
            mask = sum((int(dot>=.5)<<bit) for bit, dot in enumerate(p['dots']))
            true_mask=record['masks'][index]
            stats['matched_dot_confidence'].append(min(max(dot,1-dot) for dot in p['dots']))
            stats['matched_true_bit_probability'].append(min(dot if (true_mask>>bit)&1 else 1-dot for bit,dot in enumerate(p['dots'])))
            for bit in range(6):bit_errors[str(bit+1)]+=bool((mask^true_mask)&(1<<bit))
            by_mask[str(true_mask)]['matched']+=1
            by_mask[str(true_mask)]['correct_mask']+=mask==true_mask
            counts['matched_ground_truth_cells'] += 1
            counts['matched_correct_mask'] += mask==record['masks'][index]
            if record['rejection']=='clipped':
                counts['clipped_word_matched_cells'] += 1
                counts['clipped_word_correct_masks'] += mask==record['masks'][index]
        counts['unmatched_predictions'] += len(predictions)-len(used)
        if record['rejection']=='clipped':
            counts['clipped_with_unmatched_prediction'] += len(predictions)>len(used)
        pixels=np.asarray(Image.open(args.data/record['image']), np.float32)/255.
        luminance=pixels@np.asarray([.2126,.7152,.0722], np.float32)
        stats[evaluated_group+'_luminance_std'].append(float(luminance.std()))
    for split in ('train', 'validation'):
        records=json.loads((args.data/(split+'.json')).read_text())
        for record in records:
            boxes=np.asarray(record['boxes']).reshape(-1,4)
            stats[split+'_gt_height'].extend((boxes[:,3]-boxes[:,1]).tolist())
        # Deterministic spread through the cached dataset; no quality selection.
        for record in records[::max(1, len(records)//256)]:
            pixels=np.asarray(Image.open(args.data/record['image']), np.float32)/255.
            luminance=pixels@np.asarray([.2126,.7152,.0722], np.float32)
            stats[split+'_luminance_std'].append(float(luminance.std()))
    result={'scope':'Natural held validation geometry from its frozen oracle; no external data.' if args.validation else 'Post-test failure diagnosis only; no external gradients, threshold calibration, relabeling or fixture removal.', 'matching':'Greedy nearest unused selected proposal center, distance <= .45 ground-truth cell height; approximate diagnostic, not detector AP.', 'counts':dict(counts), 'quantiles':{k:quantiles(v) for k,v in stats.items()}, 'words_by_median_cell_height_pixels':dict(by_height)}
    result['rejection_reasons']=dict(Counter(row['rejection'] for row in report['words'] if not row['predicted']))
    result['matched_bit_errors']=dict(bit_errors)
    result['by_ground_truth_mask']=dict(by_mask)
    save_json(args.run/('validation-geometry-analysis.json' if args.validation else 'failure-analysis.json'), result)
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
