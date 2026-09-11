"""Explain rejected geometry from cached outputs without changing acceptance."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from word_data import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path('C:/bm-datasets/AngelinaDataset'))
    args=parser.parse_args()
    report_path=args.run/'public-benchmark.json'
    report=json.loads(report_path.read_text())
    totals=Counter(); rows=[]; signed=defaultdict(list)
    for record in report['words']:
        if record['rejection']!='clipped':continue
        gt=np.asarray(record['boxes'],float); centers=(gt[:,:2]+gt[:,2:])/2
        heights=gt[:,3]-gt[:,1]
        context=np.asarray(record['context_boxes'],float)
        context_centers=(context[:,:2]+context[:,2:])/2
        context_heights=context[:,3]-context[:,1]
        selection=np.asarray(record['selection'])*[480,192,480,192]
        used=set(); kinds=set(); sides=set(); cells=[]; all_masks_confident=True
        target_order=np.argsort(centers[:,0]).tolist()
        for cell in record['detected_cells']:
            box=np.asarray(cell['box'])*[480,192,480,192];center=(box[:2]+box[2:])/2
            candidates=[(float(np.linalg.norm(center-c)/heights[i]),i) for i,c in enumerate(centers) if i not in used]
            distance,index=min(candidates) if candidates else (float('inf'),-1)
            cross=[side for side,value,edge,less in zip(('left','top','right','bottom'),box,selection,(True,True,False,False)) if (value<edge if less else value>edge)]
            if min(box[0]/480,box[1]/192,1-box[2]/480,1-box[3]/192)<report['thresholds']['edgeMargin']:
                cross.append('context_edge')
            nearest_context=int(np.argmin(np.linalg.norm(context_centers-center,axis=1)/context_heights))
            context_distance=float(np.linalg.norm(context_centers[nearest_context]-center)/context_heights[nearest_context])
            if distance<=.45:
                used.add(index)
                mask=sum(int(dot>=.5)<<bit for bit,dot in enumerate(cell['dots']))
                dots_confident=min(max(dot,1-dot) for dot in cell['dots'])>=report['thresholds']['dotConfidence']
                all_masks_confident &= mask==record['masks'][index] and dots_confident and cell['score']>=report['thresholds']['acceptScore']
                kind='selected_terminal_cell' if index in (target_order[0],target_order[-1]) else 'selected_interior_cell'
                signed['x_center_per_height'].append(float((center[0]-centers[index,0])/heights[index]))
                signed['y_center_per_height'].append(float((center[1]-centers[index,1])/heights[index]))
            elif context_distance<=.45 and not np.any(np.linalg.norm(centers-context_centers[nearest_context],axis=1)<1e-5):
                kind='unselected_context_neighbor';all_masks_confident=False
            else:
                kind='unmatched_or_duplicate';all_masks_confident=False
            if cross:kinds.add(kind);sides.update(cross)
            cells.append({'prediction_box_pixels':box.tolist(),'crossed_sides':cross,'classification':kind,'selected_gt_index':index if distance<=.45 else None,'center_error_per_gt_height':distance if distance<=.45 else None,'nearest_context_index':nearest_context,'nearest_context_error_per_height':context_distance})
        all_targets_present=len(used)==len(gt)
        geometry_only_failure=all_targets_present and len(record['detected_cells'])==len(gt) and all_masks_confident
        totals.update(kinds)
        totals['clipped_words']+=1
        totals['all_target_centers_matched']+=all_targets_present
        totals['complete_correct_confident_cells_rejected_by_geometry']+=geometry_only_failure
        totals['ground_truth_selected_boxes_cross_selection']+=bool(np.any(gt[:,:2]<selection[:2]) or np.any(gt[:,2:]>selection[2:]))
        for side in sides:totals['side_'+side]+=1
        rows.append({'image':record['image'],'source':record['source'],'expected':record['expected'],'selection_pixels':selection.tolist(),'classification':sorted(kinds),'all_target_centers_matched':all_targets_present,'complete_correct_confident_cells_rejected_by_geometry':geometry_only_failure,'cells':cells})
    negatives=[]
    training=json.loads((args.data/'train.json').read_text())
    for negative in report['negative_rows']:
        if not negative['predicted']:continue
        source=args.dataset/negative['source']
        with Image.open(source) as original:
            source_size=list(original.size)
            pixels=np.asarray(original.convert('RGB').resize((480,192),Image.Resampling.BILINEAR))
        matches=[record for record in training if record['source']==negative['source']]
        equality=[]
        for match in matches:
            with Image.open(args.data/match['image']) as cached:
                equality.append(bool(np.array_equal(np.asarray(cached.convert('RGB')),pixels)))
        negatives.append(dict(negative,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_size=source_size,benchmark_crop='entire_source_image',benchmark_selection=[0.,.1875,1.,.8125],training_records=matches,training_and_benchmark_rgb_pixels_identical=equality))
    result={'scope':'Post-failure cached geometry diagnosis only. No model inference, threshold changes, fixture changes, or acceptance changes. Approximate center matching is not an alternative success metric.','benchmark_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'model_sha256':report['model_sha256'],'matching':'Greedy nearest unused selected GT center within .45 GT height; otherwise nearest unselected context center within .45 context height, else unmatched/duplicate.','counts':dict(totals),'signed_center_quantiles_for_clipped_matched_targets':{key:dict(zip(('p05','p25','median','p75','p95'),np.quantile(value,[.05,.25,.5,.75,.95]).tolist())) for key,value in signed.items()},'wrong_negative_inputs':negatives,'words':rows}
    save_json(args.run/'clipped-geometry-analysis.json',result)
    print(json.dumps({key:value for key,value in result.items() if key!='words'},indent=2))


if __name__=='__main__':main()
