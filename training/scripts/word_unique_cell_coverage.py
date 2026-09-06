"""Audit unique original source+CSV-row training coverage without inference.

Use --data for the frozen source-split directory, --manifest for a proposed
training manifest, and --output for a separate evidence directory. Every cached
annotation is inverse-mapped to its source; complete-cell eligibility uses exact
float32 canvas bounds. Output includes per-mask/family/source summaries and
full covered/uncovered row identities. Source images are opened for dimensions
only; this script does not evaluate recognition or claim physical independence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,required=True,help='Directory containing frozen source-split.json')
parser.add_argument('--dataset',type=Path,default=Path('C:/bm-datasets/AngelinaDataset'))
parser.add_argument('--manifest',type=Path,required=True,help='Training manifest to audit without mutation')
parser.add_argument('--output',type=Path,required=True,help='Separate audit output directory')
args=parser.parse_args()
DATA=args.data
DATASET=args.dataset
MANIFEST=args.manifest
HERE=args.output
if HERE.resolve()==DATA.resolve():
    raise ValueError('Use a separate output directory for audit evidence')
HERE.mkdir(parents=True,exist_ok=True)
SPLIT=DATA/'source-split.json'
MATCH_TOLERANCE_PX=.001
WIDTH,HEIGHT=480,192


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(source):
    return '/'.join(Path(source).parts[:2])


def ratio_count(total,covered,occurrences):
    return {'source_unique_cells':total,'complete_unique_cells':covered,'uncovered_unique_cells':total-covered,'complete_unique_fraction':covered/total if total else None,'complete_patch_occurrences':occurrences}


source_split=json.loads(SPLIT.read_text())
sources=source_split['images']['train']
heldout=set(source_split['images']['validation'])|set(source_split['images']['external'])
assert not set(sources)&heldout
records=json.loads(MANIFEST.read_text())
pages={}
original={}
csv_hashes={}
for source in sources:
    path=DATASET/source
    with Image.open(path) as image:
        size=image.size
    csv_path=path.with_suffix('.csv')
    rows=np.loadtxt(csv_path,delimiter=';',ndmin=2)
    boxes=rows[:,:4]*np.asarray(list(size)*2,np.float64)
    masks=rows[:,4].astype(int)
    pages[source]=(boxes,masks)
    csv_hashes[source]=sha(csv_path)
    for row_index,(box,mask) in enumerate(zip(boxes,masks),1):
        original[(source,row_index)]={'mask':int(mask),'box':box.tolist()}

complete_counts=Counter()
any_counts=Counter()
partial_counts=Counter()
unmatched=[]
ambiguous=[]
flag_disagreements=[]
metadata_disagreements=[]
source_scope_errors=[]
occurrences=[]
max_inverse_error=0.
negative_records=0
complete_occurrences=0
all_occurrences=0
runtime_complete_occurrences=0
float32_vs_float64_complete_differences=[]
for record_index,record in enumerate(records):
    boxes64=np.asarray(record['boxes'],np.float64).reshape(-1,4)
    boxes32=boxes64.astype(np.float32)
    masks=np.asarray(record['masks'],int)
    assert len(boxes32)==len(masks)
    if not len(boxes32):
        negative_records+=1
        continue
    source=record['source']
    if source not in pages:
        source_scope_errors.append({'record_index':record_index,'source':source})
        continue
    left,top,right,bottom=record['crop']
    assert right>left and bottom>top
    inverse=boxes32.astype(np.float64)*np.asarray([(right-left)/WIDTH,(bottom-top)/HEIGHT]*2)+np.asarray([left,top]*2)
    source_boxes,source_masks=pages[source]
    intersects=(boxes32[:,2]>0)&(boxes32[:,3]>0)&(boxes32[:,0]<WIDTH)&(boxes32[:,1]<HEIGHT)
    full32=(boxes32[:,0]>=0)&(boxes32[:,1]>=0)&(boxes32[:,2]<=WIDTH)&(boxes32[:,3]<=HEIGHT)
    full64=(boxes64[:,0]>=0)&(boxes64[:,1]>=0)&(boxes64[:,2]<=WIDTH)&(boxes64[:,3]<=HEIGHT)
    explicit=np.asarray(record.get('originally_complete',[True]*len(boxes32)),bool)
    assert len(explicit)==len(boxes32)
    eligible=intersects&full32&explicit
    runtime_complete_occurrences+=int((intersects&full32).sum())
    complete_occurrences+=int(eligible.sum())
    all_occurrences+=len(masks)
    for item_index in np.flatnonzero(full32!=full64):
        float32_vs_float64_complete_differences.append({'record_index':record_index,'item_index':int(item_index),'float64_box':boxes64[item_index].tolist(),'float32_box':boxes32[item_index].tolist(),'float32_complete':bool(full32[item_index])})
    for item_index in np.flatnonzero(full32&~explicit):
        flag_disagreements.append({'record_index':record_index,'item_index':int(item_index),'box':boxes32[item_index].tolist()})
    for item_index,(box,mask) in enumerate(zip(inverse,masks)):
        candidate_indices=np.flatnonzero(source_masks==mask)
        errors=np.max(np.abs(source_boxes[candidate_indices]-box),axis=1)
        matches=candidate_indices[errors<=MATCH_TOLERANCE_PX]
        occurrence={'record_index':record_index,'patch_image':record['image'],'item_index':item_index,'source':source,'mask':int(mask),'complete':bool(eligible[item_index]),'inverse_box':box.tolist()}
        if len(matches)==0:
            unmatched.append({**occurrence,'nearest_error_px':float(errors.min()) if len(errors) else None})
            continue
        if len(matches)>1:
            ambiguous.append({**occurrence,'matching_csv_rows':(matches+1).tolist()})
            continue
        row_index=int(matches[0])+1
        key=(source,row_index)
        error=float(np.max(np.abs(source_boxes[row_index-1]-box)))
        max_inverse_error=max(max_inverse_error,error)
        if 'annotation_rows' in record and record['annotation_rows'][item_index]!=row_index:
            metadata_disagreements.append({**occurrence,'matched_csv_row':row_index,'metadata_csv_row':record['annotation_rows'][item_index]})
        any_counts[key]+=1
        if eligible[item_index]: complete_counts[key]+=1
        else: partial_counts[key]+=1
        occurrences.append([record_index,record['image'],item_index,source,row_index,int(mask),int(eligible[item_index]),int(full32[item_index]),int(explicit[item_index]),error])

assert not source_scope_errors,(len(source_scope_errors),'non-training source with positive annotations')
assert not unmatched,(len(unmatched),'unmatched annotations')
assert not ambiguous,(len(ambiguous),'ambiguous annotation identities')
assert not metadata_disagreements,(len(metadata_disagreements),'explicit source row identity disagrees')
assert complete_occurrences==sum(complete_counts.values())
assert all_occurrences==sum(any_counts.values())

with (HERE/'occurrence-matches.csv').open('w',newline='',encoding='utf-8') as output:
    writer=csv.writer(output)
    writer.writerow(['record_index_zero_based','patch_image','item_index_zero_based','source','csv_row_one_based','mask','complete_eligible','float32_canvas_complete','manifest_originally_complete','inverse_max_error_source_px'])
    writer.writerows(occurrences)

source_stats={}
family_stats={}
mask_stats={}
for dimension in ['source','family','mask']:
    groups={}
    for key,value in original.items():
        source,row=key
        group=source if dimension=='source' else family(source) if dimension=='family' else str(value['mask'])
        stats=groups.setdefault(group,Counter())
        stats['total']+=1
        stats['covered']+=key in complete_counts
        stats['occurrences']+=complete_counts[key]
        stats['partial_only']+=key in any_counts and key not in complete_counts
        stats['never_sampled']+=key not in any_counts
    result={group:{**ratio_count(stats['total'],stats['covered'],stats['occurrences']),'partial_only_unique_cells':stats['partial_only'],'never_sampled_unique_cells':stats['never_sampled']} for group,stats in sorted(groups.items(),key=lambda pair:int(pair[0]) if dimension=='mask' else pair[0])}
    if dimension=='source': source_stats=result
    elif dimension=='family': family_stats=result
    else: mask_stats=result

uncovered=[]
with (HERE/'uncovered-source-cells.csv').open('w',newline='',encoding='utf-8') as output:
    writer=csv.writer(output)
    writer.writerow(['source','csv_row_one_based','mask','source_x0','source_y0','source_x1','source_y1','partial_patch_occurrences','reason'])
    for key,value in original.items():
        if key in complete_counts: continue
        source,row=key
        reason='partial_only' if key in any_counts else 'never_sampled'
        writer.writerow([source,row,value['mask'],*value['box'],partial_counts[key],reason])
        uncovered.append({'source':source,'annotation_row':row,'mask':value['mask'],'box':value['box'],'partial_patch_occurrences':partial_counts[key],'reason':reason})
(HERE/'uncovered-source-cells.json').write_text(json.dumps(uncovered,indent=2)+'\n')
with (HERE/'covered-source-cells.csv').open('w',newline='',encoding='utf-8') as output:
    writer=csv.writer(output)
    writer.writerow(['source','csv_row_one_based','mask','complete_patch_occurrences','partial_patch_occurrences'])
    for key in sorted(complete_counts):
        writer.writerow([key[0],key[1],original[key]['mask'],complete_counts[key],partial_counts[key]])

summary={
    'scope':'Unique original source+CSV-row coverage in cached unaugmented training contexts. Read-only: no model inference, no prediction data, no fixture or training changes. This is availability for positive supervision, not a count of effective training updates after stochastic augmentation.',
    'source_split_sha256':sha(SPLIT),'training_manifest_sha256':sha(MANIFEST),'manifest':str(MANIFEST),'dataset_root':str(DATASET),'source_csv_sha256':csv_hashes,
    'method':{'source_identity':'training source relative path + one-based CSV row; masks must match exactly','inverse_transform':'Runtime float32 box -> float64 for inverse scaling [cropW/480,cropH/192] and offset [left,top]; compare against original CSV float64 pixel bounds','matching_max_linf_tolerance_original_px':MATCH_TOLERANCE_PX,'maximum_observed_inverse_error_original_px':max_inverse_error,'complete_rule':'Exact float32 intersects and x0>=0,y0>=0,x1<=480,y1<=192, combined with manifest originally_complete when present. Matching tolerance is NEVER used for completeness.','negative_records':'Empty-annotation training records are excluded from positive-cell denominator.','runtime_note':'Current word_train.Patches recomputes initially complete from float32 bounds; manifest flags are additionally honored here. Separate disagreement count detects any effect.'},
    'patch_records':len(records),'negative_empty_annotation_records':negative_records,'training_source_pages':len(sources),'all_patch_cell_occurrences':all_occurrences,'complete_patch_occurrences':complete_occurrences,'runtime_float32_complete_occurrences':runtime_complete_occurrences,
    'overall':ratio_count(len(original),len(complete_counts),complete_occurrences),
    'partial_only_unique_cells':sum(key in any_counts and key not in complete_counts for key in original),'never_sampled_unique_cells':sum(key not in any_counts for key in original),'repeat_complete_occurrences_beyond_unique':complete_occurrences-len(complete_counts),
    'matching_errors':{'unmatched_count':len(unmatched),'ambiguous_count':len(ambiguous),'metadata_row_disagreements':len(metadata_disagreements),'positive_nontraining_source_count':len(source_scope_errors),'manifest_flag_vs_runtime_complete_disagreements':len(flag_disagreements)},
    'float32_vs_float64_complete_differences':float32_vs_float64_complete_differences,'by_family':family_stats,'by_mask':mask_stats,'by_source':source_stats,
    'outputs':{'occurrences':'occurrence-matches.csv','covered_unique':'covered-source-cells.csv','uncovered_unique':'uncovered-source-cells.csv','uncovered_json':'uncovered-source-cells.json'},
}
(HERE/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({key:summary[key] for key in ['patch_records','negative_empty_annotation_records','training_source_pages','all_patch_cell_occurrences','complete_patch_occurrences','overall','partial_only_unique_cells','never_sampled_unique_cells','repeat_complete_occurrences_beyond_unique','matching_errors','method','by_family']},indent=2))
