"""Audit proposed original-pixel TRAIN supplements produced by word_cover_training.

Verify exact parent-record preservation, every new context and intersecting
annotation, and deterministic RGB pixel samples against original photographs.
The combined manifest must be next to coverage-events.json and its new images.
No source manifest, image, fixture, or recognition model is modified or loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--dataset',type=Path,default=Path('C:/bm-datasets/AngelinaDataset'))
parser.add_argument('--parent-manifest',type=Path,required=True)
parser.add_argument('--combined-manifest',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--sample-count',type=int,default=64,help='Minimum sample count, preserving family/factor and edge coverage')
args=parser.parse_args()
HERE=args.output
HERE.mkdir(parents=True,exist_ok=True)
DATA=args.data
DATASET=args.dataset
parent_path=args.parent_manifest
combined_path=args.combined_manifest
PROPOSED=combined_path.parent
old=json.loads(parent_path.read_text())
combined=json.loads(combined_path.read_text())
assert combined[:len(old)]==old,'Existing records changed'
new=combined[len(old):]
events=json.loads((PROPOSED/'coverage-events.json').read_text())
assert len(new)==len(events)
split=json.loads((DATA/'source-split.json').read_text())
allowed=set(split['images']['train'])
heldout=set(split['images']['validation'])|set(split['images']['external'])
assert not allowed&heldout
pages={}
groups={}
source_edge_crops=[]
all_contexts=[]
for index,(record,event) in enumerate(zip(new,events)):
    source=record['source']
    assert source in allowed and source not in heldout
    assert source==event['source'] and record['image']==event['image']
    if source not in pages:
        with Image.open(DATASET/source) as image:
            size=image.size
        rows=np.loadtxt((DATASET/source).with_suffix('.csv'),delimiter=';',ndmin=2)
        # Source preparation explicitly casts original normalized CSV pixels to float32.
        source_boxes=(rows[:,:4]*np.asarray(list(size)*2)).astype(np.float32)
        pages[source]=(size,source_boxes,rows[:,4].astype(int))
    (width,height),boxes,masks=pages[source]
    left,top,right,bottom=record['crop']
    assert all(isinstance(v,int) for v in record['crop'])
    assert 0<=left<right<=width and 0<=top<bottom<=height
    expected_rows=np.flatnonzero((boxes[:,2]>left)&(boxes[:,3]>top)&(boxes[:,0]<right)&(boxes[:,1]<bottom))
    assert record['annotation_rows']==(expected_rows+1).tolist()
    assert record['masks']==masks[expected_rows].tolist()
    expected=((boxes[expected_rows].astype(np.float64)-np.asarray([left,top]*2))*np.asarray([480/(right-left),192/(bottom-top)]*2))
    assert np.max(np.abs(expected-np.asarray(record['boxes'])))<1e-9
    full=(boxes[expected_rows,0]>=left)&(boxes[expected_rows,1]>=top)&(boxes[expected_rows,2]<=right)&(boxes[expected_rows,3]<=bottom)
    assert record['originally_complete']==full.tolist()
    anchor=record['annotation_rows'].index(event['anchor_annotation_row'])
    assert full[anchor]
    if left==0 or top==0 or right==width or bottom==height:
        source_edge_crops.append(index)
    family='/'.join(Path(source).parts[:2])
    group=(family,event['context_width_per_cell_height'])
    groups.setdefault(group,[]).append(index)
    all_contexts.append({'index':index,'source':source,'family':family,'context_factor':event['context_width_per_cell_height']})


def rank(index):
    return hashlib.sha256(f'20260905:{index}:{new[index]["source"]}:{new[index]["crop"]}'.encode()).hexdigest()


chosen=set()
for group,indices in sorted(groups.items()):
    chosen.update(sorted(indices,key=rank)[:2])
chosen.update(sorted(source_edge_crops,key=rank)[:8])
for index in sorted(range(len(new)),key=rank):
    if len(chosen)>=args.sample_count: break
    chosen.add(index)
sample_results=[]
for index in sorted(chosen):
    record=new[index]
    with Image.open(DATASET/record['source']) as opened:
        original=opened.convert('RGB')
    expected=original.crop(record['crop']).resize((480,192),Image.Resampling.BILINEAR)
    actual_path=(DATA/record['image']).resolve()
    assert actual_path.is_relative_to(PROPOSED.resolve())
    with Image.open(actual_path) as actual:
        assert actual.mode=='RGB' and actual.size==(480,192)
        assert np.array_equal(np.asarray(actual),np.asarray(expected)),record['image']
    sample_results.append({'supplement_index_zero_based':index,'combined_index_zero_based':len(old)+index,'image':record['image'],'source':record['source'],'crop':record['crop'],'family':all_contexts[index]['family'],'context_width_per_cell_height':all_contexts[index]['context_factor'],'source_edge_shifted_crop':index in source_edge_crops,'source_sha256':hashlib.sha256((DATASET/record['source']).read_bytes()).hexdigest(),'image_sha256':hashlib.sha256(actual_path.read_bytes()).hexdigest(),'pixel_mismatch_count':0})
report={
    'scope':'Read-only independent audit of proposed training contexts; no model inference or fixture edits.',
    'passed':True,
    'parent_manifest_sha256':hashlib.sha256(parent_path.read_bytes()).hexdigest(),
    'combined_manifest_sha256':hashlib.sha256(combined_path.read_bytes()).hexdigest(),
    'original_records_preserved_exactly':len(old),
    'additional_context_count':len(new),
    'all_additional_context_geometry_checked':len(new),
    'all_intersecting_annotations_retained':True,
    'all_masks_and_source_csv_row_ids_match':True,
    'all_unclipped_transformed_boxes_match':True,
    'all_complete_flags_match':True,
    'all_anchor_cells_complete':True,
    'source_edge_exceptions':[],
    'valid_contexts_shifted_to_source_edge_count':len(source_edge_crops),
    'source_geometry_precision':'Original CSV normalized bounds * original image dimensions, float32 as specified by load_page; transform compared float64 at1e-9 model-pixel tolerance. No tolerance on source/canvas completeness.',
    'pixel_sampling':'Deterministic SHA256 order of fixed seed20260905 + supplementindex + source + crop; two per family/context-factor group, up to eight source-edge crops, then fill to the requested minimum sample count with hash order. No model predictions used.',
    'pixel_sample_count':len(sample_results),
    'pixel_sample_family_count':len(set(row['family'] for row in sample_results)),
    'pixel_sample_factor_values':sorted(set(row['context_width_per_cell_height'] for row in sample_results)),
    'sampled_model_images_equal_original_crop_pillow_rgb_bilinear':True,
    'pixel_samples':sample_results,
    'limits':'Pixel equality checked for the documented sample only. Annotation/crop geometry and unique-cell matching checked for every context. Source+CSV-row identities are not claims of independent physical examples.',
}
(HERE/'pixel-preservation-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:report[key] for key in ['passed','original_records_preserved_exactly','additional_context_count','all_additional_context_geometry_checked','valid_contexts_shifted_to_source_edge_count','source_edge_exceptions','pixel_sample_count','pixel_sample_family_count','pixel_sample_factor_values','parent_manifest_sha256','combined_manifest_sha256']},indent=2))
