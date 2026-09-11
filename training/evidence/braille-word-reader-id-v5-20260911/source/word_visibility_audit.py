"""Compare augmentation labels before/after the visibility fix on 262 fixed training inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training/scripts'))
import word_data


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--previous-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('word_data_before_visibility_fix',args.previous_source)
    previous=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    manifest=args.data/'train-coverage-v2.json'
    records=json.loads(manifest.read_text(encoding='utf-8'))
    indices=list(range(0,6283,24))
    assert len(indices)==262 and max(indices)<len(records)
    rows=[]
    for index in indices:
        record=records[index]
        with Image.open(args.data/record['image']) as source:
            image=source.convert('RGB')
        boxes=np.asarray(record['boxes'],np.float32).reshape(-1,4)
        masks=np.asarray(record['masks'],int)
        complete=(boxes[:,0]>=0)&(boxes[:,1]>=0)&(boxes[:,2]<=word_data.WIDTH)&(boxes[:,3]<=word_data.HEIGHT)
        old_rng=np.random.default_rng(20260905+index)
        new_rng=np.random.default_rng(20260905+index)
        half_turn=old_rng.random()<.5
        assert (new_rng.random()<.5)==half_turn
        old_image,old_boxes,old_masks=image,boxes.copy(),masks.copy()
        new_image,new_boxes,new_masks=image,boxes.copy(),masks.copy()
        if half_turn:
            old_image,old_boxes,old_masks=previous.rotate_patch_180(old_image,old_boxes,old_masks)
            new_image,new_boxes,new_masks=word_data.rotate_patch_180(new_image,new_boxes,new_masks)
        old_pixels,old_boxes=previous.augment_patch(old_image,old_boxes,old_rng,illumination_augmentation=True)
        new_pixels,new_boxes,eligible=word_data.augment_patch(new_image,new_boxes,new_rng,illumination_augmentation=True,originally_complete=complete)
        pixels_match=np.array_equal(old_pixels,new_pixels)
        boxes_match=np.array_equal(old_boxes,new_boxes)
        masks_match=np.array_equal(old_masks,new_masks)
        random_state_match=old_rng.bit_generator.state==new_rng.bit_generator.state
        assert pixels_match and boxes_match and masks_match and random_state_match, record['image']
        assert not np.any(eligible & ~complete)
        old_final=complete & (old_boxes[:,0]>=0)&(old_boxes[:,1]>=0)&(old_boxes[:,2]<=word_data.WIDTH)&(old_boxes[:,3]<=word_data.HEIGHT)
        newly_partial=np.flatnonzero(old_final & ~eligible)
        old_target=previous.make_target(old_boxes,old_masks,complete)
        new_target=word_data.make_target(new_boxes,new_masks,eligible)
        old_count=int((old_target[...,0]==1).sum())
        new_count=int((new_target[...,0]==1).sum())
        assert old_count-new_count==len(newly_partial), record['image']
        rows.append({'index':index,'image':record['image'],'source':record['source'],'half_turn':bool(half_turn),
                     'original_complete_cells':int(complete.sum()),'old_positive_centers':old_count,'new_positive_centers':new_count,
                     'newly_partial_indices':newly_partial.tolist(),'newly_partial_final_boxes':new_boxes[newly_partial].tolist(),
                     'newly_partial_masks':new_masks[newly_partial].tolist(),
                     'pixels_match':bool(pixels_match),'boxes_match':bool(boxes_match),'masks_match':bool(masks_match),'random_state_match':random_state_match})
    assert not any(name=='torch' or name=='tensorflow' or name.startswith('tensorflow.') for name in sys.modules)
    affected=sum(bool(row['newly_partial_indices']) for row in rows)
    removed=sum(len(row['newly_partial_indices']) for row in rows)
    old_count=sum(row['old_positive_centers'] for row in rows)
    report={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Fixed 262 training-input augmentation comparison only; no external images, model, GPU, or accuracy inference.',
            'selection':{'indices':'range(0,6283,24)','rng_seed_per_image':'20260905 + index','half_turn':'consume first rng.random(), rotate if <.5','augmentation':'illumination_augmentation=True'},
            'training_manifest_sha256':digest(manifest),'previous_source':str(args.previous_source),'previous_source_sha256':digest(args.previous_source),
            'current_source_sha256':digest(ROOT/'training/scripts/word_data.py'),'harness_sha256':digest(__file__),
            'numpy_version':np.__version__,'pillow_version':PIL.__version__,'samples':len(rows),'affected_samples':affected,
            'affected_sample_fraction':affected/len(rows),'old_positive_centers':old_count,'new_positive_centers':sum(row['new_positive_centers'] for row in rows),
            'removed_incorrect_complete_labels':removed,'fraction_of_old_positive_centers':removed/old_count if old_count else 0.,
            'pixels_boxes_masks_rng_match_all':True,'monotonic_eligibility_passed':True,'torch_imported':False,'tensorflow_imported':False,
            'limitations':['Counts cover these fixed augmented training inputs, not every epoch or external recognition errors.',
                           'Unchanged pixels and boxes isolate the effect to supervision eligibility; this does not measure model accuracy after retraining.'],
            'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='rows'},indent=2))


if __name__=='__main__':
    main()
