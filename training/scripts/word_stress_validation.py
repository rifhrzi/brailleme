"""Freeze camera-stress validation from held validation pixels before inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from word_data import WIDTH, HEIGHT, affine_patch, save_json

VARIANTS = (
    {'name':'soft_narrow', 'horizontal_scale':.75, 'contrast':.55, 'blur':1.2, 'brightness':1.},
    {'name':'faint', 'horizontal_scale':.9, 'contrast':.35, 'blur':.65, 'brightness':.9},
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args=parser.parse_args()
    target=args.data/'validation-stress-selection.json'
    if target.exists():
        raise SystemExit('Preserve the frozen stress manifest; do not overwrite it')
    source=args.data/'validation-words.json'
    natural=json.loads(source.read_text())['records']
    records=[dict(row, validation_group='natural') for row in natural]
    folder=args.data/'validation-stress-words'
    folder.mkdir(exist_ok=True)
    for index,row in enumerate(natural):
        for variant in VARIANTS:
            image=Image.open(args.data/row['image']).convert('RGB')
            selection=np.asarray(row['selection'])*[WIDTH,HEIGHT,WIDTH,HEIGHT]
            aspect=(selection[2]-selection[0])/(selection[3]-selection[1])
            # Actual guide supports aspect >=1. Preserve this bound without
            # expanding a selection into an adjacent annotated word.
            horizontal=min(1.,max(variant['horizontal_scale'],1/aspect))
            boxes=np.asarray(row['boxes'])
            context=np.asarray(row['context_boxes'])
            count=len(boxes)
            all_boxes=np.concatenate((boxes,context,selection[None]))
            image, transformed=affine_patch(image,all_boxes,horizontal,1.)
            chosen=transformed[-1]
            width=max(chosen[2]-chosen[0],(chosen[3]-chosen[1])*WIDTH/HEIGHT)
            height=width*HEIGHT/WIDTH
            cx,cy=(chosen[:2]+chosen[2:])/2
            crop=np.asarray([max(0,int(np.floor(cx-width/2))),max(0,int(np.floor(cy-height/2))),min(WIDTH,int(np.ceil(cx+width/2))),min(HEIGHT,int(np.ceil(cy+height/2)))])
            scale=np.asarray([WIDTH/(crop[2]-crop[0]),HEIGHT/(crop[3]-crop[1])]*2)
            transformed=(transformed-crop[:2].tolist()*2)*scale
            image=image.crop(tuple(crop)).resize((WIDTH,HEIGHT),Image.Resampling.BILINEAR)
            image=ImageEnhance.Brightness(image).enhance(variant['brightness'])
            image=ImageEnhance.Contrast(image).enhance(variant['contrast'])
            image=image.filter(ImageFilter.GaussianBlur(variant['blur']))
            new_selection=(transformed[-1]/[WIDTH,HEIGHT,WIDTH,HEIGHT]).tolist()
            if min(new_selection)>=-1e-12 and max(new_selection)<=1+1e-12:
                new_selection=np.clip(new_selection,0,1).tolist()
            if not (0<=new_selection[0]<new_selection[2]<=1 and 0<=new_selection[1]<new_selection[3]<=1):
                raise ValueError('Transformed selection leaves its input')
            if not np.all((transformed[:count,:2]>=transformed[-1,:2]) & (transformed[:count,2:]<=transformed[-1,2:])):
                raise ValueError('Stress transformation cut a known complete word')
            name=f'{index:06d}-{variant["name"]}.png'
            image.save(folder/name,compress_level=1)
            records.append(dict(row,image='validation-stress-words/'+name,selection=new_selection,
                                boxes=transformed[:count].tolist(),context_boxes=transformed[count:-1].tolist(),
                                validation_group='stress', natural_image=row['image'],
                                stress_transform=dict(variant,actual_horizontal_scale=horizontal,recontext_crop=crop.tolist()),
                                source_crop_is_pre_transform=True))
    protocol={'scope':'Held-validation-only deterministic stress; no external source or model prediction used.',
              'natural_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'variants':VARIANTS,'natural_count':len(natural),'stress_count':len(records)-len(natural),
              'selection_rule':'Require accepted accuracy >=95% separately on natural and stress groups, then maximize equal-group mean whole-word exactness, with all quality/decoder rejections counted as failures.',
              'records':records}
    save_json(target,protocol)
    print(json.dumps({'file':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'natural':len(natural),'stress':len(records)-len(natural)},indent=2))


if __name__=='__main__':
    main()
