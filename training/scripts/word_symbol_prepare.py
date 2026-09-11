"""Freeze the old reader and make source-disjoint, explicitly recomposed symbol data.

Photo patches supervise six-dot shape only. Indonesian capitalization/number
semantics come from the specified composition, never from Russian/math text.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
from urllib.parse import quote
from urllib.request import urlopen
from PIL import Image, ImageStat, ImageOps
from word_data import WIDTH, HEIGHT, LETTER_BY_MASK


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def restore_dsbi(repo):
    manifest = json.loads((repo/'training/evidence/braille-word-reader-v1/dsbi-training-preparation/download-manifest.json').read_text())
    root = Path(manifest['sourceDirectory'])
    def fetch(row):
        target = root/row['file']
        if not target.exists() or digest(target) != row['sha256']:
            url = 'https://raw.githubusercontent.com/yeluo1994/DSBI/'+manifest['revision']+'/'+quote(row['file'])
            with urlopen(url, timeout=90) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != row['sha256']:
                raise ValueError('Pinned DSBI hash mismatch: '+row['file'])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return dict(file=row['file'], sha256=digest(target))
    with ThreadPoolExecutor(max_workers=8) as executor:
        verified = list(executor.map(fetch, manifest['sourceFiles']))
    return dict(root=str(root), revision=manifest['revision'], files=verified, test_pixels_downloaded=False)


def cell_pool(data, records):
    pool = defaultdict(list)
    # Deduplicate source-space boxes because the old corpus contains overlapping crops.
    seen = set()
    for row in records:
        for index, (box, mask) in enumerate(zip(row['boxes'], row['masks'])):
            x0,y0,x1,y1 = box
            if not (1 <= x0 < x1 <= WIDTH-1 and 1 <= y0 < y1 <= HEIGHT-1):
                continue
            if x1-x0 < 8 or y1-y0 < 16:
                continue
            key = (row['source'], tuple(row.get('crop', [])), tuple(round(v,2) for v in box))
            if key in seen:
                continue
            seen.add(key)
            pool[int(mask)].append((row,index))
    return pool


def complete_synthetic_pool(pool, masks):
    """Fill absent patterns only using explicit mask-preserving image transforms.

    Reflection/half-turn are synthetic six-dot geometry, not natural photo truth.
    The donor stays inside the previously assigned source split.
    """
    original=dict(pool)
    transforms=[('horizontal',[3,4,5,0,1,2]),('half_turn',[5,4,3,2,1,0]),('vertical',[2,1,0,5,4,3])]
    for missing in sorted(set(masks)-set(pool)):
        for transform,mapping in transforms:
            donors=[mask for mask in original if sum(((mask>>old)&1)<<new for old,new in enumerate(mapping))==missing]
            if donors:
                for donor in donors:
                    for row,index in original[donor]:
                        copied=dict(row,synthetic_transform=transform,original_mask=donor)
                        pool[missing].append((copied,index))
                break
    return pool


def make_case(data, pool, masks, rng, path):
    selected = [rng.choice(pool[mask]) for mask in masks]
    # Consistent cell size preserves spacing; patches remain public-photo-derived.
    cell_h = rng.randint(72,112)
    cell_w = min(int(cell_h*.56), int((WIDTH-64)/(len(masks)*1.32)))
    cell_h = min(cell_h, int(cell_w/0.43))
    gap = max(4,int(cell_w*.30))
    total = len(masks)*cell_w+(len(masks)-1)*gap
    left = (WIDTH-total)//2
    top = (HEIGHT-cell_h)//2 + rng.randint(-8,8)
    crops = []
    provenance = []
    for row,index in selected:
        box = row['boxes'][index]
        with Image.open(data/row['image']) as image:
            crop = image.convert('RGB').crop(tuple(round(v) for v in box))
        transform=row.get('synthetic_transform')
        if transform=='horizontal':
            crop=ImageOps.mirror(crop)
        elif transform=='vertical':
            crop=ImageOps.flip(crop)
        elif transform=='half_turn':
            crop=crop.transpose(Image.Transpose.ROTATE_180)
        crops.append(crop.resize((cell_w,cell_h),Image.Resampling.BILINEAR))
        provenance.append(dict(source=row['source'],image=row['image'],box=box,original_mask=row['masks'][index],synthetic_transform=transform))
    background = tuple(round(v) for v in ImageStat.Stat(crops[0]).median)
    image = Image.new('RGB',(WIDTH,HEIGHT),background)
    boxes = []
    for index,crop in enumerate(crops):
        x=left+index*(cell_w+gap)
        image.paste(crop,(x,top))
        boxes.append([x,top,x+cell_w,top+cell_h])
    path.parent.mkdir(parents=True,exist_ok=True)
    image.save(path)
    return boxes,provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo=args.repo.resolve(); output=args.output.resolve()
    if output.exists() and (output/'corpus-audit.json').exists():
        raise SystemExit('Use a new symbol experiment directory; do not overwrite frozen evidence')
    output.mkdir(parents=True,exist_ok=True)
    paths=[repo/'training/checkpoints/braille-word-reader-v1',repo/'training/evidence/braille-word-reader-v1',repo/'assets/models/braille-word-reader-v1']
    old_files={str(path.relative_to(repo)):digest(path) for root in paths for path in root.rglob('*') if path.is_file()}
    if not (output/'baseline-freeze.json').exists():
        write_json(output/'baseline-freeze.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),files=old_files))
        shutil.copy2(repo/'training/checkpoints/braille-word-reader-v1/best-word.keras',output/'baseline.keras')
    write_json(output/'dsbi-restoration.json',restore_dsbi(repo))
    data=repo/'.braille-training-work/word-scan-v1-20260905/data'
    split=json.loads((data/'source-split.json').read_text())
    train=json.loads((data/'../combined-training-v5/combined-training-manifest.json').read_text())
    validation=json.loads((data/'validation.json').read_text())
    train_sources={row['source'] for row in train}; val_sources={row['source'] for row in validation}
    external_sources=set(split['images']['external'])
    if train_sources & val_sources or (train_sources|val_sources)&external_sources:
        raise ValueError('Source split overlap before recomposition')
    pools={name:cell_pool(data,rows) for name,rows in [('train',train),('validation',validation)]}
    letters={letter.lower():mask for mask,letter in LETTER_BY_MASK.items()}
    for name,pool in pools.items():
        complete_synthetic_pool(pool,set(letters.values())|{32,60})
        missing=(set(letters.values())|{32,60})-set(pool)
        if missing:
            raise ValueError(f'{name} lacks masks {sorted(missing)}; never invent public source labels')
    rng=random.Random(20260908)
    manifests={}; composition_counts={}
    for split_name,repeat in [('train',24),('validation',4)]:
        rows=[]
        # Each group has A-Z coverage or all digits plus leading zero/long examples.
        words=['abc','braille','baca','huruf','angka','uji','aab','zz','abcdefghijkl']+list(letters)
        numbers=list('0123456789')+['2026','001','123','9876543210','000','012345678901']
        for group,items in [('lowercase',words),('capital_single',words),('capital_word',words),('numbers',numbers)]:
            for repetition in range(repeat):
                for value in items:
                    if group=='numbers':
                        masks=[60]+[letters['abcdefghij'[int(c)-1]] if c!='0' else letters['j'] for c in value]
                        expected=value
                    else:
                        masks=[letters[c] for c in value]
                        prefix=[] if group=='lowercase' else [32] if group=='capital_single' else [32,32]
                        masks=prefix+masks
                        expected=value if group=='lowercase' else value[0].upper()+value[1:] if group=='capital_single' else value.upper()
                    index=len(rows); path=output/'data'/split_name/f'{index:06d}.png'
                    boxes,provenance=make_case(data,pools[split_name],masks,rng,path)
                    rows.append(dict(image=str(path),source=f'synthetic-id/{split_name}/{index:06d}',source_family=f'synthetic-id/{group}',boxes=boxes,masks=masks,expected=expected,symbol_group=group,origin='synthetic_recomposed_public_cell_photographs',source_cells=provenance))
        # Mandatory parser-invalid sequences still supervise every real physical cell.
        for masks in [[32],[60],[32,32],[1,60,1],[60,1,32,3],[60,5],[1,32,32,3],[32,60,1]]:
            for repetition in range(repeat):
                index=len(rows);path=output/'data'/split_name/f'{index:06d}.png'
                boxes,provenance=make_case(data,pools[split_name],masks,rng,path)
                rows.append(dict(image=str(path),source=f'synthetic-id/{split_name}/{index:06d}',source_family='synthetic-id/rejection',boxes=boxes,masks=masks,expected='',symbol_group='rejection',origin='synthetic_recomposed_public_cell_photographs',source_cells=provenance))
        manifests[split_name]=rows
        composition_counts[split_name]=dict(Counter(row['symbol_group'] for row in rows))
        write_json(output/f'{split_name}-symbols.json',dict(scope='synthetic_recomposed_public_cell_photographs_not_natural_indonesian_camera_truth',records=rows))
    write_json(output/'corpus-audit.json',dict(status='passed',seed=20260908,dataset_revisions=dict(Angelina=split['dataset_revision'],DSBI='6937eb8973efe5ff337e5c018485cb626303f5c2'),source_split_sha256=digest(data/'source-split.json'),split_before_crops=True,source_overlap=[],train_sources=sorted(train_sources),validation_sources=sorted(val_sources),external_sources_excluded=sorted(external_sources),cell_pool_mask_counts={k:{str(mask):len(v) for mask,v in pool.items()} for k,pool in pools.items()},composition_counts=composition_counts,indicator_label_audit='Mask32/60 public source cells are used as six-dot shape patches only. Their original Russian/math semantic roles are not asserted. Indonesian semantics are explicit in newly composed synthetic sequences.',physical_camera_validated=False))
    print(json.dumps(composition_counts),flush=True)


if __name__=='__main__':
    main()
