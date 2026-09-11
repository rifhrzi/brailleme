"""Compose new labelled feeds using a frozen pixel mapping; no model inference."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'training/scripts'))
from word_symbols import parse_word_masks, DECODER_VERSION

MAPPING = Path(__file__).with_name('feed-mapping-v1.json')
OLD_MANIFEST = Path('C:/bm-symbols-apk-20260908/evidence/labelled-feeds-v1/manifest.json')
ASSET_MANIFEST = ROOT/'assets/models/braille-word-reader-v1/manifest.json'
DESTINATION = Path('C:/bm-symbols-apk-20260908/evidence/calibrated-feeds-v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    mapping = json.loads(MAPPING.read_text())
    original = json.loads(OLD_MANIFEST.read_text())
    native = json.loads(ASSET_MANIFEST.read_text())['nativeSelfTest']
    if DESTINATION.exists():
        raise FileExistsError('Preserve prior feed evidence; destination must be new')
    center = mapping['estimated_visible_feed_center']
    polygon = mapping['observed_visible_feed_polygon']
    visible = [max(0.,polygon[0][0],polygon[3][0]), max(0.,polygon[0][1],polygon[1][1]),
               min(1024.,polygon[1][0],polygon[2][0]), min(1024.,polygon[2][1],polygon[3][1])]
    margin = 32
    allowed = [visible[2]-visible[0]-2*margin, visible[3]-visible[1]-2*margin]
    matrix = mapping['feed_to_screenshot']['matrix']
    translation = mapping['feed_to_screenshot']['translation']
    preserved = {MAPPING:sha(MAPPING), OLD_MANIFEST:sha(OLD_MANIFEST), ASSET_MANIFEST:sha(ASSET_MANIFEST)}
    # Freeze every original feed, including those not chosen for a camera action.
    for old in original['rows']:
        path=OLD_MANIFEST.parent/old['output']
        if sha(path)!=old['output_sha256']:
            raise ValueError('Original labelled feed changed: '+str(path))
        preserved[path]=sha(path)

    cases=[]
    for index,old in enumerate(original['rows']):
        source=Path(old['source']['image']) if old.get('source') else OLD_MANIFEST.parent/old['output']
        expected_hash=old['source_sha256'] if old.get('source') else old['output_sha256']
        if sha(source)!=expected_hash:
            raise ValueError('Labelled source hash differs: '+str(source))
        preserved[source]=expected_hash
        symbols=[]
        if old.get('source'):
            parsed=parse_word_masks(old['source']['masks'])
            if parsed['status']!='success' or parsed['text']!=old['expected']:
                raise ValueError('Existing label annotation disagrees with the semantic grammar')
            symbols=parsed['symbols']
        cases.append(dict(output=old['output'],source=source,source_sha256=expected_hash,
            expected_text=old['expected'],classification='synthetic_recomposed_label' if old.get('source') else 'uniform_blank',
            background_rgb=[128,128,128],symbols=symbols,
            symbol_roles_origin='Deterministic grammar on preserved annotated masks, not image inference' if symbols else 'No annotated cells',
            original_manifest_row=index,original_annotation=old.get('source'),
            expected_runtime=old.get('expected_runtime')))
    source=ROOT/'assets/models/braille-word-reader-v1'/native['file']
    if sha(source)!=native['sha256']:
        raise ValueError('Startup source hash changed')
    preserved[source]=sha(source)
    cases.append(dict(output='startup-multirow-context.png',source=source,source_sha256=sha(source),
        expected_text=None,classification='multirow_startup_context',background_rgb=[83,83,83],
        symbols=None,symbol_roles_origin='No whole-scene cell annotation asserted',
        roi_expected_text=native['expectedText'],roi_source_normalized=native['roi'],
        roi_annotation_origin='Existing nativeSelfTest ROI, not a label for the whole 450x257 image',
        is_natural_whole_word_benchmark=False))

    DESTINATION.mkdir(exist_ok=False)
    rows=[]
    for case in cases:
        with Image.open(case['source']) as image:
            source=image.convert('RGB')
        width,height=source.size
        scale=min(1.,allowed[0]/width,allowed[1]/height)
        scaled_size=[max(1,int(math.floor(width*scale))),max(1,int(math.floor(height*scale)))]
        scaled=source if scaled_size==[width,height] else source.resize(scaled_size,Image.Resampling.BILINEAR)
        left=int(math.floor(center[0]-scaled.width/2))
        top=int(math.floor(center[1]-scaled.height/2))
        placement=[left,top,left+scaled.width,top+scaled.height]
        if not (0<=left<placement[2]<=1024 and 0<=top<placement[3]<=1024):
            raise ValueError('Composition would crop source pixels')
        canvas=Image.new('RGB',(1024,1024),tuple(case['background_rgb']))
        canvas.paste(scaled,(left,top))
        if canvas.crop(tuple(placement)).tobytes()!=scaled.tobytes():
            raise ValueError('Composed source pixels differ')
        corners=[[left,top],[placement[2],top],[placement[2],placement[3]],[left,placement[3]]]
        projected=[[matrix[0][0]*x+matrix[0][1]*y+translation[0],
                    matrix[1][0]*x+matrix[1][1]*y+translation[1]] for x,y in corners]
        preview=mapping['preview_rectangle_xywh']
        if not all(preview[0]<=x<=preview[0]+preview[2] and preview[1]<=y<=preview[1]+preview[3] for x,y in projected):
            raise ValueError('Frozen mapping predicts a clipped source')
        path=DESTINATION/case['output']
        canvas.save(path)
        rows.append({**case,'source':str(case['source']), 'source_size':[width,height],
            'output_sha256':sha(path),'output_size':[1024,1024],
            'scale':scale,'realized_scale_xy':[scaled.width/width,scaled.height/height],
            'scaled_size':scaled_size,'placement':placement,'source_pixels_cropped':False,
            'unscaled_source_pixels_exact':scale==1.,'interpolation':'none' if scale==1. else 'pillow_bilinear',
            'projected_screenshot_corners':projected,'prediction_observed_for_selection':False})
    if rows[-1]['source_size']!=[450,257] or rows[-1]['placement']!=[62,383,512,640] or rows[-1]['scale']!=1:
        raise ValueError('Full startup source does not match the frozen geometry proposal')
    for path,expected in preserved.items():
        if sha(path)!=expected:
            raise ValueError('Existing input changed during preparation: '+str(path))
    report=dict(schema_version=1,created_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='New static emulator geometry calibration; no runtime/model acceptance claim',
        mapping_report=str(MAPPING),mapping_report_sha256=sha(MAPPING),
        original_label_manifest=str(OLD_MANIFEST),original_label_manifest_sha256=sha(OLD_MANIFEST),
        native_asset_manifest_sha256=sha(ASSET_MANIFEST),
        generator_sha256=sha(Path(__file__)),semantic_parser_sha256=sha(ROOT/'training/scripts/word_symbols.py'),
        decoder_version=DECODER_VERSION,source_selection='Exactly all seven preselected label records, existing blank, and the complete existing startup source; no model predictions consulted',
        frozen_rule=dict(canvas=[1024,1024],feed_visible_rectangle=visible,
            feed_visible_center=center,feed_pixel_margin_each_side=margin,
            maximum_source_composition_size=allowed,
            scale='min(1, available_width/source_width, available_height/source_height)',
            scaled_dimensions='floor(original dimensions times scale)',
            placement='floor(inverse-mapped preview center minus half scaled dimensions)',
            backgrounds=dict(labels_and_blank=[128,128,128],startup=[83,83,83])),
        actual_preview_verified=False,emulator_modified=False,launcher_modified=False,
        original_files_verified_unchanged=len(preserved),rows=rows,
        limitations=['The mapping is fitted from one previous screenshot and must be visually rechecked after relaunch/layout changes.',
                     'Full startup image contains multiple rows; rawenstwa applies only to its recorded ROI.',
                     'Only geometry and label provenance are asserted; no predictions, camera pass, or physical accuracy are asserted.'])
    with (DESTINATION/'manifest.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(DESTINATION),cases=len(rows),manifest_sha256=sha(DESTINATION/'manifest.json'),
        rows=[{key:row[key] for key in ['output','classification','source_size','scale','placement']} for row in rows]),indent=2))


if __name__=='__main__':
    main()
