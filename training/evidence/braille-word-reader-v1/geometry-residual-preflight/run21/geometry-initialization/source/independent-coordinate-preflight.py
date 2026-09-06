"""Independent raw-TXT and saved-graph coordinate check; no ML imports/inference."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from PIL import Image

WORK = Path(__file__).resolve().parent
def read(path):
    return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

feature = read(WORK/'feature-lattice-audit.json')
slot = read(WORK/'slot-prior-audit.json')
checkpoint = WORK/'selected-v13/best-word.keras'
assert sha(checkpoint)==feature['checkpoint_sha256']
with zipfile.ZipFile(checkpoint) as archive:
    config_bytes=archive.read('config.json')
    config=json.loads(config_bytes)
assert hashlib.sha256(config_bytes).hexdigest()==feature['saved_config_sha256']
backbone=next(layer for layer in config['config']['layers'] if layer['name']=='mobilenetv2_half')
layers={layer['name']:layer for layer in backbone['config']['layers']}
conv=layers['Conv1']['config'];pad=layers['block_1_pad']['config'];depth=layers['block_1_depthwise']['config']
assert conv['kernel_size']==[3,3] and conv['strides']==[2,2] and conv['padding']=='same'
assert pad['padding']==[[0,1],[0,1]]
assert depth['kernel_size']==[3,3] and depth['strides']==[2,2] and depth['padding']=='valid'
for name in ('expanded_conv_depthwise','block_2_depthwise'):
    layer=layers[name]['config']
    assert layer['strides']==[1,1] and layer['padding']=='same' and layer['kernel_size']==[3,3]

# Independently propagate sample indices with center-only 1-D three-tap kernels.
# Conv1 SAME on even input adds the single extra zero at the end, followed by
# the explicit zero-at-end block1 padding. Other fine-branch kernels are odd,
# stride1 SAME or1x1 and leave the nominal center lattice unchanged.
ramps=[]
for length in (192,480):
    first=np.pad(np.arange(length,dtype=float),(0,1))[1:-1:2]
    fourth=np.pad(first,(0,1))[1:-1:2]
    np.testing.assert_array_equal(fourth,3+4*np.arange(length//4))
    ramps.append({'input_size':length,'output_size':len(fourth),'first':float(fourth[0]),'last':float(fourth[-1]),'jump':4})
assert feature['backbone_outputs']['block_3_expand_relu']=={'size':[48,120],'jump':[4,4],'origin':[3.,3.],'conv_support':[19,19]}

index_path=WORK/'dsbi-training-contexts-v2/source-index.json'
manifest_path=WORK/'dsbi-training-contexts-v2/train.json'
point_path=WORK/'dsbi-point-metadata-v1/points.json'
assert sha(manifest_path)==slot['manifest_sha256'] and sha(point_path)==slot['point_metadata_sha256']
index=read(index_path);dataset=Path(index['root']);metadata=read(point_path)['records'];records=read(manifest_path)
truth={};raw_hashes=[];source_lookup={}
for source in index['sources']:
    source_lookup[source['source']]=source
    annotation=dataset/source['original_annotation']
    assert sha(annotation)==source['annotation_sha256'] and source['split']=='train'
    raw_hashes.append({'source':source['source'],'sha256':sha(annotation)})
    lines=annotation.read_text(encoding='utf-8-sig').splitlines()
    if not lines:continue
    angle=float(lines[0]);xs=list(map(float,lines[1].split()));ys=list(map(float,lines[2].split()))
    assert np.isfinite([angle,*xs,*ys]).all() and len(xs)%2==0 and len(ys)%3==0
    for line_number,line in enumerate(lines[3:],4):
        if not line.strip():continue
        terms=line.split();assert len(terms)==8
        assert all(re.fullmatch(r'[01]',v) for v in terms[2:])
        row,col=map(int,terms[:2]);mask=sum(int(v)<<bit for bit,v in enumerate(terms[2:]))
        if not mask:continue
        xx=xs[2*(col-1):2*col];yy=ys[3*(row-1):3*row]
        assert len(xx)==2 and len(yy)==3
        points=np.array([[x,y] for x in xx for y in yy])
        margin=(xx[1]-xx[0])*.3
        box=np.array([xx[0]-margin,yy[0]-margin,xx[1]+margin,yy[2]+margin])
        fractions=(points+.5-box[:2])/(box[2:]-box[:2])
        key=(source['source'],row,col)
        assert key not in truth
        truth[key]={'line':line_number,'mask':mask,'points':points,'box':box,'fractions':fractions}
assert len(truth)==9983
priors=np.median(np.stack([row['fractions'] for row in truth.values()]),axis=0)
np.testing.assert_allclose(priors,slot['median_corner_fractions'],atol=1e-13,rtol=0)
all_fractions=np.stack([row['fractions'] for row in truth.values()])
for percentile,expected in slot['fraction_percentiles'].items():
    np.testing.assert_allclose(np.percentile(all_fractions,int(percentile),axis=0),expected,atol=1e-13,rtol=0)
seen=Counter();complete_seen=set();complete_errors=[];max_box_error=0.;max_q_error=0.;max_f_error=0.;markers=[]
for record in records:
    entry=metadata[record['image']]
    assert entry['source']==record['source'] and entry['source_cells']==record['source_cells'] and entry['masks']==record['masks']
    crop=np.asarray(record['crop']);scale=np.array([480,192])/(crop[2:]-crop[:2])
    for i,identity in enumerate(record['source_cells']):
        key=(record['source'],identity['row'],identity['col']);original=truth[key]
        assert identity['annotation_line']==original['line'] and record['masks'][i]==original['mask']
        np.testing.assert_array_equal(entry['source_points'][i],original['points'])
        box=(original['box']-np.tile(crop[:2],2))*np.tile(scale,2)
        actual_box=np.asarray(record['boxes'][i]);max_box_error=max(max_box_error,float(np.max(np.abs(box-actual_box))))
        q=(original['points']+.5-crop[:2])*scale-.5
        reconstructed=actual_box[:2]+original['fractions']*(actual_box[2:]-actual_box[:2])-.5
        max_q_error=max(max_q_error,float(np.max(np.abs(q-reconstructed))))
        measured_f=(q+.5-actual_box[:2])/(actual_box[2:]-actual_box[:2])
        max_f_error=max(max_f_error,float(np.max(np.abs(measured_f-original['fractions']))))
        seen[key]+=1
        if record['originally_complete'][i]:
            complete_seen.add(key)
            # Algebraic source-fraction difference, independent of root's q subtraction.
            complete_errors.append((priors-original['fractions'])*(actual_box[2:]-actual_box[:2]))
            family=record['source'].rsplit('/',1)[0]
            local=original['points'][0]-crop[:2]
            if family not in {m['family'] for m in markers} and np.all(local>8) and np.all(local<(crop[2:]-crop[:2]-9)):
                size=source_lookup[record['source']]['image_size']
                image=Image.new('RGB',tuple(size),(0,0,0));point=original['points'][0]
                assert np.array_equal(point,np.round(point))
                image.putpixel(tuple(point.astype(int)),(255,255,255))
                pixels=np.asarray(image.crop(tuple(crop)).resize((480,192),Image.Resampling.BILINEAR),np.float64)[...,0]
                image.close();yy,xx=np.indices(pixels.shape)
                center=np.array([(pixels*xx).sum(),(pixels*yy).sum()])/pixels.sum()
                error=np.abs(center-q[0])
                markers.append({'family':family,'record':record['image'],'source_row':identity['row'],'source_col':identity['col'],
                                'crop':crop.tolist(),'point':point.tolist(),'scale':scale.tolist(),'expected':q[0].tolist(),
                                'pillow_rgb_energy_centroid':center.tolist(),'absolute_error_pixels':error.tolist()})
assert set(seen)==set(truth) and complete_seen==set(truth)
assert len(complete_errors)==slot['complete_cached_occurrences']==13911
assert max_box_error<1e-10 and max_q_error<1e-10 and max_f_error<1e-12
errors=np.stack(complete_errors)
for percentile,expected in slot['fixed_prior_input_pixel_absolute_error_percentiles'].items():
    np.testing.assert_allclose(np.percentile(np.abs(errors),int(percentile),axis=(0,1)),expected,atol=1e-10,rtol=0)
assert len(markers)==4 and max(max(row['absolute_error_pixels']) for row in markers)<.1
for source,expected in slot['per_source'].items():
    values=np.stack([row['fractions'] for key,row in truth.items() if key[0]==source])
    assert len(values)==expected['unique_known_cells']
    np.testing.assert_allclose(np.median(values,axis=0),expected['median_corner_fractions'],atol=1e-13,rtol=0)
result={'status':'passed','scope':'Independent raw26 TRAIN TXT arithmetic, all cached known-cell coordinates, saved config center-tap lattice and four synthetic Pillow RGB markers. No TensorFlow/Torch/GPU imports, model execution, training or shared-source edits.',
        'audited_files':{name:sha(WORK/name) for name in ('audit_feature_lattice.py','feature-lattice-audit.json','audit_slot_priors.py','slot-prior-audit.json')},
        'script_sha256':sha(__file__),'checkpoint_sha256':sha(checkpoint),'source_index_sha256':sha(index_path),
        'manifest_sha256':sha(manifest_path),'point_metadata_sha256':sha(point_path),'raw_annotations':raw_hashes,
        'stride4_center_tap_ramps':ramps,'unique_raw_nonzero_cells':len(truth),'cached_occurrences':sum(seen.values()),
        'complete_cached_occurrences':len(complete_errors),'unique_complete_seen':len(complete_seen),
        'max_cached_box_error_pixels':max_box_error,'max_original_fraction_to_cached_q_error_pixels':max_q_error,
        'max_cached_fraction_vs_unique_source_fraction_difference':max_f_error,'raw_source_medians':priors.tolist(),
        'all_reported_fraction_and_pixel_error_percentiles_match':True,'pillow_markers':markers,'defects':[],
        'interpretation_limits':['Origin3 is the fine-branch nominal convolution lattice; fused nearest-upsampled features do not possess one exact physical center.',
                                 'Priors weight each9983 unique known nonzero cell once, including unoccupied grid slots. They are not per-family balanced.',
                                 'Pixel-error percentiles weight13911 complete cached occurrences times six slots and report per-axis absolute errors, not radial or unique-cell errors.',
                                 'Median absolute pixel errors x=.0282885,y=.246437; p95 x=.118431,y=1.521288; worst y=8.332971. Low median is not a universal point-localization bound.',
                                 'DSBI deskewed TRAIN grid priors do not prove Angelina occupied-point coordinates, rotated box priors, learned sampler improvement or converted graph parity.']}
destination=WORK/'independent-coordinate-preflight.json'
destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value for key,value in result.items() if key not in ('raw_annotations',)},indent=2))
print('report_sha256',sha(destination))
