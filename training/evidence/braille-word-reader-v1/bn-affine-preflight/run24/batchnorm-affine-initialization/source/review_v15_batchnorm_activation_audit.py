"""Independent saved-report/graph/weights review; no ML backend or inference."""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path,PurePosixPath
import sys
from zipfile import ZipFile

import h5py
import numpy as np
from PIL import Image

W=Path('.braille-training-work/word-scan-v1-20260905')
folder=W/'v15-train-batchnorm-activations'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((folder/'protocol.json').read_text())
report=json.loads((folder/'report.json').read_text())
manifest=W/'combined-training-v5/combined-training-manifest.json'
checkpoint=W/'selected-v15/best-word.keras'
source=W/'audit_v15_batchnorm_activations.py'
for key,path in [('protocol_sha256',folder/'protocol.json'),('script_sha256',source),('checkpoint_sha256',checkpoint)]:
    assert report[key]==sha(path)
assert protocol['manifest_sha256']==sha(manifest)
assert protocol['script_sha256']==report['script_sha256']
assert report['word_model_source_sha256']==sha(W/'selected-v15/source/word_model.py')
records=json.loads(manifest.read_text())
families={}
for i,r in enumerate(records):
    if r['boxes']:families.setdefault(str(PurePosixPath(r['source'].replace('\\','/')).parent),[]).append(i)
expected=[(family,indices[(len(indices)-1)*q//4]) for family,indices in sorted(families.items()) for q in [1,2,3]]
assert len(expected)==36 and len(families)==12
assert [(r['family'],r['manifest_index']) for r in protocol['samples']]==expected
split=json.loads((W/'data/source-split.json').read_text())
supplement=json.loads((W/'dsbi-training-contexts-v2/source-index.json').read_text())
allowed=set(split['images']['train'])|{r['source'] for r in supplement['sources']}
assert all(records[i]['source'] in allowed for _,i in expected)
max_image_stats_error=0.
for sample,stats in zip(protocol['samples'],report['image_statistics'],strict=True):
    row=records[sample['manifest_index']]
    assert sample['source']==row['source'] and sample['image']==row['image']
    assert stats['manifest_index']==sample['manifest_index'] and stats['family']==sample['family']
    image=W/'data'/row['image'];assert sha(image)==sample['sha256']
    with Image.open(image) as im: x=np.asarray(im.convert('RGB'),np.float32)/255.
    assert x.shape==(192,480,3)
    # Independent NumPy rendition of the serialized prefix, without any Keras.
    shifted=x-x[:1,:1,:]
    centered=shifted-shifted.mean(axis=1,keepdims=True).mean(axis=0,keepdims=True)
    variance=np.square(centered).mean(axis=1,keepdims=True).mean(axis=0,keepdims=True)
    gray=(centered/np.sqrt(np.maximum(variance,.001**2))).mean(axis=-1,keepdims=True)
    y=np.repeat(np.clip(.5+.05*gray,0,1),3,axis=-1)*2-1
    measured={'raw_mean':x.mean(),'raw_std':x.std(),'backbone_input_mean':y.mean(),'backbone_input_std':y.std()}
    for key,value in measured.items():
        error=abs(float(value)-stats[key]);assert error<2e-5,(key,error)
        max_image_stats_error=max(max_image_stats_error,error)

with ZipFile(checkpoint) as z:
    config=json.loads(z.read('config.json'))['config']
    layers={l['config']['name']:l for l in config['layers']}
    backbone=layers['mobilenetv2_half']
    assert backbone['inbound_nodes'][0]['kwargs']['training'] is False
    history=lambda layer:layer['inbound_nodes'][0]['args'][0]['config']['keras_history'][0]
    assert history(backbone)=='mobilenet_normalization'
    assert history(layers['mobilenet_normalization'])=='camera_normalization'
    assert history(layers['camera_normalization'])=='rgb_0_to_1'
    assert layers['mobilenet_normalization']['config']['scale']==2 and layers['mobilenet_normalization']['config']['offset']==-1
    assert layers['camera_normalization']['config']['target_std']==.05
    assert layers['camera_normalization']['config']['std_floor']==.001
    inner=backbone['config']['layers']
    bns={l['config']['name']:l for l in inner if l['class_name']=='BatchNormalization'}
    relus={l['config']['name']:l for l in inner if l['class_name']=='ReLU'}
    assert len(bns)==52 and len(bns)==report['batchnorm_layers']
    assert {r['layer'] for r in report['batchnorm_statistics']}==set(bns)
    assert set(report['relu_counts'])==set(relus)
    with h5py.File(io.BytesIO(z.read('model.weights.h5')),'r') as h5:
        saved={}
        def collect(name,obj):
            if isinstance(obj,h5py.Group) and obj.attrs.get('name') in bns:
                saved[obj.attrs['name']]=[obj[str(i)][:] for i in range(4)]
        h5.visititems(collect)
    assert set(saved)==set(bns)
    qnames=['min','p05','median','p95','max']
    max_quantile_error=0.
    for row in report['batchnorm_statistics']:
        layer=bns[row['layer']];params=layer['config']
        assert params['axis']==-1 and params['trainable'] is False
        shape=layer['build_config']['input_shape']
        assert row['channels']==shape[-1]
        assert row['spatial_sample_count_per_channel']==36*shape[1]*shape[2]
        mean,var=np.asarray(row['observed_mean']),np.asarray(row['observed_variance'])
        sm,sv=saved[row['layer']][2:]
        np.testing.assert_array_equal(row['saved_mean'],sm)
        np.testing.assert_array_equal(row['saved_variance'],sv)
        assert np.isfinite(mean).all() and np.isfinite(var).all() and np.all(var>=0)
        denominator=np.sqrt(sv.astype(np.float64)+params['epsilon'])
        quantities={'absolute_mean_shift_per_saved_std':abs(mean-sm)/denominator,
                    'observed_std_per_saved_std':np.sqrt(var)/denominator}
        for key,values in quantities.items():
            q=np.quantile(values,[0,.05,.5,.95,1]);old=np.asarray([row[key][n] for n in qnames])
            error=float(np.max(abs(q-old)));assert error<1e-12
            max_quantile_error=max(max_quantile_error,error)
        assert row['channels_abs_shift_over_one_saved_std']==int(np.count_nonzero(abs(mean-sm)>denominator))
        assert row['channels_observed_std_below_quarter_saved']==int(np.count_nonzero(np.sqrt(var)<.25*denominator))
    for name,counts in report['relu_counts'].items():
        layer=relus[name]
        assert layer['config']['max_value']==6 and layer['config']['negative_slope']==0 and layer['config']['threshold']==0
        shape=layer['inbound_nodes'][0]['args'][0]['config']['shape']
        assert counts['values']==36*np.prod(shape[1:])
        assert 0<=counts['zero']<=counts['values'] and 0<=counts['at_or_above_six']<=counts['values']
        assert counts['zero']+counts['at_or_above_six']<=counts['values']
        assert counts['zero_fraction']==counts['zero']/counts['values']
        assert counts['at_or_above_six_fraction']==counts['at_or_above_six']/counts['values']

assert report['all_model_values_unchanged'] is True
assert sha(checkpoint)==protocol['checkpoint_sha256']
assert all(name not in sys.modules for name in ['keras','torch','tensorflow'])
selected={r['layer']:{k:r[k] for k in ['channels','spatial_sample_count_per_channel','absolute_mean_shift_per_saved_std','observed_std_per_saved_std']} for r in report['batchnorm_statistics'] if r['layer'] in ['bn_Conv1','block_13_depthwise_BN']}
answer={'status':'passed_with_descriptive_limits','script_sha256':sha(__file__),
 'reviewed_files':{str(p):sha(p) for p in [source,folder/'protocol.json',folder/'report.json',manifest,checkpoint]},
 'sampling':{'cached_inputs':36,'positive_families':12,'unique_source_photos':len({r['source'] for r in protocol['samples']}),'family_counts':dict(Counter(r['family'] for r in protocol['samples'])),'all_hashes_and_TRAIN_membership_verified':True},
 'checks':{'actual_checkpoint_prefix_and_full_nested_backbone':True,'BN_axis_last_population_moments_formula_reviewed':True,'all52_saved_moments_match_HDF5':True,'all52_channel_counts_quantiles_recomputed':True,'all_relu6_shapes_counts_fractions_consistent':True,'checkpoint_hash_unchanged':True,'max_prefix_numpy_statistics_error':max_image_stats_error,'max_recomputed_quantile_error':max_quantile_error},
 'selected_statistics':selected,'block13_relu6':report['relu_counts']['block_13_depthwise_relu'],
 'defects':[],
 'interpretation':'Valid descriptive motivation for a separately controlled gamma/beta adaptation experiment while keeping moving moments and inference-mode backbone fixed. BN inference is an affine per-channel mapping; gamma/beta can change that mapping without recalibrating moments. This audit does not establish that those parameters are the bottleneck or predict any accuracy gain.',
 'limitations':['36 cached positives are only 3 manifest-order quantile samples per family, with repeated source photos and possible overlapping contexts; this is not random or source-independent population estimation.','Moments pool all spatial positions, including paper/background/verso/unknown regions. They do not isolate foreground Braille activations, and no negatives or stochastic augmentation are sampled.','Mean-shift and std ratios use sqrt(saved_variance+epsilon), not unadjusted stored standard deviation, and are channel medians. ReLU6 saturation is a pooled element fraction.','The report retains observed means/variances, not raw activations or their sums. This review checks the accumulation formula, shapes, stored moments and derived arithmetic; it does not independently reproduce deep-layer moments without inference.','The script checks both all model weight bytes before/after and checkpoint file hash, with gradients disabled and no optimizer/save. Its in-memory before/after equality is evidence from the original run; this review separately verifies the retained checkpoint and its BN moments.','Gamma/beta-only training requires an explicit trainable-variable set and training=False for the nested backbone; merely unfreezing BN under training=True would update moments and test a different hypothesis.']}
out=W/'v15-batchnorm-activation-independent-review.json'
assert not out.exists()
out.write_text(json.dumps(answer,indent=2)+'\n')
print(json.dumps({'report':str(out),'sha256':sha(out),'sampling':answer['sampling'],'selected_statistics':selected,'block13_relu6':answer['block13_relu6']},indent=2))
