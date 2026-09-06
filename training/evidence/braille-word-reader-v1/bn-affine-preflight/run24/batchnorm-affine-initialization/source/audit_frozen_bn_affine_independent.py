"""Tiny CPU fixed-moment BN proof and correction to saved-kwargs advice."""
import os
os.environ['KERAS_BACKEND']='torch'
os.environ['KERAS_TORCH_DEVICE']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import hashlib
import json
from pathlib import Path
import shutil
import sys

W=Path('.braille-training-work/word-scan-v1-20260905')
OUT=W/'bn-affine-independent-preflight'
OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
helper=Path('training/scripts/word_bn_affine.py')
assert sha(helper)=='fa117f65beda3f6d159f707a5cd45d3a367b8f5b436b8c43c0d7621071297dff'
shutil.copyfile(helper,OUT/helper.name)
shutil.copyfile('training/tests/test_word_bn_affine.py',OUT/'original_test_word_bn_affine.py')
protocol={'scope':'Independent tiny CPU Torch/Keras model only; no actual reader execution, training data, optimizer step on retained candidates, TensorFlow or CUDA.',
 'script_sha256':sha(__file__),'helper_sha256':sha(helper),
 'checks':['Reproduce plain trainable BN updating moments despite saved nested training=False.',
           'Clone exact initial values/outputs without sharing variables or changing parent.',
           'Real SGD updates gamma/beta while all moving moments remain byte exact under outer training=True.',
           'Save/load trained tiny model preserves forced-moment class, outputs and moments.',
           'Actual Keras train_on_batch with existing convolution trainability also preserves every moving moment.']}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
sys.path.insert(0,str(OUT.resolve()))
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import keras
from word_bn_affine import FrozenMomentBatchNormalization,affine_backbone_spec,build_bn_affine_parent
assert 'tensorflow' not in sys.modules
keras.utils.set_random_seed(531)

inputs=keras.Input((6,7,3))
x=keras.layers.Conv2D(4,3,padding='same',name='conv_a')(inputs)
x=keras.layers.BatchNormalization(trainable=False,name='bn_a')(x)
x=keras.layers.Activation('tanh')(x)
x=keras.layers.Conv2D(4,1,name='conv_b')(x)
x=keras.layers.BatchNormalization(trainable=False,name='bn_b')(x)
backbone=keras.Model(inputs,x,name='mobilenetv2_half')
outer=keras.Input((6,7,3))
parent=keras.Model(outer,keras.layers.Conv2D(11,1,activation='sigmoid',name='head')(backbone(outer,training=False)))
rng=np.random.default_rng(771)
for layer in backbone.layers:
    if isinstance(layer,keras.layers.BatchNormalization):
        layer.set_weights([rng.uniform(.5,1.5,4).astype('float32'),rng.uniform(-.2,.2,4).astype('float32'),
                           rng.uniform(-.5,.5,4).astype('float32'),rng.uniform(.4,1.6,4).astype('float32')])
pixels=rng.uniform(-.5,1.5,(3,6,7,3)).astype('float32')
targets=rng.uniform(.05,.95,(3,6,7,11)).astype('float32')
bn=lambda m:[l for l in m.get_layer('mobilenetv2_half').layers if isinstance(l,keras.layers.BatchNormalization)]
moments=lambda m:[np.array(w,copy=True) for l in bn(m) for w in [l.moving_mean.numpy(),l.moving_variance.numpy()]]
same=lambda a,b: all(np.array_equal(x,y) for x,y in zip(a,b,strict=True))
digest=lambda values:hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in values)).hexdigest()
parent_weights=parent.get_weights()
parent_flags=[l.trainable for l in backbone.layers]

unsafe=keras.models.clone_model(parent,recursive=True)
unsafe.set_weights(parent_weights)
for layer in bn(unsafe):layer.trainable=True
before_unsafe=moments(unsafe)
unsafe(pixels,training=True)
changed_unsafe=sum(not np.array_equal(x,y) for x,y in zip(before_unsafe,moments(unsafe)))
assert changed_unsafe==4
try:affine_backbone_spec(unsafe)
except ValueError as error:assert 'inside layer.call' in str(error)
else:raise AssertionError('Plain trainable BN was accepted')

model,spec=build_bn_affine_parent(parent)
assert spec['batchnorm_layers']==spec['trainable_affine_layers']==spec['layer_forced_frozen_moments']==2
assert spec['affine_parameters']==16
assert all(w.value.device.type=='cpu' for w in model.weights)
assert same(model.get_weights(),parent_weights)
assert all(a.value.data_ptr()!=b.value.data_ptr() for a,b in zip(parent.weights,model.weights,strict=True))
expected=parent(pixels,training=False).detach().numpy()
np.testing.assert_array_equal(model(pixels,training=False).detach().numpy(),expected)
np.testing.assert_array_equal(model(pixels,training=True).detach().numpy(),expected)
before_moments=moments(model)
affine=[w for l in bn(model) for w in [l.gamma,l.beta]]
affine_before=[w.numpy().copy() for w in affine]
other=[w for w in model.weights if all(w is not v for v in affine)]
other_before=[w.numpy().copy() for w in other]
optimizer=torch.optim.SGD([w.value for w in affine],lr=.05)
gradient_checks=[]
for step in range(3):
    model.zero_grad()
    loss=torch.mean((model(pixels,training=True)-torch.from_numpy(targets))**2)
    loss.backward()
    norms=[float(w.value.grad.abs().max()) for w in affine]
    assert all(np.isfinite(n) and n>0 for n in norms)
    assert all(not w.value.requires_grad and w.value.grad is None for l in bn(model) for w in [l.moving_mean,l.moving_variance])
    optimizer.step()
    assert same(before_moments,moments(model))
    gradient_checks.append(norms)
assert all(not np.array_equal(a,w.numpy()) for a,w in zip(affine_before,affine))
assert same(other_before,[w.numpy() for w in other])
assert same(parent_weights,parent.get_weights()) and parent_flags==[l.trainable for l in backbone.layers]
trained=model(pixels,training=True).detach().numpy()
delta=float(np.max(abs(trained-expected)));assert delta>0
saved=OUT/'tiny-trained-affine.keras';model.save(saved)
loaded=keras.models.load_model(saved,compile=False)
assert affine_backbone_spec(loaded)==spec
assert all(type(l) is FrozenMomentBatchNormalization for l in bn(loaded))
assert same(model.get_weights(),loaded.get_weights()) and same(before_moments,moments(loaded))
np.testing.assert_array_equal(loaded(pixels,training=True).detach().numpy(),trained)
np.testing.assert_array_equal(loaded(pixels,training=False).detach().numpy(),trained)
# Keras's real training context is exercised after loading, so this also tests
# the case which overrode plain BN despite the graph's stored False keyword.
loaded.compile(optimizer=keras.optimizers.SGD(.01),loss='mse',jit_compile=False)
loaded.train_on_batch(pixels,targets)
assert same(before_moments,moments(loaded))
assert not same(model.get_weights(),loaded.get_weights())
assert same(parent_weights,parent.get_weights())
assert sha(helper)==protocol['helper_sha256']
assert 'tensorflow' not in sys.modules
import keras.src.models.functional as functional
import keras.src.layers.normalization.batch_normalization as normalization
review={'status':'passed','protocol_sha256':sha(OUT/'protocol.json'),'script_sha256':sha(__file__),
 'helper_sha256':protocol['helper_sha256'],'keras_version':keras.__version__,'torch_version':torch.__version__,
 'installed_sources':{functional.__file__:sha(functional.__file__),normalization.__file__:sha(normalization.__file__)},
 'plain_nested_false_changed_moment_arrays':changed_unsafe,'fixed_moment_BN_layers':2,'affine_parameters':16,
 'initial_parent_clone_weights_and_train_eval_outputs_exact':True,'variables_do_not_alias_parent':True,
 'three_real_sgd_steps':{'all4_affine_tensors_changed':True,'gradient_max_per_tensor_by_step':gradient_checks,
                       'all_moving_and_non_affine_arrays_exact':True,'maximum_output_change':delta},
 'trained_save_load_weights_train_eval_outputs_exact':True,'loaded_classes_force_false_inside_call':True,
 'keras_train_on_batch_updates_weights_without_moving_moments':True,'original_parent_weights_and_flags_unchanged':True,
 'moving_arrays_before_after_sha256':digest(before_moments),'tiny_model_sha256':sha(saved),'defects':[],
 'correction_to_prior_activation_review':{'file':str(W/'v15-batchnorm-activation-independent-review.json'),
   'sha256':sha(W/'v15-batchnorm-activation-independent-review.json'),
   'statement':'The prior descriptive activation audit remains valid because its probe/prefix used training=False and every BN was frozen. Its future affine-training advice requires refinement: stored nested training=False kwargs alone are insufficient in actual Keras3.15.1 Functional training. Force inference behavior inside each BN call (as this helper does), then verify actual updates/save-load leave moments exact.'},
 'interpretation':'This utility opens only existing BN gamma/beta in addition to the already trainable convolution/head weights. It does not make the entire reader optimizer affine-only. The isolated SGD proof intentionally updated only gamma/beta to demonstrate their effect; actual train_on_batch also verifies the existing broader variable set.',
 'limits':['Tiny-model Torch CPU semantic proof only; actual full-v15 initialization and TF/TFLite parity remain separate trainer proofs.','No recognition accuracy, production readiness or benefit from affine adaptation is established.']}
(OUT/'report.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'report':str(OUT/'report.json'),'sha256':sha(OUT/'report.json'),'plain_changed_moments':changed_unsafe,'max_affine_only_output_change':delta}))
