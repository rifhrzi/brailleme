"""Clone a frozen-moment backbone while enabling only its BN affine weights.

This does not update moving moments, reset weights or alter image normalization.
Keras Functional can overwrite a nested graph's saved training=False when the
outer model trains. A layer-level override is therefore required; merely
opening the existing BN trainable flags would also update moving statistics.
"""
import numpy as np
import keras


@keras.saving.register_keras_serializable(package='BrailleMe')
class FrozenMomentBatchNormalization(keras.layers.BatchNormalization):
    """Learn gamma/beta while always using the inherited moving moments."""
    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training=False, mask=mask)


def affine_backbone_spec(model):
    backbone = model.get_layer('mobilenetv2_half')
    graph = next(layer for layer in model.get_config()['layers'] if layer['name'] == backbone.name)
    calls = [node.get('kwargs', {}).get('training') for node in graph['inbound_nodes']]
    if not calls or any(value is not False for value in calls):
        raise ValueError('Affine-only BN requires every backbone call to retain explicit training=False')
    if not model.trainable or not backbone.trainable:
        raise ValueError('Outer model and backbone must remain trainable for their existing convolution weights')
    batchnorm = [layer for layer in backbone.layers if isinstance(layer, keras.layers.BatchNormalization)]
    if not batchnorm or any(not layer.center or not layer.scale for layer in batchnorm):
        raise ValueError('Expected existing BatchNormalization layers with both gamma and beta')
    if any(isinstance(layer, keras.layers.BatchNormalization) for layer in model.layers):
        raise ValueError('Unexpected BatchNormalization outside the explicitly frozen-moment backbone')
    if any(layer.moving_mean.trainable or layer.moving_variance.trainable for layer in batchnorm):
        raise ValueError('Moving moments must remain nontrainable')
    if any(layer.trainable and not isinstance(layer, FrozenMomentBatchNormalization) for layer in batchnorm):
        raise ValueError('Trainable BN must force frozen moments inside layer.call, not only the saved backbone call')
    return {'mode': 'affine_only_frozen_moments',
            'batchnorm_layers': len(batchnorm),
            'trainable_affine_layers': sum(layer.trainable for layer in batchnorm),
            'backbone_call_training_keywords': calls,
            'layer_forced_frozen_moments': sum(isinstance(layer, FrozenMomentBatchNormalization) for layer in batchnorm),
            'affine_parameters': sum(int(np.prod(weight.shape)) for layer in batchnorm for weight in (layer.gamma, layer.beta)),
            'layer_names': [layer.name for layer in batchnorm],
            'moving_moments': 'Unchanged stored mean/variance enforced inside each replacement BN call, including outer model training=True.'}


def build_bn_affine_parent(parent):
    """Return a separate uncompiled model; the supplied parent stays untouched."""
    initial = affine_backbone_spec(parent)
    if initial['trainable_affine_layers']:
        raise ValueError('Initialization requires an entirely frozen BN parent, not a trained affine continuation')
    initial_weights = parent.get_weights()
    def clone_layer(layer):
        kind = FrozenMomentBatchNormalization if isinstance(layer, keras.layers.BatchNormalization) else type(layer)
        return kind.from_config(layer.get_config())
    clone = keras.models.clone_model(parent, clone_function=clone_layer, recursive=True)
    clone.set_weights(initial_weights)
    backbone = clone.get_layer('mobilenetv2_half')
    previous_flags = {layer.name: layer.trainable for layer in backbone.layers}
    previous_trainable = sum(int(np.prod(weight.shape)) for weight in clone.trainable_weights)
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
    result = affine_backbone_spec(clone)
    if result['trainable_affine_layers'] != result['batchnorm_layers']:
        raise AssertionError('Every BN gamma/beta must become trainable')
    for layer in backbone.layers:
        if not isinstance(layer, keras.layers.BatchNormalization) and layer.trainable != previous_flags[layer.name]:
            raise AssertionError('A non-BN backbone trainable flag changed')
    new_trainable = sum(int(np.prod(weight.shape)) for weight in clone.trainable_weights)
    if new_trainable-previous_trainable != result['affine_parameters'] or clone.count_params() != parent.count_params():
        raise AssertionError('Only existing gamma/beta parameters may change trainability')
    for before, original, copied in zip(initial_weights, parent.get_weights(), clone.get_weights(), strict=True):
        if not np.array_equal(before, original) or not np.array_equal(before, copied):
            raise AssertionError('BN affine initialization changed a model weight')
    if affine_backbone_spec(parent) != initial:
        raise AssertionError('Original parent trainability changed')
    return clone, result
