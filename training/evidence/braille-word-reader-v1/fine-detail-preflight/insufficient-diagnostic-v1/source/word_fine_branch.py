"""Add a zero-initialized stride2 detail path to the existing stride4 FPN.

The extra 3x3 SAME stride2 depthwise convolution aligns the nominal pixel
origin at 3 with FPN4. Receptive-field sizes describe the normalized tensor;
CameraNormalization itself depends on the whole raw image.
"""
from __future__ import annotations

import keras
from keras import layers
import numpy as np


def fine_branch_spec(model):
    backbone = model.get_layer('mobilenetv2_half')
    depthwise = model.get_layer('fpn_2_depthwise')
    lateral = model.get_layer('fpn_2_zero')
    if (tuple(backbone.get_layer('expanded_conv_depthwise_relu').output.shape) != (None, 96, 240, 16)
            or depthwise.kernel_size != (3, 3) or depthwise.strides != (2, 2)
            or depthwise.padding != 'same' or depthwise.use_bias
            or lateral.filters != 64 or lateral.kernel_size != (1, 1)
            or not lateral.use_bias):
        raise ValueError('Unexpected stride2 residual configuration')
    return {'tap': 'expanded_conv_depthwise_relu', 'tap_shape': [None, 96, 240, 16],
            'tap_stride': 2, 'tap_pixel_origin': 1, 'tap_local_support': 7,
            'branch_output_shape': [None, 48, 120, 64], 'output_stride': 4,
            'output_pixel_origin': 3, 'output_local_support': 11,
            'support_scope': 'Convolution support relative to the normalized tensor; raw RGB normalization is global.',
            'depthwise': {'kernel': 3, 'stride': 2, 'padding': 'same', 'bias': False},
            'lateral': {'kernel': 1, 'channels': 64, 'bias': True},
            'addition': 'Original add_1 result plus branch, before original activation',
            'added_parameters': 1232, 'added_convolution_macs': 6727680,
            'extra_batchnorm_layers': 0}


def inherited_weight_pairs(parent, result):
    """Match by stable layer identity, independent of changed variable order."""
    for old in parent.layers:
        copied = result.get_layer(old.name)
        if old.name == 'mobilenetv2_half':
            for child in old.layers:
                yield child, copied.get_layer(child.name)
        else:
            yield old, copied


def extend_fine_branch(parent, seed=20260905):
    if (tuple(parent.input_shape) != (None, 192, 480, 3)
            or tuple(parent.output_shape) != (None, 48, 120, 11)):
        raise ValueError('Require the existing RGB word-reader tensor contract')
    if any(layer.name in ('fpn_2_zero', 'geometry_residual_zero') for layer in parent.layers):
        raise ValueError('This initialization requires the plain full-tail parent')
    old_backbone = parent.get_layer('mobilenetv2_half')
    batchnorm = [layer for layer in old_backbone.layers if isinstance(layer, layers.BatchNormalization)]
    if len(batchnorm) != 52 or any(type(layer) is not layers.BatchNormalization or layer.trainable for layer in batchnorm):
        raise ValueError('Require all 52 ordinary BatchNormalization layers frozen')
    if len(old_backbone.outputs) != 4 or 'fpn_32_zero' not in [layer.name for layer in parent.layers]:
        raise ValueError('Require the four-scale full-tail reader')
    backbone = keras.models.clone_model(old_backbone)
    backbone.set_weights(old_backbone.get_weights())
    for old in old_backbone.layers:
        backbone.get_layer(old.name).trainable = old.trainable
    features = keras.Model(backbone.input,
                           list(backbone.outputs)+[backbone.get_layer('expanded_conv_depthwise_relu').output],
                           name='mobilenetv2_half')
    inputs = layers.Input((192, 480, 3), name='rgb_0_to_1')

    def copy_layer(name, value):
        old = parent.get_layer(name)
        copied = old.__class__.from_config(old.get_config())
        result = copied(value)
        copied.set_weights(old.get_weights())
        copied.trainable = old.trainable
        return result

    pixels = copy_layer('camera_normalization', inputs)
    low, middle, high, tail, fine = features(copy_layer('mobilenet_normalization', pixels), training=False)
    high = copy_layer('fpn_16', high)
    # This inherited lateral was trained in run18. Never re-zero it.
    tail = copy_layer('fpn_32_upsample', copy_layer('fpn_32_zero', tail))
    high = copy_layer('fpn_16_with_32', [high, tail])
    middle = copy_layer('add', [copy_layer('fpn_8', middle), copy_layer('up_sampling2d', high)])
    low = copy_layer('add_1', [copy_layer('fpn_4', low), copy_layer('up_sampling2d_1', middle)])
    fine = layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False,
                                 depthwise_initializer=keras.initializers.HeNormal(seed=seed),
                                 name='fpn_2_depthwise')(fine)
    fine = layers.Conv2D(64, 1, kernel_initializer='zeros', bias_initializer='zeros', name='fpn_2_zero')(fine)
    output = copy_layer('activation', layers.Add(name='fpn_4_with_2')([low, fine]))
    for index in range(2):
        output = copy_layer(f'reader_features_{index}', output)
    output = copy_layer('word_cells', [copy_layer(name, output)
                        for name in ('cell_probability', 'cell_geometry', 'dot_probabilities')])
    result = keras.Model(inputs, output, name='braille_word_reader_fine_branch')
    for old, copied in inherited_weight_pairs(parent, result):
        if old.trainable != copied.trainable:
            raise ValueError(f'Inherited trainability changed: {old.name}')
        for before, after in zip(old.get_weights(), copied.get_weights(), strict=True):
            np.testing.assert_array_equal(before, after)
    assert result.count_params()-parent.count_params() == 1232
    fine_branch_spec(result)
    return result
