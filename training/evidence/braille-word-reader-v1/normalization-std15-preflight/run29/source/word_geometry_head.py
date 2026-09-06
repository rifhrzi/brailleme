"""Optional geometry-conditioned residual for inherited cell/box/dot logits.

Sampling uses approximate TRAIN-only point-corner fractions, not asserted
Angelina keypoints. The stride-4 fine-grid nominal pixel-index origin is 3;
coarse FPN fusion and subsequent convolutions do not share a unique physical
receptive-field center. A zero final residual preserves the inherited logits.
"""
from __future__ import annotations

import keras
from keras import layers, ops
import numpy as np


def bilinear_feature_samples(features, x, y):
    """Sample [B,H,W,C] at [B,N] fine-grid coordinates; clamp at the border."""
    batch, height, width, channels = ops.shape(features)
    x = ops.clip(x, 0., float(width-1))
    y = ops.clip(y, 0., float(height-1))
    x0, y0 = ops.floor(x), ops.floor(y)
    x1, y1 = ops.minimum(x0+1., float(width-1)), ops.minimum(y0+1., float(height-1))
    wx, wy = ops.expand_dims(x-x0, -1), ops.expand_dims(y-y0, -1)
    flattened = ops.reshape(features, (batch, height*width, channels))

    def gather(xx, yy):
        indices = ops.cast(yy*width+xx, 'int32')
        return ops.take_along_axis(flattened, ops.expand_dims(indices, -1), axis=1)

    top = gather(x0, y0)*(1.-wx)+gather(x1, y0)*wx
    bottom = gather(x0, y1)*(1.-wx)+gather(x1, y1)*wx
    return top*(1.-wy)+bottom*wy


@keras.saving.register_keras_serializable(package='BrailleMe')
class GeometrySlotSampler(layers.Layer):
    """Sample six projected features from the unmodified inherited geometry."""
    def __init__(self, slot_fractions, feature_origin=3., stride=4, **kwargs):
        super().__init__(**kwargs)
        fractions = np.asarray(slot_fractions, np.float64)
        if fractions.shape != (6, 2) or not np.isfinite(fractions).all() or np.any(fractions<0) or np.any(fractions>1):
            raise ValueError('Exactly six finite point-corner fractions in [0,1] are required')
        if stride != 4 or feature_origin != 3.:
            raise ValueError('This experiment requires the audited stride4/origin3 fine grid')
        self.slot_fractions = fractions.tolist()
        self.feature_origin, self.stride = float(feature_origin), int(stride)

    def call(self, inputs):
        features, probabilities = inputs
        batch, grid_h, grid_w, channels = ops.shape(features)
        slots = ops.convert_to_tensor(self.slot_fractions, dtype=features.dtype)
        gx = ops.reshape(ops.arange(grid_w, dtype=features.dtype), (1, 1, grid_w, 1))
        gy = ops.reshape(ops.arange(grid_h, dtype=features.dtype), (1, grid_h, 1, 1))
        center_x = (gx+probabilities[..., 1:2])*self.stride
        center_y = (gy+probabilities[..., 2:3])*self.stride
        width = probabilities[..., 3:4]*(grid_w*self.stride)
        height = probabilities[..., 4:5]*(grid_h*self.stride)
        # Parent rectangles use the same linear edge coordinates as targets.
        # Convert point-corner coordinates to pixel indices exactly once.
        px = center_x+(slots[:, 0]-.5)*width-.5
        py = center_y+(slots[:, 1]-.5)*height-.5
        x = ops.reshape((px-self.feature_origin)/self.stride, (batch, -1))
        y = ops.reshape((py-self.feature_origin)/self.stride, (batch, -1))
        samples = bilinear_feature_samples(features, x, y)
        return ops.reshape(samples, (batch, grid_h, grid_w, 6*channels))

    def compute_output_shape(self, input_shape):
        features = input_shape[0]
        return features[:3]+(features[3]*6,)

    def get_config(self):
        return dict(super().get_config(), slot_fractions=self.slot_fractions,
                    feature_origin=self.feature_origin, stride=self.stride)


def extend_geometry_head(parent, slot_fractions):
    """Reuse all old feature weights and copy head kernels into linear logits."""
    if any(layer.name == 'geometry_residual_zero' for layer in parent.layers):
        raise ValueError('Parent already has the geometry-conditioned residual')
    if tuple(parent.input_shape) != (None, 192, 480, 3) or tuple(parent.output_shape) != (None, 48, 120, 11):
        raise ValueError('Unexpected parent word-reader tensor contract')
    features = parent.get_layer('reader_features_1').output
    inherited = []
    for name in ('cell_probability', 'cell_geometry', 'dot_probabilities'):
        old = parent.get_layer(name)
        config = old.get_config()
        if config['activation'] != 'sigmoid':
            raise ValueError('Inherited logits must come directly from sigmoid Conv2D heads')
        config['activation'] = 'linear'
        copied = layers.Conv2D.from_config(config)
        inherited.append(copied(features))
        copied.set_weights(old.get_weights())
        copied.trainable = old.trainable
    logits = layers.Concatenate(name='inherited_cell_logits')(inherited)
    probabilities = layers.Activation('sigmoid', name='inherited_cell_probabilities')(logits)
    projected = layers.Conv2D(8, 1, name='geometry_feature_projection')(features)
    sampled = GeometrySlotSampler(slot_fractions, name='geometry_slot_samples')([projected, probabilities])
    hidden = layers.Dense(32, activation='relu', name='geometry_feature_mlp')(sampled)
    residual = layers.Dense(11, kernel_initializer='zeros', bias_initializer='zeros', name='geometry_residual_zero')(hidden)
    corrected = layers.Add(name='geometry_corrected_logits')([logits, residual])
    result = keras.Model(parent.input, layers.Activation('sigmoid', name='word_cells')(corrected),
                         name='braille_word_reader_geometry_residual')
    for old in parent.layers:
        if not old.weights:
            continue
        copied = result.get_layer(old.name)
        for before, after in zip(old.get_weights(), copied.get_weights(), strict=True):
            np.testing.assert_array_equal(before, after)
    assert result.count_params()-parent.count_params() == 2451
    return result
