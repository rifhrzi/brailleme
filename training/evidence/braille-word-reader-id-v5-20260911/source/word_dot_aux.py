"""Conditional training-only occupancy head; never imported by exported reader."""
import keras
from keras import layers, ops

from word_geometry_head import bilinear_feature_samples

AUXILIARY_DOT_COEFFICIENT = .25
AUXILIARY_DOT_INITIALIZER_SEED = 20260905


@keras.saving.register_keras_serializable(package='BrailleMe')
class AnnotatedDotSampler(layers.Layer):
    def call(self, inputs):
        logits, queries = inputs
        return bilinear_feature_samples(logits, queries[..., 0], queries[..., 1])

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)


@keras.saving.register_keras_serializable(package='BrailleMe')
def known_dot_bce(labels, logits):
    """Stable logit BCE with an independent known-point denominator."""
    known = ops.cast(labels >= 0, 'float32')
    target = ops.maximum(labels, 0.)
    bce = ops.maximum(logits, 0.) - logits * target + ops.log(1. + ops.exp(-ops.abs(logits)))
    return ops.sum(bce * known) / ops.maximum(ops.sum(known), 1.)


def wrap_dot_training(reader, main_loss, learning_rate):
    """Share the exact existing reader; do not rebuild or copy its backbone."""
    if tuple(reader.input_shape) != (None, 192, 480, 3) or tuple(reader.output_shape) != (None, 48, 120, 11):
        raise ValueError('Unexpected reader contract')
    normalization = reader.get_layer('camera_normalization')
    backbone = reader.get_layer('mobilenetv2_half')
    batchnorm = [layer for layer in backbone.layers if isinstance(layer, layers.BatchNormalization)]
    if (reader.count_params() != 829307 or normalization.target_std != .05 or normalization.std_floor != .001
            or sorted(192 // int(output.shape[1]) for output in backbone.outputs) != [4, 8, 16, 32]
            or len(batchnorm) != 52 or any(layer.trainable for layer in batchnorm)):
        raise ValueError('Require the unchanged full-tail reader with .05 normalization and frozen BatchNorm')
    fine = reader.get_layer('fpn_4').input
    if tuple(fine.shape[1:3]) != (48, 120):
        raise ValueError('The existing pre-FPN stride4 feature tensor is required')
    points = layers.Input((None, 2), name='annotated_dot_queries')
    logits = layers.Conv2D(1, 1, activation=None, name='auxiliary_dot_logits',
                           kernel_initializer=keras.initializers.GlorotUniform(seed=AUXILIARY_DOT_INITIALIZER_SEED),
                           bias_initializer='zeros')(fine)
    samples = AnnotatedDotSampler(name='annotated_dot_logits')([logits, points])
    training = keras.Model((reader.input, points), (reader.output, samples), name='braille_reader_with_training_dot_task')
    training.compile(optimizer=keras.optimizers.Adam(learning_rate, clipnorm=10.),
                     loss=(main_loss, known_dot_bce), loss_weights=(1., AUXILIARY_DOT_COEFFICIENT),
                     jit_compile=False)
    return training
