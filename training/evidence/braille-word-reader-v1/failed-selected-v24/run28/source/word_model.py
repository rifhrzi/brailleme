"""Portable Keras 3 MobileNetV2 + feature pyramid joint Braille reader."""
import keras
from keras import layers, ops

from word_data import GRID_H, GRID_W, HEIGHT, WIDTH


@keras.saving.register_keras_serializable(package="BrailleMe")
class CameraNormalization(layers.Layer):
    """Weight-free per-image illumination normalization with explicit bounds."""
    def __init__(self, target_std=.05, std_floor=.001, **kwargs):
        super().__init__(**kwargs)
        self.target_std, self.std_floor = target_std, std_floor

    def call(self, inputs):
        # Algebraically identical per-channel standardization. Subtract one
        # reference pixel before reduction, then reduce each spatial axis in
        # sequence: summing 92,160 near-white values directly loses precision
        # in TensorFlow and disagrees with the Torch training backend.
        shifted=inputs-inputs[:,:1,:1,:]
        mean=ops.mean(ops.mean(shifted,axis=2,keepdims=True),axis=1,keepdims=True)
        centered=shifted-mean
        variance=ops.mean(ops.mean(ops.square(centered),axis=2,keepdims=True),axis=1,keepdims=True)
        std=ops.sqrt(ops.maximum(variance,self.std_floor**2))
        gray=ops.mean(centered/std,axis=-1,keepdims=True)
        return ops.repeat(ops.clip(.5+self.target_std*gray,0.,1.),3,axis=-1)

    def get_config(self):
        return dict(super().get_config(),target_std=self.target_std,std_floor=self.std_floor)


def image_normalization_spec(model):
    layer=next((layer for layer in model.layers if isinstance(layer,CameraNormalization)),None)
    if layer is None:
        return {'kind':'none'}
    return {'kind':'per_channel_standardize_grayscale','location':'inside_model_before_backbone_rescaling',
            'variance':'population','stdFloor':layer.std_floor,'targetMean':.5,'targetStd':layer.target_std,
            'clip':[0,1],'grayscale':'mean_of_standardized_channels','outputChannels':3}


@keras.saving.register_keras_serializable(package="BrailleMe")
def word_loss(truth, prediction):
    return joint_word_loss(truth,prediction,relative_sizes=False)


@keras.saving.register_keras_serializable(package="BrailleMe")
def word_relative_loss(truth,prediction):
    return joint_word_loss(truth,prediction,relative_sizes=True)


def joint_word_loss(truth, prediction, relative_sizes):
    heatmap = truth[..., :1]
    probability = ops.clip(prediction[..., :1], 1e-5, 1-1e-5)
    centers = ops.cast(heatmap >= .999, "float32")
    valid = ops.cast(heatmap >= 0, "float32")
    normalizer = ops.maximum(ops.sum(centers), 1.)
    positive = -centers * ops.square(1-probability) * ops.log(probability)
    negative = -(1-centers) * valid * ops.power(1-ops.maximum(heatmap, 0), 4) * ops.square(probability) * ops.log(1-probability)
    localization = ops.sum(positive+negative) / normalizer
    if relative_sizes:
        offset_delta=truth[...,1:3]-prediction[...,1:3]
        size_delta=ops.log(ops.maximum(truth[...,3:5],1e-3))-ops.log(ops.maximum(prediction[...,3:5],1e-3))
        delta=ops.concatenate((offset_delta,size_delta),axis=-1)
    else:
        delta = (truth[..., 1:5]-prediction[..., 1:5]) * ops.convert_to_tensor([1., 1., 4., 2.])
    absolute = ops.abs(delta)
    huber = ops.where(absolute < 1., .5*ops.square(delta), absolute-.5)
    geometry = ops.sum(huber*centers) / normalizer
    bits = ops.clip(prediction[..., 5:], 1e-5, 1-1e-5)
    known_bits = ops.cast(truth[..., 5:] >= 0, 'float32')
    labels = ops.maximum(truth[..., 5:], 0.)
    bit_bce = -(labels*ops.log(bits)+(1-labels)*ops.log(1-bits))
    dot_normalizer = ops.maximum(ops.sum(centers*known_bits), 1.)
    dot_loss = ops.sum(bit_bce*centers*known_bits) / dot_normalizer
    return localization + 5*geometry + 3*dot_loss


def build_model(weights="imagenet", image_normalization="raw"):
    inputs = layers.Input((HEIGHT, WIDTH, 3), name="rgb_0_to_1")
    pixels=CameraNormalization(name='camera_normalization')(inputs) if image_normalization=='channel_gray_std05' else inputs
    normalized = layers.Rescaling(2., offset=-1., name="mobilenet_normalization")(pixels)
    backbone = keras.applications.MobileNetV2(input_shape=(HEIGHT, WIDTH, 3), alpha=.5, include_top=False, weights=weights)
    feature_model = keras.Model(backbone.input, [backbone.get_layer(name).output for name in ("block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu")], name="mobilenetv2_half")
    # Small batches and strongly varying illumination make moving-BN updates
    # unstable; retain pretrained statistics while fine-tuning convolution weights.
    for layer in feature_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    low, middle, high = feature_model(normalized, training=False)
    high = layers.Conv2D(64, 1, padding="same", name="fpn_16")(high)
    middle = layers.Add()([layers.Conv2D(64, 1, padding="same", name="fpn_8")(middle), layers.UpSampling2D(2, interpolation="nearest")(high)])
    low = layers.Add()([layers.Conv2D(64, 1, padding="same", name="fpn_4")(low), layers.UpSampling2D(2, interpolation="nearest")(middle)])
    x = layers.Activation("relu")(low)
    for index in range(2):
        x = layers.SeparableConv2D(64, 3, padding="same", activation="relu", name=f"reader_features_{index}")(x)
    objectness = layers.Conv2D(1, 1, activation="sigmoid", bias_initializer=keras.initializers.Constant(-4.), name="cell_probability")(x)
    geometry = layers.Conv2D(4, 1, activation="sigmoid", bias_initializer=keras.initializers.Constant([0., 0., -2., -1.]), name="cell_geometry")(x)
    dots = layers.Conv2D(6, 1, activation="sigmoid", name="dot_probabilities")(x)
    model = keras.Model(inputs, layers.Concatenate(name="word_cells")([objectness, geometry, dots]), name="braille_word_reader_v1")
    assert tuple(model.output_shape) == (None, GRID_H, GRID_W, 11), model.output_shape
    return model
