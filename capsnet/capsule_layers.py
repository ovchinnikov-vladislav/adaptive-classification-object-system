import keras.backend as backend
import tensorflow as tf
from keras import initializers, layers
from keras.utils.conv_utils import conv_output_length, deconv_length
import numpy as np


# TODO: Utils
class UtilsCapsNet:
    @staticmethod
    def update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim, num_routing):
        pass

    @staticmethod
    def squash(input_tensor):
        pass


# TODO: Length layer
class Length(layers.Layer):
    def __init__(self, num_classes, seg=True, **kwargs):
        pass

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def get_config(self):
        pass


# TODO: Mask layer
class Mask(layers.Layer):
    def __init__(self, resize_masks=False, **kwargs):
        pass

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def get_config(self):
        pass


# TODO: ConvCapsule layer
class ConvCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, num_capsule, num_atoms, strides=1, padding='same', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        pass

    def build(self, input_shape):
        pass

    def call(self, input_tensor, training=None):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def get_config(self):
        pass


# TODO: DeconvCapsule layer
class DeconvCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, num_capsule, num_atoms, scalling=2, upsamp_type='deconv', padding='same', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        pass

    def build(self, input_shape):
        pass

    def call(self, input_tensor, training=None):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def get_config(self):
        pass
