import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon, sqrt, sum, square
from tensorflow.keras.layers import BatchNormalization, Add
from libs.capsnets.utls import squash
from libs.capsnets.layers import basic


def block_caps(x, routings, classes, kernel_size=9, strides=1, num_capsule=12,
               primary_dim_capsule=8, dim_capsule=6, padding='valid'):
    x, capsules = PrimaryCapsule2DWithConvOutput(num_capsules=num_capsule, dim_capsules=primary_dim_capsule,
                                                 kernel_size=kernel_size, strides=strides,
                                                 padding=padding, do_reshape=True)(x)
    capsules = Capsule(num_capsules=classes, dim_capsules=dim_capsule, routings=routings)(capsules)

    return x, capsules


def residual_primary_caps_block(x, num_capsules, dim_capsules, kernel_size=5):
    _, capsules = PrimaryCapsule2DWithConvOutput(num_capsules=num_capsules, dim_capsules=dim_capsules,
                                                 kernel_size=kernel_size, padding='same', strides=3)(x)
    _, capsules = PrimaryCapsule2DWithConvOutput(num_capsules=num_capsules, dim_capsules=dim_capsules,
                                                 kernel_size=kernel_size, padding='same', strides=3)(capsules)

    out = Add()([x, capsules])
    return out


class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return sqrt(sum(square(inputs), -1) + epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Capsule(basic.Capsule):
    def __init__(self, num_capsules, dim_capsules, routings=3, **kwargs):
        super(Capsule, self).__init__(num_capsules, dim_capsules, routings, **kwargs)


class PrimaryCapsule2DWithConvOutput(layers.Layer):
    def __init__(self, num_capsules, dim_capsules, kernel_size, strides, padding='valid', do_reshape=False, **kwargs):
        super(PrimaryCapsule2DWithConvOutput, self).__init__(**kwargs)
        self.do_reshape = do_reshape
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        num_filters = num_capsules * dim_capsules
        self.conv = layers.Conv2D(filters=num_filters,
                                  kernel_size=kernel_size,
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                  strides=strides,
                                  padding=padding)
        self.batch = BatchNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        output = self.conv(inputs)
        output = self.batch(output)

        if not self.do_reshape:
            shape = output.shape[1] * output.shape[2] * output.shape[3] / self.dim_capsules
            outputs = layers.Reshape(target_shape=(int(np.sqrt(shape)), int(np.sqrt(shape)), self.dim_capsules))(output)
            return output, layers.Lambda(squash)(outputs)

        outputs = layers.Reshape(target_shape=(-1, self.dim_capsules))(output)
        return output, layers.Lambda(squash)(outputs)

    def get_config(self):
        return super().get_config()
