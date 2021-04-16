import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon, sqrt, sum, square
from tensorflow.keras.layers import BatchNormalization, Dropout, Conv2D
from libs.capsnets.utls import squash
from libs.capsnets.layers import basic


def res_block_caps(x, routings, classes, kernel_size=9, strides=1, num_capsule=12, primary_dim_capsule=8,
                   dim_capsule=6, padding='valid'):
    x, capsules = PrimaryCapsule2D(num_capsules=num_capsule, dim_capsules=primary_dim_capsule, kernel_size=kernel_size,
                                   strides=strides, padding=padding)(x)
    capsules = Capsule(num_capsules=classes, dim_capsules=dim_capsule, routings=routings)(capsules)

    return x, capsules


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


class PrimaryCapsule2D(layers.Layer):
    def __init__(self, num_capsules, dim_capsules, kernel_size, strides, padding='valid', **kwargs):
        super(PrimaryCapsule2D, self).__init__(**kwargs)
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

        outputs = layers.Reshape(target_shape=(-1, self.dim_capsules))(output)

        return output, squash(outputs)

    def get_config(self):
        return super().get_config()
