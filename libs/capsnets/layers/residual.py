import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from libs.capsnets.utls import squash
from libs.capsnets.layers import basic


def bottleneck(inputs, filters, kernel, e, stride, activation):
    def _relu6(x):
        return tf.keras.backend.relu(x, max_value=6.0)

    def _hard_swish(x):
        return x * tf.keras.backend.relu(x + 3.0, max_value=6.0) / 6.0

    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    input_shape = tf.keras.backend.int_shape(inputs)
    tchannel = input_shape[channel_axis] * e

    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    x = Conv2D(tchannel, (1, 1), padding='same', strides=(1, 1))(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    if activation == 'hard_swish':
        x = tf.keras.layers.Activation(_hard_swish)(x)
    if activation == 'relu':
        x = tf.keras.layers.Activation(_relu6)(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(stride, stride), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if activation == 'hard_swish':
        x = tf.keras.layers.Activation(_hard_swish)(x)
    if activation == 'relu':
        x = tf.keras.layers.Activation(_relu6)(x)
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    return tf.keras.layers.Concatenate(axis=-1)([inputs, x])


def res_block_caps(x, routings, classes, kernel_size=9, strides=2, num_capsule=12, dim_capsule=6):
    x, capsules = PrimaryCapsule2D(num_capsules=num_capsule, dim_capsules=8, kernel_size=kernel_size, strides=strides)(x)
    capsules = Capsule(num_capsules=classes, dim_capsules=dim_capsule, routings=routings)(capsules)

    return x, capsules


class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        tf.print(tf.reduce_sum(tf.square(inputs), -1) + epsilon())
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1) + epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Capsule(basic.Capsule):
    def __init__(self, num_capsules, dim_capsules, routings=3, **kwargs):
        super(Capsule, self).__init__(num_capsules, dim_capsules, routings, **kwargs)


class PrimaryCapsule2D(layers.Layer):
    def __init__(self, num_capsules, dim_capsules, kernel_size, strides, **kwargs):
        super(PrimaryCapsule2D, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.kernel_size = kernel_size
        self.strides = strides
        num_filters = num_capsules * dim_capsules
        self.conv = layers.Conv2D(filters=num_filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='valid')
        self.batch = BatchNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        output = self.batch(inputs)
        output = layers.Activation('relu')(output)
        output = self.conv(output)

        outputs = layers.Reshape(target_shape=(-1, self.dim_capsules))(output)

        return output, squash(outputs)

    def get_config(self):
        return super().get_config()
