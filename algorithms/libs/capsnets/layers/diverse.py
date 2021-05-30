import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
import numpy as np

from tflearn.layers.conv import global_avg_pool
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Lambda
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras import backend as K
from libs.capsnets.utils import squash
from libs.capsnets.layers import basic

weight_decay = 1E-4


# Mobilenet V3 bottleneck
def bottleneck(inputs, filters, kernel, e, s, squeeze, nl):
    def _relu6(x):
        return K.relu(x, max_value=6.0)

    def _hard_swish(x):
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _return_activation(x, nl):
        if nl == 'HS':
            x = Activation(_hard_swish)(x)
        if nl == 'RE':
            x = Activation(_relu6)(x)
        return x

    def _conv_block(inputs, filters, kernel, strides, nl):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)
        return _return_activation(x, nl)

    def _squeeze(inputs):
        input_channels = int(inputs.shape[-1])

        x = GlobalAveragePooling2D()(inputs)
        x = Dense(input_channels, activation='relu')(x)
        x = Dense(input_channels, activation='hard_sigmoid')(x)
        return x

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)
    tchannel = input_shape[channel_axis] * e
    x = _conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    if squeeze:
        x = Lambda(lambda x: x * _squeeze(x))(x)
    x = _return_activation(x, nl)
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    return x


class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1) + epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Capsule(basic.Capsule):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(Capsule, self).__init__(num_capsule, dim_capsule, routings, **kwargs)


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

        self.dense_first = layers.Dense(units=int(num_capsules / 5))
        self.dense_second = layers.Dense(units=num_capsules)

    def call(self, inputs, **kwargs):
        output = BatchNormalization(axis=-1)(inputs)
        output = layers.Activation('relu')(output)
        output = self.conv(output)

        outputs = layers.Reshape(target_shape=(-1, self.dim_capsules))(output)

        length = tf.sqrt(tf.reduce_sum(tf.square(outputs), -1) + epsilon())
        data_size = int(inputs.shape[1])
        strides = self.strides[0]
        data_size = int(np.floor((data_size - self.kernel_size) / strides + 1))
        length = layers.Reshape(target_shape=(data_size, data_size, self.num_capsules))(length)

        squeeze = global_avg_pool(length)
        excitation = self.dense_first(squeeze)
        excitation = tf.nn.relu(excitation)
        excitation = self.dense_second(excitation)
        excitation = layers.Reshape(target_shape=(-1, self.num_capsules))(excitation)
        excitation = tf.reduce_mean(excitation, axis=-1, keepdims=True)
        excitation = tf.nn.sigmoid(excitation)
        excitation = layers.Reshape(target_shape=(-1, 1))(excitation)

        return output, layers.Lambda(squash)(outputs), excitation

    def get_config(self):
        return super().get_config()
