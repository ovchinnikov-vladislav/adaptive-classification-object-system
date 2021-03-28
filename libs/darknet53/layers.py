import tensorflow as tf
from tensorflow.keras.layers import Concatenate, ZeroPadding2D, Conv2D, BatchNormalization, LeakyReLU, Add, Input, MaxPool2D
from tensorflow.keras.regularizers import l2


class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return inputs * tf.keras.backend.tanh(tf.keras.backend.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def darknet_conv_leaky(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def darknet_conv_mish(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = Mish()(x)
    return x


def darknet_residual_leaky(x, filters):
    prev = x
    x = darknet_conv_leaky(x, filters // 2, 1)
    x = darknet_conv_leaky(x, filters, 3)
    x = Add()([prev, x])
    return x


def darknet_residual_mish(x, filters, all_narrow=True):
    prev = x
    x = darknet_conv_mish(x, filters // 2, 1)
    x = darknet_conv_mish(x, filters // 2 if all_narrow else filters, 3)
    x = Add()([prev, x])
    return x


def darknet_block_leaky(x, filters, blocks):
    x = darknet_conv_leaky(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = darknet_residual_leaky(x, filters)
    return x


def darknet_block_mish(x, filters, blocks, all_narrow=True):
    preconv = darknet_conv_mish(x, filters, 3, strides=2)
    shortconv = darknet_conv_mish(preconv, filters // 2 if all_narrow else filters, 1)
    x = darknet_conv_mish(preconv, filters // 2 if all_narrow else filters, 1)
    for _ in range(blocks):
        x = darknet_residual_mish(x, filters, all_narrow)
    postconv = darknet_conv_mish(x, filters // 2 if all_narrow else filters, 1)
    route = Concatenate()([postconv, shortconv])
    return darknet_conv_mish(route, filters, 1)


def darknet_leaky(name=None):
    x = inputs = Input([None, None, 3])
    x = darknet_conv_leaky(x, 32, 3)
    x = darknet_block_leaky(x, 64, 1)
    x = darknet_block_leaky(x, 128, 2)  # skip connection
    x = x_36 = darknet_block_leaky(x, 256, 8)  # skip connection
    x = x_61 = darknet_block_leaky(x, 512, 8)
    x = darknet_block_leaky(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def darknet_mish(name=None):
    x = inputs = Input([None, None, 3])
    x = darknet_conv_mish(x, 32, 3)
    x = darknet_block_mish(x, 64, 1, False)
    x = darknet_block_mish(x, 128, 2)  # skip connection
    x = x76 = darknet_block_mish(x, 256, 8)  # skip connection
    x = x38 = darknet_block_mish(x, 512, 8)
    x = darknet_block_mish(x, 1024, 4)
    return tf.keras.Model(inputs, (x76, x38, x), name=name)


def darknet_tiny_leaky(name=None):
    x = inputs = Input([None, None, 3])
    x = darknet_conv_leaky(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = darknet_conv_leaky(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = darknet_conv_leaky(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = darknet_conv_leaky(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = darknet_conv_leaky(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = darknet_conv_leaky(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = darknet_conv_leaky(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


if __name__ == '__main__':
    darknet = darknet_mish()
    darknet.summary()