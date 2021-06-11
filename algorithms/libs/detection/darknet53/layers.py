import tensorflow as tf
from tensorflow.keras.layers import Concatenate, ZeroPadding2D, Conv2D, BatchNormalization, LeakyReLU, Add, Input, \
    MaxPool2D
from tensorflow.keras.regularizers import l2


def darknet_conv(x, filters, size, down_sampling=False,
                 activation='leaky', batch_norm=True):
    def mish(inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    if down_sampling:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1

    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        if activation == 'mish':
            x = mish(x)
        elif activation == 'leaky':
            x = LeakyReLU(alpha=0.1)(x)

    return x


def darknet_residual(x, first_filters, second_filters, activation='leaky'):
    prev = x
    x = darknet_conv(x, first_filters, 1, activation=activation)
    x = darknet_conv(x, second_filters, 3, activation=activation)
    x = Add()([prev, x])
    return x


def darknet_block(x, filters, blocks):
    x = darknet_conv(x, filters, 3, down_sampling=True)
    for _ in range(blocks):
        x = darknet_residual(x, filters // 2, filters)
    return x


def darknet53(name=None, size=None, channels=3):
    x = inputs = Input([size, size, channels])
    x = darknet_conv(x, 32, 3)
    x = darknet_block(x, 64, 1)
    x = darknet_block(x, 128, 2)  # skip connection
    x = x_36 = darknet_block(x, 256, 8)  # skip connection
    x = x_61 = darknet_block(x, 512, 8)
    x = darknet_block(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def darknet53_tiny(name=None, channels=3):
    x = inputs = Input([None, None, channels])
    x = darknet_conv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = darknet_conv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = darknet_conv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = darknet_conv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = darknet_conv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = darknet_conv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = darknet_conv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def csp_block(x, residual_out, repeat, residual_bottleneck=False):
    route = x
    route = darknet_conv(route, residual_out, 1, activation="mish")
    x = darknet_conv(x, residual_out, 1, activation="mish")
    for i in range(repeat):
        x = darknet_residual(x, residual_out // 2 if residual_bottleneck else residual_out,
                             residual_out, activation="mish")
    x = darknet_conv(x, residual_out, 1, activation="mish")

    x = Concatenate()([x, route])
    return x


def cspdarknet53(name=None, size=None, channels=3):
    x = inputs = Input([size, size, channels])

    x = darknet_conv(x, 32, 3)
    x = darknet_conv(x, 64, 3, down_sampling=True)

    x = csp_block(x, residual_out=64, repeat=1, residual_bottleneck=True)
    x = darknet_conv(x, 64, 1, activation='mish')
    x = darknet_conv(x, 128, 3, activation='mish', down_sampling=True)

    x = csp_block(x, residual_out=64, repeat=2)
    x = darknet_conv(x, 128, 1, activation='mish')
    x = darknet_conv(x, 256, 3, activation='mish', down_sampling=True)

    x = csp_block(x, residual_out=128, repeat=8)
    x = darknet_conv(x, 256, 1, activation='mish')
    route0 = x
    x = darknet_conv(x, 512, 3, activation='mish', down_sampling=True)

    x = csp_block(x, residual_out=256, repeat=8)
    x = darknet_conv(x, 512, 1, activation='mish')
    route1 = x
    x = darknet_conv(x, 1024, 3, activation='mish', down_sampling=True)

    x = csp_block(x, residual_out=512, repeat=4)

    x = darknet_conv(x, 1024, 1, activation="mish")

    x = darknet_conv(x, 512, 1)
    x = darknet_conv(x, 1024, 3)
    x = darknet_conv(x, 512, 1)

    x = Concatenate()([MaxPool2D(pool_size=13, strides=1, padding='same')(x),
                       MaxPool2D(pool_size=9, strides=1, padding='same')(x),
                       MaxPool2D(pool_size=5, strides=1, padding='same')(x), x])
    x = darknet_conv(x, 512, 1)
    x = darknet_conv(x, 1024, 3)
    route2 = darknet_conv(x, 512, 1)
    return tf.keras.Model(inputs, [route0, route1, route2], name=name)

