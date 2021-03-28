import tensorflow as tf
from tensorflow.keras.layers import Concatenate, ZeroPadding2D, Conv2D, BatchNormalization, LeakyReLU, Add, Input, MaxPool2D
from tensorflow.keras.regularizers import l2


def darknet_conv(x, filters, size, strides=1, activation='leaky', batch_norm=True):
    def mish(inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

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
    if activation == 'mish':
        x = mish(x)
    elif activation == 'leaky':
        x = LeakyReLU(alpha=0.1)(x)
    return x


def darknet_residual(x, filters, activation='leaky'):
    prev = x
    x = darknet_conv(x, filters // 2, 1, activation=activation)
    x = darknet_conv(x, filters, 3, activation=activation)
    x = Add()([prev, x])
    return x


def darknet53(name=None):
    x = inputs = Input([None, None, 3])
    x = darknet_conv(x, 32, 3)
    x = darknet_conv(x, 64, 3, strides=2)
    for _ in range(1):
        x = darknet_residual(x, 64)

    x = darknet_conv(x, 128, 3, strides=2)
    for _ in range(2):
        x = darknet_residual(x, 128)

    x = darknet_conv(x, 256, 3, strides=2)
    for _ in range(8):
        x = darknet_residual(x, 256)
    route_1 = x

    x = darknet_conv(x, 512, 3, strides=2)
    for _ in range(8):
        x = darknet_residual(x, 512)
    route_2 = x

    x = darknet_conv(x, 1024, 3, strides=2)
    for _ in range(4):
        x = darknet_residual(x, 1024)

    return tf.keras.Model(inputs, (route_1, route_2, x), name=name)


def darknet_tiny_leaky(name=None):
    x = inputs = Input([None, None, 3])
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


if __name__ == '__main__':
    darknet = darknet53()
    darknet.summary()
