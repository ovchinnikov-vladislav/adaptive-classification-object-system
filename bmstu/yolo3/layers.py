from functools import wraps
from tensorflow.keras import backend
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from bmstu.yolo3.utils import compose
import numpy as np
import tensorflow as tf


class DarknetConv2D(Layer):
    def __init__(self, *args, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(DarknetConv2D, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4),
                               'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
        darknet_conv_kwargs.update(kwargs)
        self.darknet_conv_2d = Conv2D(*args, **darknet_conv_kwargs)

    def call(self, inputs, **kwargs):
        return self.darknet_conv_2d(inputs)

    def get_config(self):
        return self.darknet_conv_2d.get_config()


class DarknetConv2DBNLeakyRelu(Layer):
    def __init__(self, *args, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(DarknetConv2DBNLeakyRelu, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        no_bias_kwargs = {'use_bias': False}
        no_bias_kwargs.update(kwargs)
        self.darknet_conv_2d = DarknetConv2D(*args, **no_bias_kwargs)

    def call(self, inputs, **kwargs):
        return compose(self.darknet_conv_2d, BatchNormalization(), LeakyReLU(alpha=0.1))

    def get_config(self):
        return self.darknet_conv_2d.get_config()


def residual_block_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2DBNLeakyRelu(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(DarknetConv2DBNLeakyRelu(num_filters // 2, (1, 1)),
                    DarknetConv2DBNLeakyRelu(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    x = DarknetConv2DBNLeakyRelu(32, (3, 3))(x)
    x = residual_block_body(x, 64, 1)
    x = residual_block_body(x, 128, 2)
    x = residual_block_body(x, 256, 8)
    x = residual_block_body(x, 512, 8)
    x = residual_block_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    x = compose(DarknetConv2DBNLeakyRelu(num_filters, (1, 1)),
                DarknetConv2DBNLeakyRelu(num_filters * 2, (3, 3)),
                DarknetConv2DBNLeakyRelu(num_filters, (1, 2)),
                DarknetConv2DBNLeakyRelu(num_filters * 2, (3, 3)),
                DarknetConv2DBNLeakyRelu(num_filters, (1, 1)))(x)

    y = compose(DarknetConv2DBNLeakyRelu(num_filters * 2, (3, 3)),
                DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(DarknetConv2DBNLeakyRelu(256, (1, 1)),
                UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(DarknetConv2DBNLeakyRelu(128, (1, 1)),
                UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])