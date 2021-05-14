import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, UpSampling2D, ZeroPadding2D, Concatenate,
                                     Conv2D, BatchNormalization, LeakyReLU, Add, Lambda)
from tensorflow.keras.regularizers import l2
from libs.capsnets.layers.basic import PrimaryCapsule2D, Capsule


def conv(x, filters, size, down_sampling=False,
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


def residual(x, first_filters, second_filters, activation='leaky'):
    prev = x
    x = conv(x, first_filters, 1, activation=activation)
    x = conv(x, second_filters, 3, activation=activation)
    x = Add()([prev, x])
    return x


def block(x, filters, blocks):
    x = conv(x, filters, 3, down_sampling=True)
    for _ in range(blocks):
        x = residual(x, filters // 2, filters)
    return x


def conv_net(name=None, size=None, channels=3):
    x = inputs = Input([size, size, channels])
    x = conv(x, 32, 3)
    x = block(x, 64, 1)
    x = block(x, 128, 2)  # skip connection
    x = x_36 = block(x, 256, 8)  # skip connection
    x = x_61 = block(x, 512, 8)
    x = block(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def yolo_conv_input_tuple(x_in, filters):
    inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
    x, x_skip = inputs

    # concat with skip connection
    x = conv(x, filters, 1)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x_skip])

    return x, inputs


def yolo_conv(x_in, filters, name=None):
    if isinstance(x_in, tuple):
        x, inputs = yolo_conv_input_tuple(x_in, filters)
    else:
        x = inputs = Input(x_in.shape[1:])

    x = conv(x, filters, 1)
    x = conv(x, filters * 2, 3)
    x = conv(x, filters, 1)
    x = conv(x, filters * 2, 3)
    x = conv(x, filters, 1)
    return Model(inputs, x, name=name)(x_in)


def yolo_conv_tiny(x_in, filters, name=None):
    if isinstance(x_in, tuple):
        x, inputs = yolo_conv_input_tuple(x_in, filters)
    else:
        x = inputs = Input(x_in.shape[1:])
        x = conv(x, filters, 1)

    return Model(inputs, x, name=name)(x_in)


def yolo_output(x_in, filters, anchors, classes, name=None):
    x = inputs = Input(x_in.shape[1:])
    # x = conv(x, filters * 2, 3)
    # x = conv(x, anchors * (classes + 5), 1, batch_norm=False)

    x = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=3, strides=1, do_reshape=True)(x)
    capsules = Capsule(num_capsules=anchors * (classes + 5), dim_capsules=filters, routings=1)(x)

    x = Lambda(lambda inp: tf.reshape(inp, (-1, filters, filters, anchors, classes + 5)))(capsules)
    model = tf.keras.Model(inputs, x, name=name)
    # model.summary()
    return model(x_in)


def capsules_yolo(anchors, size, channels, classes, training=False):
    masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    x = inputs = Input([size, size, channels], name='input')
    x_36, x_61, x = conv_net(name='yolo_conv_net', size=size, channels=channels)(x)

    output_0 = yolo_output(x, 13, len(masks[0]), classes, name='yolo_output_0')

    output_1 = yolo_output(x, 26, len(masks[1]), classes, name='yolo_output_1')

    output_2 = yolo_output(x, 52, len(masks[2]), classes, name='yolo_output_2')

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    from libs.yolo.utils import yolo_boxes, yolo_nms
    boxes_0 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda inp: yolo_nms(inp, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


if __name__ == '__main__':
    import config
    from libs.yolo.utils import get_anchors
    anchors = get_anchors(config.yolo_caps_anchors)

    model = capsules_yolo(anchors=anchors, size=416, channels=3, classes=1, training=True)
    model.summary()
