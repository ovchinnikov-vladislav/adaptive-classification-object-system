import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv2D, BatchNormalization, MaxPooling2D, Concatenate, Input, Lambda, UpSampling2D)
from libs.yolo3.utils import yolo_boxes, yolo_nms
from libs.darknet53.layers import darknet_conv_leaky, darknet_mish

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90),
                         (156, 198), (373, 326)], np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)], np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def yolo_conv_input_tuple(x_in, filters):
    inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
    x, x_skip = inputs

    # concat with skip connection
    x = darknet_conv_leaky(x, filters, 1)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x_skip])

    return x, inputs


def yolo_conv(x_in, filters, name=None):
    if isinstance(x_in, tuple):
        x, inputs = yolo_conv_input_tuple(x_in, filters)
    else:
        x = inputs = Input(x_in.shape[1:])

    x = darknet_conv_leaky(x, filters, 1)
    x = darknet_conv_leaky(x, filters * 2, 3)
    x = darknet_conv_leaky(x, filters, 1)
    x = darknet_conv_leaky(x, filters * 2, 3)
    x = darknet_conv_leaky(x, filters, 1)
    return Model(inputs, x, name=name)(x_in)


def yolo_conv_tiny(x_in, filters, name=None):
    if isinstance(x_in, tuple):
        x, inputs = yolo_conv_input_tuple(x_in, filters)
    else:
        x = inputs = Input(x_in.shape[1:])
        x = darknet_conv_leaky(x, filters, 1)

    return Model(inputs, x, name=name)(x_in)


def yolo_output(x_in, filters, anchors, classes, name=None):
    x = inputs = Input(x_in.shape[1:])
    x = darknet_conv_leaky(x, filters * 2, 3)
    x = darknet_conv_leaky(x, anchors * (classes + 5), 1, batch_norm=False)
    x = Lambda(lambda inp: tf.reshape(inp, (-1, tf.shape(inp)[1], tf.shape(inp)[2],
                                            anchors, classes + 5)))(x)
    return tf.keras.Model(inputs, x, name=name)(x_in)


def yolo_v4(size=None, channels=3, anchors=yolo_anchors,
            masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    darknet = darknet_mish(name='yolo_darknet')
    print(f'131: {darknet.layers[131].name}')
    print(f'204: {darknet.layers[204].name}')
    x76, x38, x = darknet(x)
    print(f'x76: {x76}')
    print(f'x38: {x38}')

    # 19x19 head
    y19 = darknet_conv_leaky(x, 512, 1)
    y19 = darknet_conv_leaky(y19, 1024, 3)
    y19 = darknet_conv_leaky(y19, 512, 1)
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(y19)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(y19)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(y19)
    y19 = Concatenate()([maxpool1, maxpool2, maxpool3, y19])
    y19 = darknet_conv_leaky(y19, 512, 1)
    y19 = darknet_conv_leaky(y19, 1024, 3)
    y19 = darknet_conv_leaky(y19, 512, 1)

    y19_upsample = darknet_conv_leaky(y19, 256, 1)
    y19_upsample = UpSampling2D(2)(y19_upsample)

    # 38x38 head
    y38 = darknet_conv_leaky(x38, 256, 1)
    y38 = Concatenate()([y38, y19_upsample])
    y38 = darknet_conv_leaky(y38, 256, 1)
    y38 = darknet_conv_leaky(y38, 512, 3)
    y38 = darknet_conv_leaky(y38, 256, 1)
    y38 = darknet_conv_leaky(y38, 512, 3)
    y38 = darknet_conv_leaky(y38, 256, 1)

    y38_upsample = darknet_conv_leaky(y38, 128, 1)
    y38_upsample = UpSampling2D(2)(y38_upsample)

    # 76x76 head
    y76 = darknet_conv_leaky(x76, 128, 1)
    y76 = Concatenate()([y76, y38_upsample])
    y76 = darknet_conv_leaky(y76, 128, 1)
    y76 = darknet_conv_leaky(y76, 256, 3)
    y76 = darknet_conv_leaky(y76, 128, 1)
    y76 = darknet_conv_leaky(y76, 256, 3)
    y76 = darknet_conv_leaky(y76, 128, 1)

    # 76x76 output
    y76_output = darknet_conv_leaky(y76, 256, 3)
    output_2 = darknet_conv_leaky(y76_output, len(yolo_anchors) * (classes + 5), 1, batch_norm=False)

    # 38x38 output
    y76_downsample = darknet_conv_leaky(y76, 256, 3, strides=(2, 2))
    y38 = Concatenate()([y76_downsample, y38])
    y38 = darknet_conv_leaky(y38, 256, 1)
    y38 = darknet_conv_leaky(y38, 512, 3)
    y38 = darknet_conv_leaky(y38, 256, 1)
    y38 = darknet_conv_leaky(y38, 512, 3)
    y38 = darknet_conv_leaky(y38, 256, 1)

    y38_output = darknet_conv_leaky(y38, 512, 3)
    output_1 = darknet_conv_leaky(y38_output, len(yolo_anchors) * (classes + 5), 1, batch_norm=False)

    # 19x19 output
    y38_downsample = darknet_conv_leaky(y38, 512, 3, strides=(2, 2))
    y19 = Concatenate()([y38_downsample, y19])
    y19 = darknet_conv_leaky(y19, 512, 1)
    y19 = darknet_conv_leaky(y19, 1024, 3)
    y19 = darknet_conv_leaky(y19, 512, 1)
    y19 = darknet_conv_leaky(y19, 1024, 3)
    y19 = darknet_conv_leaky(y19, 512, 1)

    y19_output = darknet_conv_leaky(y19, 1024, 3)
    output_0 = darknet_conv_leaky(y19_output, len(yolo_anchors) * (classes + 5), 1, batch_norm=False)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov4')

    boxes_0 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda inp: yolo_nms(inp, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov4')


