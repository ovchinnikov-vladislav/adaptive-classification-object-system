import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Input, Lambda, UpSampling2D
from libs.yolo3.utils import yolo_boxes, yolo_nms
from libs.darknet53.layers import darknet_conv, cspdarknet53


def yolo_output(x_in, filters, anchors, classes, name=None):
    x = inputs = Input(x_in.shape[1:])
    x = darknet_conv(x, filters * 2, 3)
    x = darknet_conv(x, anchors * (classes + 5), 1, activation=None, batch_norm=False)
    x = Lambda(lambda inp: tf.reshape(inp, (-1, tf.shape(inp)[1], tf.shape(inp)[2],
                                            anchors, classes + 5)))(x)
    return tf.keras.Model(inputs, x, name=name)(x_in)


def yolo_v4(size=None, channels=3, classes=80, training=False):
    anchors = np.array([(12, 16), (19, 36), (40, 28), (36, 75), (76, 55), (72, 146),
                        (142, 110), (192, 243), (459, 401)], np.float32) / 416
    masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    x = inputs = Input([size, size, channels], name='input')

    route0, route1, route2 = cspdarknet53(name='yolo_darknet')(x)

    route_input = route2
    x = darknet_conv(route2, 256, 1)
    x = UpSampling2D()(x)
    route1 = darknet_conv(route1, 256, 1)
    x = Concatenate()([route1, x])

    x = darknet_conv(x, 256, 1)
    x = darknet_conv(x, 512, 3)
    x = darknet_conv(x, 256, 1)
    x = darknet_conv(x, 512, 3)
    x = darknet_conv(x, 256, 1)

    route1 = x
    x = darknet_conv(x, 128, 1)
    x = UpSampling2D()(x)
    route0 = darknet_conv(route0, 128, 1)
    x = Concatenate()([route0, x])

    x = darknet_conv(x, 128, 1)
    x = darknet_conv(x, 256, 3)
    x = darknet_conv(x, 128, 1)
    x = darknet_conv(x, 256, 3)
    x = darknet_conv(x, 128, 1)

    route0 = x
    output_2 = yolo_output(x, 128, len(masks[2]), classes, name='yolo_output_2')

    x = darknet_conv(route0, 256, 3, down_sampling=True)
    x = Concatenate()([x, route1])

    x = darknet_conv(x, 256, 1)
    x = darknet_conv(x, 512, 3)
    x = darknet_conv(x, 256, 1)
    x = darknet_conv(x, 512, 3)
    x = darknet_conv(x, 256, 1)

    route1 = x
    output_1 = yolo_output(x, 256, len(masks[1]), classes, name='yolo_output_1')

    x = darknet_conv(route1, 512, 3, down_sampling=True)
    x = Concatenate()([x, route_input])

    x = darknet_conv(x, 512, 1)
    x = darknet_conv(x, 1024, 3)
    x = darknet_conv(x, 512, 1)
    x = darknet_conv(x, 1024, 3)
    x = darknet_conv(x, 512, 1)

    output_0 = yolo_output(x, 512, len(masks[0]), classes, name='yolo_output_0')

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


