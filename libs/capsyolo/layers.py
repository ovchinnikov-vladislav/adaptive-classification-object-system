import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, UpSampling2D, ZeroPadding2D, Concatenate,
                                     Conv2D, BatchNormalization, LeakyReLU, Add, Lambda)
from tensorflow.keras.regularizers import l2
from libs.capsnets.layers.residual import block_caps

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90),
                         (156, 198), (373, 326)], np.float32) / 416
yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)], np.float32) / 416


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * tf.convert_to_tensor(anchors)

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes, yolo_max_boxes=30, yolo_iou_threshold=0.5, yolo_score_threshold=0.5):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections


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


def conv_net(name=None, channels=3):
    x = inputs = Input([None, None, channels])
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


def yolo_output(x_in, anchors, classes, name=None):
    x = inputs = Input(x_in.shape[1:])
    x, capsules = block_caps(x, routings=1, classes=anchors * (classes + 5), num_capsule=8, dim_capsule=8)

    x = Lambda(lambda inp: tf.reshape(inp, (-1, tf.shape(inp)[1], tf.shape(inp)[2], anchors, classes + 5)))(capsules)
    return tf.keras.Model(inputs, x, name=name)(x_in)


def capsules_yolo(anchors, size, channels, classes, training=False):
    masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    x = inputs = Input([size, size, channels], name='input')
    x_36, x_61, x = conv_net(name='yolo_conv_net', channels=channels)(x)

    x = yolo_conv(x, 512, name='yolo_conv_0')
    output_0 = yolo_output(x, len(masks[0]), classes, name='yolo_output_0')

    x = yolo_conv((x, x_61), 256, name='yolo_conv_1')
    output_1 = yolo_output(x, len(masks[1]), classes, name='yolo_output_1')

    x = yolo_conv((x, x_36), 128, name='yolo_conv_2')
    output_2 = yolo_output(x, len(masks[2]), classes, name='yolo_output_2')

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda inp: yolo_nms(inp, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


if __name__ == '__main__':
    model = capsules_yolo(anchors=yolo_anchors, size=416, channels=1, classes=10, training=True)
    model.summary()
