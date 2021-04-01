import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv2D, BatchNormalization, MaxPooling2D, Concatenate, Input, Lambda, UpSampling2D)
from libs.yolo3.utils import yolo_boxes, yolo_nms
from libs.darknet53.layers import darknet_conv, cspdarknet53


yolo_anchors = np.array([(12, 16), (19, 36), (40, 28), (36, 75), (76, 55), (72, 146), (142, 110),
                         (192, 243), (459, 401)], np.float32) / 608
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
yolo_xyscale = [1.2, 1.1, 1.05]


def yolo_output(x_in, filters, anchors, classes, name=None):
    x = inputs = Input(x_in.shape[1:])
    x = darknet_conv(x, filters * 2, 3)
    x = darknet_conv(x, anchors * (classes + 5), 1, activation=None, batch_norm=False)
    x = Lambda(lambda inp: tf.reshape(inp, (-1, tf.shape(inp)[1], tf.shape(inp)[2],
                                            anchors, classes + 5)))(x)
    return tf.keras.Model(inputs, x, name=name)(x_in)


def yolo_v4(size=None, channels=3, anchors=yolo_anchors,
            masks=yolo_anchor_masks, classes=80, training=False):
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


def yolov4_head(yolo_neck_outputs, classes, anchors, xyscale):
    bbox0, object_probability0, class_probabilities0, pred_box0 = get_boxes(yolo_neck_outputs[0],
                                                                            anchors=anchors[0, :, :], classes=classes,
                                                                            grid_size=52, strides=8,
                                                                            xyscale=xyscale[0])
    bbox1, object_probability1, class_probabilities1, pred_box1 = get_boxes(yolo_neck_outputs[1],
                                                                            anchors=anchors[1, :, :], classes=classes,
                                                                            grid_size=26, strides=16,
                                                                            xyscale=xyscale[1])
    bbox2, object_probability2, class_probabilities2, pred_box2 = get_boxes(yolo_neck_outputs[2],
                                                                            anchors=anchors[2, :, :], classes=classes,
                                                                            grid_size=13, strides=32,
                                                                            xyscale=xyscale[2])
    x = [bbox0, object_probability0, class_probabilities0, pred_box0,
         bbox1, object_probability1, class_probabilities1, pred_box1,
         bbox2, object_probability2, class_probabilities2, pred_box2]

    return x


def get_boxes(pred, anchors, classes, grid_size, strides, xyscale):
    """

    :param pred:
    :param anchors:
    :param classes:
    :param grid_size:
    :param strides:
    :param xyscale:
    :return:
    """
    pred = tf.reshape(pred,
                      (tf.shape(pred)[0],
                       grid_size,
                       grid_size,
                       3,
                       5 + classes))  # (batch_size, grid_size, grid_size, 3, 5+classes)
    box_xy, box_wh, obj_prob, class_prob = tf.split(
        pred, (2, 2, 1, classes), axis=-1
    )  # (?, 52, 52, 3, 2) (?, 52, 52, 3, 2) (?, 52, 52, 3, 1) (?, 52, 52, 3, 80)

    box_xy = tf.sigmoid(box_xy)  # (?, 52, 52, 3, 2)
    obj_prob = tf.sigmoid(obj_prob)  # (?, 52, 52, 3, 1)
    class_prob = tf.sigmoid(class_prob)  # (?, 52, 52, 3, 80)
    pred_box_xywh = tf.concat((box_xy, box_wh), axis=-1)  # (?, 52, 52, 3, 4)

    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))  # (52, 52) (52, 52)
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # (52, 52, 1, 2)
    grid = tf.cast(grid, dtype=tf.float32)

    box_xy = ((box_xy * xyscale) - 0.5 * (xyscale - 1) + grid) * strides  # (?, 52, 52, 1, 4)

    box_wh = tf.exp(box_wh) * anchors  # (?, 52, 52, 3, 2)
    box_x1y1 = box_xy - box_wh / 2  # (?, 52, 52, 3, 2)
    box_x2y2 = box_xy + box_wh / 2  # (?, 52, 52, 3, 2)
    pred_box_x1y1x2y2 = tf.concat([box_x1y1, box_x2y2], axis=-1)  # (?, 52, 52, 3, 4)
    return pred_box_x1y1x2y2, obj_prob, class_prob, pred_box_xywh
    # pred_box_x1y1x2y2: absolute xy value


def nms(model_ouputs, input_shape, num_class, iou_threshold=0.413, score_threshold=0.3):
    """
    Apply Non-Maximum suppression
    ref: https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
    :param model_ouputs: yolo model model_ouputs
    :param input_shape: size of input image
    :return: nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
    """
    bs = tf.shape(model_ouputs[0])[0]
    boxes = tf.zeros((bs, 0, 4))
    confidence = tf.zeros((bs, 0, 1))
    class_probabilities = tf.zeros((bs, 0, num_class))

    for output_idx in range(0, len(model_ouputs), 4):
        output_xy = model_ouputs[output_idx]
        output_conf = model_ouputs[output_idx + 1]
        output_classes = model_ouputs[output_idx + 2]
        boxes = tf.concat([boxes, tf.reshape(output_xy, (bs, -1, 4))], axis=1)
        confidence = tf.concat([confidence, tf.reshape(output_conf, (bs, -1, 1))], axis=1)
        class_probabilities = tf.concat([class_probabilities, tf.reshape(output_classes, (bs, -1, num_class))], axis=1)

    scores = confidence * class_probabilities
    boxes = tf.expand_dims(boxes, axis=-2)
    boxes = boxes / input_shape[0]  # box normalization: relative img size
    print(f'nms iou: {iou_threshold} score: {score_threshold}')
    (nmsed_boxes,  # [bs, max_detections, 4]
     nmsed_scores,  # [bs, max_detections]
     nmsed_classes,  # [bs, max_detections]
     valid_detections  # [batch_size]
     ) = tf.image.combined_non_max_suppression(
        boxes=boxes,  # y1x1, y2x2 [0~1]
        scores=scores,
        max_output_size_per_class=100,
        max_total_size=100,  # max_boxes: Maximum nmsed_boxes in a single img.
        iou_threshold=iou_threshold,  # iou_threshold: Minimum overlap that counts as a valid detection.
        score_threshold=score_threshold,  # # Minimum confidence that counts as a valid detection.
    )
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections

