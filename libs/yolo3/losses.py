import tensorflow as tf
from tensorflow.keras import backend as keras
from tensorflow.keras.losses import (binary_crossentropy, sparse_categorical_crossentropy, categorical_crossentropy)
from libs.yolo3.utils import yolo_boxes
import math


def softmax_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)

    # Calculate Cross Entropy
    cross_entropy = -y_true * tf.math.log(y_pred)

    # Calculate Focal Loss
    return alpha * tf.pow(1 - y_pred, gamma) * cross_entropy


def sigmoid_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    sigmoid_loss = keras.binary_crossentropy(y_true, y_pred, from_logits=True)

    pred_prob = tf.sigmoid(y_pred)
    p_t = ((y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob)))
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

    return modulating_factor * alpha_weight_factor * sigmoid_loss


def box_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def box_giou(b_true, b_pred):
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh/2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh/2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = keras.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = keras.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = keras.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + keras.epsilon())

    # get enclosed area
    enclose_mins = keras.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = keras.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = keras.maximum(enclose_maxes - enclose_mins, 0.0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    # calculate GIoU, add epsilon in denominator to avoid dividing by 0
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + keras.epsilon())
    giou = keras.expand_dims(giou, -1)

    return giou


def box_diou(b_true, b_pred, use_ciou=False):
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh/2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh/2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = keras.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = keras.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = keras.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + keras.epsilon())

    # box center distance
    center_distance = keras.sum(keras.square(b_true_xy - b_pred_xy), axis=-1)
    # get enclosed area
    enclose_mins = keras.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = keras.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = keras.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = keras.sum(keras.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * center_distance / (enclose_diagonal + keras.epsilon())

    if use_ciou:
        v = 4 * keras.square(tf.math.atan2(
            b_true_wh[..., 0], b_true_wh[..., 1]) - tf.math.atan2(
            b_pred_wh[..., 0], b_pred_wh[..., 1])) / (math.pi * math.pi)

        v = v * tf.stop_gradient(b_pred_wh[..., 0] * b_pred_wh[..., 0] + b_pred_wh[..., 1] * b_pred_wh[..., 1])

        alpha = v / (1.0 - iou + v)
        diou = diou - alpha*v

    diou = keras.expand_dims(diou, -1)
    return diou


def _smooth_labels(y_true, label_smoothing):
    label_smoothing = keras.constant(label_smoothing, dtype=keras.floatx())
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing


def yolo_loss(anchors, classes=80, ignore_thresh=0.5, label_smoothing=0, use_focal_loss=False,
              use_focal_obj_loss=False, use_softmax_loss=False, use_giou_loss=False, use_diou_loss=True):
    def calc_yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(lambda x: tf.reduce_max(
            box_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1),
                             (pred_box, true_box, obj_mask), tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        if label_smoothing:
            true_class_idx = _smooth_labels(true_class_idx, label_smoothing)
            true_obj = _smooth_labels(true_obj, label_smoothing)

        # 5. calculate all losses
        if use_focal_obj_loss:
            obj_loss = sigmoid_focal_loss(true_obj, pred_obj)
        else:
            obj_loss = binary_crossentropy(true_obj, pred_obj)
            obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

        if use_focal_loss:
            if use_softmax_loss:
                class_loss = softmax_focal_loss(true_class_idx, pred_class)
            else:
                class_loss = sigmoid_focal_loss(true_class_idx, pred_class)
        else:
            if use_softmax_loss:
                class_loss = obj_mask * tf.expand_dims(
                    categorical_crossentropy(true_class_idx, pred_class, from_logits=True))
            else:
                class_loss = obj_mask * binary_crossentropy(true_class_idx, pred_class, from_logits=True)

        if use_giou_loss:
            giou = box_giou(true_box, pred_box)
            giou_loss = obj_mask * box_loss_scale * (1 - giou)
            giou_loss = tf.reduce_sum(giou_loss, axis=(1, 2, 3))
            location_loss = giou_loss
        elif use_diou_loss:
            diou = box_diou(true_box, pred_box)
            diou_loss = obj_mask * box_loss_scale * (1 - diou)
            diou_loss = tf.reduce_sum(diou_loss, axis=(1, 2, 3))
            location_loss = diou_loss
        else:
            xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
            wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
            xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
            wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
            location_loss = xy_loss + wh_loss

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        loss = location_loss + obj_loss + class_loss
        return loss, location_loss, obj_loss, class_loss

    return calc_yolo_loss
