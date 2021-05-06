import tensorflow as tf
import tensorflow.keras.backend as keras


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = keras.reshape(keras.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = keras.shape(feats)[1:3]  # height, width
    grid_y = keras.tile(keras.reshape(keras.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = keras.tile(keras.reshape(keras.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = keras.concatenate([grid_x, grid_y])
    grid = keras.cast(grid, keras.dtype(feats))

    feats = keras.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (keras.sigmoid(feats[..., :2]) + grid) / keras.cast(grid_shape[::-1], keras.dtype(feats))
    box_wh = keras.exp(feats[..., 2:4]) * anchors_tensor / keras.cast(input_shape[::-1], keras.dtype(feats))
    box_confidence = keras.sigmoid(feats[..., 4:5])
    box_class_probs = keras.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh

    return box_xy, box_wh, box_confidence, box_class_probs


def box_iou(b1, b2):
    """Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    """

    # Expand dim to apply broadcasting.
    b1 = keras.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = keras.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = keras.maximum(b1_mins, b2_mins)
    intersect_maxes = keras.minimum(b1_maxes, b2_maxes)
    intersect_wh = keras.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    """Return yolo_loss tensor
    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true_input: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    Returns
    -------
    loss: tensor, shape=(1,)
    """

    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = keras.cast(keras.shape(yolo_outputs[0])[1:3] * 32, keras.dtype(y_true[0]))
    grid_shapes = [keras.cast(keras.shape(yolo_outputs[l])[1:3], keras.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = keras.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = keras.cast(m, keras.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = keras.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = keras.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = keras.switch(object_mask, raw_true_wh, keras.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(keras.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = keras.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = keras.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, keras.cast(best_iou < ignore_thresh, keras.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = keras.expand_dims(ignore_mask, -1)

        # keras.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * keras.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                           from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * keras.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * keras.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * keras.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                        from_logits=True) * ignore_mask
        class_loss = object_mask * keras.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = keras.sum(xy_loss) / mf
        wh_loss = keras.sum(wh_loss) / mf
        confidence_loss = keras.sum(confidence_loss) / mf
        class_loss = keras.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss

    return loss

