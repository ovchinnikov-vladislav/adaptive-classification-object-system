from tensorflow.keras import backend
from bmstu.yolo3.utils import yolo_head, box_iou
import tensorflow as tf


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = backend.cast(backend.shape(yolo_outputs[0])[1:3] * 32, backend.dtype(y_true[0]))
    grid_shapes = [backend.cast(backend.shape(yolo_outputs[i])[1:3], backend.dtype(y_true[0]))
                   for i in range(num_layers)]
    loss = 0
    m = backend.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = backend.cast(m, backend.dtype(yolo_outputs[0]))

    for i in range(num_layers):
        object_mask = y_true[i][..., 4:5]
        true_class_probs = y_true[i][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[i], anchors[anchor_mask[i]],
                                                     num_classes, input_shape, calc_loss=True)
        pred_box = backend.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss
        raw_true_xy = y_true[i][..., :2] * grid_shapes[i][::-1] - grid
        raw_true_wh = backend.log(y_true[i][..., 2:4] / anchors[anchor_mask[i]] * input_shape[::-1])
        raw_true_wh = backend.switch(object_mask, raw_true_wh, backend.zeros_like(raw_true_wh))
        box_loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(backend.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = backend.cast(object_mask, 'bool')

        def loop_body(b, ign_mask):
            true_box = tf.boolean_mask(y_true[i][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = backend.max(iou, axis=-1)
            ign_mask = ign_mask.write(b, backend.cast(best_iou < ignore_thresh, backend.dtype(true_box)))
            return b + 1, ign_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = backend.expand_dims(ignore_mask, -1)

        # backend.binary_crossentropy is helpful to avoid exp overflow
        xy_loss = object_mask * box_loss_scale * backend.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                             from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * backend.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * backend.binary_crossentropy(
            object_mask, raw_pred[..., 4:5], from_logits=True) + (1 - object_mask) * backend.binary_crossentropy(
            object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * backend.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = backend.sum(xy_loss) / mf
        wh_loss = backend.sum(wh_loss) / mf
        confidence_loss = backend.sum(confidence_loss) / mf
        class_loss = backend.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss

    return loss
