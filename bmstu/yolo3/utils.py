from functools import reduce
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters"""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params
    anchors_tensor = backend.reshape(backend.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = backend.shape(feats)[1:3]  # height, width
    grid_y = backend.tile(backend.reshape(backend.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                          [1, grid_shape[1], 1, 1])
    grid_x = backend.tile(backend.reshape(backend.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                          [grid_shape[0], 1, 1, 1])
    grid = backend.concatenate([grid_x, grid_y])
    type_feats = feats.dtype
    grid = backend.cast(grid, type_feats)

    feats = backend.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust predictions to each spatial grid point and anchor size
    box_xy = (backend.sigmoid(feats[..., :2]) + grid) / backend.cast(grid_shape[::-1], backend.dtype(feats))
    box_wh = backend.exp(feats[..., 2:4]) * anchors_tensor / backend.cast(input_shape[::-1], backend.dtype(feats))
    box_confidence = backend.sigmoid(feats[..., 4:5])
    box_class_probs = backend.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = backend.cast(input_shape, backend.dtype(box_yx))
    image_shape = backend.cast(image_shape, backend.dtype(box_yx))
    new_shape = backend.round(image_shape * backend.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = backend.concatenate([
        box_mins[..., 0:1],   # y_min
        box_mins[..., 1:2],   # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]   # x_max
    ])

    # Scale boxes back to original image shape
    boxes *= backend.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """Process Convolutional layer output"""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = backend.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = backend.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = backend.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for i in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[i], anchors[anchor_mask[i]],
                                                    num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = backend.concatenate(boxes, axis=0)
    box_scores = backend.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = backend.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores,
                                                 max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = backend.gather(class_boxes, nms_index)
        class_box_scores = backend.gather(class_box_scores, nms_index)
        classes = backend.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = backend.concatenate(boxes_, axis=0)
    scores_ = backend.concatenate(scores_, axis=0)
    classes_ = backend.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def box_iou(b1, b2):
    # Expand dim to apply broadcasting
    b1 = backend.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting
    b2 = backend.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = backend.minimum(b1_mins, b2_mins)
    intersect_maxes = backend.minimum(b1_maxes, b2_maxes)
    intersect_wh = backend.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """Preprocess true boxes to training input format"""
    assert (true_boxes[..., 4] < num_classes), 'class id must be less then num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='float32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[i] for i in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[i][0], grid_shapes[i][1], len(anchor_mask[i]), 5 + num_classes), dtype='float32')
              for i in range(num_layers)]

    # Expand dim to apply broadcasting
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = - anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for i in range(num_layers):
                if n in anchor_mask[i]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[i][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[i][0]).astype('int32')
                    k = anchor_mask[i].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[i][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[i][b, j, i, k, 4] = 1
                    y_true[i][b, j, i, k, 5 + c] = 1
    return y_true


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image


def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, random=True,
                    max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    """random preprocessing for real-time data augmentation"""
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip:
            box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
