import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from libs.yolo3.layers import yolo_v3, yolo_v3_tiny, yolo_anchors, yolo_anchor_masks, yolo_tiny_anchors, \
    yolo_tiny_anchor_masks
from libs.yolo3.losses import YoloLoss
from libs.yolo3.utils import freeze_all, transform_images, transform_targets
import tensorflow_datasets as tfds
from libs.datasets.wider_faces import wider_dataset_annotations

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--tiny', default=False, type=bool, help='yolov3 or yolov3-tiny')
parser.add_argument('--weights', default='./model_data/yolov3.tf', help='path to weights file')
parser.add_argument('--classes', default='./model_data/coco_classes.txt', help='path to classes file')
parser.add_argument('--dataset_path', default='D:\\tensorflow_datasets', help='path to download dataset')

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5,
                    proc_img=True):
    '''random preprocessing for real-time data augmentation'''
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
            if len(box) > max_boxes: box = box[:max_boxes]
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
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
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
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format
    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value
    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
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
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


if __name__ == '__main__':
    args = parser.parse_args()
    size = 416
    batch_size = 10

    if args.tiny:
        model = yolo_v3_tiny(size, training=True, classes=1)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = yolo_v3(size, training=True, classes=1)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    annotation_train_path, annotation_test_path, annotation_val_path = \
        wider_dataset_annotations(args.dataset_path, is_prepare_annotation=True)

    # # Configure the model for transfer learning
    # if FLAGS.transfer == 'none':
    #     pass  # Nothing to do
    # elif FLAGS.transfer in ['darknet', 'no_output']:
    #     # Darknet transfer is a special case that works
    #     # with incompatible number of classes
    #
    #     # reset top layers
    #     if FLAGS.tiny:
    #         model_pretrained = YoloV3Tiny(
    #             FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
    #     else:
    #         model_pretrained = YoloV3(
    #             FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
    #     model_pretrained.load_weights(FLAGS.weights)
    #
    #     if FLAGS.transfer == 'darknet':
    #         model.get_layer('yolo_darknet').set_weights(
    #             model_pretrained.get_layer('yolo_darknet').get_weights())
    #         freeze_all(model.get_layer('yolo_darknet'))
    #
    #     elif FLAGS.transfer == 'no_output':
    #         for l in model.layers:
    #             if not l.name.startswith('yolo_output'):
    #                 l.set_weights(model_pretrained.get_layer(
    #                     l.name).get_weights())
    #                 freeze_all(l)
    #
    # else:
    #     # All other transfer require matching classes
    #     model.load_weights(FLAGS.weights)
    #     if FLAGS.transfer == 'fine_tune':
    #         # freeze darknet and fine tune other layers
    #         darknet = model.get_layer('yolo_darknet')
    #         freeze_all(darknet)
    #     elif FLAGS.transfer == 'frozen':
    #         # freeze everything
    #         freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    loss = [YoloLoss(anchors[mask], classes=1) for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss)
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf', verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]

    with open(annotation_train_path) as f:
        train_lines = f.readlines()
    with open(annotation_val_path) as f:
        val_lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.shuffle(val_lines)
    num_val = len(val_lines)
    num_train = len(train_lines)
    input_shape = (416, 416)
    num_classes = 1

    history = model.fit(data_generator(train_lines, batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(val_lines, batch_size, input_shape, anchors,
                                                       num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=callbacks)
