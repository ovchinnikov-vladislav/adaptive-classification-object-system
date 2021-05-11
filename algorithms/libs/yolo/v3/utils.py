from absl import logging
from PIL import ImageDraw, ImageFont, Image
import numpy as np
import tensorflow as tf
import cv2
import config
import base64
from io import BytesIO
import time

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


class ObjectDetection:
    def __init__(self, clazz, box, score, num=-1, img=None):
        self.clazz = clazz
        self.box = box
        self.score = score
        self.num = num
        self.img = img

    def get_class(self):
        return self.clazz

    def get_box(self):
        return self.box

    def get_score(self):
        return self.score

    def get_num(self):
        return self.num

    def get_img(self):
        return self.img

    def __repr__(self):
        return f'ObjectDetection[class = {self.clazz}, box = {self.box}, score = {self.score}, num = {self.num}]'

    def __str__(self):
        return f'ObjectDetection[class = {self.clazz}, box = {self.box}, score = {self.score}, num = {self.num}]'


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


def yolo_nms(outputs, yolo_max_boxes=30, yolo_iou_threshold=0.5, yolo_score_threshold=0.5):
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

    return boxes[0], scores[0], classes[0], valid_detections[0]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def analyze_detection_outputs(img, outputs, class_names, colors):
    boxes, scores, classes, nums = outputs
    wh = np.flip(img.shape[0:2])
    img = Image.fromarray(img)
    font = ImageFont.truetype(font=config.font_cv,
                              size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
    thickness = (img.size[0] + img.size[1]) // 300
    object_detection = []
    for i in range(nums):
        predicted_class = class_names[int(classes[i])]
        box = boxes[i]
        score = scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(img)
        label_size = draw.textsize(label, font)

        x1, y1 = tuple((np.array(box[0:2]) * wh).astype(np.int32))
        x2, y2 = tuple((np.array(box[2:4]) * wh).astype(np.int32))

        if y1 - label_size[1] >= 0:
            text_origin = np.array([x1, y1 - label_size[1]])
        else:
            text_origin = np.array([x1, y1 + 1])

        object_detection.append(ObjectDetection(class_names[int(classes[i])], (x1, y1, x2, y2), scores[i]))
        # My kingdom for a good redistributable image drawing library.
        color = colors[int(classes[i])]
        for j in range(thickness):
            draw.rectangle([x1 + j, y1 + j, x2 - j, y2 - j], outline=color)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.asarray(img), object_detection


def analyze_tracks_outputs(img, tracks, colors):
    before_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    before_img = Image.fromarray(before_img)
    img = Image.fromarray(img)
    font = ImageFont.truetype(font=config.font_cv,
                              size=np.floor((3e-2 * img.size[1] + 0.5) / 2).astype('int32'))
    draw = ImageDraw.Draw(img)

    thickness = 1
    object_detection = []
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        predicted_class = track.get_class()
        bbox = track.to_tlbr()

        label = f'{predicted_class} - №{track.track_id} - {track.score:.2f}'

        label_size = draw.textsize(label, font)

        x1, y1 = bbox[0], bbox[1]
        x2, y2 = bbox[2], bbox[3]

        if y1 - label_size[1] >= 0:
            text_origin = np.array([x1, y1 - label_size[1]])
        else:
            text_origin = np.array([x1, y1 + 5])

        img_crop = before_img.crop((int(x1) + (int(x1) - int(x2)) // 2,
                                    int(y1) + (int(y1) - int(y2)) // 2,
                                    int(x2) + (int(x2) - int(x1)) // 2,
                                    int(y2) + (int(y2) - int(y1)) // 2))
        buffered = BytesIO()
        img_crop.save(buffered, format="JPEG")
        img_bytes = base64.b64encode(buffered.getvalue())
        img_str = img_bytes.decode('utf-8')

        object_detection.append(ObjectDetection(predicted_class, (x1, y1, x2, y2), track.score,
                                                track.track_id, img_str))

        # My kingdom for a good redistributable image drawing library.
        color = colors[int(track.track_id) % len(colors)]
        color = [int(i * 255) for i in color]
        for j in range(thickness):
            draw.rectangle([x1 + j, y1 + j, x2 - j, y2 - j], outline=tuple(color))
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=tuple(color))
        draw.text(text_origin, label, fill=(255, 255, 255), font=font)
    del draw

    return np.asarray(img), object_detection


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true_input: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float64)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float64)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def convert_boxes(image, boxes):
    returned_boxes = []
    for box in boxes:
        box[0] = (box[0] * image.shape[1]).astype(int)
        box[1] = (box[1] * image.shape[0]).astype(int)
        box[2] = (box[2] * image.shape[1]).astype(int)
        box[3] = (box[3] * image.shape[0]).astype(int)
        box[2] = int(box[2] - box[0])
        box[3] = int(box[3] - box[1])
        box = box.astype(int)
        box = box.tolist()
        if box != [0, 0, 0, 0]:
            returned_boxes.append(box)
    return returned_boxes


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors, np.float32).reshape(-1, 2) / 416


def get_random_data(annotation_line, input_shape, random=False, max_boxes=20, jitter=.3,
                    hue=.1, sat=1.5, val=1.5, proc_img=True):
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])

    # resize image
    image_data = 0
    if proc_img:
        np_image = np.array(image)
        if np_image.ndim == 2:
            np_image = tf.expand_dims(np_image, -1)
        new_image = tf.image.resize(np_image, (w, h))
        image_data = np.array(new_image) / 255.

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box[:, [0, 2]] = box[:, [0, 2]] / iw
        box[:, [1, 3]] = box[:, [1, 3]] / ih
        box_data[:len(box)] = box

    return image_data, box_data


def data_generator(annotation_lines, batch_size, input_shape, anchors):
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
        y_true = transform_targets(box_data, anchors, np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]]), input_shape[0])
        yield image_data, y_true


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors)
