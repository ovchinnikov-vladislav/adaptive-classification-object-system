import tensorflow as tf
import numpy as np
from PIL import ImageDraw, ImageFont, Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
import base64
from io import BytesIO
import colorsys
from libs.detection.yolo.v3.layers import yolo_v3, yolo_v3_tiny
from libs.detection.yolo.v4.layers import yolo_v4
import config
from libs.deepsort import preprocessing, nn_matching
from libs.deepsort.detection import Detection
from libs.deepsort.tracker import Tracker
from libs.deepsort.box_encoder import create_box_encoder
import matplotlib.pyplot as plt


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


class ObjectDetectionModel:
    def __init__(self, model='yolo3-person', weights=config.object_detection_weights, use_tracking=True, classes=None, size=416):
        if classes is None:
            classes = []

        self.class_names = classes
        num_classes = len(self.class_names)

        if num_classes == 0:
            raise Exception('num classes equals 0')

        self.size = size
        self.use_tracking = use_tracking

        if model == 'yolo3':
            anchors = get_anchors(config.yolo_v3_anchors)
            self.object_detection_model = yolo_v3(anchors, size=size, channels=3, classes=num_classes)
        elif model == 'yolo3-person':
            anchors = get_anchors(config.yolo_v3_anchors)
            self.object_detection_model = yolo_v3(anchors, size=size, channels=3, classes=num_classes)
        elif model == 'yolo4':
            anchors = get_anchors(config.yolo_v4_anchors)
            self.object_detection_model = yolo_v4(anchors, size=size, channels=3, classes=num_classes)
        elif model == 'yolo3_tiny':
            anchors = get_anchors(config.yolo_v3_tiny_anchors)
            self.object_detection_model = yolo_v3_tiny(anchors, size=size, channels=3, classes=num_classes)
        elif model == 'yolo2':
            raise Exception('undefined yolo2')
        elif model == 'yolo_caps':
            raise Exception('undefined yolo_caps')
        else:
            raise Exception(f'undefined {model}')

        self.object_detection_model.load_weights(weights).expect_partial()
        self.object_detection_model.predict(np.zeros((1, size, size, 3)))

        if use_tracking:
            max_cosine_distance = 0.5
            nn_budget = None
            self.nms_max_overlap = 1.0
            self.size = size

            self.encoder = create_box_encoder(config.deepsort_model, batch_size=1)
            self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
            self.tracker = Tracker(self.metric, num_classes=1)
        else:
            # Generate colors for drawing bounding boxes.
            hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
            np.random.seed(10101)
            np.random.shuffle(self.colors)
            np.random.seed(None)

    def clear_tracker(self):
        if self.use_tracking:
            self.tracker = Tracker(self.metric, num_classes=1)
        else:
            raise Exception('it is detection model')

    def detect_image(self, image):
        img = tf.expand_dims(image, 0)
        img = transform_images(img, self.size)

        boxes, scores, classes, nums = self.object_detection_model.predict(img)

        if not self.use_tracking:
            img = np.array(image)
            img, det_info = analyze_detection_outputs(img, (boxes, scores, classes, nums),
                                                      self.class_names, self.colors)
        else:
            names = []
            for i in range(len(classes)):
                names.append(self.class_names[int(classes[i])])
            names = np.array(names)
            converted_boxes = convert_boxes(image, boxes)
            features = self.encoder(image, converted_boxes)
            detections = [Detection(bbox, score, class_name, int(class_id), feature)
                          for bbox, score, class_name, class_id, feature
                          in zip(converted_boxes, scores, names, classes, features)
                          if class_name == 'человек' or class_name == 'person']

            # run non-maxima suppression
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxes, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # call the tracker
            self.tracker.predict()
            self.tracker.update(detections)

            img, det_info = analyze_tracks_outputs(image, self.tracker.tracks, colors)

        return img, det_info


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for i in model.layers:
            freeze_all(i, frozen)


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true_input: (n, boxes, (x1, y1, x2, y2, class, best_anchor))
    n = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    y_true_out = tf.zeros((n, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(n):
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
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
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


def rand(a=0.0, b=1.0):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20,
                    jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
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


IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}


def parse_tfrecord(tfrecord, class_table, size, yolo_max_boxes=30):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

    paddings = [[0, yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def load_tfrecord_dataset(file_pattern, class_file, size=416):
    line_number = -1  # TODO: use tf.lookup.TextFileIndex.line_number
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, line_number, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))


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

        img_crop = before_img.crop((int(x1), int(y1), int(x2), int(y2)))

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


def yolo_nms(outputs, yolo_max_boxes=18, yolo_iou_threshold=0.5, yolo_score_threshold=0.5, num_classes=80):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    if num_classes > 1:
        scores = confidence * class_probs
    else:
        scores = confidence

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
    )

    return boxes[0], scores[0], classes[0], valid_detections[0]
