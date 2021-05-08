import colorsys
import tensorflow as tf
import numpy as np
from libs.yolo3.utils import transform_images, convert_boxes, analyze_outputs, get_anchors
from libs.yolo3.layers import yolo_v3
import config
import time
from PIL import Image, ImageFont, ImageDraw
from libs.deepsort import preprocessing, nn_matching
from libs.deepsort.detection import Detection
from libs.deepsort.tracker import Tracker
from libs.deepsort.box_encoder import create_box_encoder
import matplotlib.pyplot as plt


class YoloDetectionModel:
    def __init__(self, num_classes=80,
                 weights=config.yolo_v3_weights,
                 classes=config.coco_classes_ru,
                 anchors_path=config.yolo_anchors,
                 size=416):
        anchors = get_anchors(anchors_path)
        self.yolo = yolo_v3(anchors, size=size, channels=3, classes=num_classes)
        self.yolo.make_predict_function()
        self.yolo.load_weights(weights).expect_partial()
        self.class_names = [c.strip() for c in open(classes).readlines()]
        self.size = size

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

    def detect_image(self, image):
        img = tf.expand_dims(image, 0)
        img = transform_images(img, self.size)

        t1 = time.time()
        boxes, scores, classes, nums = self.yolo.predict(img)
        t2 = time.time()
        print('Только время детекции: ' + str(int(1000 * (t2 - t1))))
        boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]

        img = np.array(image)
        t1 = time.time()
        img, object_detection = analyze_outputs(img, (boxes, scores, classes, nums),
                                                self.class_names, self.colors)
        t2 = time.time()
        print('Время разбора детекции: ' + str(int(1000 * (t2 - t1))))

        return img, object_detection


def output(img, tracks, colors):
    img = Image.fromarray(img)
    font = ImageFont.truetype(font=config.font_cv,
                              size=np.floor((3e-2 * img.size[1] + 0.5) / 2).astype('int32'))
    thickness = 1
    detection_obj = []
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        predicted_class = track.get_class()
        bbox = track.to_tlbr()

        label = f'{predicted_class} - №{track.track_id} - {track.score:.2f}'
        draw = ImageDraw.Draw(img)
        label_size = draw.textsize(label, font)

        x1, y1 = bbox[0], bbox[1]
        x2, y2 = bbox[2], bbox[3]

        if y1 - label_size[1] >= 0:
            text_origin = np.array([x1, y1 - label_size[1]])
        else:
            text_origin = np.array([x1, y1 + 5])

        # My kingdom for a good redistributable image drawing library.
        color = colors[int(track.track_id) % len(colors)]
        color = [int(i * 255) for i in color]
        for j in range(thickness):
            draw.rectangle([x1 + j, y1 + j, x2 - j, y2 - j], outline=tuple(color))
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=tuple(color))
        draw.text(text_origin, label, fill=(255, 255, 255), font=font)
        del draw

    return np.asarray(img), detection_obj


class YoloTrackingModel:
    def __init__(self, num_classes=80,
                 deepsort_weights=config.deepsort_model,
                 yolo_weights=config.yolo_v3_weights,
                 classes=config.coco_classes_ru,
                 anchors_path=config.yolo_anchors,
                 size=416):
        max_cosine_distance = 0.5
        nn_budget = None
        self.nms_max_overlap = 1.0
        self.size = size

        self.encoder = create_box_encoder(deepsort_weights, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)

        anchors = get_anchors(anchors_path)
        self.yolo = yolo_v3(anchors, size, channels=3, classes=num_classes)
        self.yolo.load_weights(yolo_weights)

        self.class_names = [c.strip() for c in open(classes).readlines()]

    def clear_tracker(self):
        self.tracker = Tracker(self.metric)

    def detect_image(self, image):
        img_in = tf.expand_dims(image, 0)
        img_in = transform_images(img_in, self.size)

        boxes, scores, classes, nums = self.yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(self.class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(image, boxes[0])
        features = self.encoder(image, converted_boxes)
        detections = [Detection(bbox, score, class_name, int(class_id), feature)
                      for bbox, score, class_name, class_id, feature
                      in zip(converted_boxes, scores[0], names, classes, features)
                      if class_name == 'человек']

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        img, det_info = output(image, self.tracker.tracks, colors)

        return img, det_info
