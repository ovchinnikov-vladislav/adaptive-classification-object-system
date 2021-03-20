import cv2
import colorsys
import tensorflow as tf
import numpy as np
from bmstu.yolo3.utils import transform_images, draw_outputs

from bmstu.yolo3.layers import yolo_v3, yolo_v3_tiny


class ObjectDetection:
    def __init__(self, clazz, box, score):
        self.clazz = clazz
        self.box = box
        self.score = score

    def get_class(self):
        return self.clazz

    def get_box(self):
        return self.box

    def get_score(self):
        return self.score

    def __repr__(self):
        return f'ObjectDetection[class = {self.clazz}, box = {self.box}, score = {self.score}]'

    def __str__(self):
        return f'ObjectDetection[class = {self.clazz}, box = {self.box}, score = {self.score}]'


class YoloModel:
    def __init__(self, num_classes=80,
                 weights='./model_data/yolov3.tf',
                 classes='./model_data/coco_classes.txt',
                 size=416):
        self.yolo = yolo_v3(classes=num_classes)
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

        boxes, scores, classes, nums = self.yolo(img)

        boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]
        object_detection = []
        wh = np.flip(img.shape[0:2])
        for i in range(nums):
            x1, y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2, y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            object_detection.append(ObjectDetection(self.class_names[int(classes[i])], (x1, y1, x2, y2), scores[i]))

        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), self.class_names, self.colors)

        return img, object_detection
