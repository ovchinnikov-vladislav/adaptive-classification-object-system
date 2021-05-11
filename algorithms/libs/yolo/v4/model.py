import cv2
import colorsys
import numpy as np
from libs.yolo.v4.utils import analyze_outputs
from libs.yolo.v4.layers import yolo_v4


class YoloModel:
    def __init__(self, num_classes=80,
                 weights='./model_data/yolov4.tf',
                 classes='./model_data/coco_classes_ru.txt',
                 size=416):
        self.yolo = yolo_v4(size=size, classes=num_classes)
        self.yolo.load_weights(weights)

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
        img = cv2.resize(image, (self.size, self.size))
        img = img / 255.
        img = np.expand_dims(img, axis=0)
        boxes, scores, classes, nums = self.yolo.predict(img)
        boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]

        img = np.array(image)
        img, object_detection = analyze_outputs(img, (boxes, scores, classes, nums),
                                                self.class_names, self.colors)

        return img, object_detection
