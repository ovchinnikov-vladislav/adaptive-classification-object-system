import colorsys
import tensorflow as tf
import numpy as np
from libs.yolo3.utils import transform_images, analyze_outputs, get_anchors
from libs.yolo3.layers import yolo_v3, yolo_v3_tiny


class YoloModel:
    def __init__(self, num_classes=80,
                 weights='./model_data/yolov3.tf',
                 classes='./model_data/coco_classes.txt',
                 anchors_path='./model_data/yolo_anchors.txt',
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

        boxes, scores, classes, nums = self.yolo.predict(img)
        boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]

        img = np.array(image)
        img, object_detection = analyze_outputs(img, (boxes, scores, classes, nums),
                                                self.class_names, self.colors)

        return img, object_detection
