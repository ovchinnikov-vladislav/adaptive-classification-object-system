import cv2
import colorsys
import tensorflow as tf
import numpy as np
from libs.yolo4.utils import transform_images, analyze_outputs, load_weights
from libs.yolo4.layers import yolov4_neck, yolov4_head, nms, yolo_anchors, yolo_xyscale


class YoloModel:
    def __init__(self, num_classes=80,
                 weights='./model_data/yolov4.weights',
                 classes='./model_data/coco_classes.txt',
                 size=416):
        input_layer = tf.keras.layers.Input([size, size, 3])
        yolov4_output = yolov4_neck(input_layer, num_classes)
        self.yolo_model = tf.keras.models.Model(input_layer, yolov4_output)
        load_weights(self.yolo_model, weights)
        self.anchors = np.array(yolo_anchors).reshape((3, 3, 2))
        # Build inference model
        yolov4_output = yolov4_head(yolov4_output, num_classes, self.anchors, yolo_xyscale)
        # output: [boxes, scores, classes, valid_detections]
        self.yolo = tf.keras.models.Model(input_layer,
                                          nms(yolov4_output, [size, size, 3], num_classes,
                                              iou_threshold=0.413,
                                              score_threshold=0.3))

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
