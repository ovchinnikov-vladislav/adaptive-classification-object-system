import colorsys
import os
from timeit import default_timer as timer
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw

from bmstu.yolo3.utils import yolo_eval, letterbox_image
from bmstu.yolo3.layers import yolo_body

tf.compat.v1.disable_eager_execution()

class YoloModel:

    def __init__(self, model_path="model_data/yolo.dat",
                 anchors_path="model_data/yolo_anchors.txt",
                 classes_path="model_data/coco_classes.txt",
                 score=0.3,
                 iou=0.45,
                 model_image_size=(416, 416)):
        self.model_path = model_path
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.score = score
        self.iou = iou
        self.model_image_size = model_image_size
        self.class_names = self._get_class(classes_path)
        self.anchors = self._get_anchors(anchors_path)

        # try:
        #     # TODO: не моя модель, надо переписать конвертер
        #     self.yolo_model = load_model(model_path, compile=False)
        # except:
        # TODO: моя модель, не сработает по причине не корректно заданных весов
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        model_path = os.path.expanduser(self.model_path)
        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
        self.yolo_model.load_weights(model_path)
        self.yolo_model.summary()

        # self.model = tf.keras.Model(self.yolo_model.input, tf.keras.layers.Lambda(
        #     lambda x: yolo_eval(x, self.anchors, len(self.class_names),
        #                         (480, 640), score_threshold=self.score, iou_threshold=self.iou))(self.yolo_model.output))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.result_predictions = self.generate()

    @staticmethod
    def _get_class(classes_path):
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as file:
            class_names = file.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @staticmethod
    def _get_anchors(anchors_path):
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as file:
            anchors = file.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        #   assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.input_image_shape = backend.placeholder(shape=(2,))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, len(self.class_names),
                                           self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)

        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        print(image.size, image.size)
        # out_boxes, out_scores, out_classes = self.generate()(image_data, [image.size[1], image.size[0]])
        # out_boxes, out_scores, out_classes = self.model.predict(image_data)
        out_boxes, out_scores, out_classes = self.sess.run(self.result_predictions,
                                                           feed_dict={
                                                               self.yolo_model.input: image_data,
                                                               self.input_image_shape: [image.size[1], image.size[0]]
                                                           })

        #  print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/Roboto-Regular.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            if score < 0.7:
                continue

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #   print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print(end - start)
        return image
