from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from bmstu.yolo3.layers import yolo_body
from bmstu.yolo3.losses import yolo_loss
import numpy as np


def get_classes(classes_path):
    with open(classes_path) as file:
        class_names = file.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as file:
        anchors = file.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained, freeze_body=2, weights_path=''):
    h, w = input_shape
    image_input = Input(shape=(w, h, 3))
    num_anchors = len(anchors)

    y_true = [
        Input(shape=(h // {0: 32, 1: 16, 2: 8}[i], w // {0: 32, 1: 16, 2: 8}[i], num_anchors // 3, num_classes + 5))
        for i in range(3)
    ]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if freeze_body in [1, 2]:
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False

    model_loss = Lambda(
        yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model
