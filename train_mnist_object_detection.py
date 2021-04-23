from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from libs.yolo3.layers import (yolo_v3, yolo_v3_tiny)
from libs.yolo3.losses import yolo_loss
from libs.yolo3.losses import YoloLoss
from libs.yolo3.utils import get_anchors, data_generator_wrapper
from libs.datasets.wider_faces import wider_dataset_annotations
import tensorflow as tf
import numpy as np
import argparse


if __name__ == '__main__':
    size = 416
    batch_size = 1
    epochs = 30
    channels = 1
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    training_path = './'
    num_classes = len(class_names)

    input_shape = (size, size)

    anchors = get_anchors('./model_data/yolo_anchors.txt')
    masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    model_body = yolo_v3(anchors, size=size, channels=channels, classes=10, training=True)

    num_anchors = len(anchors)

    y_true_input = [tf.keras.layers.Input(shape=(size // {0: 32, 1: 16, 2: 8}[i], size // {0: 32, 1: 16, 2: 8}[i],
                                                 num_anchors // 3, num_classes + 5)) for i in range(3)]
    model_loss = tf.keras.layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                        arguments={'anchors': anchors, 'num_classes': num_classes,
                                                   'ignore_thresh': 0.5})([*model_body.output, *y_true_input])
    model = tf.keras.Model([model_body.input, *y_true_input], model_loss)

    ann_train_path, ann_test_path = './model_data/mnist_detection_train_annotation.txt', './model_data/mnist_detection_test_annotation.txt'

    with open(ann_train_path) as f:
        train_lines = f.readlines()
    with open(ann_test_path) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint(training_path + '/checkpoints/yolov3_train_{epoch}.tf', verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]

    history = model.fit(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator_wrapper(val_lines, batch_size, input_shape,
                                                               anchors, num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=epochs,
                        initial_epoch=0,
                        callbacks=callbacks)
    model.save_weights(f'{training_path}/yolov3_wider.tf')
