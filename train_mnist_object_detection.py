from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from libs.yolo3.layers import (yolo_v3, yolo_v3_tiny)
from libs.yolo3.losses import yolo_loss
from libs.yolo3.utils import get_anchors, data_generator_wrapper
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

    ann_train_path, ann_test_path = './model_data/mnist_detection_train_annotation.txt', \
                                    './model_data/mnist_detection_test_annotation.txt'

    with open(ann_train_path) as f:
        train_lines = f.readlines()
    with open(ann_test_path) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    anchors = get_anchors('./model_data/yolo_anchors.txt')
    masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    model = yolo_v3(anchors, size=size, channels=channels, classes=10, training=True)

    grid_size = size // 32
    shape_input_image = (None, size, size, channels)
    shape_output_0_image = (None, grid_size, grid_size, channels, 6)
    shape_output_1_image = (None, grid_size * 2, grid_size * 2, channels, 6)
    shape_output_2_image = (None, grid_size * 4, grid_size * 4, channels, 6)

    dataset = tf.data.Dataset.from_generator(
        generator=lambda: map(tuple,
                              data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes)),
        output_types=(tf.float32, (tf.float32, tf.float32, tf.float32)),
        output_shapes=(shape_input_image, (shape_output_0_image, shape_output_1_image, shape_output_2_image)))
    val_dataset = tf.data.Dataset.from_generator(
        generator=lambda: map(tuple, data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes)),
        output_types=(tf.float32, (tf.float32, tf.float32, tf.float32)),
        output_shapes=(shape_input_image, (shape_output_0_image, shape_output_1_image, shape_output_2_image)))

    loss = [yolo_loss(anchors[mask], classes=num_classes) for mask in masks]
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(optimizer=optimizer, loss=loss)
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint(training_path + '/checkpoints/yolov3_train_{epoch}.tf', verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]

    history = model.fit(dataset,
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=val_dataset,
                        validation_steps=max(1, num_val // batch_size),
                        epochs=epochs,
                        initial_epoch=0,
                        callbacks=callbacks)
    model.save_weights(f'{training_path}/yolov3_mnist.tf')
