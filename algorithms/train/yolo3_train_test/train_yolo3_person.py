from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from libs.detection.yolo.v3.layers import (yolo_v3, yolo_v3_tiny)
from libs.detection.losses import yolo_standard_loss
from libs.detection.utils import get_anchors, data_generator_wrapper
from libs.datasets.coco import coco_dataset_annotations
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--tiny', default=False, type=bool, help='yolov3 or yolov3-tiny')
parser.add_argument('--weights', default='./data/yolov3.tf', help='path to weights file')
parser.add_argument('--dataset_path', default='./', help='path to download dataset')
parser.add_argument('--pretrained', default=False, type=bool, help='pretrained model')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--size', default=416, type=int, help='size image')
parser.add_argument('--channels', default=3, type=int, help='channels')
parser.add_argument('--training_path', default='./', help='train data path')
parser.add_argument('--update_annotation', default=1, type=int, help='update annotation path to files')
parser.add_argument('--epochs', default=100, type=int, help='epochs number')

if __name__ == '__main__':
    args = parser.parse_args()
    size = args.size
    batch_size = args.batch_size
    epochs = args.epochs
    channels = args.channels
    classes = {'person'}

    training_path = args.training_path
    update_annotation = True if args.update_annotation == 1 else False
    pretrained = True if args.pretrained == 1 else False
    num_classes = len(classes)

    input_shape = (size, size)

    ann_train_path, ann_test_path, ann_val_path = coco_dataset_annotations(classes, args.dataset_path,
                                                                           update_annotation)

    with open(ann_train_path) as f:
        train_lines = f.readlines()
    with open(ann_test_path) as f:
        test_lines = f.readlines()
    with open(ann_val_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if args.tiny:
        anchors = get_anchors('resources/data/tiny_yolo_anchors.txt')
        masks = np.array([[3, 4, 5], [0, 1, 2]])
        model = yolo_v3_tiny(anchors, size=size, channels=channels, classes=1, training=True)
    else:
        anchors = get_anchors('resources/data/yolo_anchors.txt')
        masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        model = yolo_v3(anchors, size=size, channels=channels, classes=1, training=True)

    grid_size = size // 32
    shape_input_image = (None, size, size, channels)
    shape_output_0_image = (None, grid_size, grid_size, channels, 6)
    shape_output_1_image = (None, grid_size * 2, grid_size * 2, channels, 6)
    shape_output_2_image = (None, grid_size * 4, grid_size * 4, channels, 6)

    dataset = tf.data.Dataset.from_generator(
        generator=lambda: map(tuple,
                              data_generator_wrapper(train_lines, batch_size, input_shape, anchors)),
        output_types=(tf.float32, (tf.float32, tf.float32, tf.float32)),
        output_shapes=(shape_input_image, (shape_output_0_image, shape_output_1_image, shape_output_2_image)))
    val_dataset = tf.data.Dataset.from_generator(
        generator=lambda: map(tuple, data_generator_wrapper(val_lines, batch_size, input_shape, anchors)),
        output_types=(tf.float32, (tf.float32, tf.float32, tf.float32)),
        output_shapes=(shape_input_image, (shape_output_0_image, shape_output_1_image, shape_output_2_image)))

    loss = [yolo_standard_loss(anchors[mask], classes=num_classes) for mask in masks]
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(optimizer=optimizer, loss=loss)
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint(training_path + '/checkpoints/yolov3_train_{epoch}.tf', verbose=1, save_weights_only=True),
        TensorBoard(log_dir='../../resources/data/logs')
    ]

    history = model.fit(dataset,
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=val_dataset,
                        validation_steps=max(1, num_val // batch_size),
                        epochs=args.epochs,
                        initial_epoch=0,
                        callbacks=callbacks)
    model.save_weights(f'{training_path}/yolov3_person.tf')
