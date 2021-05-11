from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from libs.yolo.v3.layers import (yolo_v3_tiny)
from libs.yolo.caps.layers import capsules_yolo
from libs.yolo3 import yolo_loss
from libs.yolo3 import get_anchors, data_generator_wrapper
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--tiny', default=False, type=bool, help='yolov3 or yolov3-tiny')
parser.add_argument('--weights', default='./model_data/yolov3.tf', help='path to weights file')
parser.add_argument('--classes', default='./model_data/wider_classes.txt', help='path to classes file')
parser.add_argument('--dataset_path', default='D:/tensorflow_datasets/', help='path to download dataset')
parser.add_argument('--pretrained', default=False, type=bool, help='pretrained model')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--size', default=416, type=int, help='size image')
parser.add_argument('--channels', default=3, type=int, help='channels')
parser.add_argument('--training_path', default='./', help='training data path')
parser.add_argument('--download_dataset', default=1, type=int, )
parser.add_argument('--update_annotation', default=1, type=int, help='update annotation path to files')
parser.add_argument('--epochs', default=100, type=int, help='epochs number')

if __name__ == '__main__':
    args = parser.parse_args()
    size = args.size
    batch_size = args.batch_size
    epochs = args.epochs
    channels = args.channels
    class_names = ['face']
    training_path = args.training_path
    download_dataset = True if args.download_dataset == 1 else False
    update_annotation = True if args.update_annotation == 1 else False
    pretrained = True if args.pretrained == 1 else False
    num_classes = len(class_names)

    # ann_train, ann_test, ann_val = wider_dataset_annotations(args.dataset_path, download_dataset, update_annotation)

    ann_train, ann_test, ann_val = './wider_face/wider_face_train_annotation.txt', \
                                   './wider_face/wider_face_test_annotation.txt', \
                                   './wider_face/wider_face_val_annotation.txt'

    with open(ann_train) as f:
        train_lines = f.readlines()
    with open(ann_test) as f:
        test_lines = f.readlines()
    with open(ann_val) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    input_shape = (size, size)

    if args.tiny:
        anchors = get_anchors('./model_data/tiny_yolo_anchors.txt')
        masks = np.array([[3, 4, 5], [0, 1, 2]])
        model = yolo_v3_tiny(anchors, size=size, channels=channels, classes=num_classes, training=True)
    else:
        anchors = get_anchors('./model_data/yolo_anchors.txt')
        masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        model = capsules_yolo(anchors, size=size, channels=channels, classes=num_classes, training=True)

    grid_size = size // 32
    shape_input_image = (None, size, size, channels)
    shape_output_0_image = (None, grid_size, grid_size, 3, 6)
    shape_output_1_image = (None, grid_size * 2, grid_size * 2, 3, 6)
    shape_output_2_image = (None, grid_size * 4, grid_size * 4, 3, 6)

    dataset = tf.data.Dataset.from_generator(
        generator=lambda: map(tuple,
                              data_generator_wrapper(train_lines, batch_size, input_shape, anchors)),
        output_types=(tf.float32, (tf.float32, tf.float32, tf.float32)),
        output_shapes=(shape_input_image, (shape_output_0_image, shape_output_1_image, shape_output_2_image)))
    val_dataset = tf.data.Dataset.from_generator(
        generator=lambda: map(tuple, data_generator_wrapper(val_lines, batch_size, input_shape, anchors)),
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
                        epochs=args.epochs,
                        initial_epoch=0,
                        callbacks=callbacks)
    model.save_weights(f'{training_path}/yolov3_wider.tf')
