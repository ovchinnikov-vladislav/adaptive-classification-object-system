from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from libs.yolo3.layers import (yolo_v3, yolo_v3_tiny)
from libs.yolo3.losses import yolo_loss
from libs.yolo3.losses import YoloLoss
from libs.yolo3.utils import get_anchors, data_generator_wrapper
from libs.datasets.wider_faces import wider_dataset_annotations
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--tiny', default=False, type=bool, help='yolov3 or yolov3-tiny')
parser.add_argument('--weights', default='./model_data/yolov3.tf', help='path to weights file')
parser.add_argument('--classes', default='./model_data/wider_classes.txt', help='path to classes file')
parser.add_argument('--dataset_path', default='D:/tensorflow_datasets/', help='path to download dataset')
parser.add_argument('--pretrained', default=False, type=bool, help='pretrained model')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--size', default=416, type=int, help='size image')
parser.add_argument('--channels', default=3, type=int, help='channels')
parser.add_argument('--training_path', default='./', help='training data path')
parser.add_argument('--update_annotation', default=0, type=int, help='update annotation path to files')
parser.add_argument('--epochs', default=100, type=int, help='epochs number')

if __name__ == '__main__':
    args = parser.parse_args()
    size = args.size
    batch_size = args.batch_size
    epochs = args.epochs
    channels = args.channels
    class_names = ['face']
    training_path = args.training_path
    update_annotation = True if args.update_annotation == 1 else False
    pretrained = True if args.pretrained == 1 else False
    num_classes = len(class_names)

    input_shape = (size, size)

    if args.tiny:
        anchors = get_anchors('./model_data/tiny_yolo_anchors.txt')
        masks = np.array([[3, 4, 5], [0, 1, 2]])
        model_body = yolo_v3_tiny(anchors, size=size, channels=channels, classes=1, training=True)
    else:
        anchors = get_anchors('./model_data/yolo_anchors.txt')
        masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        model_body = yolo_v3(anchors, size=size, channels=channels, classes=1, training=True)

    num_anchors = len(anchors)
    # y_true_input = [tf.keras.layers.Input(shape=(size // {0: 32, 1: 16, 2: 8}[i], size // {0: 32, 1: 16, 2: 8}[i],
    #                                              num_anchors // 3, num_classes + 5)) for i in range(3)]
    # model_loss = tf.keras.layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
    #                                     arguments={'anchors': anchors, 'num_classes': num_classes,
    #                                                'ignore_thresh': 0.5})([*model_body.output, *y_true_input])
    # model = tf.keras.Model([model_body.input, *y_true_input], model_loss)
    model = model_body
    loss = [YoloLoss(anchors[mask], classes=num_classes) for mask in masks]

    ann_train_path, ann_test_path, ann_val_path = wider_dataset_annotations(args.dataset_path, update_annotation)

    with open(ann_train_path) as f:
        train_lines = f.readlines()
    with open(ann_test_path) as f:
        test_lines = f.readlines()
    with open(ann_val_path) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(optimizer=optimizer, loss=loss)
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
                        epochs=args.epochs,
                        initial_epoch=12,
                        callbacks=callbacks)
    model.save_weights(f'{training_path}/yolov3_wider.tf')
