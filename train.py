import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import Model
from bmstu.yolo3.layers import yolo_body
from bmstu.yolo3.losses import yolo_loss
from bmstu.yolo3.utils import preprocess_true_boxes, get_random_data
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

import argparse

parser = argparse.ArgumentParser(description='Train WIDER YOLO')
parser.add_argument('--root_dataset', default='D:/tensorflow_datasets/',
                    help='path dataset ')
parser.add_argument('--prepare_annotation', default=True, help='prepare annotation')


def _get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as file:
        anchors = file.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def prepare_annotation(filename, dataset):
    with open(filename, 'w') as file:
        for example in dataset:
            string = train_path + example['image/filename'].numpy().decode()
            height, width, _ = example['image'].shape
            bbox = example['faces']['bbox'].numpy()
            bboxs = ''
            for i in range(len(bbox)):
                bbox_string = f'{int(bbox[i][1] * width)},{int(bbox[i][0] * height)},' \
                              f'{int(bbox[i][3] * width)},{int(bbox[i][2] * height)},{0}'
                bboxs += ' ' + bbox_string
            string += bboxs
            print(string)
            file.write(string + '\n')


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    args = parser.parse_args()
    root_path = args.root_dataset
    train_path = f'{root_path}downloads/extracted/ZIP.ucexport_download_id_0B6eKvaijfFUDQUUwd21EckhU4jt2EpyCXK-ui-lE9lMQsuG6HHaIWv5zLxecQeXtbVk/WIDER_train/images/'
    test_path = f'{root_path}downloads/extracted/ZIP.ucexport_download_id_0B6eKvaijfFUDbW4tdGpaYjgzOwMT4R6ikuxYiUtHrEwFA7Iw4SVAMwhF1wp3mCQfiNM/WIDER_test/images/'
    val_path = f'{root_path}downloads/extracted/ZIP.ucexport_download_id_0B6eKvaijfFUDd3dIRmpvSk8t-e-9CfKMXS2IS-jA6u85ZxWMhmpZP8NqsEE-SypYoXo/WIDER_val/images/'
    train_ds = tfds.load('wider_face', split='train', data_dir=root_path)
    val_ds = tfds.load('wider_face', split='validation', data_dir=root_path)
    test_ds = tfds.load('wider_face', split='test', data_dir=root_path)

    annotation_train_path = 'model_data/wider_face_train_annotation.txt'
    annotation_test_path = 'model_data/wider_face_test_annotation.txt'
    annotation_val_path = 'model_data/wider_face_val_annotation.txt'

    is_prepare_annotation = args.prepare_annotation

    if is_prepare_annotation:
        prepare_annotation(annotation_train_path, train_ds)
        prepare_annotation(annotation_test_path, test_ds)
        prepare_annotation(annotation_val_path, val_ds)

    anchors_path = 'model_data/yolo_anchors.txt'
    log_dir = 'logs/000/'
    num_classes = 1
    anchors = _get_anchors(anchors_path)

    input_shape = (416, 416)

    model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path='model_data/yolo.dat')

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss',
                                 save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1

    with open(annotation_train_path) as f:
        train_lines = f.readlines()
    with open(annotation_val_path) as f:
        val_lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.shuffle(val_lines)
    num_val = len(val_lines)
    num_train = len(train_lines)

    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        batch_size = 32
        print(f'Train on {num_train} samples, val on {num_val} samples, with batch size {batch_size}.')
        model.fit_generator(generator=data_generator_wrapper(train_lines, batch_size, input_shape,
                                                             anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape,
                                                                   anchors, num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Unfreeze all of the layers.')
        batch_size = 32
        print(f'Train on {num_train} samples, val on {num_val} samples, with batch size {batch_size}.')
        model.fit_generator(generator=data_generator_wrapper(train_lines, batch_size, input_shape,
                                                             anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape,
                                                                   anchors, num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=100,
                            initial_epoch=50,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')
