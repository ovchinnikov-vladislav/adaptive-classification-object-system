import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from libs.yolo3.layers import yolo_v3, yolo_v3_tiny, yolo_anchors, yolo_anchor_masks, yolo_tiny_anchors, yolo_tiny_anchor_masks
from libs.yolo3.losses import YoloLoss
from libs.yolo3.utils import freeze_all, transform_images, transform_targets
import tensorflow_datasets as tfds

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--tiny', default=False, type=bool, help='yolov3 or yolov3-tiny')
parser.add_argument('--weights', default='./model_data/yolov3.tf', help='path to weights file')
parser.add_argument('--classes', default='./model_data/coco_classes.txt', help='path to classes file')
parser.add_argument('--dataset_path', default='D:\\tensorflow_datasets', help='path to download dataset')

if __name__ == '__main__':
    args = parser.parse_args()
    size = 416
    batch_size = 10

    if args.tiny:
        model = yolo_v3_tiny(size, training=True, classes=1)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = yolo_v3(size, training=True, classes=1)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    train_dataset = tfds.load('wider_face', split='train', data_dir=args.dataset_path)
    val_dataset = tfds.load('wider_face', split='validation', data_dir=args.dataset_path)
    test_dataset = tfds.load('wider_face', split='test', data_dir=args.dataset_path)

    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(lambda x: (transform_images(x['image'], size),
                                                 transform_targets(x['faces']['bbox'], anchors, anchor_masks, size)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.map(lambda x, y: (transform_images(x, size),
                                                transform_targets(y, anchors, anchor_masks, size)))

    # # Configure the model for transfer learning
    # if FLAGS.transfer == 'none':
    #     pass  # Nothing to do
    # elif FLAGS.transfer in ['darknet', 'no_output']:
    #     # Darknet transfer is a special case that works
    #     # with incompatible number of classes
    #
    #     # reset top layers
    #     if FLAGS.tiny:
    #         model_pretrained = YoloV3Tiny(
    #             FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
    #     else:
    #         model_pretrained = YoloV3(
    #             FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
    #     model_pretrained.load_weights(FLAGS.weights)
    #
    #     if FLAGS.transfer == 'darknet':
    #         model.get_layer('yolo_darknet').set_weights(
    #             model_pretrained.get_layer('yolo_darknet').get_weights())
    #         freeze_all(model.get_layer('yolo_darknet'))
    #
    #     elif FLAGS.transfer == 'no_output':
    #         for l in model.layers:
    #             if not l.name.startswith('yolo_output'):
    #                 l.set_weights(model_pretrained.get_layer(
    #                     l.name).get_weights())
    #                 freeze_all(l)
    #
    # else:
    #     # All other transfer require matching classes
    #     model.load_weights(FLAGS.weights)
    #     if FLAGS.transfer == 'fine_tune':
    #         # freeze darknet and fine tune other layers
    #         darknet = model.get_layer('yolo_darknet')
    #         freeze_all(darknet)
    #     elif FLAGS.transfer == 'frozen':
    #         # freeze everything
    #         freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    loss = [YoloLoss(anchors[mask], classes=1) for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss)
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf', verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]

    history = model.fit(train_dataset,
                        epochs=10,
                        callbacks=callbacks,
                        validation_data=val_dataset)
