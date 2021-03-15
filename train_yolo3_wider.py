import numpy as np
from bmstu.datasets.wider_faces import wider_dataset_annotations
from bmstu.yolo3.train import get_anchors, create_model, data_generator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from bmstu.capsnets.losses import margin_loss

import argparse

parser = argparse.ArgumentParser(description='Train WIDER YOLO')
parser.add_argument('--root_dataset', default='D:/tensorflow_datasets/',
                    help='path dataset ')
parser.add_argument('--batch_size', default=10)
parser.add_argument('--prepare_annotation', default=True, help='prepare annotation')
parser.add_argument('--weights', default='model_data/yolo.dat', help='weights')


if __name__ == '__main__':
    args = parser.parse_args()
    root_path = args.root_dataset
    annotation_train_path, annotation_test_path, annotation_val_path = \
        wider_dataset_annotations('D:/tensorflow_datasets/', is_prepare_annotation=True)

    anchors_path = 'model_data/yolo_anchors.txt'
    log_dir = 'logs/000/'
    num_classes = 1
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)

    model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=args.weights)

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

    # TODO: вернуть для тренировки на базе натренированной сети
    # if True:
    #     model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    #     batch_size = 10
    #     print(f'Train on {num_train} samples, val on {num_val} samples, with batch size {batch_size}.')
    #     model.fit_generator(generator=data_generator(train_lines, batch_size, input_shape, anchors, num_classes),
    #                         steps_per_epoch=max(1, num_train // batch_size),
    #                         validation_data=data_generator(val_lines, batch_size, input_shape, anchors, num_classes),
    #                         validation_steps=max(1, num_val // batch_size),
    #                         epochs=50,
    #                         initial_epoch=0,
    #                         callbacks=[logging, checkpoint])
    #     model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss=margin_loss)
        print('Unfreeze all of the layers.')
        batch_size = args.batch_size
        print(f'Train on {num_train} samples, val on {num_val} samples, with batch size {batch_size}.')
        model.fit_generator(generator=data_generator(train_lines, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator(val_lines, batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')
