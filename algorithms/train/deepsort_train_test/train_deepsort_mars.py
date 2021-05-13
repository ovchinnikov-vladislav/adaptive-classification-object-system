import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras import optimizers
from libs.datasets.deepsort import utils, mars
from libs.deepsort.net import losses
import libs.deepsort.net.network_definition as net


IMAGE_SHAPE = (128, 64, 3)


class Mars(object):

    def __init__(self, dataset_dir, num_validation_y=0.1, seed=1234):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed

    def read_train(self):
        filenames, ids, camera_indices, _ = mars.read_train_split_to_str(self._dataset_dir)
        train_indices, _ = utils.create_validation_split(np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in train_indices]
        ids = [ids[i] for i in train_indices]
        camera_indices = [camera_indices[i] for i in  train_indices]
        return filenames, ids, camera_indices

    def read_validation(self):
        filenames, ids, camera_indices, _ = mars.read_train_split_to_str(self._dataset_dir)
        _, valid_indices = utils.create_validation_split(np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in valid_indices]
        ids = [ids[i] for i in valid_indices]
        camera_indices = [camera_indices[i] for i in valid_indices]
        return filenames, ids, camera_indices

    def read_test_filenames(self):
        filename = os.path.join(self._dataset_dir, "info", "test_name.txt")
        with open(filename, "r") as file_handle:
            content = file_handle.read()
            lines = content.splitlines()

        image_dir = os.path.join(self._dataset_dir, "bbox_test")
        return [os.path.join(image_dir, f[:4], f) for f in lines]


if __name__ == '__main__':
    dataset = Mars('./mars', num_validation_y=0.1, seed=1234)
    train_x, train_y, _ = dataset.read_train()

    # model = net.create_network(inputs_shape=IMAGE_SHAPE, num_classes=mars.MAX_LABEL + 1, add_logits=True)
    # model.summary()
    #
    # model.compile(optimizer=optimizers.Adam(lr=0.003),
    #               loss=[sparse_categorical_crossentropy, losses.magnet_loss],
    #               metrics='accuracy')
    #
    # history = model.fit((train_x, train_y),
    #                     validation_data=,
    #                     epochs=epochs)