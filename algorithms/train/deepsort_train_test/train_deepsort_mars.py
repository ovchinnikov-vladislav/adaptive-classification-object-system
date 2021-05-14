import os
import numpy as np
import config
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
    dataset = Mars(config.mars_datasets, num_validation_y=0.1, seed=1234)
    dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(config.mars_datasets, 'bbox_train'), image_size=(128, 64))
    dataset_val = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(config.mars_datasets, 'bbox_test'), image_size=(128, 64))

    model = net.create_network(inputs_shape=IMAGE_SHAPE, num_classes=mars.MAX_LABEL + 1, add_logits=False)
    model.summary()

    losses = {
        "features_output": sparse_categorical_crossentropy
        # "logits_output": losses.magnet_loss,
    }

    model.compile(optimizer=optimizers.Adam(lr=0.003),
                  loss=losses,
                  metrics='accuracy')

    history = model.fit(dataset_train,
                        epochs=100,
                        batch_size=64)
