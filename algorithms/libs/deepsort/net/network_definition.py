import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Input, Conv2D, Flatten, Dropout, Dense,
                                     MaxPooling2D, Layer, BatchNormalization, Lambda)
from tensorflow.keras.models import Model
from libs.deepsort.net import residual_net
import os
import numpy as np


class Logits(Layer):
    def __init__(self, num_classes, feature_dim, **kwargs):
        super(Logits, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.w = None
        self.scale = None

    def build(self, input_shape):
        self.w = self.add_weight(shape=[self.feature_dim, self.num_classes],
                                 dtype=tf.float32,
                                 initializer=TruncatedNormal(stddev=0.1),
                                 trainable=True)
        self.scale = self.add_weight(shape=(),
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.),
                                     regularizer=l2(1e-1))
        self.scale = tf.nn.softplus(self.scale)

        self.built = True

    def call(self, inputs, **kwargs):
        weights_normed = tf.nn.l2_normalize(self.w, axis=0)

        return self.scale * tf.matmul(inputs, weights_normed)

    def get_config(self):
        return super(Logits, self).get_config()


def create_network(input_shape, num_classes=None, add_logits=True, weight_decay=1e-8):
    nonlinearity = tf.nn.elu
    conv_weight_init = TruncatedNormal(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = l2(weight_decay)
    fc_weight_init = TruncatedNormal(stddev=1e-3)
    fc_bias_init = tf.zeros_initializer()
    fc_regularizer = l2(weight_decay)

    network = inputs = Input(shape=input_shape)
    network = Conv2D(32, (3, 3), strides=1, padding='same',
                     kernel_initializer=conv_weight_init,
                     bias_initializer=conv_bias_init,
                     kernel_regularizer=conv_regularizer)(network)
    network = BatchNormalization()(network)
    network = nonlinearity(network)

    network = Conv2D(32, (3, 3), strides=1, padding='same',
                     kernel_initializer=conv_weight_init,
                     bias_initializer=conv_bias_init,
                     kernel_regularizer=conv_regularizer)(network)
    network = BatchNormalization()(network)
    network = nonlinearity(network)

    network = MaxPooling2D((3, 3), (2, 2), padding='same')(network)

    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=False, is_first=True)
    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=False)

    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=True)
    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=False)

    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=True)
    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=False)

    feature_dim = network.shape[-1]
    network = Flatten()(network)

    network = Dropout(0.6)(network)
    network = Dense(feature_dim, kernel_regularizer=fc_regularizer,
                    kernel_initializer=fc_weight_init,
                    bias_initializer=fc_bias_init)(network)
    network = BatchNormalization()(network)
    network = nonlinearity(network)

    features = network

    features = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), name='features_output')(features)

    if add_logits:
        logits = Logits(num_classes, feature_dim, name='logits_output')(features)
        return Model(inputs, [features, logits])
    else:
        return Model(inputs, features)


def preprocess(image, is_training=False, input_is_bgr=False):
    if input_is_bgr:
        image = image[:, :, ::-1]  # BGR to RGB
    image = tf.divide(tf.cast(image, tf.float32), 255.0)
    if is_training:
        image = tf.image.random_flip_left_right(image)
    return image


def create_losses_metrics(mode, monitor_magnet=True, monitor_triplet=True):
    from libs.deepsort.net.losses import MagnetLoss
    from libs.deepsort.net.losses import soft_margin_triplet_loss

    loss = {
        'logits_output': None,
        'features_output': None
    }
    metrics = {
        'logits_output': [],
        'features_output': []
    }
    if mode == 'cosine-softmax':
        loss['logits_output'] = tf.keras.losses.sparse_categorical_crossentropy
        metrics['logits_output'].append(tf.keras.metrics.categorical_accuracy)
    elif mode == 'magnet':
        loss['features_output'] = MagnetLoss()
    elif mode == 'triplet':
        loss['features_output'] = soft_margin_triplet_loss
    else:
        raise ValueError(f'Unknown loss mode: {mode}')

    if monitor_magnet and mode != "magnet":
        metrics['features_output'].append(MagnetLoss())
    if monitor_triplet and mode != 'triplet':
        metrics['features_output'].append(soft_margin_triplet_loss)

    return loss, metrics


class ReIdentificationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, num_classes, batch_size=32, num_samples_per_id=4, num_fa_images=0, shuffle=True, shape=(128, 64, 3)):
        assert (batch_size - num_fa_images) % num_samples_per_id == 0
        self.num_ids_per_batch = int((batch_size - num_fa_images) / num_samples_per_id)
        self.num_classes = num_classes
        self.num_samples_per_id = num_samples_per_id
        self.directory = directory
        self.batch_size = batch_size
        self.indexes = None
        self.shuffle = shuffle
        self.shape = shape
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(os.listdir(self.directory)) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        dirs = os.listdir(self.directory)
        dirs = [str(os.path.abspath(os.path.join(self.directory, dirs[k]))) for k in indexes]
        X, y = self.__data_generation(dirs)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(os.listdir(self.directory)))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, dirs):
        X = np.empty((self.batch_size, *self.shape))
        y = np.empty(self.batch_size, dtype=int)

        indices = np.random.choice(len(dirs), self.num_ids_per_batch, replace=False)
        dirs_unique = []
        for i in indices:
            dirs_unique.append(dirs[i])

        e = 0
        for i, directory in enumerate(dirs_unique):
            images = os.listdir(directory)
            num_samples = min(self.num_samples_per_id, len(images))
            indices = np.random.choice(len(images), num_samples, replace=False)
            images_record = []
            for j in indices:
                images_record.append(images[j])

            s, e = e, e + num_samples

            images_array = []
            y_array = []
            for record in images_record:
                image = tf.image.decode_jpeg(tf.io.read_file(os.path.join(directory, record)), channels=self.shape[2])
                image = tf.image.resize(image, self.shape[:2])
                image = preprocess(image, is_training=True)
                images_array.append(np.array(image))
                label = os.path.basename(directory).split('.')[0]
                y_array.append(int(label))

            X[s:e] = np.array(images_array)
            y[s:e] = y_array

        return X, np.array(y)


if __name__ == '__main__':
    model = create_network(input_shape=(128, 64, 3), num_classes=1501, add_logits=True, weight_decay=1e-8)
    model.summary()

    losses, accuracies = create_losses_metrics('cosine-softmax')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=losses,
                  metrics=accuracies)

    train_generator = ReIdentificationDataGenerator('D:/MasterDissertation/dataset/mars/bbox_train', num_classes=1501)
    test_generator = ReIdentificationDataGenerator('D:/MasterDissertation/dataset/mars/bbox_test', num_classes=1501)

    model.fit(train_generator,
              validation_data=test_generator,
              batch_size=32,
              epochs=25)
