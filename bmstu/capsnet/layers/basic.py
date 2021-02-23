import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
import numpy as np

from bmstu.capsnet.utls import squash


class PrimaryCapsule2D(layers.Layer):
    """
    :param capsules: количество первичных капсул
    :param dim_capsules: размер капсул
    :param kernel_size: размер ядра свертки
    :param strides: шаг свертки
    :param name: имя слоя
    """

    def __init__(self, capsules, dim_capsules, kernel_size, strides, **kwargs):
        super(PrimaryCapsule2D, self).__init__(**kwargs)
        assert capsules % dim_capsules == 0, "Invalid size of capsules and dim_capsules"

        num_filters = capsules * dim_capsules
        self.conv2d = layers.Conv2D(filters=num_filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    activation=None,
                                    padding='valid')
        self.reshape = layers.Reshape(target_shape=(-1, dim_capsules))

    def call(self, inputs, **kwargs):
        x = self.conv2d(inputs)
        x = self.reshape(x)
        x = squash(x)
        return x

    def get_config(self):
        return super(PrimaryCapsule2D, self).get_config()


class Capsule(layers.Layer):
    """
    :param capsules: количество капсул
    :param dim_capsules: размер капсул
    :param routings: количество итераций маршрутизации
    :param name: имя слоя
    """

    def __init__(self, capsules, dim_capsules, routings, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.capsules = capsules
        self.dim_capsules = dim_capsules
        self.routings = routings
        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight(shape=[self.capsules, input_shape[1], self.dim_capsules, input_shape[2]],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.1),
                                 trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.capsules, 1, 1])
        inputs_tiled = tf.expand_dims(inputs_tiled, 4)

        inputs_hat = tf.map_fn(lambda x: tf.matmul(self.w, x), elems=inputs_tiled)

        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.capsules, inputs.shape[1], 1, 1])

        assert self.routings > 0, 'The routings should be > 0.'
        outputs = None
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)

            outputs = tf.multiply(c, inputs_hat)
            outputs = tf.reduce_sum(outputs, axis=2, keepdims=True)
            outputs = squash(outputs, axis=-2)

            if i < self.routings - 1:
                outputs_tiled = tf.tile(outputs, [1, 1, inputs.shape[1], 1, 1])
                agreement = tf.matmul(inputs_hat, outputs_tiled, transpose_a=True)
                b = tf.add(b, agreement)

        return tf.squeeze(outputs, [2, 4])

    def get_config(self):
        return super(Capsule, self).get_config()


class Decoder(layers.Layer):
    def __init__(self, classes, output_shape, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.classes = classes
        self.shape = output_shape
        self.masked = Mask()
        self.decoder = tf.keras.models.Sequential()
        self.decoder.add(layers.Dense(512, activation='relu', input_dim=16 * self.classes))
        self.decoder.add(layers.Dense(1024, activation='relu'))
        self.decoder.add(layers.Dense(np.prod(self.shape), activation='sigmoid'))
        self.decoder.add(layers.Reshape(target_shape=self.shape))

    def call(self, inputs, **kwargs):
        return self.decoder(self.masked(inputs))

    def get_config(self):
        return super(Decoder, self).get_config()


class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1) + epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            mask = tf.one_hot(indices=tf.argmax(x, 1), depth=x.shape[1])

        masked = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config
