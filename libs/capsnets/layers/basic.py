import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
import numpy as np

from libs.capsnets.utls import squash


class PrimaryCapsule2D(layers.Layer):
    """
    :param capsules: количество первичных капсул
    :param dim_capsules: размер капсул
    :param kernel_size: размер ядра свертки
    :param strides: шаг свертки
    :param name: имя слоя
    """

    def __init__(self, num_capsules, dim_capsules, kernel_size, strides, **kwargs):
        super(PrimaryCapsule2D, self).__init__(**kwargs)

        num_filters = num_capsules * dim_capsules
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

    def __init__(self, num_capsules, dim_capsules, routings, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routings = routings
        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight(shape=[self.num_capsules, input_shape[1], self.dim_capsules, input_shape[2]],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.1),
                                 trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_expand = tf.expand_dims(inputs, 1)
        u_i = tf.tile(inputs_expand, [1, self.num_capsules, 1, 1])
        u_i = tf.expand_dims(u_i, 4)  # u_i

        u_ji_hat = tf.map_fn(lambda x: tf.matmul(self.w, x), elems=u_i)  # u_j|i_hat = Wij * u_i

        # for all capsule i in layer l and capsule j in layer(l + 1): b_ij <- 0
        b_ij = tf.zeros(shape=[tf.shape(u_ji_hat)[0], self.num_capsules, inputs.shape[1], 1, 1])  # b_ij <- 0

        assert self.routings > 0, 'The routings should be > 0.'
        v_j = None
        # for r iterations do
        for i in range(self.routings):
            # for all capsule i in layer l: c_i <- softmax(b_i)
            c_i = tf.nn.softmax(b_ij, axis=1)

            # for all capsule j in layer (l + 1): s_j <- Sum_i (c_ij * u_j|i_hat)
            s_j = tf.multiply(c_i, u_ji_hat)  # c_ij * u_j|i_hat
            s_j = tf.reduce_sum(s_j, axis=2, keepdims=True)  # s_j <- Sum_i (c_ij * u_j|i_hat)

            # for all capsule j in layer (l + 1): v_j <- squash(s_j)
            v_j = squash(s_j, axis=-2)  # v_j <- squash(s_j)

            # Небольшая оптимизация по причине того, что финальным тензором является v_j <- squash(s_j)
            if i < self.routings - 1:
                # for all capsule i in layer l and capsule j in layer (l + 1): b_ij <- b_ij + u_j|i_hat * v_j
                outputs_tiled = tf.tile(v_j, [1, 1, inputs.shape[1], 1, 1])
                agreement = tf.matmul(u_ji_hat, outputs_tiled, transpose_a=True)  # u_j|i_hat * v_j
                b_ij = tf.add(b_ij, agreement)  # b_ij <- b_ij + u_j|i_hat * v_j

        return tf.squeeze(v_j, [2, 4])  # return v_j

    def get_config(self):
        return super(Capsule, self).get_config()


class Decoder(layers.Layer):
    def __init__(self, num_classes, output_shape, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.shape = output_shape
        self.masked = Mask()
        self.decoder = tf.keras.models.Sequential()
        self.decoder.add(layers.Dense(512, activation='relu', input_dim=16 * self.num_classes))
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
