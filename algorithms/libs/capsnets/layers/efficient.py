import tensorflow as tf
from tensorflow.keras import layers
from libs.capsnets.utils import efficient_squash


class PrimaryCapsule2D(layers.Layer):
    """
    :param capsules: количество первичных капсул
    :param dim_capsules: размер капсул
    :param kernel_size: размер ядра свертки
    :param strides: шаг свертки
    :param name: имя слоя
    """

    def __init__(self, filters, kernel_size, num_capsules, dim_capsules, strides=1, padding='valid', **kwargs):
        super(PrimaryCapsule2D, self).__init__(**kwargs)

        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.dw_conv2d = layers.Conv2D(filters, kernel_size, strides, activation='linear', padding=padding)

    def call(self, inputs, **kwargs):
        output = self.dw_conv2d(inputs)
        outputs = layers.Reshape(target_shape=(self.num_capsules, self.dim_capsules))(output)

        return efficient_squash(outputs)

    def get_config(self):
        return super(PrimaryCapsule2D, self).get_config()


class Capsule(layers.Layer):
    """
    :param capsules: количество капсул
    :param dim_capsules: размер капсул
    :param routings: количество итераций маршрутизации
    :param name: имя слоя
    """

    def __init__(self, num_capsules, dim_capsules, kernel_initializer='he_normal', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.w = None
        self.b = None

    def build(self, input_shape):
        input_n = input_shape[-2]
        input_d = input_shape[-1]

        self.w = self.add_weight(shape=[self.num_capsules, input_n, input_d, self.dim_capsules],
                                 dtype=tf.float32,
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.b = self.add_weight(shape=[self.num_capsules, input_n, 1],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer(),
                                 name='b')

        self.built = True

    def call(self, inputs, **kwargs):
        u = tf.einsum('...ji,kjiz->...kjz', inputs, self.w)  # u shape=(None,N,H*W*input_N,D)

        c = tf.einsum('...ij,...kj->...i', u, u)[..., None]  # b shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c / tf.sqrt(tf.cast(self.dim_capsules, tf.float32))
        c = tf.nn.softmax(c, axis=1)  # c shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c + self.b
        s = tf.reduce_sum(tf.multiply(u, c), axis=-2)  # s shape=(None,N,D)

        return efficient_squash(s)  # v shape=(None,N,D)

    def get_config(self):
        return super(Capsule, self).get_config()
