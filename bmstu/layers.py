import tensorflow as tf


class Norm(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        x = tf.norm(inputs, axis=-1)
        return x


class Flatten(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        x = tf.reshape(inputs, [inputs.shape[0], -1])
        return x


class UnFlatten(tf.keras.layers.Layer):
    def __init__(self, c, w, h, **kwargs):
        super(UnFlatten, self).__init__(**kwargs)
        self.c = c
        self.w = w
        self.h = h

    def call(self, inputs, **kwargs):
        x = tf.reshape(inputs, [-1, self.c, self.w, self.h])
        return x
