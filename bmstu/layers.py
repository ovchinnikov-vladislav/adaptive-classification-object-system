import tensorflow as tf


class Norm(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        x = tf.norm(inputs, axis=-1)
        return x
