from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, initializers, regularizers, constraints
from tensorflow.python.keras.utils import conv_utils
from libs.capsnets.utls import own_batch_dot
from tensorflow.keras.backend import epsilon


def squeeze(vectors):
    vectors_q = tf.reduce_sum(tf.square(vectors), axis=-1, keepdims=True)
    return (vectors_q / (1 + vectors_q)) * (vectors / tf.sqrt(vectors_q + epsilon()))


class ConvertToCapsule(layers.Layer):
    def __init__(self, **kwargs):
        super(ConvertToCapsule, self).__init__(**kwargs)
        # self.input_spec = InputSpec(min_ndim=2)

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape.insert(len(output_shape), 1)
        return tuple(output_shape)

    def get_config(self):
        return super(ConvertToCapsule, self).get_config()


class FlattenCapsule(layers.Layer):
    def __init__(self, **kwargs):
        super(FlattenCapsule, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=4)

    def call(self, inputs, **kwargs):
        shape = inputs.shape
        return tf.reshape(inputs, shape=(-1, np.prod(shape[1:-1]), shape[-1]))

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "FlattenCaps" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return input_shape[0], np.prod(input_shape[1:-1]), input_shape[-1]
