import tensorflow as tf
import numpy as np
from tensorflow.keras.backend import epsilon


# v = ((||sj||^2) / (1 + ||sj||^2)) * (sj / ||sj||)
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)

    additional_squashing = s_squared_norm / (1 + s_squared_norm)
    unit_scaling = vectors / tf.sqrt(s_squared_norm + epsilon())

    return additional_squashing * unit_scaling


def efficient_squash(vectors, axis=-1):
    norm = tf.norm(vectors, axis=-1, keepdims=True)
    return 1 - 1 / (tf.math.exp(norm) + epsilon()) * (vectors / (norm + epsilon()))
