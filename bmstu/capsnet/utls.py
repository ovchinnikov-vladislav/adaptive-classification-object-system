import tensorflow as tf
from tensorflow.keras.backend import epsilon


# v = ((||sj||^2) / (1 + ||sj||^2)) * (sj / ||sj||)
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)

    additional_squashing = s_squared_norm / (1 + s_squared_norm)
    unit_scaling = vectors / tf.sqrt(s_squared_norm + epsilon())

    return additional_squashing * unit_scaling


def margin_loss(y_true, y_pred):
    correction = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) \
                 * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_sum(tf.reduce_sum(correction, 1))
