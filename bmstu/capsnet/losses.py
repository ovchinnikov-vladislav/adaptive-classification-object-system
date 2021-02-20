import tensorflow as tf


def margin_loss(y_true, y_pred):
    correction = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) \
                 * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_sum(tf.reduce_sum(correction, 1))
