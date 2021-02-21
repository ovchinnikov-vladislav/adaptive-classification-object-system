import tensorflow as tf


def margin_loss(y_true, y_pred, m_plus=0.9, m_minus=0.1, down_weighting=0.5):
    correction = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + down_weighting * (1 - y_true) \
                 * tf.square(tf.maximum(0., y_pred - m_minus))

    return tf.reduce_sum(tf.reduce_sum(correction, 1))


def compute_loss(y_true, y_pred, reconstruction, x, reconstruction_weight=0.0005):
    num_classes = tf.shape(y_pred)[1]

    loss = margin_loss(y_pred, tf.one_hot(y_true, num_classes))
    loss = tf.reduce_mean(loss)

    x_1d = tf.keras.layers.Flatten()(x)
    distance = tf.square(reconstruction - x_1d)
    reconstruction_loss = tf.reduce_sum(distance, axis=-1)
    reconstruction_loss = reconstruction_weight * tf.reduce_mean(reconstruction_loss)

    loss = loss + reconstruction_loss

    return loss, reconstruction_loss
