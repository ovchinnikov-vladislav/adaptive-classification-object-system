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


def cross_entropy_loss(y_true, y_pred, x):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = tf.reduce_mean(loss)
    num_classes = int(y_pred.shape[-1])
    data_size = int(x.shape[1])

    y_true = tf.one_hot(y_true, num_classes, dtype=tf.float32)
    y_true = tf.expand_dims(y_true, axis=2)
    y_pred = tf.expand_dims(y_pred, axis=2)
    y_pred = tf.reshape(tf.multiply(y_pred, y_true), shape=[-1])

    y_pred = tf.keras.layers.Dense(512, trainable=True)(y_pred)
    y_pred = tf.keras.layers.Dense(1024, trainable=True)(y_pred)
    y_pred = tf.keras.layers.Dense(data_size * data_size, trainable=True, activation=tf.sigmoid)(y_pred)

    x = tf.reshape(x, shape=[-1])
    reconstruction_loss = tf.reduce_mean(tf.square(y_pred - x))

    loss_all = tf.add_n([loss] + [0.0005 * reconstruction_loss])

    return loss_all, reconstruction_loss, y_pred


def spread_loss(y_true, y_pred, pose_out, x, m):
    num_classes = int(y_pred.shape[-1])
    data_size = int(x.shape[1])

    y_true = tf.one_hot(y_true, num_classes, dtype=tf.float32)

    output = tf.reshape(y_pred, shape=[1, num_classes])
    y_true = tf.expand_dims(y_true, axis=2)
    at = tf.matmul(output, y_true)

    loss = tf.square(tf.maximum(0., m - (at - output)))
    loss = tf.matmul(loss, 1. - y_true)
    loss = tf.reduce_mean(loss)

    pose_out = tf.reshape(tf.multiply(pose_out, y_true), shape=[-1])

    pose_out = tf.keras.layers.Dense(512, trainable=True,
                                     kernel_regularizer=tf.keras.regularizers.L2(5e-04),
                                     activity_regularizer=tf.keras.regularizers.L2(5e-04))(pose_out)
    pose_out = tf.keras.layers.Dense(1024, trainable=True,
                                     kernel_regularizer=tf.keras.regularizers.L2(5e-04),
                                     activity_regularizer=tf.keras.regularizers.L2(5e-04))(pose_out)
    pose_out = tf.keras.layers.Dense(data_size * data_size, trainable=True, activation=tf.sigmoid,
                                     kernel_regularizer=tf.keras.regularizers.L2(5e-04),
                                     activity_regularizer=tf.keras.regularizers.L2(5e-04))(pose_out)

    x = tf.reshape(x, shape=[-1])
    reconstruction_loss = tf.reduce_mean(tf.square(pose_out - x))

    loss_all = tf.add_n([loss] + [0.0005 * data_size * data_size * reconstruction_loss])

    return loss_all, loss, reconstruction_loss, pose_out

