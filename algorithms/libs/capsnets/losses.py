import tensorflow as tf
from tensorflow.keras import activations


def margin_loss(y_true, y_pred, m_plus=0.9, m_minus=0.1, down_weighting=0.5):
    correction = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + down_weighting * (1 - y_true) \
                 * tf.square(tf.maximum(0., y_pred - m_minus))

    return tf.reduce_mean(tf.reduce_sum(correction, axis=1))


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


def cross_ent_loss(output, x, y, regularization):
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y, logits=output)
    loss = tf.reduce_mean(loss)
    num_class = output.shape[-1]
    data_size = x.shape[1]

    # reconstruction loss
    y = tf.one_hot(y, num_class, dtype=tf.float32)
    y = tf.expand_dims(y, axis=2)
    output = tf.expand_dims(output, axis=2)
    output = tf.reshape(tf.multiply(output, y), shape=[x.shape[0], -1])

    output = tf.keras.layers.Dense(512, trainable=True)(output)
    output = tf.keras.layers.Dense(1024, trainable=True)(output)
    output = tf.keras.layers.Dense(data_size * data_size, trainable=True,
                                   activation=activations.sigmoid)(output)

    x = tf.reshape(x, shape=[x.shape[0], -1])
    reconstruction_loss = tf.reduce_mean(tf.square(output - x))

    loss_all = tf.add_n([loss] + [0.0005 * reconstruction_loss] + regularization)

    return loss_all, reconstruction_loss, output


def spread_loss_old(output, pose_out, x, y, m, regularization=None):
    num_class = output.shape[-1]
    data_size = x.shape[1]

    y = tf.one_hot(tf.cast(y, tf.int32), num_class, dtype=tf.float32)
    # spread loss
    output1 = tf.reshape(output, shape=[x.shape[0], 1, num_class])
    y = tf.expand_dims(y, axis=2)
    at = tf.matmul(output1, y)

    loss = tf.square(tf.maximum(0., m - (at - output1)))
    loss = tf.matmul(loss, 1. - y)
    loss = tf.reduce_mean(loss)

    # reconstruction loss
    # pose_out = tf.reshape(tf.matmul(pose_out, y, transpose_a=True), shape=[x.shape[0], -1])
    # pose_out = tf.reshape(tf.multiply(pose_out, y), shape=[x.shape[0], -1])
    #
    # pose_out = tf.keras.layers.Dense(512, trainable=True, kernel_regularizer=tf.keras.regularizers.L2(5e-04))(pose_out)
    # pose_out = tf.keras.layers.Dense(1024, trainable=True, kernel_regularizer=tf.keras.regularizers.L2(5e-04))(pose_out)
    # pose_out = tf.keras.layers.Dense(data_size * data_size, trainable=True,
    #                                  kernel_regularizer=tf.keras.regularizers.L2(5e-04))(pose_out)
    #
    # x = tf.reshape(x, shape=[x.shape[0], -1])
    # reconstruction_loss = tf.reduce_mean(tf.square(pose_out - x))

    # if regularization is not None:
    #     loss_all = tf.add_n([loss] + [0.0005 * data_size * data_size * reconstruction_loss] + regularization)
    # else:
    #     loss_all = tf.add_n([loss] + [0.0005 * data_size * data_size * reconstruction_loss])

    # return loss_all, loss, reconstruction_loss, pose_out
    return loss


def spread_loss(y_true, y_pred, margin):
    activations_shape = y_pred.shape
    mask_t = tf.equal(y_true, 1)
    mask_i = tf.equal(y_true, 0)

    activations_t = tf.reshape(tf.boolean_mask(y_pred, mask_t), shape=(tf.shape(y_pred)[0], 1))
    activations_i = tf.reshape(tf.boolean_mask(y_pred, mask_i), [tf.shape(y_pred)[0], activations_shape[1] - 1])

    loss = tf.reduce_sum(tf.square(tf.maximum(0.0, margin - (activations_t - activations_i))))

    return loss
