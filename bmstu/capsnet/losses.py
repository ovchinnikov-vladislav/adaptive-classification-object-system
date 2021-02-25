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


def spread_loss(y_true, y_pred, margin):
    """Spread loss
    :param margin:
    :param y_true: (24, 10] in one-hot vector
    :param y_pred: [24, 10], activation for each class
    :return: spread loss

    margin: increment from 0.2 to 0.9 during training
    """

    activations_shape = y_pred.shape

    # mask_t, mask_f Tensor (?, 10)
    mask_t = tf.equal(y_true, 1)  # Mask for the true label
    mask_i = tf.equal(y_true, 0)  # Mask for the non-true label

    # Activation for the true label
    # activations_t (?, 1)
    activations_t = tf.reshape(tf.boolean_mask(y_pred, mask_t), shape=(tf.shape(y_pred)[0], 1))

    # Activation for the other classes
    # activations_i (?, 9)
    activations_i = tf.reshape(tf.boolean_mask(y_pred, mask_i),
                               [tf.shape(y_pred)[0], activations_shape[1] - 1])

    loss = tf.reduce_sum(tf.square(tf.maximum(0.0, margin - (activations_t - activations_i))))
    return loss
