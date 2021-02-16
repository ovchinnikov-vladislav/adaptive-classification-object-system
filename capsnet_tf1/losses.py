import tensorflow as tf


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    # return tf.reduce_mean(tf.square(y_pred))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))


def spread_loss(labels, activations, iterations_per_epoch, global_step, name):
    """Spread loss
    :param labels: (24, 10] in one-hot vector
    :param activations: [24, 10], activation for each class
    :param margin: increment from 0.2 to 0.9 during training

    :return: spread loss
    """

    # Margin schedule
    # Margin increase from 0.2 to 0.9 by an increment of 0.1 for every epoch
    margin = tf.compat.v1.train.piecewise_constant(tf.cast(global_step, dtype=tf.int32),
                                                   boundaries=[
                                                       (iterations_per_epoch * x) for x in range(1, 8)
                                                   ],
                                                   values=[
                                                       x / 10.0 for x in range(2, 10)
                                                   ]
                                                   )

    activations_shape = activations.shape

    # mask_t, mask_f Tensor (?, 10)
    mask_t = tf.equal(labels, 1)  # Mask for the true label
    mask_i = tf.equal(labels, 0)  # Mask for the non-true label

    # Activation for the true label
    # activations_t (?, 1)
    activations_t = tf.reshape(tf.boolean_mask(activations, mask_t), shape=(tf.shape(activations)[0], 1))

    # Activation for the other classes
    # activations_i (?, 9)
    activations_i = tf.reshape(tf.boolean_mask(activations, mask_i),
                               [tf.shape(activations)[0], activations_shape[1] - 1])
    l = tf.reduce_sum(tf.square(tf.maximum(0.0, margin - (activations_t - activations_i))))

    return l
