import tensorflow as tf


def _pdist(a, b=None):
    sq_sum_a = tf.reduce_sum(tf.square(a), axis=1)
    if b is None:
        return -2 * tf.matmul(a, tf.transpose(a)) + tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_a, (1, -1))
    sq_sum_b = tf.reduce_sum(tf.square(b), axis=1)
    return -2 * tf.matmul(a, tf.transpose(b)) + tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_b, (1, -1))


def soft_margin_triplet_loss(y_true, y_pred):
    eps = tf.constant(1e-5, tf.float32)
    nil = tf.constant(0., tf.float32)
    almost_inf = tf.constant(1e+10, tf.float32)

    squared_distance_mat = _pdist(y_pred)
    distance_mat = tf.sqrt(tf.maximum(nil, eps + squared_distance_mat))
    label_mat = tf.cast(tf.equal(tf.reshape(y_true, (-1, 1)), tf.reshape(y_true, (1, -1))), tf.float32)

    positive_distance = tf.reduce_max(label_mat * distance_mat, axis=1)
    negative_distance = tf.reduce_min((label_mat * almost_inf) + distance_mat, axis=1)
    loss = tf.nn.softplus(positive_distance - negative_distance)

    return tf.reduce_mean(loss)


def magnet_loss(y_true, y_pred, margin=1.0, unique_labels=None):
    nil = tf.constant(0., tf.float32)
    one = tf.constant(1., tf.float32)
    minus_two = tf.constant(-2., tf.float32)
    eps = tf.constant(1e-4, tf.float32)
    margin = tf.constant(margin, tf.float32)

    num_per_class = None
    if unique_labels is None:
        unique_labels, sample_to_unique_y, num_per_class = tf.unique_with_counts(y_true)
        num_per_class = tf.cast(num_per_class, tf.float32)

    y_mat = tf.cast(tf.equal(tf.reshape(y_true, (-1, 1)), tf.reshape(unique_labels, (1, -1))), dtype=tf.float32)

    # If class_means is None, compute from batch data.
    if num_per_class is None:
        num_per_class = tf.reduce_sum(y_mat, axis=0)
    class_means = tf.reduce_sum(tf.expand_dims(tf.transpose(y_mat), -1) * tf.expand_dims(y_pred, 0),
                                axis=1) / tf.expand_dims(num_per_class, -1)

    squared_distance = _pdist(y_pred, class_means)

    num_samples = tf.cast(tf.shape(y_true)[0], tf.float32)
    variance = tf.reduce_sum(y_mat * squared_distance) / (num_samples - one)

    const = one / (minus_two * (variance + eps))
    linear = const * squared_distance - y_mat * margin

    maxi = tf.reduce_max(linear, axis=1, keepdims=True)
    loss_mat = tf.exp(linear - maxi)

    a = tf.reduce_sum(y_mat * loss_mat, axis=1)
    b = tf.reduce_sum((one - y_mat) * loss_mat, axis=1)
    loss = tf.maximum(nil, -tf.math.log(eps + a / (eps + b)))
    return tf.reduce_mean(loss), class_means, variance
