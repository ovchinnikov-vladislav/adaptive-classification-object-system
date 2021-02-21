import tensorflow as tf


def t_score(c_ij):
    out_caps = tf.cast(tf.shape(c_ij)[1], tf.float32)
    c_ij = tf.squeeze(c_ij, axis=3)
    c_ij = tf.transpose(c_ij, [0, 2, 1])

    epsilon = 1e-12
    entropy = -tf.reduce_sum(c_ij * tf.math.log(c_ij + epsilon), axis=-1)
    T = 1 - entropy / -tf.math.log(1 / out_caps)
    return tf.reduce_mean(T)


def d_score(v_j):
    v_j_norm = tf.norm(v_j, axis=-1)
    v_j_std = tf.math.reduce_std(v_j_norm, axis=0)
    return tf.reduce_max(v_j_std)


def v_map(v_j):
    v_j_norm = tf.norm(v_j, axis=-1)
    v_j_norm = tf.expand_dims(v_j_norm, 0)
    v_j_norm = tf.expand_dims(v_j_norm, -1)
    return v_j_norm
