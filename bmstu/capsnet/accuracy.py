import tensorflow as tf


def matrix_accuracy(y_true, y_pred):
    logits_idx = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
    logits_idx = tf.reshape(logits_idx, shape=(y_true[0], -1))
    correct_preds = tf.equal(tf.cast(y_true, tf.int32), logits_idx)
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / y_true[0]

    return accuracy
