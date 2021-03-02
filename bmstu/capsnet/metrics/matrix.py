import tensorflow as tf


def matrix_accuracy(logits, labels):
    logits_idx = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    logits_idx = tf.reshape(logits_idx, shape=(logits.shape[0], ))
    correct_preds = tf.equal(tf.cast(labels, tf.int32), logits_idx)
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / logits.shape[0]

    return accuracy
