import tensorflow as tf


def matrix_accuracy(labels, logits):
    logits_idx = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    logits_idx = tf.reshape(logits_idx, shape=(logits.shape[0], ))
    labels_idx = tf.cast(tf.argmax(labels, axis=1), tf.int32)
    labels_idx = tf.reshape(labels_idx, shape=(labels.shape[0],))
    correct_preds = tf.equal(labels_idx, logits_idx)
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / logits.shape[0]

    return accuracy
