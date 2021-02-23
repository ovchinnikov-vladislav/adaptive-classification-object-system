import tensorflow as tf


def matrix_accuracy(y_true, y_pred):
    logits = tf.identity(y_pred, name="logits")
    labels = tf.identity(y_true, name="labels")
    batch_size = int(logits.shape[0])
    logits_idx = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    logits_idx = tf.reshape(logits_idx, shape=(batch_size,))
    correct_preds = tf.equal(tf.cast(labels, tf.int32), logits_idx)
    accuracy = (tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / batch_size)

    return accuracy
