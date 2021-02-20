import tensorflow as tf


def accuracy(logits, labels):
    logits = tf.identity(logits, name="logits")
    labels = tf.identity(labels, name="labels")
    batch_size = int(logits.get_shape()[0])
    logits_idx = tf.to_int32(tf.argmax(logits, axis=1))
    logits_idx = tf.reshape(logits_idx, shape=(batch_size,))
    correct_preds = tf.equal(tf.to_int32(labels), logits_idx)
    return tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / batch_size
