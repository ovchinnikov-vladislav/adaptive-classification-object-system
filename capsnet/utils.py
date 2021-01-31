import tensorflow.keras.backend as keras_backend
import tensorflow as tf
from keras import layers
import numpy as np


def keras_squeeze(inputs):
    newdim = tuple([x for x in inputs.shape.as_list() if x != 1 and x is not None])
    return layers.Reshape(newdim)(inputs)


def update_locally_constrained_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim, num_routing):
    if num_dims == 6:
        votes_t_shape = [5, 0, 1, 2, 3, 4]
        r_t_shape = [1, 2, 3, 4, 5, 0]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = tf.transpose(votes, votes_t_shape)
    _, _, _, height, width, caps = votes_trans.get_shape()

    def _body(i, logits, activations):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        route = tf.nn.softmax(logits)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        activation = squash(preactivate)
        activations = activations.write(i, activation)
        act_3d = keras_backend.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=-1)
        logits += distances
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < num_routing,
        _body,
        loop_vars=[i, logits, activations],
        swap_memory=True)

    return keras_backend.cast(activations.read(num_routing - 1), dtype='float32')


def squash(vectors, axis=-1):
    """
    Функция нелинейной активации, используемая в капсулах. Она приводит длину большого вектора к значению близкому 1,
    а малый вектор к значению до 0.
    :param vectors: некоторые векторы для сжатия (операции squash), N-разерный тензор
    :param axis: ось для сжатия (операции squash)
    :return: на входе тензор той же формы (shape) что и входные векторы
    """

    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + keras_backend.epsilon())
    return scale * vectors
