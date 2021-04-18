import tensorflow as tf
from tensorflow.keras.backend import epsilon, ndim, expand_dims
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


# v = ((||sj||^2) / (1 + ||sj||^2)) * (sj / ||sj||)
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)

    additional_squashing = s_squared_norm / (1 + s_squared_norm)
    unit_scaling = vectors / tf.sqrt(s_squared_norm + epsilon())

    return additional_squashing * unit_scaling


# v = (1 - 1 / e^(||s||)) * (s / ||s||)
def efficient_squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    s_norm = tf.sqrt(s_squared_norm + epsilon())

    return (1 - 1 / (tf.math.exp(s_norm))) * (vectors / s_norm)


def own_batch_dot(x, y, axes=None):
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [x_ndim - 1, y_ndim - 2]
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = array_ops.reshape(y,
                              array_ops.concat(
                                  [array_ops.shape(y), [1] * diff], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = array_ops.reshape(x,
                              array_ops.concat(
                                  [array_ops.shape(x), [1] * diff], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = math_ops.reduce_sum(math_ops.multiply(x, y), axes[0])
        else:
            out = math_ops.reduce_sum(
                math_ops.multiply(array_ops.transpose(x, [1, 0]), y), axes[1])
    else:
        adj_x = None if axes[0] == ndim(x) - 1 else True
        adj_y = True if axes[1] == ndim(y) - 1 else None
        out = math_ops.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = array_ops.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out
