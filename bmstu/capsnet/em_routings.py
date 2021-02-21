import tensorflow as tf
import numpy as np
from tensorflow.keras.backend import epsilon


def kernel_tile(inputs, kernel, stride):
    """This constructs a primary capsule layer using regular convolution layer.
    :param inputs: shape (?, 14, 14, 32, 4, 4)
    :param kernel: 3
    :param stride: 2
    :return output: (50, 5, 5, 3x3=9, 136)
    """

    # (?, 14, 14, 32x(16)=512)
    input_shape = inputs.shape
    size = input_shape[4]*input_shape[5] if len(input_shape) > 5 else 1
    inputs = tf.reshape(inputs, shape=[-1, input_shape[1], input_shape[2], input_shape[3] * size])

    tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3], kernel * kernel], dtype=np.float32)
    for i in range(kernel):
        for j in range(kernel):
            # (3, 3, 512, 9)
            tile_filter[i, j, :, i * kernel + j] = 1.0

    # (3, 3, 512, 9)
    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)

    # (?, 6, 6, 4608)
    output = tf.nn.depthwise_conv2d(inputs, tile_filter_op, strides=[
                                    1, stride, stride, 1], padding='VALID')

    output_shape = output.get_shape()
    output = tf.reshape(output, shape=[-1, output_shape[1], output_shape[2], input_shape[3], kernel * kernel])
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

    # (?, 6, 6, 9, 512)
    return output


def mat_transform(inputs, output_cap_size, size):
    """Compute the vote.
    :param inputs: shape (size, 288, 16)
    :param output_cap_size: 32
    :param size:
    :return votes: (24, 5, 5, 3x3=9, 136)
    """

    # 288
    caps_num_i = int(inputs.shape[1])
    # (size, 288, 1, 4, 4)
    output = tf.reshape(inputs, shape=[size, caps_num_i, 1, 4, 4])

    truncated_normal_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0)
    # (1, 288, 32, 4, 4)
    w = tf.Variable(initial_value=truncated_normal_initializer(shape=[1, caps_num_i, output_cap_size, 4, 4],
                                                               dtype=tf.float32), name='w')
    # (24, 288, 32, 4, 4)
    w = tf.tile(w, [size, 1, 1, 1, 1])
    # (size, 288, 32, 4, 4)
    output = tf.tile(output, [1, 1, output_cap_size, 1, 1])
    # (24, 288, 32, 4, 4)
    votes = tf.matmul(output, w)
    # (size, 288, 32, 16)
    votes = tf.reshape(votes, [size, caps_num_i, output_cap_size, 16])

    return votes


def coord_addition(votes, H, W):
    """Coordinate addition.
    :param votes: (24, 4, 4, 32, 10, 16)
    :param H: spatial height 4
    :param W: spatial width 4
    :return votes: (24, 4, 4, 32, 10, 16)
    """
    coordinate_offset_hh = tf.reshape((tf.range(H, dtype=tf.float32) + 0.50) / H, [1, H, 1, 1, 1])
    coordinate_offset_h0 = tf.constant(0.0, shape=[1, H, 1, 1, 1], dtype=tf.float32)
    # (1, 4, 1, 1, 1, 16)
    coordinate_offset_h = tf.stack([coordinate_offset_hh, coordinate_offset_h0]
                                   + [coordinate_offset_h0 for _ in range(14)], axis=-1)

    coordinate_offset_ww = tf.reshape((tf.range(W, dtype=tf.float32) + 0.50) / W, [1, 1, W, 1, 1])
    coordinate_offset_w0 = tf.constant(0.0, shape=[1, 1, W, 1, 1], dtype=tf.float32)
    # (1, 1, 4, 1, 1, 16)
    coordinate_offset_w = tf.stack([coordinate_offset_w0, coordinate_offset_ww]
                                   + [coordinate_offset_w0 for _ in range(14)], axis=-1)

    # (24, 4, 4, 32, 10, 16)
    votes = votes + coordinate_offset_h + coordinate_offset_w

    return votes
