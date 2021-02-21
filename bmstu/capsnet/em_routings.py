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
    size = input_shape[4] * input_shape[5] if len(input_shape) > 5 else 1
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


def matrix_capsules_em_routing(votes, activations, beta_v, beta_a, routings):
    """The EM routing between input capsules (i) and output capsules (j).
    :param votes: (N, OH, OW, kh x kw x i, o, 4 x 4) = (24, 6, 6, 3x3*32=288, 32, 16)
    :param activations: activation from Level L (24, 6, 6, 288)
    :param beta_v: (1, 1, 1, 32)
    :param beta_a: (1, 1, 1, 32)
    :param routings: number of iterations in EM routing, often 3.
    :return: (pose, activation) of output capsules.
    """

    votes_shape = votes.shape

    # Match rr (routing assignment) shape, i_activations shape with votes shape for broadcasting in EM routing

    # rr: [3x3x32=288, 32, 1]
    # rr: routing matrix from each input capsule (i) to each output capsule (o)
    rr = tf.constant(1.0 / votes_shape[-2], shape=votes_shape[-3:-1] + [1], dtype=tf.float32)

    # i_activations: expand_dims to (24, 6, 6, 288, 1, 1)
    activations = activations[..., tf.newaxis, tf.newaxis]

    # beta_v and beta_a: expand_dims to (1, 1, 1, 1, 32, 1]
    beta_v = beta_v[..., tf.newaxis, :, tf.newaxis]
    beta_a = beta_a[..., tf.newaxis, :, tf.newaxis]

    # inverse_temperature schedule (min, max)
    it_min = 1.0
    it_max = min(routings, 3.0)
    o_mean = o_activations = None
    for it in range(routings):
        inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, routings - 1.0)
        o_mean, o_stdv, o_activations = m_step(rr, votes, activations, beta_v, beta_a,
                                               inverse_temperature=inverse_temperature)
        if it < routings - 1:
            rr = e_step(o_mean, o_stdv, o_activations, votes)

    # pose: (N, OH, OW, o 4 x 4) via squeeze o_mean (24, 6, 6, 32, 16)
    poses = tf.squeeze(o_mean, axis=-3)

    # activation: (N, OH, OW, o) via squeeze o_activations [24, 6, 6, 32]
    activations = tf.squeeze(o_activations, axis=[-3, -1])

    return poses, activations


def m_step(rr, votes, activations, beta_v, beta_a, inverse_temperature):
    """The M-Step in EM Routing from input capsules i to output capsule j.
    i: input capsules (32)
    o: output capsules (32)
    h: 4x4 = 16
    output spatial dimension: 6x6
    :param rr: routing assignments. shape = (kh x kw x i, o, 1) =(3x3x32, 32, 1) = (288, 32, 1)
    :param votes: shape = (N, OH, OW, kh x kw x i, o, 4x4) = (24, 6, 6, 288, 32, 16)
    :param activations: input capsule activation (at Level L). (N, OH, OW, kh x kw x i, 1, 1) = (24, 6, 6, 288, 1, 1)
       with dimensions expanded to match votes for broadcasting.
    :param beta_v: Trainable parameters in computing cost (1, 1, 1, 1, 32, 1)
    :param beta_a: Trainable parameters in computing next level activation (1, 1, 1, 1, 32, 1)
    :param inverse_temperature: lambda, increase over each iteration by the caller.
    :return: (o_mean, o_stdv, o_activation)
    """

    rr_prime = rr * activations

    # rr_prime_sum: sum over all input capsule i
    rr_prime_sum = tf.reduce_sum(rr_prime, axis=-3, keep_dims=True, name='rr_prime_sum')

    # Mean of the output capsules: o_mean(24, 6, 6, 1, 32, 16)
    o_mean = tf.reduce_sum(rr_prime * votes, axis=-3, keep_dims=True) / rr_prime_sum

    # Standard deviation of the output capsule:  o_stdv (24, 6, 6, 1, 32, 16)
    o_stdv = tf.sqrt(tf.reduce_sum(rr_prime * tf.square(votes - o_mean), axis=-3, keep_dims=True) / rr_prime_sum)

    # o_cost_h: (24, 6, 6, 1, 32, 16)
    o_cost_h = (beta_v + tf.math.log(o_stdv + epsilon)) * rr_prime_sum

    # o_cost: (24, 6, 6, 1, 32, 1)
    # o_activations_cost = (24, 6, 6, 1, 32, 1)
    # For numeric stability.
    o_cost = tf.reduce_sum(o_cost_h, axis=-1, keep_dims=True)
    o_cost_mean = tf.reduce_mean(o_cost, axis=-2, keep_dims=True)
    o_cost_stdv = tf.sqrt(tf.reduce_sum(tf.square(o_cost - o_cost_mean), axis=-2, keep_dims=True) / o_cost.shape[-2])
    o_activations_cost = beta_a + (o_cost_mean - o_cost) / (o_cost_stdv + epsilon)

    # (24, 6, 6, 1, 32, 1)
    o_activations = tf.nn.sigmoid(inverse_temperature * o_activations_cost)

    return o_mean, o_stdv, o_activations


def e_step(o_mean, o_stdv, o_activations, votes):
    """The E-Step in EM Routing.
    :param o_mean: (24, 6, 6, 1, 32, 16)
    :param o_stdv: (24, 6, 6, 1, 32, 16)
    :param o_activations: (24, 6, 6, 1, 32, 1)
    :param votes: (24, 6, 6, 288, 32, 16)
    :return: rr
    """

    o_p_unit0 = - tf.reduce_sum(tf.square(votes - o_mean) / (2 * tf.square(o_stdv)), axis=-1, keep_dims=True)

    o_p_unit2 = - tf.reduce_sum(tf.math.log(o_stdv + epsilon), axis=-1, keep_dims=True)

    # o_p is the probability density of the h-th component of the vote from i to j
    # (24, 6, 6, 1, 32, 16)
    o_p = o_p_unit0 + o_p_unit2

    # rr: (24, 6, 6, 288, 32, 1)
    zz = tf.math.log(o_activations + epsilon) + o_p
    rr = tf.nn.softmax(zz, dim=len(zz.get_shape().as_list()) - 2)

    return rr
