import tensorflow as tf
import numpy as np

epsilon = 1e-9


def kernel_tile(input, kernel, stride):
    # input (50, 12, 12, 8x(16+1) = 136)
    # output (50, 5, 5, 3x3=9, 136)

    input_shape = input.shape
    tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                  kernel * kernel], dtype=np.float32)
    for i in range(kernel):
        for j in range(kernel):
            tile_filter[i, j, :, i * kernel + j] = 1.0

    # (3, 3, 136, 9)
    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)

    # (50, 5, 5, 1224)
    output = tf.nn.depthwise_conv2d(input, tile_filter_op,
                                    strides=[1, stride, stride, 1], padding='VALID')

    output_shape = output.shape
    output = tf.reshape(output, shape=[output_shape[0], output_shape[1], output_shape[2], input_shape[3],
                                       kernel * kernel])
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

    # (50, 5, 5, 9, 136)
    return output


def mat_transform(input, caps_num_c, w):
    # input (1250, 72, 16), caps_num_c 16, w (1, 72, 16, 4, 4)
    batch_size = input.shape[0]  # 1250
    caps_num_i = input.shape[1]  # 72

    # (1250, 72, 1, 4, 4)
    output = tf.reshape(input, shape=[batch_size, caps_num_i, 1, 4, 4])

    # (1250, 72, 16, 4, 4)
    w = tf.tile(w, [batch_size, 1, 1, 1, 1])

    # (1250, 72, 16, 4, 4)
    output = tf.tile(output, [1, 1, caps_num_c, 1, 1])

    votes = tf.reshape(tf.matmul(output, w), [batch_size, caps_num_i, caps_num_c, 16])

    # (1250, 72, 16, 16)
    return votes


def em_routing(votes, activation, caps_num_c, beta_v, beta_a, routings=3, ac_lambda0=0.01):
    # votes (1250, 3x3x8=72, 16, 4x4), activation (1250, 72, 1), capsule 16, routings 3, ac_lambda 0.01
    # beta_v (16, 16), beta_a (16,)

    test = []

    batch_size = votes.shape[0]  # 1250
    caps_num_i = activation.shape[1]  # 72
    n_channels = votes.shape[-1]  # 16

    sigma_square = []
    miu = []
    activation_out = []

    votes_in = votes
    activation_in = activation

    for iters in range(routings):

        # e-step
        if iters == 0:
            # (1250, 72, 16)
            r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / caps_num_c)
        else:
            # (1250, 72, 16, 16)
            log_p_c_h = -tf.math.log(tf.sqrt(sigma_square)) - (tf.square(votes_in - miu) / (2 * sigma_square))

            # (1250, 72, 16, 16)
            log_p_c_h = log_p_c_h - (tf.reduce_max(log_p_c_h, axis=[2, 3], keepdims=True) - tf.math.log(10.0))
            p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3))

            # (1250, 72, 16) * (1250, 1, 16) -> (1250, 72, 16)
            ap = p_c * tf.reshape(activation_out, shape=[-1, 1, caps_num_c])

            # (1250, 72, 16)
            r = ap / (tf.reduce_sum(ap, axis=2, keepdims=True) + epsilon)

        # m-step
        r = r * activation_in
        r = r / (tf.reduce_sum(r, axis=2, keepdims=True) + epsilon)

        # (1250, 1, 16)
        r_sum = tf.reduce_sum(r, axis=1, keepdims=True)

        # (1250, 72, 16, 1)
        r1 = tf.reshape(r / (r_sum + epsilon), shape=[-1, caps_num_i, caps_num_c, 1])

        # (1250, 1, 16, 16)
        miu = tf.reduce_sum(votes_in * r1, axis=1, keepdims=True)

        # (1250, 1, 16, 16)
        sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1, axis=1, keepdims=True) + epsilon

        if iters == routings - 1:
            # (1250, 16, 1)
            r_sum = tf.reshape(r_sum, [-1, caps_num_c, 1])

            # (1250, 16, 16)
            cost_h = (beta_v + tf.math.log(tf.sqrt(
                tf.reshape(sigma_square, shape=[-1, caps_num_c, n_channels])))) * r_sum

            # (1250, 16)
            activation_out = tf.nn.softmax(ac_lambda0 * (beta_a - tf.reduce_sum(cost_h, axis=2)))
        else:
            # (1250, 1, 16)
            activation_out = tf.nn.softmax(r_sum)

    # (1250, 1, 16, 16), (1250, 16)
    return miu, activation_out, test
