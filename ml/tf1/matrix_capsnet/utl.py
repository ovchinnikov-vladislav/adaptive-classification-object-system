"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com
"""

import tensorflow as tf
import numpy as np

def create_routing_map(child_space, k, s):
    parent_space = int((child_space - k) / s + 1)
    binmap = np.zeros((child_space ** 2, parent_space ** 2))
    for r in range(parent_space):
        for c in range(parent_space):
            p_idx = r * parent_space + c
            for i in range(k):
                # c_idx stand for child_index; p_idx is parent_index
                c_idx = r * s * child_space + c * s + child_space * i
                binmap[(c_idx):(c_idx + k), p_idx] = 1
    return binmap


def kernel_tile(inputs, kernel_size, strides, batch_size):
    input_shape = inputs.get_shape()
    spatial_size = int(input_shape[1])
    n_capsules = int(input_shape[3])
    parent_spatial_size = int((spatial_size - kernel_size) / strides + 1)
    assert input_shape[1] == input_shape[2]
    if len(input_shape) > 5:
        size = input_shape[4] * input_shape[5]
    else:
        size = 1
    child_parent_matrix = create_routing_map(spatial_size, kernel_size, strides)

    child_to_parent_idx = group_children_by_parent(child_parent_matrix)

    inputs = tf.reshape(inputs, [batch_size, spatial_size * spatial_size, -1])

    tiled = tf.gather(inputs, child_to_parent_idx, axis=1)

    tiled = tf.squeeze(tiled)
    tiled = tf.reshape(tiled, [batch_size, parent_spatial_size, parent_spatial_size, kernel_size * kernel_size,
                               n_capsules, -1])

    return tiled, child_parent_matrix


def compute_votes(poses_i, o, regularizer, tag=False):
    batch_size = int(poses_i.get_shape()[0])  # 64*5*5
    kh_kw_i = int(poses_i.get_shape()[1])  # 9*8

    output = tf.reshape(poses_i, shape=[batch_size, kh_kw_i, 1, 4, 4])
    initializer = tf.truncated_normal_initializer(mean=0.0,stddev=1.0)
    w = tf.Variable(lambda: initializer(shape=[1, kh_kw_i, o, 4, 4], dtype=tf.float32), name='w')
    w = tf.tile(w, [batch_size, 1, 1, 1, 1])
    output = tf.tile(output, [1, 1, o, 1, 1])
    mult = tf.matmul(output, w)
    votes = tf.reshape(mult, [batch_size, kh_kw_i, o, 16])

    return votes


def group_children_by_parent(bin_routing_map):
    tmp = np.where(np.transpose(bin_routing_map))
    children_per_parent = np.reshape(tmp[1], [bin_routing_map.shape[1], -1])

    return children_per_parent


def init_rr(spatial_routing_matrix, child_caps, parent_caps):
    parent_space_2 = int(spatial_routing_matrix.shape[1])
    parent_space = int(np.sqrt(parent_space_2))
    child_space_2 = int(spatial_routing_matrix.shape[0])
    child_space = int(np.sqrt(child_space_2))

    parents_per_child = np.sum(spatial_routing_matrix, axis=1, keepdims=True)

    rr_initial = (spatial_routing_matrix
                  / (parents_per_child * parent_caps + 1e-9))

    mask = spatial_routing_matrix.astype(bool)
    rr_initial = rr_initial.T[mask.T]
    rr_initial = np.reshape(rr_initial, [parent_space, parent_space, -1])

    rr_initial = rr_initial[..., np.newaxis, np.newaxis]
    rr_initial = np.tile(rr_initial, [1, 1, 1, child_caps, parent_caps])

    rr_initial = np.expand_dims(rr_initial, 0)

    dropped_child_caps = np.sum(np.sum(spatial_routing_matrix, axis=1) < 1e-9)
    effective_child_cap = ((child_space * child_space - dropped_child_caps)
                           * child_caps)

    sum_routing_weights = np.sum(rr_initial)

    assert np.abs(sum_routing_weights - effective_child_cap) < 1e-3

    return rr_initial


def to_sparse(probs, spatial_routing_matrix, sparse_filler=tf.log(1e-20)):
    shape = probs.get_shape().as_list()
    batch_size = shape[0]
    parent_space = shape[1]
    kk = shape[3]
    child_caps = shape[4]
    parent_caps = shape[5]

    child_space_2 = int(spatial_routing_matrix.shape[0])
    parent_space_2 = int(spatial_routing_matrix.shape[1])

    probs_unroll = tf.reshape(
        probs,
        [batch_size, parent_space_2, kk, child_caps, parent_caps])

    child_to_parent_idx = group_children_by_parent(spatial_routing_matrix)

    child_sparse_idx = child_to_parent_idx
    child_sparse_idx = child_sparse_idx[np.newaxis, ...]
    child_sparse_idx = np.tile(child_sparse_idx, [batch_size, 1, 1])

    parent_idx = np.arange(parent_space_2)
    parent_idx = np.reshape(parent_idx, [-1, 1])
    parent_idx = np.repeat(parent_idx, kk)
    parent_idx = np.tile(parent_idx, batch_size)
    parent_idx = np.reshape(parent_idx, [batch_size, parent_space_2, kk])

    batch_idx = np.arange(batch_size)
    batch_idx = np.reshape(batch_idx, [-1, 1])
    batch_idx = np.tile(batch_idx, parent_space_2 * kk)
    batch_idx = np.reshape(batch_idx, [batch_size, parent_space_2, kk])

    indices = np.stack((batch_idx, parent_idx, child_sparse_idx), axis=3)
    indices = tf.constant(indices)

    shape = [batch_size, parent_space_2, child_space_2, child_caps, parent_caps]
    sparse = tf.scatter_nd(indices, probs_unroll, shape)

    zeros_in_log = tf.ones_like(sparse, dtype=tf.float32) * sparse_filler
    sparse = tf.where(tf.equal(sparse, 0.0), zeros_in_log, sparse)

    sparse = tf.reshape(sparse, [batch_size, parent_space, parent_space, child_space_2, child_caps, parent_caps])

    assert sparse.get_shape().as_list() == [batch_size, parent_space, parent_space, child_space_2, child_caps,
                                            parent_caps]

    return sparse


def normalise_across_parents(probs_sparse, spatial_routing_matrix):
    shape = probs_sparse.get_shape().as_list()
    batch_size = shape[0]
    parent_space = shape[1]
    child_space_2 = shape[3]  # squared
    child_caps = shape[4]
    parent_caps = shape[5]

    rr_updated = probs_sparse / (tf.reduce_sum(probs_sparse,
                                               axis=[1, 2, 5],
                                               keepdims=True) + 1e-9)

    assert (rr_updated.get_shape().as_list()
            == [batch_size, parent_space, parent_space, child_space_2,
                child_caps, parent_caps])

    dropped_child_caps = np.sum(np.sum(spatial_routing_matrix, axis=1) < 1e-9)
    effective_child_caps = (child_space_2 - dropped_child_caps) * child_caps
    effective_child_caps = tf.to_double(effective_child_caps)

    sum_routing_weights = tf.reduce_sum(tf.to_double(rr_updated),
                                        axis=[1, 2, 3, 4, 5])

    pct_delta = tf.abs((effective_child_caps - sum_routing_weights)
                       / effective_child_caps)

    return rr_updated


def softmax_across_parents(probs_sparse, spatial_routing_matrix):
    shape = probs_sparse.get_shape().as_list()
    batch_size = shape[0]
    parent_space = shape[1]
    child_space_2 = shape[3]  # squared
    child_caps = shape[4]
    parent_caps = shape[5]

    sparse = tf.transpose(probs_sparse, perm=[0, 3, 4, 1, 2, 5])

    sparse = tf.reshape(sparse, [batch_size, child_space_2, child_caps, -1])

    parent_softmax = tf.nn.softmax(sparse, axis=-1)

    parent_softmax = tf.reshape(
        parent_softmax,
        [batch_size, child_space_2, child_caps, parent_space, parent_space,
         parent_caps])

    parent_softmax = tf.transpose(parent_softmax, perm=[0, 3, 4, 1, 2, 5])

    rr_updated = parent_softmax

    assert (rr_updated.get_shape().as_list()
            == [batch_size, parent_space, parent_space, child_space_2,
                child_caps, parent_caps])

    total_child_caps = tf.to_float(child_space_2 * child_caps * batch_size)
    sum_routing_weights = tf.round(tf.reduce_sum(rr_updated))

    return rr_updated


def to_dense(sparse, spatial_routing_matrix):
    shape = sparse.get_shape().as_list()
    batch_size = shape[0]
    parent_space = shape[1]
    child_space_2 = shape[3]  # squared
    child_caps = shape[4]
    parent_caps = shape[5]

    kk = int(np.sum(spatial_routing_matrix[:, 0]))

    sparse_unroll = tf.reshape(sparse, [batch_size, parent_space * parent_space,
                                        child_space_2, child_caps, parent_caps])

    dense = tf.boolean_mask(sparse_unroll,
                            tf.transpose(spatial_routing_matrix), axis=1)

    dense = tf.reshape(dense, [batch_size, parent_space, parent_space, kk,
                               child_caps, parent_caps])

    assert (dense.get_shape().as_list()
            == [batch_size, parent_space, parent_space, kk, child_caps,
                parent_caps])

    return dense


def logits_one_vs_rest(logits, positive_class=0):
    logits_positive = tf.reshape(logits[:, positive_class], [-1, 1])
    logits_rest = tf.concat([logits[:, :positive_class],
                             logits[:, (positive_class + 1):]], axis=1)
    logits_rest_max = tf.reduce_max(logits_rest, axis=1, keepdims=True)
    logits_one_vs_rest = tf.concat([logits_positive, logits_rest_max], axis=1)
    return logits_one_vs_rest


def em_routing(votes_ij, activations_i, batch_size, spatial_routing_matrix, routings, final_lambda):
    N = batch_size
    votes_shape = votes_ij.shape
    OH = np.sqrt(int(votes_shape[0]) / N)
    OH = int(OH)
    OW = np.sqrt(int(votes_shape[0]) / N)
    OW = int(OW)
    kh_kw_i = int(votes_shape[1])
    o = int(votes_shape[2])
    n_channels = int(votes_shape[3])

    kk = int(np.sum(spatial_routing_matrix[:, 0]))

    parent_caps = o
    child_caps = int(kh_kw_i / kk)

    rt_mat_shape = spatial_routing_matrix.shape
    child_space_2 = rt_mat_shape[0]
    child_space = int(np.sqrt(child_space_2))
    parent_space_2 = rt_mat_shape[1]
    parent_space = int(np.sqrt(parent_space_2))

    votes_ij = tf.reshape(votes_ij, [N, OH, OW, kh_kw_i, o, n_channels])
    activations_i = tf.reshape(activations_i, [N, OH, OW, kh_kw_i, 1, 1])

    xavier_initializer = tf.contrib.layers.xavier_initializer()
    truncated_initializer = tf.truncated_normal_initializer(mean=-1000.0, stddev=500.0)
    beta_v = tf.Variable(lambda: xavier_initializer(shape=[1, 1, 1, 1, o, 1], dtype=tf.float32), name='beta_v')
    beta_a = tf.Variable(lambda: truncated_initializer(shape=[1, 1, 1, 1, o, 1], dtype=tf.float32), name='beta_a')

    rr = init_rr(spatial_routing_matrix, child_caps, parent_caps)
    rr = np.reshape(rr, [1, parent_space, parent_space, kk * child_caps, parent_caps, 1])
    rr = tf.constant(rr, dtype=tf.float32)

    poses_j = activations_j = None
    for it in range(routings):
        final_lambda = final_lambda
        inverse_temperature = (final_lambda * (1 - tf.pow(0.95, tf.cast(it + 1, tf.float32))))

        activations_j, mean_j, stdv_j, var_j = m_step(rr, votes_ij, activations_i, beta_v, beta_a,
                                                      inverse_temperature=inverse_temperature)
        if it < routings - 1:
            rr = e_step(votes_ij, activations_j, mean_j, stdv_j, var_j, spatial_routing_matrix)
        poses_j = tf.squeeze(mean_j, axis=-3, name="poses")
        activations_j = tf.squeeze(activations_j, axis=-3, name="activations")

    return poses_j, activations_j


def m_step(rr, votes, activations_i, beta_v, beta_a, inverse_temperature):
    rr_prime = rr * activations_i
    rr_prime = tf.identity(rr_prime, name="rr_prime")
    rr_prime_sum = tf.reduce_sum(rr_prime, axis=-3, keepdims=True, name='rr_prime_sum')

    child_caps = float(rr_prime.get_shape().as_list()[-3])
    parent_caps = float(rr_prime.get_shape().as_list()[-2])
    ratio_child_to_parent = child_caps / parent_caps
    layer_norm_factor = 100 / ratio_child_to_parent

    mean_j_numerator = tf.reduce_sum(rr_prime * votes, axis=-3, keepdims=True, name="mean_j_numerator")
    mean_j = tf.div(mean_j_numerator, rr_prime_sum + 1e-9, name="mean_j")

    var_j_numerator = tf.reduce_sum(rr_prime * tf.square(votes - mean_j), axis=-3, keepdims=True,
                                    name="var_j_numerator")
    var_j = tf.div(var_j_numerator, rr_prime_sum + 1e-9, name="var_j")

    var_j = tf.identity(var_j + 1e-9, name="var_j_epsilon")
    stdv_j = None
    cost_j_h = (beta_v + 0.5 * tf.log(var_j)) * rr_prime_sum * layer_norm_factor
    cost_j_h = tf.identity(cost_j_h, name="cost_j_h")

    cost_j = tf.reduce_sum(cost_j_h, axis=-1, keepdims=True, name="cost_j")
    activations_j_cost = tf.identity(beta_a - cost_j, name="activations_j_cost")

    activations_j = tf.sigmoid(inverse_temperature * activations_j_cost, name="sigmoid")

    return activations_j, mean_j, stdv_j, var_j


def e_step(votes_ij, activations_j, mean_j, stdv_j, var_j, spatial_routing_matrix):
    o_p_unit0 = - tf.reduce_sum(tf.square(votes_ij - mean_j, name="num") / (2 * var_j), axis=-1, keepdims=True,
                                name="o_p_unit0")

    o_p_unit2 = - 0.5 * tf.reduce_sum(tf.log(2 * np.pi * var_j), axis=-1, keepdims=True, name="o_p_unit2")

    o_p = o_p_unit0 + o_p_unit2
    zz = tf.log(activations_j + 1e-9) + o_p

    zz_shape = zz.get_shape().as_list()
    batch_size = zz_shape[0]
    parent_space = zz_shape[1]
    kh_kw_i = zz_shape[3]
    parent_caps = zz_shape[4]
    kk = int(np.sum(spatial_routing_matrix[:, 0]))
    child_caps = int(kh_kw_i / kk)

    zz = tf.reshape(zz, [batch_size, parent_space, parent_space, kk, child_caps, parent_caps])

    sparse_filler = tf.minimum(tf.reduce_min(zz), -100)
    zz_sparse = to_sparse(zz, spatial_routing_matrix, sparse_filler=sparse_filler)
    rr_sparse = softmax_across_parents(zz_sparse, spatial_routing_matrix)
    rr_dense = to_dense(rr_sparse, spatial_routing_matrix)

    rr = tf.reshape(rr_dense, [batch_size, parent_space, parent_space, kh_kw_i, parent_caps, 1])

    return rr


def coord_addition(votes):
    height = votes.get_shape().as_list()[1]
    width = votes.get_shape().as_list()[2]
    dims = votes.get_shape().as_list()[-1]

    w_offset_vals = (np.arange(width) + 0.50) / float(width)
    h_offset_vals = (np.arange(height) + 0.50) / float(height)

    w_offset = np.zeros([width, dims])  # (5, 16)
    w_offset[:, 3] = w_offset_vals
    w_offset = np.reshape(w_offset, [1, 1, width, 1, 1, dims])

    h_offset = np.zeros([height, dims])
    h_offset[:, 7] = h_offset_vals
    h_offset = np.reshape(h_offset, [1, height, 1, 1, 1, dims])

    offset = w_offset + h_offset

    offset = tf.constant(offset, dtype=tf.float32)

    votes = tf.add(votes, offset, name="votes_with_coord_add")

    return votes
