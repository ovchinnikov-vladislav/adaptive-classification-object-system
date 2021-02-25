import tensorflow as tf
import numpy as np

epsilon = 1e-9


def create_routing_map(child_space, k, s):
    """Generate TFRecord for train and test datasets from .mat files.
    Create a binary map where the rows are capsules in the lower layer (children)
    and the columns are capsules in the higher layer (parents). The binary map
    shows which children capsules are connected to which parent capsules along the   spatial dimension.
    Args:
      child_space: spatial dimension of lower capsule layer
      k: kernel size
      s: stride
    Returns:
      binmap:
        A 2D numpy matrix containing mapping between children capsules along the
        rows, and parent capsules along the columns.
        (child_space^2, parent_space^2)
        (7*7, 5*5)
    """

    parent_space = int((child_space - k) / s + 1)
    binmap = np.zeros((child_space ** 2, parent_space ** 2))
    for r in range(parent_space):
        for c in range(parent_space):
            p_idx = r * parent_space + c
            for i in range(k):
                # c_idx stand for child_index; p_idx is parent_index
                c_idx = r * s * child_space + c * s + child_space * i
                binmap[c_idx:(c_idx + k), p_idx] = 1
    return binmap


def group_children_by_parent(bin_routing_map):
    """Groups children capsules by parent capsule.
    Rearrange the bin_routing_map so that each row represents one parent capsule,   and the entries in the row
    are indexes of the children capsules that route to   that parent capsule. This mapping is only along the spatial
    dimension, each child capsule along in spatial dimension will actually contain many capsules,   e.g. 32.
    The grouping that we are doing here tell us about the spatial
    routing, e.g. if the lower layer is 7x7 in spatial dimension, with a kernel of
    3 and stride of 1, then the higher layer will be 5x5 in the spatial dimension.
    So this function will tell us which children from the 7x7=49 lower capsules
    map to each of the 5x5=25 higher capsules. One child capsule can be in several
    different parent capsules, children in the corners will only belong to one
    parent, but children towards the center will belong to several with a maximum   of kernel*kernel (e.g. 9),
    but also depending on the stride.
    Args:
      bin_routing_map:
        binary routing map with children as rows and parents as columns
    Returns:
      children_per_parents:
        parents are rows, and the indexes in the row are which children belong to       that parent
    """

    tmp = np.where(np.transpose(bin_routing_map))
    children_per_parent = np.reshape(tmp[1], [bin_routing_map.shape[1], -1])

    return children_per_parent


def kernel_tile(inputs, kernel, stride):
    """Tile the children poses/activations so that the children for each parent occur in one axis.
    Args:
      inputs:
        tensor of child poses or activations
        poses (N, child_space, child_space, i, 4, 4) -> (64, 7, 7, 8, 4, 4)
        activations (N, child_space, child_space, i, 1) -> (64, 7, 7, 8, 16)
      kernel:
      stride:
    Returns:
      tiled:
        (N, parent_space, parent_space, kh*kw, i, 16 or 1)
        (64, 5, 5, 9, 8, 16 or 1)
      child_parent_matrix:
        A 2D numpy matrix containing mapping between children capsules along the
        rows, and parent capsules along the columns.
        (child_space^2, parent_space^2)
        (7*7, 5*5)
    """

    input_shape = inputs.shape
    batch_size = int(input_shape[0])
    spatial_size = int(input_shape[1])
    n_capsules = int(input_shape[3])
    parent_spatial_size = int((spatial_size - kernel) / stride + 1)

    # Matrix showing which children map to which parent. Children are rows,
    # parents are columns.
    child_parent_matrix = create_routing_map(spatial_size, kernel, stride)

    # Convert from np to tf
    # child_parent_matrix = tf.constant(child_parent_matrix)

    # Each row contains the children belonging to one parent
    child_to_parent_idx = group_children_by_parent(child_parent_matrix)

    # Spread out spatial dimension of children
    inputs = tf.reshape(inputs, [batch_size, spatial_size * spatial_size, -1])

    # Select which children go to each parent capsule
    tiled = tf.gather(inputs, child_to_parent_idx, axis=1)

    tiled = tf.squeeze(tiled)
    tiled = tf.reshape(tiled, [batch_size, parent_spatial_size, parent_spatial_size, kernel * kernel, n_capsules, -1])

    return tiled, child_parent_matrix


def compute_votes(poses_i, capsules, w):
    """Compute the votes by multiplying input poses by transformation matrix.
    Multiply the poses of layer i by the transform matrix to compute the votes for
    layer j.
    Args:
      poses_i:
        poses in layer i tiled according to the kernel
        (N*OH*OW, kh*kw*i, 16)
        (64*5*5, 9*8, 16)
      capsules: number of output capsules, also called "parent_caps"
      w:
    Returns:
      votes:
        (N*OH*OW, kh*kw*i, o, 16)
        (64*5*5, 9*8, 32, 16)
    """

    batch_size = int(poses_i.shape[0])  # 64*5*5
    kh_kw_i = int(poses_i.shape[1])  # 9*8

    # (64*5*5, 9*8, 16) -> (64*5*5, 9*8, 1, 4, 4)
    output = tf.reshape(poses_i, shape=[batch_size, kh_kw_i, 1, 4, 4])

    # the output of capsule is miu, the mean of a Gaussian, and activation, the
    # sum of probabilities it has no relationship with the absolute values of w
    # and votes using weights with bigger stddev helps numerical stability
    # (1, 9*8, 32, 4, 4) -> (64*5*5, 9*8, 32, 4, 4)
    w = tf.tile(w, [batch_size, 1, 1, 1, 1])

    # (64*5*5, 9*8, 1, 4, 4) -> (64*5*5, 9*8, 32, 4, 4)
    output = tf.tile(output, [1, 1, capsules, 1, 1])

    # (64*5*5, 9*8, 32, 4, 4) x (64*5*5, 9*8, 32, 4, 4)
    # -> (64*5*5, 9*8, 32, 4, 4)
    mult = tf.matmul(output, w)

    # (64*5*5, 9*8, 32, 4, 4) -> (64*5*5, 9*8, 32, 16)
    votes = tf.reshape(mult, [batch_size, kh_kw_i, capsules, 16])

    # tf.summary.histogram('w', w)

    return votes


def matrix_capsules_em_routing(votes, activation, spatial_routing_matrix,
                               batch_size, routings, final_lambda, beta_a, beta_v):
    N = batch_size
    votes_shape = votes.shape
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
    parent_space_2 = rt_mat_shape[1]
    parent_space = int(np.sqrt(parent_space_2))

    votes = tf.reshape(votes, [N, OH, OW, kh_kw_i, o, n_channels])

    # (N*OH*OW, kh*kw*i, 1) -> (N, OH, OW, kh*kw*i, o, n_channels)
    #              (24, 6, 6, 288, 1, 1)
    activation = tf.reshape(activation, [N, OH, OW, kh_kw_i, 1, 1])

    # Initialise routing assignments
    # rr (1, 6, 6, 9, 8, 16)
    #  (1, parent_space, parent_space, kk, child_caps, parent_caps)
    rr = init_rr(spatial_routing_matrix, child_caps, parent_caps)

    # Need to reshape (1, 6, 6, 9, 8, 16) -> (1, 6, 6, 9*8, 16, 1)
    rr = np.reshape(rr, [1, parent_space, parent_space, kk * child_caps, parent_caps, 1])

    # Convert rr from np to tf
    rr = tf.constant(rr, dtype=tf.float32)

    mean_j = activations_j = None
    for it in range(routings):
        # AG 17/09/2018: modified schedule for inverse_temperature (lambda) based
        # on Hinton's response to questions on OpenReview.net:
        # https://openreview.net/forum?id=HJWLfGWRb
        # "the formula we used for lambda is:
        # lambda = final_lambda * (1 - tf.pow(0.95, tf.cast(i + 1, tf.float32)))
        # where 'i' is the routing iteration (range is 0-2). Final_lambda is set
        # to 0.01."
        # final_lambda = 0.01
        inverse_temperature = (final_lambda *
                               (1 - tf.pow(0.95, tf.cast(it + 1, tf.float32))))

        # AG 26/06/2018: added var_j
        activations_j, mean_j, stdv_j, var_j = m_step(rr, votes, activation,
                                                      beta_v, beta_a,
                                                      inverse_temperature=inverse_temperature)

        # We skip the e_step call in the last iteration because we only need to
        # return the a_j and the mean from the m_stp in the last iteration to
        # compute the output capsule activation and pose matrices
        if it < routings - 1:
            rr = e_step(votes, activations_j, mean_j, var_j, spatial_routing_matrix)

    # pose: (N, OH, OW, o, 4 x 4) via squeeze mean_j (24, 6, 6, 32, 16)
    poses_j = tf.squeeze(mean_j, axis=-3, name="poses")

    # activation: (N, OH, OW, o, 1) via squeeze o_activation is
    # [24, 6, 6, 32, 1]
    activations_j = tf.squeeze(activations_j, axis=-3, name="activations")

    return poses_j, activations_j


def init_rr(spatial_routing_matrix, child_caps, parent_caps):
    # Get spatial dimension of parent & child
    parent_space_2 = int(spatial_routing_matrix.shape[1])
    parent_space = int(np.sqrt(parent_space_2))
    child_space_2 = int(spatial_routing_matrix.shape[0])
    child_space = int(np.sqrt(child_space_2))

    # Count the number of parents that each child belongs to
    parents_per_child = np.sum(spatial_routing_matrix, axis=1, keepdims=True)

    # Divide the vote of each child by the number of parents that it belongs to
    # If the striding causes the filter not to fit, it will result in some
    # "dropped" child capsules, which effectively means child capsules that do not
    # have any parents. This would create a divide by 0 scenario, so need to add
    # 1e-9 to prevent NaNs.
    rr_initial = (spatial_routing_matrix / (parents_per_child * parent_caps + 1e-9))

    # Convert the sparse matrix to be compatible with votes.
    # This is done by selecting the child capsules belonging to each parent, which
    # is achieved by selecting the non-zero values down each column. Need the
    # combination of two transposes so that order is correct when reshaping
    mask = spatial_routing_matrix.astype(bool)
    rr_initial = rr_initial.T[mask.T]
    rr_initial = np.reshape(rr_initial, [parent_space, parent_space, -1])

    # Copy values across depth dimensions
    # i.e. the number of child_caps and the number of parent_caps
    # (5, 5, 9) -> (5, 5, 9, 8, 32)
    rr_initial = rr_initial[..., np.newaxis, np.newaxis]
    rr_initial = np.tile(rr_initial, [1, 1, 1, child_caps, parent_caps])

    # Add one mode dimension for batch size
    rr_initial = np.expand_dims(rr_initial, 0)

    # Check the total of the routing weights is equal to the number of child
    # capsules
    # child_space * child_space * child_caps (minus the dropped ones)
    dropped_child_caps = np.sum(np.sum(spatial_routing_matrix, axis=1) < 1e-9)
    effective_child_cap = ((child_space * child_space - dropped_child_caps)
                           * child_caps)

    sum_routing_weights = np.sum(rr_initial)

    assert np.abs(sum_routing_weights - effective_child_cap) < 1e-3

    return rr_initial


def m_step(rr, votes, activations_i, beta_v, beta_a, inverse_temperature):
    rr_prime = rr * activations_i
    rr_prime = tf.identity(rr_prime, name="rr_prime")

    # rr_prime_sum: sum over all input capsule i
    rr_prime_sum = tf.reduce_sum(rr_prime, axis=-3, keepdims=True, name='rr_prime_sum')

    # AG 13/12/2018: normalise amount of information
    # The amount of information given to parent capsules is very different for
    # the final "class-caps" layer. Since all the spatial capsules give output
    # to just a few class caps, they receive a lot more information than the
    # convolutional layers. So in order for lambda and beta_v/beta_a settings to
    # apply to this layer, we must normalise the amount of information.
    # activ from convcaps1 to convcaps2 (64*5*5, 144, 16, 1) 144/16 = 9 info
    # (N*OH*OW, kh*kw*i, o, 1)
    # activ from convcaps2 to classcaps (64, 1, 1, 400, 5, 1) 400/5 = 80 info
    # (N, 1, 1, IH*IW*i, n_classes, 1)
    child_caps = float(rr_prime.get_shape().as_list()[-3])
    parent_caps = float(rr_prime.get_shape().as_list()[-2])
    ratio_child_to_parent = child_caps / parent_caps
    layer_norm_factor = 100 / ratio_child_to_parent
    # logger.info("ratio_child_to_parent: {}".format(ratio_child_to_parent))
    # rr_prime_sum = rr_prime_sum/ratio_child_to_parent

    # mean_j: (24, 6, 6, 1, 32, 16)
    mean_j_numerator = tf.reduce_sum(rr_prime * votes, axis=-3, keepdims=True, name="mean_j_numerator")
    mean_j = tf.divide(mean_j_numerator, rr_prime_sum + epsilon, name="mean_j")

    # ----- AG 26/06/2018 START -----#
    # Use variance instead of standard deviation, because the sqrt seems to
    # cause NaN gradients during backprop.
    # See original implementation from Suofei below
    var_j_numerator = tf.reduce_sum(rr_prime * tf.square(votes - mean_j), axis=-3,
                                    keepdims=True, name="var_j_numerator")
    var_j = tf.divide(var_j_numerator, rr_prime_sum + epsilon, name="var_j")

    # Set the minimum variance (note: variance should always be positive)
    # This should allow me to remove the FLAGS.epsilon safety from log and div
    # that follow
    # var_j = tf.maximum(var_j, FLAGS.epsilon)
    # var_j = var_j + FLAGS.epsilon

    ###################
    # var_j = var_j + 1e-5
    var_j = tf.identity(var_j + 1e-9, name="var_j_epsilon")
    ###################

    # Compute the stdv, but it shouldn't actually be used anywhere
    # stdv_j = tf.sqrt(var_j)
    stdv_j = None

    # layer_norm_factor
    cost_j_h = (beta_v + 0.5 * tf.math.log(var_j)) * rr_prime_sum * layer_norm_factor
    cost_j_h = tf.identity(cost_j_h, name="cost_j_h")

    # ----- END ----- #

    # cost_j: (24, 6, 6, 1, 32, 1)
    # activations_j_cost = (24, 6, 6, 1, 32, 1)
    # yg: This is done for numeric stability.
    # It is the relative variance between each channel determined which one
    # should activate.
    cost_j = tf.reduce_sum(cost_j_h, axis=-1, keepdims=True, name="cost_j")
    # cost_j_mean = tf.reduce_mean(cost_j, axis=-2, keepdims=True)
    # cost_j_stdv = tf.sqrt(
    #  tf.reduce_sum(
    #    tf.square(cost_j - cost_j_mean), axis=-2, keepdims=True
    #  ) / cost_j.get_shape().as_list()[-2]
    # )

    # AG 17/09/2018: trying to remove normalisation
    # activations_j_cost = beta_a + (cost_j_mean - cost_j) / (cost_j_stdv)
    activations_j_cost = tf.identity(beta_a - cost_j,
                                     name="activations_j_cost")

    # (24, 6, 6, 1, 32, 1)
    activations_j = tf.sigmoid(inverse_temperature * activations_j_cost,
                               name="sigmoid")

    # AG 26/06/2018: added var_j to return
    return activations_j, mean_j, stdv_j, var_j


# AG 26/06/2018: added var_j
def e_step(votes_ij, activations_j, mean_j, var_j, spatial_routing_matrix):
    # AG 26/06/2018: changed stdv_j to var_j
    o_p_unit0 = - tf.reduce_sum(
        tf.square(votes_ij - mean_j, name="num") / (2 * var_j),
        axis=-1,
        keepdims=True,
        name="o_p_unit0")

    o_p_unit2 = - 0.5 * tf.reduce_sum(tf.math.log(2 * np.pi * var_j), axis=-1, keepdims=True, name="o_p_unit2")

    # (24, 6, 6, 288, 32, 1)
    o_p = o_p_unit0 + o_p_unit2
    zz = tf.math.log(activations_j + epsilon) + o_p

    # AG 13/11/2018: New implementation of normalising across parents
    # ----- Start -----#
    zz_shape = zz.get_shape().as_list()
    batch_size = zz_shape[0]
    parent_space = zz_shape[1]
    kh_kw_i = zz_shape[3]
    parent_caps = zz_shape[4]
    kk = int(np.sum(spatial_routing_matrix[:, 0]))
    child_caps = int(kh_kw_i / kk)

    zz = tf.reshape(zz, [batch_size, parent_space, parent_space, kk,
                         child_caps, parent_caps])

    # Fill the sparse matrix with the smallest value in zz (at least -100)
    sparse_filler = tf.minimum(tf.reduce_min(zz), -100)
    #       sparse_filler = -100
    zz_sparse = to_sparse(zz, spatial_routing_matrix, sparse_filler=sparse_filler)

    rr_sparse = softmax_across_parents(zz_sparse)

    rr_dense = to_dense(rr_sparse, spatial_routing_matrix)

    rr = tf.reshape(rr_dense, [batch_size, parent_space, parent_space, kh_kw_i, parent_caps, 1])

    return rr


def to_sparse(probs, spatial_routing_matrix, sparse_filler=tf.math.log(1e-20)):
    # Get shapes of probs
    shape = probs.get_shape().as_list()
    batch_size = shape[0]
    parent_space = shape[1]
    kk = shape[3]
    child_caps = shape[4]
    parent_caps = shape[5]

    # Get spatial dimesion of child capsules
    child_space_2 = int(spatial_routing_matrix.shape[0])
    parent_space_2 = int(spatial_routing_matrix.shape[1])

    # Unroll the probs along the spatial dimension
    # e.g. (64, 6, 6, 3*3, 8, 32) -> (64, 6*6, 3*3, 8, 32)
    probs_unroll = tf.reshape(
        probs,
        [batch_size, parent_space_2, kk, child_caps, parent_caps])

    # Each row contains the children belonging to one parent
    child_to_parent_idx = group_children_by_parent(spatial_routing_matrix)

    # Create an index mapping each capsule to the correct sparse location
    # Each element of the index must contain [batch_position,
    # parent_space_position, child_sparse_position]
    # E.g. [63, 24, 49] maps image 63, parent space 24, sparse position 49
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

    # Combine the 3 coordinates
    indices = np.stack((batch_idx, parent_idx, child_sparse_idx), axis=3)
    indices = tf.constant(indices)

    # Convert each spatial location to sparse
    shape = [batch_size, parent_space_2, child_space_2, child_caps, parent_caps]
    sparse = tf.scatter_nd(indices, probs_unroll, shape)

    # scatter_nd pads the output with zeros, but since we are operating
    # in log space, we need to replace 0 with log(0), or log(1e-9)
    zeros_in_log = tf.ones_like(sparse, dtype=tf.float32) * sparse_filler
    sparse = tf.where(tf.equal(sparse, 0.0), zeros_in_log, sparse)

    # Reshape
    # (64, 5*5, 7*7, 8, 32) -> (64, 6, 6, 14*14, 8, 32)
    sparse = tf.reshape(sparse, [batch_size, parent_space, parent_space, child_space_2, child_caps, parent_caps])

    assert sparse.shape == [batch_size, parent_space, parent_space, child_space_2, child_caps, parent_caps]

    return sparse


def normalise_across_parents(probs_sparse):
    # e.g. (1, 5, 5, 49, 8, 32)
    # (batch_size, parent_space, parent_space, child_space*child_space, child_caps, parent_caps)
    shape = probs_sparse.get_shape().as_list()
    batch_size = shape[0]
    parent_space = shape[1]
    child_space_2 = shape[3]  # squared
    child_caps = shape[4]
    parent_caps = shape[5]

    rr_updated = probs_sparse / (tf.reduce_sum(probs_sparse, axis=[1, 2, 5], keepdims=True) + 1e-9)

    assert (rr_updated.shape == [batch_size, parent_space, parent_space, child_space_2, child_caps, parent_caps])

    return rr_updated


def softmax_across_parents(probs_sparse):
    # e.g. (1, 5, 5, 49, 8, 32)
    # (batch_size, parent_space, parent_space, child_space*child_space,
    # child_caps, parent_caps)
    shape = probs_sparse.shape
    batch_size = shape[0]
    parent_space = shape[1]
    child_space_2 = shape[3]  # squared
    child_caps = shape[4]
    parent_caps = shape[5]

    # Move parent space dimensions, and parent depth dimension to end
    # (1, 5, 5, 49, 8, 32)  -> (1, 49, 4, 5, 5, 3)
    sparse = tf.transpose(probs_sparse, perm=[0, 3, 4, 1, 2, 5])

    # Combine parent
    # (1, 49, 4, 75)
    sparse = tf.reshape(sparse, [batch_size, child_space_2, child_caps, -1])

    # Perform softmax across parent capsule dimension
    parent_softmax = tf.nn.softmax(sparse, axis=-1)

    # Uncombine parent space and depth
    # (1, 49, 4, 5, 5, 3)
    parent_softmax = tf.reshape(parent_softmax, [batch_size, child_space_2, child_caps,
                                                 parent_space, parent_space, parent_caps])
    # Return to original order
    # (1, 5, 5, 49, 8, 32)
    parent_softmax = tf.transpose(parent_softmax, perm=[0, 3, 4, 1, 2, 5])

    rr_updated = parent_softmax

    assert (rr_updated.shape == [batch_size, parent_space, parent_space, child_space_2, child_caps, parent_caps])

    return rr_updated


def to_dense(sparse, spatial_routing_matrix):
    # Get shapes of probs
    shape = sparse.shape
    batch_size = shape[0]
    parent_space = shape[1]
    child_space_2 = shape[3]  # squared
    child_caps = shape[4]
    parent_caps = shape[5]

    # Calculate kernel size by adding up column of spatial routing matrix
    kk = int(np.sum(spatial_routing_matrix[:, 0]))

    # Unroll parent spatial dimensions
    # (64, 5, 5, 49, 8, 32) -> (64, 5*5, 49, 8, 32)
    sparse_unroll = tf.reshape(sparse, [batch_size, parent_space * parent_space,
                                        child_space_2, child_caps, parent_caps])

    # Apply boolean_mask on axis 1 and 2
    # sparse_unroll: (64, 5*5, 49, 8, 32)
    # spatial_routing_matrix: (49, 25) -> (25, 49)
    # dense: (64, 5*5, 49, 8, 32) -> (64, 5*5*9, 8, 32)
    dense = tf.boolean_mask(sparse_unroll, tf.transpose(spatial_routing_matrix), axis=1)

    # Reshape
    dense = tf.reshape(dense, [batch_size, parent_space, parent_space, kk,
                               child_caps, parent_caps])

    assert (dense.shape == [batch_size, parent_space, parent_space, kk, child_caps, parent_caps])

    return dense


def logits_one_vs_rest(logits, positive_class=0):
    logits_positive = tf.reshape(logits[:, positive_class], [-1, 1])
    logits_rest = tf.concat([logits[:, :positive_class],
                             logits[:, (positive_class + 1):]], axis=1)
    logits_rest_max = tf.reduce_max(logits_rest, axis=1, keepdims=True)
    return tf.concat([logits_positive, logits_rest_max], axis=1)


def coord_addition(votes):
    # get spacial dimension of votes
    height = votes.shape[1]
    width = votes.shape[2]
    dims = votes.shape[-1]

    # Generate offset coordinates
    # The difference here is that the coordinate won't be exactly in the middle of
    # the receptive field, but will be evenly spread out
    w_offset_vals = (np.arange(width) + 0.50) / float(width)
    h_offset_vals = (np.arange(height) + 0.50) / float(height)

    w_offset = np.zeros([width, dims])  # (5, 16)
    w_offset[:, 3] = w_offset_vals
    # (1, 1, 5, 1, 1, 16)
    w_offset = np.reshape(w_offset, [1, 1, width, 1, 1, dims])

    h_offset = np.zeros([height, dims])
    h_offset[:, 7] = h_offset_vals
    # (1, 5, 1, 1, 1, 16)
    h_offset = np.reshape(h_offset, [1, height, 1, 1, 1, dims])

    # Combine w and h offsets using broadcasting
    # w is (1, 1, 5, 1, 1, 16)
    # h is (1, 5, 1, 1, 1, 16)
    # together (1, 5, 5, 1, 1, 16)
    offset = w_offset + h_offset

    # Convent from numpy to tensor
    offset = tf.constant(offset, dtype=tf.float32)

    votes = tf.add(votes, offset, name="votes_with_coord_add")

    return votes


def init_beta(layer, caps):
    beta_a = layer.add_weight(
        name='beta_a',
        shape=[1, 1, 1, 1, caps, 1],
        dtype=tf.float32,
        initializer=tf.keras.initializers.TruncatedNormal(mean=-1000.0, stddev=500.0))
    beta_v = layer.add_weight(
        name='beta_v',
        shape=[1, 1, 1, 1, caps, 1],
        dtype=tf.float32,
        initializer=tf.keras.initializers.GlorotUniform(),
        regularizer=None)

    return beta_a, beta_v
