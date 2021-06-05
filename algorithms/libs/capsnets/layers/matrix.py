import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from functools import reduce
from libs.capsnets.losses import spread_loss

# parameters for the EM-routing operation
inv_temp = 0.5
inv_temp_delta = 0.1
epsilon = 1e-7


class PrimaryCapsule2D(layers.Layer):
    """
    Класс, описывающий первичные капсулы с использованием трехмерных сверточных слоев
    :param inputs: входной тензор
    :param channels: количество капсул
    :param kernel_size: размер ядра свертки для выделения карты признаков. Должен описывать два измерения (K_h, K_w).
    :param strides: шаг свертки при использовании ядра свертки. Должен описывать два измерения (S_h, S_w)
    :param name: имя слоя
    :param padding: требуется ли восполнять размер карты признаков путем добавления строк с помощью 'valid' или нет 'same'.
    :param activation: функция активации для позиционной матрицы.
    :return: возвращается капсула, как матрица поз и матрица активаций.
    Позы имеют размерность (N, H_out, W_out, C_out, M) гда M является высота*ширина матрицы поз.
    Актвации имеют размерность (N, H_out, W_out, C_out, 1).
    """

    def __init__(self, channels, kernel_size, strides, name='primary_caps_3d',
                 matrix_dim=(4, 4), padding='valid', activation=None,
                 pose_kernel_initializer=None, activation_kernel_initializer=None,
                 pose_bias_initializer=None, activation_bias_initializer=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.matrix_dim = matrix_dim[0] * matrix_dim[1]
        self.poses_conv = None
        self.activations_conv = None
        self.pose_kernel_initializer = pose_kernel_initializer
        self.activation_kernel_initializer = activation_kernel_initializer
        self.pose_bias_initializer = pose_bias_initializer
        self.activation_bias_initializer = activation_bias_initializer

    def build(self, input_shape):
        self.poses_conv = layers.Conv2D(filters=self.channels * self.matrix_dim, kernel_size=self.kernel_size,
                                        strides=self.strides, padding=self.padding, activation=self.activation,
                                        kernel_initializer=self.pose_kernel_initializer,
                                        bias_initializer=self.pose_bias_initializer,
                                        name=self.name + '_pose')
        self.activations_conv = layers.Conv2D(filters=self.channels, kernel_size=self.kernel_size,
                                              strides=self.strides, padding=self.padding,
                                              activation=tf.nn.sigmoid,
                                              kernel_initializer=self.activation_kernel_initializer,
                                              bias_initializer=self.activation_bias_initializer,
                                              name=self.name + '_activation')
        self.built = True

    def call(self, inputs, **kwargs):
        if self.poses_conv is None or self.activations_conv is None:
            raise Exception('PrimaryCapsule3D build is failed')
        batch_size = tf.shape(inputs)[0]

        poses = self.poses_conv(inputs)

        _, h, w, _ = poses.shape
        h, w = map(int, [h, w])

        pose = tf.reshape(poses, (batch_size, h, w, self.channels, self.matrix_dim))

        acts = self.activations_conv(inputs)
        activation = tf.reshape(acts, (batch_size, h, w, self.channels, 1))

        return pose, activation

    def get_config(self):
        return super().get_config()


class PrimaryCapsule3D(layers.Layer):
    """
    Класс, описывающий первичные капсулы с использованием трехмерных сверточных слоев
    :param inputs: входной тензор
    :param channels: количество капсул
    :param kernel_size: размер ядра свертки для выделения карты признаков. Должен описывать три измерения (K_t, K_h, K_w).
    :param strides: шаг свертки при использовании ядра свертки. Должен описывать три измерения (S_t, S_h, S_w)
    :param name: имя слоя
    :param padding: требуется ли восполнять размер карты признаков путем добавления строк с помощью 'valid' или нет 'same'.
    :param activation: функция активации для позиционной матрицы.
    :return: возвращается капсула, как матрица поз и матрица активаций.
    Позы имеют размерность (N, D_out, H_out, W_out, C_out, M) гда M является высота*ширина матрицы поз.
    Актвации имеют размерность (N, D_out, H_out, W_out, C_out, 1).
    """

    def __init__(self, channels, kernel_size, strides, name='primary_caps_3d',
                 matrix_dim=(4, 4), padding='valid', activation=None,
                 pose_kernel_initializer=None, activation_kernel_initializer=None,
                 pose_bias_initializer=None, activation_bias_initializer=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.matrix_dim = matrix_dim[0] * matrix_dim[1]
        self.poses_conv = None
        self.activations_conv = None
        self.pose_kernel_initializer = pose_kernel_initializer
        self.activation_kernel_initializer = activation_kernel_initializer
        self.pose_bias_initializer = pose_bias_initializer
        self.activation_bias_initializer = activation_bias_initializer

    def build(self, input_shape):
        self.poses_conv = layers.Conv3D(filters=self.channels * self.matrix_dim, kernel_size=self.kernel_size,
                                        strides=self.strides, padding=self.padding, activation=self.activation,
                                        kernel_initializer=self.pose_kernel_initializer,
                                        bias_initializer=self.pose_bias_initializer,
                                        name=self.name + '_pose')
        self.activations_conv = layers.Conv3D(filters=self.channels, kernel_size=self.kernel_size,
                                              strides=self.strides, padding=self.padding,
                                              activation=tf.nn.sigmoid,
                                              kernel_initializer=self.activation_kernel_initializer,
                                              bias_initializer=self.activation_bias_initializer,
                                              name=self.name + '_activation')
        self.built = True

    def call(self, inputs, **kwargs):
        if self.poses_conv is None or self.activations_conv is None:
            raise Exception('PrimaryCapsule3D build is failed')
        batch_size = tf.shape(inputs)[0]

        poses = self.poses_conv(inputs)

        _, d, h, w, _ = poses.shape
        d, h, w = map(int, [d, h, w])

        pose = tf.reshape(poses, (batch_size, d, h, w, self.channels, self.matrix_dim))

        acts = self.activations_conv(inputs)
        activation = tf.reshape(acts, (batch_size, d, h, w, self.channels, 1))

        return pose, activation

    def get_config(self):
        return super().get_config()


class ConvolutionalCapsule2D(layers.Layer):
    def __init__(self, channels, kernel_size, strides, name='conv_caps_3d',
                 padding='valid', subset_routing=-1, route_min=0.0, coord_add=False,
                 rel_center=True, route_mean=True, ch_same_w=True,
                 beta_v_initializer=tf.initializers.random_normal(stddev=0.1),
                 beta_a_initializer=tf.initializers.random_normal(stddev=0.1),
                 weights_initializer=tf.initializers.random_normal(stddev=0.1), **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.subset_routing = subset_routing
        self.route_min = route_min
        self.coord_add = coord_add
        self.rel_center = rel_center
        self.route_mean = route_mean
        self.ch_same_w = ch_same_w
        self.dense_caps = None
        self.weights_initializer = weights_initializer
        self.beta_v_initializer = beta_v_initializer
        self.beta_a_initializer = beta_a_initializer

    def build(self, input_shape):
        if self.route_mean:
            self.dense_caps = ClassCapsule(self.channels, self.name + 'dense_caps',
                                           route_min=self.route_min, ch_same_w=self.ch_same_w,
                                           weights_initializer=self.weights_initializer,
                                           beta_v_initializer=self.beta_v_initializer,
                                           beta_a_initializer=self.beta_a_initializer)
        else:
            self.dense_caps = ClassCapsule(self.channels, self.name + '_dense_caps', subset_routing=self.subset_routing,
                                           route_min=self.route_min, coord_add=self.coord_add,
                                           rel_center=self.rel_center, ch_same_w=self.ch_same_w,
                                           weights_initializer=self.weights_initializer,
                                           beta_v_initializer=self.beta_v_initializer,
                                           beta_a_initializer=self.beta_a_initializer)

        self.built = True

    def call(self, inputs, **kwargs):
        if self.dense_caps is None:
            raise Exception('ConvolutionalCapsule3D build is failed')

        inputs = tf.concat(inputs, axis=-1)

        # pads the input
        if self.padding == 'same':
            h_padding, w_padding = int(float(self.kernel_size[0]) / 2), int(float(self.kernel_size[1]) / 2)
            u_padded = tf.pad(inputs,
                              [[0, 0], [h_padding, h_padding], [w_padding, w_padding], [0, 0],
                               [0, 0]])
        else:
            u_padded = inputs

        batch_size = tf.shape(u_padded)[0]
        matrix_dim = u_padded.shape[-1]
        _, h, w, ch, _ = u_padded.get_shape().as_list()
        h, w, ch = map(int, [h, w, ch])

        # gets indices for kernels
        h_offsets = [[(h_ + k) for k in range(self.kernel_size[0])] for h_ in
                     range(0, h + 1 - self.kernel_size[0], self.strides[0])]
        w_offsets = [[(w_ + k) for k in range(self.kernel_size[1])] for w_ in
                     range(0, w + 1 - self.kernel_size[1], self.strides[1])]

        # output dimensions
        h_out, w_out = len(h_offsets), len(w_offsets)

        # gathers the capsules into shape (N, H2, W2, KD, KH, KW, Ch_in, 17)
        h_gathered = tf.gather(u_padded, h_offsets, axis=1)
        w_gathered = tf.gather(h_gathered, w_offsets, axis=3)
        w_gathered = tf.transpose(w_gathered, [0, 1, 3, 2, 4, 5, 6])

        # obtains the next layer of capsules
        if self.route_mean:
            kernels_reshaped = tf.reshape(w_gathered, [batch_size * h_out * w_out,
                                                       self.kernel_size[0] * self.kernel_size[1],
                                                       ch, matrix_dim])
            kernels_reshaped = tf.reduce_mean(kernels_reshaped, axis=1)
            capsules = self.dense_caps((kernels_reshaped[:, :, :-1], kernels_reshaped[:, :, -1:]))
        else:
            kernels_reshaped = tf.reshape(w_gathered,
                                          [batch_size * h_out * w_out, self.kernel_size[0], self.kernel_size[1], ch,
                                           matrix_dim])
            capsules = self.dense_caps((kernels_reshaped[:, :, :, :, :, :-1], kernels_reshaped[:, :, :, :, :, -1:]))

        # reshape capsules back into the 3d shape
        poses = tf.reshape(capsules[0][:, :, :(matrix_dim - 1)],
                           (batch_size, h_out, w_out, self.channels, matrix_dim - 1))
        activations = tf.reshape(capsules[1], (batch_size, h_out, w_out, self.channels, 1))

        return poses, activations

    def get_config(self):
        return super().get_config()


class ConvolutionalCapsule3D(layers.Layer):
    def __init__(self, channels, kernel_size, strides, name='conv_caps_3d',
                 padding='valid', subset_routing=-1, route_min=0.0, coord_add=False,
                 rel_center=True, route_mean=True, ch_same_w=True,
                 beta_v_initializer=tf.initializers.random_normal(stddev=0.1),
                 beta_a_initializer=tf.initializers.random_normal(stddev=0.1),
                 weights_initializer=tf.initializers.random_normal(stddev=0.1), **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.subset_routing = subset_routing
        self.route_min = route_min
        self.coord_add = coord_add
        self.rel_center = rel_center
        self.route_mean = route_mean
        self.ch_same_w = ch_same_w
        self.dense_caps = None
        self.weights_initializer = weights_initializer
        self.beta_v_initializer = beta_v_initializer
        self.beta_a_initializer = beta_a_initializer

    def build(self, input_shape):
        if self.route_mean:
            self.dense_caps = ClassCapsule(self.channels, self.name + 'dense_caps',
                                           route_min=self.route_min, ch_same_w=self.ch_same_w,
                                           weights_initializer=self.weights_initializer,
                                           beta_v_initializer=self.beta_v_initializer,
                                           beta_a_initializer=self.beta_a_initializer)
        else:
            self.dense_caps = ClassCapsule(self.channels, self.name + '_dense_caps', subset_routing=self.subset_routing,
                                           route_min=self.route_min, coord_add=self.coord_add,
                                           rel_center=self.rel_center, ch_same_w=self.ch_same_w,
                                           weights_initializer=self.weights_initializer,
                                           beta_v_initializer=self.beta_v_initializer,
                                           beta_a_initializer=self.beta_a_initializer)

        self.built = True

    def call(self, inputs, **kwargs):
        if self.dense_caps is None:
            raise Exception('ConvolutionalCapsule3D build is failed')

        inputs = tf.concat(inputs, axis=-1)

        # pads the input
        if self.padding == 'same':
            d_padding, h_padding, w_padding = int(float(self.kernel_size[0]) / 2), int(
                float(self.kernel_size[1]) / 2), int(float(self.kernel_size[2]) / 2)
            u_padded = tf.pad(inputs,
                              [[0, 0], [d_padding, d_padding], [h_padding, h_padding], [w_padding, w_padding], [0, 0],
                               [0, 0]])
        else:
            u_padded = inputs

        batch_size = tf.shape(u_padded)[0]
        matrix_dim = u_padded.shape[-1]
        _, d, h, w, ch, _ = u_padded.get_shape().as_list()
        d, h, w, ch = map(int, [d, h, w, ch])

        # gets indices for kernels
        d_offsets = [[(d_ + k) for k in range(self.kernel_size[0])] for d_ in
                     range(0, d + 1 - self.kernel_size[0], self.strides[0])]
        h_offsets = [[(h_ + k) for k in range(self.kernel_size[1])] for h_ in
                     range(0, h + 1 - self.kernel_size[1], self.strides[1])]
        w_offsets = [[(w_ + k) for k in range(self.kernel_size[2])] for w_ in
                     range(0, w + 1 - self.kernel_size[2], self.strides[2])]

        # output dimensions
        d_out, h_out, w_out = len(d_offsets), len(h_offsets), len(w_offsets)

        # gathers the capsules into shape (N, D2, H2, W2, KD, KH, KW, Ch_in, 17)
        d_gathered = tf.gather(u_padded, d_offsets, axis=1)
        h_gathered = tf.gather(d_gathered, h_offsets, axis=3)
        w_gathered = tf.gather(h_gathered, w_offsets, axis=5)
        w_gathered = tf.transpose(w_gathered, [0, 1, 3, 5, 2, 4, 6, 7, 8])

        # obtains the next layer of capsules
        if self.route_mean:
            kernels_reshaped = tf.reshape(w_gathered, [batch_size * d_out * h_out * w_out,
                                                       self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
                                                       ch, matrix_dim])
            kernels_reshaped = tf.reduce_mean(kernels_reshaped, axis=1)
            capsules = self.dense_caps((kernels_reshaped[:, :, :-1], kernels_reshaped[:, :, -1:]))
        else:
            kernels_reshaped = tf.reshape(w_gathered,
                                          [batch_size * d_out * h_out * w_out, self.kernel_size[0], self.kernel_size[1],
                                           self.kernel_size[2], ch, matrix_dim])
            capsules = self.dense_caps((kernels_reshaped[:, :, :, :, :, :-1], kernels_reshaped[:, :, :, :, :, -1:]))

        # reshape capsules back into the 3d shape
        poses = tf.reshape(capsules[0][:, :, :(matrix_dim - 1)],
                           (batch_size, d_out, h_out, w_out, self.channels, matrix_dim - 1))
        activations = tf.reshape(capsules[1], (batch_size, d_out, h_out, w_out, self.channels, 1))

        return poses, activations

    def get_config(self):
        return super().get_config()


class ClassCapsule(layers.Layer):
    def __init__(self, n_caps_j, name='class_capsule', subset_routing=-1, route_min=0.0,
                 coord_add=False, rel_center=False, ch_same_w=True,
                 beta_v_initializer=tf.initializers.random_normal(stddev=0.1),
                 beta_a_initializer=tf.initializers.random_normal(stddev=0.1),
                 weights_initializer=tf.initializers.random_normal(stddev=0.1), **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_caps_j = n_caps_j
        self.subset_routing = subset_routing
        self.route_min = route_min
        self.coord_add = coord_add
        self.rel_center = rel_center
        self.ch_same_w = ch_same_w
        self.matrix_dim = None
        self.ch = None
        self.n_capsch_i = None
        self.w = None
        self.beta_v = None
        self.beta_a = None
        self.weights_initializer = weights_initializer
        self.beta_v_initializer = beta_v_initializer
        self.beta_a_initializer = beta_a_initializer

    def build(self, input_shape):
        pose_shape, activation_shape = input_shape
        shape_list = [int(x) for x in pose_shape[1:]]
        self.matrix_dim = pose_shape[-1]
        self.ch = int(shape_list[-2])
        self.n_capsch_i = 1 if len(shape_list) == 2 else reduce((lambda x, y: x * y), shape_list[:-2])

        mdim = int(np.sqrt(self.matrix_dim))
        if self.ch_same_w:
            self.w = self.add_weight(name=self.name + '_weights',
                                     shape=(self.ch, self.n_caps_j, mdim, mdim),
                                     initializer=self.weights_initializer,
                                     regularizer=tf.keras.regularizers.L2(0.1))
        else:
            self.w = self.add_weight(name=self.name + '_weights',
                                     shape=(self.n_capsch_i, self.ch, self.n_caps_j, mdim, mdim),
                                     initializer=self.weights_initializer,
                                     regularizer=tf.keras.regularizers.L2(0.1))

        # creates the routing parameters
        self.beta_v = self.add_weight(name=self.name + '_beta_v', shape=(self.n_caps_j, self.matrix_dim),
                                      initializer=self.beta_v_initializer,
                                      regularizer=tf.keras.regularizers.L2(0.1))

        self.beta_a = self.add_weight(name=self.name + '_beta_a', shape=(self.n_caps_j, 1),
                                      initializer=self.beta_a_initializer,
                                      regularizer=tf.keras.regularizers.L2(0.1))

        self.built = True

    def call(self, inputs, **kwargs):
        if self.ch is None or self.n_capsch_i is None or self.w is None or self.beta_v is None or self.beta_a is None:
            raise Exception('ClassCapsule build is failed')

        pose, activation = inputs
        batch_size = tf.shape(pose)[0]
        mdim = int(np.sqrt(self.matrix_dim))

        u_i = tf.reshape(pose, (batch_size, self.n_capsch_i, self.ch, self.matrix_dim))
        activation = tf.reshape(activation, (batch_size, self.n_capsch_i, self.ch, 1))
        coords = create_coords_mat(pose, self.rel_center) if self.coord_add else tf.zeros_like(u_i)

        # extracts a subset of capsules to be routed
        if self.subset_routing != -1:
            u_i, coords, activation = get_subset(u_i, coords, activation, k=self.subset_routing)
            self.n_capsch_i = self.subset_routing

        # reshapes the input capsules
        u_i = tf.reshape(u_i, (batch_size, self.n_capsch_i, self.ch, mdim, mdim))
        u_i = tf.expand_dims(u_i, axis=-3)
        u_i = tf.tile(u_i, [1, 1, 1, self.n_caps_j, 1, 1])

        # calculates votes
        if self.ch_same_w:
            votes = tf.einsum('ijab,ntijbc->ntijac', self.w, u_i)
            votes = tf.reshape(votes, (batch_size, self.n_capsch_i * self.ch, self.n_caps_j, self.matrix_dim))
        else:
            votes = tf.einsum('tijab,ntijbc->ntijac', self.w, u_i)
            votes = tf.reshape(votes, (batch_size, self.n_capsch_i * self.ch, self.n_caps_j, self.matrix_dim))

        # performs coordinate addition
        if self.coord_add:
            coords = tf.reshape(coords, (batch_size, self.n_capsch_i * self.ch, 1, self.matrix_dim))
            votes = votes + tf.tile(coords, [1, 1, self.n_caps_j, 1])

        # performs thresholding, so only capsules with activations over the threshold "route_min" are routed
        acts = tf.reshape(activation, (batch_size, self.n_capsch_i * self.ch, 1))
        activations = tf.where(tf.greater_equal(acts, tf.constant(self.route_min)), acts, tf.zeros_like(acts))

        # performs EM-routing and returns the new capsules
        return em_routing(votes, activations, self.beta_v, self.beta_a)

    def get_config(self):
        return super().get_config()


def create_coords_mat(pose, rel_center):
    """
    Create matrices for coordinate addition. The output of this function should be added to the vote matrix.
    :param pose: The incoming map of pose matrices, shape (N, ..., Ch_i, 16) where ... is the dimensions of the map can
    be 1, 2 or 3 dimensional.
    :param rel_center: whether or not the coordinates are relative to the center of the map
    :return: Returns the coordinates (padded to 16) for the incoming capsules.
    """
    batch_size = tf.shape(pose)[0]
    shape_list = [int(x) for x in pose.get_shape().as_list()[1:-2]]
    matrix_dim = pose.shape[-1]
    ch = int(pose.get_shape().as_list()[-2])
    n_dims = len(shape_list)

    if n_dims == 3:
        d, h, w = shape_list
    elif n_dims == 2:
        d = 1
        h, w = shape_list
    else:
        d, h = 1, 1
        w = shape_list[0]

    subs = [0, 0, 0]
    if rel_center:
        subs = [int(d / 2), int(h / 2), int(w / 2)]

    c_mats = []
    if n_dims >= 3:
        c_mats.append(
            tf.tile(tf.reshape(tf.range(d, dtype=tf.float32), (1, d, 1, 1, 1, 1)), [batch_size, 1, h, w, ch, 1]) - subs[
                0])
    if n_dims >= 2:
        c_mats.append(
            tf.tile(tf.reshape(tf.range(h, dtype=tf.float32), (1, 1, h, 1, 1, 1)), [batch_size, d, 1, w, ch, 1]) - subs[
                1])
    if n_dims >= 1:
        c_mats.append(
            tf.tile(tf.reshape(tf.range(w, dtype=tf.float32), (1, 1, 1, w, 1, 1)), [batch_size, d, h, 1, ch, 1]) - subs[
                2])
    add_coords = tf.concat(c_mats, axis=-1)
    add_coords = tf.cast(tf.reshape(add_coords, (batch_size, d * h * w, ch, n_dims)), dtype=tf.float32)

    zeros = tf.zeros((batch_size, d * h * w, ch, matrix_dim - n_dims))

    return tf.concat([zeros, add_coords], axis=-1)


def get_subset(u_i, coords, activation, k):
    """
    Gets a subset of k capsules of each capsule type, based on their activation. When k=1, this is equivalent to
    "max-pooling" where the most active capsule for each capsule type is used.
    :param u_i: The incoming pose matrices shape (N, K, Ch_i, M)
    :param coords: The coords for these pose matrices (N, K, Ch_i, M)
    :param activation: The activations of the capsules (N, K, Ch_i, 1)
    :param k: Number of capsules which will be routed
    :return: New u_i, coords, and activation which only have k of the most active capsules per channel
    """
    batch_size, n_capsch_i, ch = tf.shape(u_i)[0], int(u_i.get_shape().as_list()[1]), tf.shape(u_i)[2]
    matrix_dim = u_i.shape[-1]

    inputs_res = tf.reshape(tf.concat([u_i, coords, activation], axis=-1),
                            (batch_size, n_capsch_i, ch, matrix_dim * 2 + 1))

    trans = tf.transpose(inputs_res, [0, 2, 1, 3])

    norms = tf.reshape(trans[:, :, :, -1], (batch_size, ch, n_capsch_i))

    inds = tf.nn.top_k(norms, k).indices

    bt = tf.reshape(tf.range(batch_size), (batch_size, 1))
    bt = tf.reshape(tf.tile(bt, [1, ch * k]), (batch_size, ch * k, 1))

    ct = tf.reshape(tf.range(ch), (ch, 1))
    ct = tf.reshape(tf.tile(ct, [1, k]), (ch, k, 1))
    ct = tf.reshape(tf.tile(ct, [batch_size, 1, 1]), (batch_size, ch * k, 1))

    conc = tf.concat([bt, ct], axis=2)
    t = tf.reshape(conc, (batch_size, ch, k, 2))

    inds = tf.reshape(inds, (batch_size, ch, k, 1))
    coords = tf.concat([t, inds], axis=3)

    top_caps = tf.gather_nd(trans, coords)

    top_caps = tf.transpose(top_caps, (0, 2, 1, 3))
    top_poses = top_caps[:, :, :, :matrix_dim]
    top_coords = top_caps[:, :, :, matrix_dim:-1]
    top_acts = top_caps[:, :, :, -1:]

    return top_poses, top_coords, top_acts


def em_routing(v, a_i, beta_v, beta_a, n_iterations=3):
    """
    Performs the EM-routing (https://openreview.net/pdf?id=HJWLfGWRb).
    Note:One change from the original algorithm is used to ensure numerical stability. The cost used to calculate the
    activations are normalized, which makes the output activations relative to each other (i.e. an activation is high if
    its cost is lower than the other capsules' costs). This works for most applications, but it leads to some issues if
    your application necessitates all, or most, capsules to be active.
    :param v: The votes for the higher level capsules. Shape - (N, C_i, C_j, M)
    :param a_i: The activations of the lower level capsules. Shape - (N, C_i, 1)
    :param beta_v: The beta_v parameter for routing (check original EM-routing paper for details)
    :param beta_a: The beta_a parameter for routing (check original EM-routing paper for details)
    :param n_iterations: Number of iterations which routing takes place.
    :return: Returns capsules of the form (poses, activations). Poses have shape (N, C_out, M) where M is the
    height*width of the pose matrix. Activations have shape (N, C_out, 1).
    """
    batch_size = tf.shape(v)[0]
    _, _, n_caps_j, mat_len = v.get_shape().as_list()
    n_caps_j, mat_len = map(int, [n_caps_j, mat_len])
    n_caps_i = tf.shape(v)[1]

    a_i = tf.expand_dims(a_i, axis=-1)

    # Prior probabilities for routing
    r = tf.ones(shape=(batch_size, n_caps_i, n_caps_j, 1), dtype=tf.float32) / float(n_caps_j)
    r = tf.multiply(r, a_i)

    den = tf.reduce_sum(r, axis=1, keepdims=True) + epsilon

    # Mean: shape=(N, 1, Ch_j, mat_len)
    m_num = tf.reduce_sum(v * r, axis=1, keepdims=True)
    m = m_num / (den + epsilon)

    # Stddev: shape=(N, 1, Ch_j, mat_len)
    s_num = tf.reduce_sum(r * tf.square(v - m), axis=1, keepdims=True)
    s = s_num / (den + epsilon)

    # cost_h: shape=(N, 1, Ch_j, mat_len)
    cost = (beta_v + tf.math.log(tf.sqrt(s + epsilon) + epsilon)) * den
    # cost_h: shape=(N, 1, Ch_j, 1)
    cost = tf.reduce_sum(cost, axis=-1, keepdims=True)

    # calculates the mean and std_deviation of the cost
    cost_mean = tf.reduce_mean(cost, axis=-2, keepdims=True)
    cost_stdv = tf.sqrt(
        tf.reduce_sum(
            tf.square(cost - cost_mean), axis=-2, keepdims=True
        ) / n_caps_j + epsilon
    )

    # calculates the activations for the capsules in layer j
    a_j = tf.sigmoid(float(inv_temp) * (beta_a + (cost_mean - cost) / (cost_stdv + epsilon)))

    # a_j = tf.sigmoid(float(config.inv_temp) * (beta_a - cost)) # may lead to numerical instability

    def route(r_mean, r_stdsqr, r_act_j, counter):
        exp = tf.reduce_sum(tf.square(v - r_mean) / (2 * r_stdsqr + epsilon), axis=-1)
        coef = 0 - .5 * tf.reduce_sum(tf.math.log(2 * np.pi * r_stdsqr + epsilon), axis=-1)
        log_p_j = coef - exp

        log_ap = tf.reshape(tf.math.log(r_act_j + epsilon), (batch_size, 1, n_caps_j)) + log_p_j
        r_ij = tf.nn.softmax(log_ap + epsilon)  # ap / (tf.reduce_sum(ap, axis=-1, keepdims=True) + epsilon)

        r_ij = tf.multiply(tf.expand_dims(r_ij, axis=-1), a_i)

        denom = tf.reduce_sum(r_ij, axis=1, keepdims=True)
        m_numer = tf.reduce_sum(v * r_ij, axis=1, keepdims=True)
        r_mean = m_numer / (denom + epsilon)

        s_numer = tf.reduce_sum(r_ij * tf.square(v - r_mean), axis=1, keepdims=True)
        r_stdsqr = s_numer / (denom + epsilon)

        cost_h = (beta_v + tf.math.log(tf.sqrt(r_stdsqr + epsilon) + epsilon)) * denom
        cost_h = tf.reduce_sum(cost_h, axis=-1, keepdims=True)

        # these are calculated for numerical stability.
        cost_h_mean = tf.reduce_mean(cost_h, axis=-2, keepdims=True)
        cost_h_stdv = tf.sqrt(
            tf.reduce_sum(
                tf.square(cost_h - cost_h_mean), axis=-2, keepdims=True
            ) / n_caps_j + epsilon
        )

        inv_t = inv_temp + counter * inv_temp_delta
        r_act_j = tf.sigmoid(inv_t * (beta_a + (cost_h_mean - cost_h) / (cost_h_stdv + epsilon)))
        # r_act_j = tf.sigmoid(inv_t * (beta_a - cost_h)) # may lead to numerical instability

        return r_mean, r_stdsqr, r_act_j, tf.add(counter, 1)

    [mean, _, act_j, _] = tf.while_loop(lambda r_mean, r_stdsqr, r_act_j, counter: tf.less(counter, n_iterations),
                                        route, [m, s, a_j, 1.0])

    return tf.reshape(mean, (batch_size, n_caps_j, mat_len)), tf.reshape(act_j, (batch_size, n_caps_j, 1))


if __name__ == '__main__':
    from libs import utils

    (x_train, y_train), (x_test, y_test) = utils.load('mnist')

    input_image = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu, name='conv2d_relu')(input_image)
    x = PrimaryCapsule2D(matrix_dim=(4, 4), kernel_size=1, channels=32, strides=1, padding='valid', name='primary_caps')(x)
    x = ConvolutionalCapsule2D(channels=32, kernel_size=(3, 3), strides=(2, 2), name='caps_conv_1')(x)
    x = ConvolutionalCapsule2D(channels=32, kernel_size=(3, 3), strides=(1, 1), name='caps_conv_2')(x)
    x = ClassCapsule(n_caps_j=10, name='class_caps')(x)
    x = layers.Lambda(lambda x: tf.reshape(x, (-1, 10)))(x[1])

    model = tf.keras.Model(input_image, x)
    model.summary(line_length=250)

    class MarginChangeCallback(tf.keras.callbacks.Callback):
        def __init__(self, margin, max_value=0.9, step=0.1):
            super().__init__()
            self.margin = margin
            self.step = step
            self.max_value = max_value

        def on_epoch_end(self, epoch, logs=None):
            if tf.keras.backend.get_value(self.margin) < self.max_value:
                tf.keras.backend.set_value(self.margin, tf.keras.backend.get_value(self.margin) + self.step)


    margin = tf.keras.backend.variable(0.2, dtype=tf.float32)

    margin_callback = MarginChangeCallback(margin)

    def caps_accuracy(y_true, y_pred):
        y_true = tf.cast(tf.argmax(input=y_true, axis=1), tf.float32)
        y_pred = tf.cast(tf.argmax(input=y_pred, axis=1), tf.float32)
        correct = tf.cast(tf.equal(y_pred, y_true), tf.float32)
        return tf.reduce_mean(correct)

    def spread_loss(y_true, y_pred):
        a_i = tf.expand_dims(y_pred, axis=1)
        y = tf.expand_dims(y_true, axis=2)
        a_t = tf.matmul(a_i, y)

        loss = tf.square(tf.maximum(0.0, margin - (a_t - a_i)))
        loss = tf.matmul(loss, 1. - y)

        class_loss = tf.reduce_mean(loss)

        return class_loss


    def margin_metric(y_true, y_pred):
        return margin

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, epsilon=1e-6),
                  loss=spread_loss,
                  metrics=[caps_accuracy, margin_metric])

    # model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=10, callbacks=[margin_callback])
    model.load_weights('D:/model_matrix.tf')

    count_error = 0
    for i in range(len(x_test)):
        y_pred = model.predict(np.expand_dims(x_test[i], axis=0))
        test_value = np.argmax(y_test[i])
        pred_value = np.argmax(y_pred)
        if test_value != pred_value:
            count_error += 1
        print(test_value, pred_value)

    print('количество ошибок', count_error, 'на', len(x_test), 'примеров')