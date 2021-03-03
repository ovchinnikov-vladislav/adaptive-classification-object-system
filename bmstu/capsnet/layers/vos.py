import tensorflow as tf
from tensorflow.keras import activations
from functools import reduce
from bmstu.capsnet.vos_utils import em_routing, em_routing_cond, create_coords_mat


class PrimaryCapsule3D(tf.keras.layers.Layer):
    def __init__(self, capsules, dim_capsules, kernel_size, strides, padding, activation, **kwargs):
        super(PrimaryCapsule3D, self).__init__(**kwargs)
        self.capsules = capsules
        self.dim_capsules = dim_capsules * dim_capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.batch_size = None
        self.conv3d_pose = tf.keras.layers.Conv3D(filters=self.capsules * self.dim_capsules,
                                                  kernel_size=self.kernel_size,
                                                  strides=self.strides,
                                                  padding=self.padding,
                                                  activation=self.activation)
        self.conv3d_activation = tf.keras.layers.Conv3D(filters=self.capsules,
                                                        kernel_size=self.kernel_size,
                                                        strides=self.strides,
                                                        padding=self.padding,
                                                        activation=activations.sigmoid)

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.built = True

    def call(self, inputs, **kwargs):
        poses = self.conv3d_pose(inputs)
        _, d, h, w, _ = poses.shape
        d, h, w = map(int, [d, h, w])

        pose = tf.reshape(poses, (self.batch_size, d, h, w, self.capsules, self.dim_capsules))

        activs = self.conv3d_activation(inputs)
        activ = tf.reshape(activs, (self.batch_size, d, h, w, self.capsules, 1))

        return pose, activ

    def get_config(self):
        return super(PrimaryCapsule3D, self).get_config()


class DenseCapsule(tf.keras.layers.Layer):
    """
    :param capsules: The number of capsules in the following layer
    :param route_min: A threshold activation to route
    :param coord_add: A boolean, whether or not to to do coordinate addition
    :param rel_center: A boolean, whether or not the coordinate addition is relative to the center
    """

    def __init__(self, capsules, dim_capsules, route_min=0.0,
                 coord_add=False, rel_center=False, ch_same_w=True, **kwargs):
        super(DenseCapsule, self).__init__(**kwargs)
        self.dim_capsules = dim_capsules
        self.dim_capsules_2 = dim_capsules * dim_capsules
        self.batch_size = None
        self.capsules = capsules
        self.route_min = route_min
        self.coord_add = coord_add
        self.rel_center = rel_center
        self.ch_same_w = ch_same_w
        self.w = self.beta_v = self.beta_a = None
        self.ch = None
        self.n_capsch_i = None

    def build(self, input_shape):
        self.batch_size = input_shape[0][0]
        shape_list = [int(x) for x in input_shape[0][1:]]
        self.ch = int(shape_list[-2])
        self.n_capsch_i = 1 if len(shape_list) == 2 else reduce((lambda x, y: x * y), shape_list[:-2])
        if self.ch_same_w:
            self.w = self.add_weight(shape=(self.ch, self.capsules, self.dim_capsules, self.dim_capsules),
                                     initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                     regularizer=tf.keras.regularizers.L2(0.1))
        else:
            self.w = self.add_weight(shape=(self.n_capsch_i, self.ch, self.capsules,
                                            self.dim_capsules, self.dim_capsules),
                                     initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                     regularizer=tf.keras.regularizers.L2(0.1))

        self.beta_v = self.add_weight(shape=(self.capsules, self.dim_capsules_2),
                                      initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                      regularizer=tf.keras.regularizers.L2(0.1))
        self.beta_a = self.add_weight(shape=(self.capsules, 1),
                                      initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                      regularizer=tf.keras.regularizers.L2(0.1))

        self.built = True

    def call(self, inputs, **kwargs):
        pose, activation = inputs

        u_i = tf.reshape(pose, (self.batch_size, self.n_capsch_i, self.ch, self.dim_capsules_2))
        activation = tf.reshape(activation, (self.batch_size, self.n_capsch_i, self.ch, 1))
        coords = create_coords_mat(pose, self.rel_center, self.dim_capsules) if self.coord_add else tf.zeros_like(u_i)

        u_i = tf.reshape(u_i, (self.batch_size, self.n_capsch_i, self.ch, self.dim_capsules, self.dim_capsules))
        u_i = tf.expand_dims(u_i, axis=-3)
        u_i = tf.tile(u_i, [1, 1, 1, self.capsules, 1, 1])

        if self.ch_same_w:
            votes = tf.einsum('ijab,ntijbc->ntijac', self.w, u_i)
        else:
            votes = tf.einsum('tijab,ntijbc->ntijac', self.w, u_i)
        votes = tf.reshape(votes, (self.batch_size, self.n_capsch_i * self.ch, self.capsules, self.dim_capsules_2))

        if self.coord_add:
            coords = tf.reshape(coords, (self.batch_size, self.n_capsch_i * self.ch, 1, self.dim_capsules_2))
            votes = votes + tf.tile(coords, [1, 1, self.capsules, 1])

        acts = tf.reshape(activation, (self.batch_size, self.n_capsch_i * self.ch, 1))
        activations_result = tf.where(tf.greater_equal(acts, tf.constant(self.route_min)), acts, tf.zeros_like(acts))

        capsules = em_routing(votes, activations_result, self.beta_v, self.beta_a)

        return capsules

    def get_config(self):
        return super(DenseCapsule, self).get_config()


class Convolutional3DCapsule(tf.keras.layers.Layer):
    def __init__(self, capsules, dim_capsules, kernel_size, strides, padding='valid',
                 coord_add=False, rel_center=True, route_mean=True, ch_same_w=True, **kwargs):
        super(Convolutional3DCapsule, self).__init__(**kwargs)
        self.capsules = capsules
        self.dim_capsules = dim_capsules
        self.dim_capsules_2 = dim_capsules * dim_capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.coord_add = coord_add
        self.rel_center = rel_center
        self.route_mean = route_mean
        self.ch_same_w = ch_same_w
        self.dense_caps = None

    def build(self, input_shape):
        if self.route_mean:
            self.dense_caps = DenseCapsule(capsules=self.capsules, ch_same_w=self.ch_same_w,
                                           dim_capsules=self.dim_capsules)
        else:
            self.dense_caps = DenseCapsule(capsules=self.capsules, coord_add=self.coord_add,
                                           rel_center=self.rel_center, ch_same_w=self.ch_same_w,
                                           dim_capsules=self.dim_capsules)

        self.built = True

    def call(self, inputs, **kwargs):
        inputs = tf.concat(inputs, axis=-1)

        if self.padding == 'same':
            d_padding, h_padding, w_padding = int(float(self.kernel_size[0]) / 2), \
                                              int(float(self.kernel_size[1]) / 2), \
                                              int(float(self.kernel_size[2]) / 2)
            u_padded = tf.pad(inputs,
                              [[0, 0], [d_padding, d_padding], [h_padding, h_padding], [w_padding, w_padding], [0, 0],
                               [0, 0]])
        else:
            u_padded = inputs

        batch_size = tf.shape(u_padded)[0]
        _, d, h, w, ch, _ = u_padded.shape
        d, h, w, ch = map(int, [d, h, w, ch])

        # gets indices for kernels
        d_offsets = [[(d_ + k) for k in range(self.kernel_size[0])]
                     for d_ in range(0, d + 1 - self.kernel_size[0], self.strides[0])]
        h_offsets = [[(h_ + k) for k in range(self.kernel_size[1])]
                     for h_ in range(0, h + 1 - self.kernel_size[1], self.strides[1])]
        w_offsets = [[(w_ + k) for k in range(self.kernel_size[2])]
                     for w_ in range(0, w + 1 - self.kernel_size[2], self.strides[2])]

        # output dimensions
        d_out, h_out, w_out = len(d_offsets), len(h_offsets), len(w_offsets)

        # gathers the capsules into shape (N, D2, H2, W2, KD, KH, KW, Ch_in, 17)
        d_gathered = tf.gather(u_padded, d_offsets, axis=1)
        h_gathered = tf.gather(d_gathered, h_offsets, axis=3)
        w_gathered = tf.gather(h_gathered, w_offsets, axis=5)
        w_gathered = tf.transpose(w_gathered, [0, 1, 3, 5, 2, 4, 6, 7, 8])

        if self.route_mean:
            kernels_reshaped = tf.reshape(w_gathered,
                                          [batch_size * d_out * h_out * w_out,
                                           self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
                                           ch, self.dim_capsules_2 + 1])
            kernels_reshaped = tf.reduce_mean(kernels_reshaped, axis=1)
            capsules = self.dense_caps([kernels_reshaped[:, :, :-1], kernels_reshaped[:, :, -1:]])
        else:
            kernels_reshaped = tf.reshape(w_gathered,
                                          [batch_size * d_out * h_out * w_out,
                                           self.kernel_size[0], self.kernel_size[1],
                                           self.kernel_size[2], ch, self.dim_capsules_2 + 1])
            capsules = self.dense_caps([kernels_reshaped[:, :, :, :, :, :-1], kernels_reshaped[:, :, :, :, :, -1:]])

        poses = tf.reshape(capsules[0][:, :, :self.dim_capsules_2], (batch_size, d_out, h_out, w_out,
                                                                     self.capsules, self.dim_capsules_2))
        activs = tf.reshape(capsules[1], (batch_size, d_out, h_out, w_out, self.capsules, 1))

        return poses, activs

    def get_config(self):
        return super(Convolutional3DCapsule, self).get_config()


class ConditionDenseCapsule(tf.keras.layers.Layer):
    def __init__(self, capsules, dim_capsules, coord_add=False, rel_center=False,
                 ch_same_w=True, n_cond_caps=0, **kwargs):
        super(ConditionDenseCapsule, self).__init__(**kwargs)
        self.capsules = capsules
        self.dim_capsules = dim_capsules
        self.dim_capsules_2 = dim_capsules * dim_capsules
        self.coord_add = coord_add
        self.rel_center = rel_center
        self.ch_same_w = ch_same_w
        self.n_cond_caps = n_cond_caps
        self.batch_size = None
        self.ch = None
        self.n_capsch_i = None
        self.w = self.beta_a = self.beta_v = None
        self.w2 = self.beta_v1 = self.beta_v2 = self.beta_a1 = self.beta_a2 = None

    def build(self, input_shape):
        pose_shape, activation_shape = input_shape
        self.batch_size = pose_shape[0]
        shape_list = [int(x) for x in pose_shape[1:]]
        self.ch = int(shape_list[-2])
        self.n_capsch_i = 1 if len(shape_list) == 2 else reduce((lambda x, y: x * y), shape_list[:-2])

        if self.ch_same_w:
            self.w = self.add_weight(shape=(self.ch, self.capsules, self.dim_capsules, self.dim_capsules),
                                     initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                     regularizer=tf.keras.regularizers.L2(0.1))
        else:
            self.w = self.add_weight(shape=(self.n_capsch_i, self.ch, self.capsules,
                                            self.dim_capsules, self.dim_capsules),
                                     initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                     regularizer=tf.keras.regularizers.L2(0.1))

        if self.n_cond_caps:
            self.beta_v = self.add_weight(shape=(self.capsules, self.dim_capsules_2),
                                          initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                          regularizer=tf.keras.regularizers.L2(0.1))
            self.beta_a = self.add_weight(shape=(self.capsules, 1),
                                          initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                          regularizer=tf.keras.regularizers.L2(0.1))
        else:
            self.beta_v1 = self.add_weight(shape=(self.capsules, self.dim_capsules_2),
                                           initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                           regularizer=tf.keras.regularizers.L2(0.1))
            self.beta_a1 = self.add_weight(shape=(self.capsules, 1),
                                           initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                           regularizer=tf.keras.regularizers.L2(0.1))
            self.beta_v2 = self.add_weight(shape=(self.capsules, self.dim_capsules_2),
                                           initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                           regularizer=tf.keras.regularizers.L2(0.1))
            self.beta_a2 = self.add_weight(shape=(self.capsules, 1),
                                           initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                           regularizer=tf.keras.regularizers.L2(0.1))
            self.w2 = self.add_weight(shape=(self.ch - self.n_cond_caps, self.n_capsch_i,
                                             self.dim_capsules, self.dim_capsules),
                                      initializer=tf.keras.initializers.random_normal(stddev=0.1),
                                      regularizer=tf.keras.regularizers.L2(0.1))

        self.built = True

    def call(self, inputs, **kwargs):
        pose, activation = inputs

        u_i = tf.reshape(pose, (self.batch_size, self.n_capsch_i, self.ch, self.dim_capsules_2))
        activation = tf.reshape(activation, (self.batch_size, self.n_capsch_i, self.ch, 1))
        coords = create_coords_mat(pose, self.rel_center, self.dim_capsules) if self.coord_add else tf.zeros_like(u_i)

        # reshapes the input capsules
        u_i = tf.reshape(u_i, (self.batch_size, self.n_capsch_i, self.ch, self.dim_capsules, self.dim_capsules))
        u_i = tf.expand_dims(u_i, axis=-3)
        u_i = tf.tile(u_i, [1, 1, 1, self.capsules, 1, 1])

        if self.ch_same_w:
            votes = tf.einsum('ijab,ntijbc->ntijac', self.w, u_i)
        else:
            votes = tf.einsum('tijab,ntijbc->ntijac', self.w, u_i)
        votes = tf.reshape(votes, (self.batch_size, self.n_capsch_i * self.ch, self.capsules, self.dim_capsules_2))

        if self.coord_add:
            coords = tf.reshape(coords, (self.batch_size, self.n_capsch_i * self.ch, 1, self.dim_capsules_2))
            votes = votes + tf.tile(coords, [1, 1, self.capsules, 1])

        if self.n_cond_caps == 0:
            acts = tf.reshape(activation, (self.batch_size, self.n_capsch_i * self.ch, 1))

            capsules1 = em_routing(votes, acts, self.beta_v, self.beta_a)
            capsules2 = capsules1
        else:
            votes = tf.reshape(votes, (self.batch_size, self.n_capsch_i, self.ch, self.capsules, self.dim_capsules_2))

            votes1 = tf.reshape(votes[:, :, :self.ch - self.n_cond_caps],
                                (self.batch_size, self.n_capsch_i *
                                 (self.ch - self.n_cond_caps), self.capsules, self.dim_capsules_2))
            votes2 = tf.reshape(votes[:, :, self.ch - self.n_cond_caps:],
                                (self.batch_size, self.n_capsch_i * self.n_cond_caps,
                                 self.capsules, self.dim_capsules_2))

            acts = tf.reshape(activation, (self.batch_size, self.n_capsch_i, self.ch, 1))

            acts1 = tf.reshape(acts[:, :, :self.ch - self.n_cond_caps],
                               (self.batch_size, self.n_capsch_i * (self.ch - self.n_cond_caps), 1))
            acts2 = tf.reshape(acts[:, :, self.ch - self.n_cond_caps:],
                               (self.batch_size, self.n_capsch_i * self.n_cond_caps, 1))

            votes_2 = tf.einsum('ijab,ntijbc->ntijac', self.w2, u_i[:, :, :self.ch - self.n_cond_caps])
            votes_2 = tf.reshape(votes_2, (self.batch_size, self.n_capsch_i * (self.ch - self.n_cond_caps),
                                           self.capsules, self.dim_capsules_2))

            capsules1, capsules2 = em_routing_cond(votes1, votes_2, acts1, votes2, acts2,
                                                   self.beta_v1, self.beta_a1, self.beta_v2, self.beta_a2)

        return capsules1, capsules2

    def get_config(self):
        return super(ConditionDenseCapsule, self).get_config()


class ConditionConvolutional3DCapsule(tf.keras.layers.Layer):
    def __init__(self, capsules, dim_capsules, kernel_size, strides, padding='same',
                 coord_add=False, rel_center=True, route_mean=True, ch_same_w=True,
                 n_cond_caps=0, **kwargs):
        super(ConditionConvolutional3DCapsule, self).__init__(**kwargs)
        self.capsules = capsules
        self.dim_capsules = dim_capsules
        self.dim_capsules_2 = dim_capsules * dim_capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.coord_add = coord_add
        self.rel_center = rel_center
        self.route_mean = route_mean
        self.ch_same_w = ch_same_w
        self.n_cond_caps = n_cond_caps
        self.dense_caps = None

    def build(self, input_shape):
        if self.route_mean:
            self.dense_caps = ConditionDenseCapsule(capsules=self.capsules, ch_same_w=self.ch_same_w,
                                                    dim_capsules=self.dim_capsules, n_cond_caps=self.n_cond_caps)
        else:
            self.dense_caps = ConditionDenseCapsule(capsules=self.capsules, coord_add=self.coord_add,
                                                    rel_center=self.rel_center, ch_same_w=self.ch_same_w,
                                                    dim_capsules=self.dim_capsules, n_cond_caps=self.n_cond_caps)

        self.built = True

    def call(self, inputs, **kwargs):
        inputs = tf.concat(inputs, axis=-1)

        if self.padding == 'same':
            d_padding, h_padding, w_padding = int(float(self.kernel_size[0]) / 2), \
                                              int(float(self.kernel_size[1]) / 2), \
                                              int(float(self.kernel_size[2]) / 2)
            u_padded = tf.pad(inputs,
                              [[0, 0], [d_padding, d_padding], [h_padding, h_padding], [w_padding, w_padding], [0, 0],
                               [0, 0]])
        else:
            u_padded = inputs

        batch_size = tf.shape(u_padded)[0]
        _, d, h, w, ch, _ = u_padded.shape
        d, h, w, ch = map(int, [d, h, w, ch])

        # gets indices for kernels
        d_offsets = [[(d_ + k) for k in range(self.kernel_size[0])]
                     for d_ in range(0, d + 1 - self.kernel_size[0], self.strides[0])]
        h_offsets = [[(h_ + k) for k in range(self.kernel_size[1])]
                     for h_ in range(0, h + 1 - self.kernel_size[1], self.strides[1])]
        w_offsets = [[(w_ + k) for k in range(self.kernel_size[2])]
                     for w_ in range(0, w + 1 - self.kernel_size[2], self.strides[2])]

        # output dimensions
        d_out, h_out, w_out = len(d_offsets), len(h_offsets), len(w_offsets)

        # gathers the capsules into shape (N, D2, H2, W2, KD, KH, KW, Ch_in, 17)
        d_gathered = tf.gather(u_padded, d_offsets, axis=1)
        h_gathered = tf.gather(d_gathered, h_offsets, axis=3)
        w_gathered = tf.gather(h_gathered, w_offsets, axis=5)
        w_gathered = tf.transpose(w_gathered, [0, 1, 3, 5, 2, 4, 6, 7, 8])

        if self.route_mean:
            kernels_reshaped = tf.reshape(w_gathered,
                                          [batch_size * d_out * h_out * w_out, self.kernel_size[0] *
                                           self.kernel_size[1] * self.kernel_size[2], ch, self.dim_capsules_2 + 1])
            kernels_reshaped = tf.reduce_mean(kernels_reshaped, axis=1)
            capsules1, capsules2 = self.dense_caps([kernels_reshaped[:, :, :-1], kernels_reshaped[:, :, -1:]])
        else:
            kernels_reshaped = tf.reshape(w_gathered,
                                          [batch_size * d_out * h_out * w_out, self.kernel_size[0],
                                           self.kernel_size[1], self.kernel_size[2], ch, self.dim_capsules_2 + 1])
            capsules1, capsules2 = self.dense_caps([kernels_reshaped[:, :, :, :, :, :-1],
                                                    kernels_reshaped[:, :, :, :, :, -1:]])
        poses1 = tf.reshape(capsules1[0][:, :, :self.dim_capsules_2],
                            (batch_size, d_out, h_out, w_out, self.capsules, self.dim_capsules_2))
        activations1 = tf.reshape(capsules1[1], (batch_size, d_out, h_out, w_out, self.capsules, 1))

        poses2 = tf.reshape(capsules2[0][:, :, :self.dim_capsules_2],
                            (batch_size, d_out, h_out, w_out, self.capsules, self.dim_capsules_2))
        activations2 = tf.reshape(capsules2[1], (batch_size, d_out, h_out, w_out, self.capsules, 1))

        return (poses1, activations1), (poses2, activations2)

    def get_config(self):
        return super(ConditionConvolutional3DCapsule, self).get_config
