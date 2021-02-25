import tensorflow as tf
import numpy as np
from tensorflow.keras import activations
from functools import reduce
from bmstu.capsnet.vos_utils import em_routing, create_coords_mat


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
        coords = create_coords_mat(pose, self.rel_center) if self.coord_add else tf.zeros_like(u_i)

        u_i = tf.reshape(u_i, (self.batch_size, self.n_capsch_i, self.ch, self.dim_capsules, self.dim_capsules))
        u_i = tf.expand_dims(u_i, axis=-3)
        u_i = tf.tile(u_i, [1, 1, 1, self.capsules, 1, 1])

        votes = tf.einsum('ijab,ntijbc->ntijac', self.w, u_i)
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
