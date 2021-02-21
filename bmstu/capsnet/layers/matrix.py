from abc import ABC

import tensorflow as tf
from bmstu.capsnet.em_utils import kernel_tile, mat_transform, matrix_capsules_em_routing


class PrimaryCapsule2D(tf.keras.Model, ABC):
    def __init__(self, capsules, kernel_size, strides, padding, pose_shape, name=''):
        super(PrimaryCapsule2D, self).__init__(name)
        self.capsules = capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.pose_shape = pose_shape

        num_filters = capsules * pose_shape[0] * pose_shape[1]
        self.conv2d_pose = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size,
                                                  strides=strides, padding=padding)
        self.conv2d_activation = tf.keras.layers.Conv2D(filters=capsules, kernel_size=kernel_size,
                                                        strides=strides, padding=padding, activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        pose = self.conv2d_pose(inputs)
        activation = self.conv2d_activation(inputs)

        pose = tf.reshape(pose, shape=[-1, inputs.shape[-3], inputs.shape[-2],
                                       self.capsules, self.pose_shape[0], self.pose_shape[1]])

        return pose, activation


class ConvolutionalCapsule(tf.keras.Model, ABC):
    def __init__(self, shape, strides, routings, name=''):
        super(ConvolutionalCapsule, self).__init__(name)
        self.shape = shape
        self.strides = strides
        self.routings = routings
        self.stride = strides[1]
        self.i_size = shape[-2]
        self.o_size = shape[-1]
        self.batch_size = None

    def call(self, inputs, training=None, mask=None):
        inputs_pose, inputs_activation = inputs
        batch_size = inputs_pose.shape[0]
        pose_size = inputs_pose.shape[-1]

        inputs_pose = kernel_tile(inputs_pose, 3, self.stride)
        inputs_activation = kernel_tile(inputs_activation, 3, self.stride)

        spatial_size = int(inputs_activation.shape[1])
        inputs_pose = tf.reshape(inputs_pose, shape=[-1, 3 * 3 * self.i_size, 16])
        inputs_activation = tf.reshape(inputs_activation, shape=[-1, spatial_size, spatial_size, 3 * 3 * self.i_size])

        votes = mat_transform(inputs_pose, self.o_size, size=batch_size * spatial_size * spatial_size)
        votes = tf.reshape(votes, shape=[batch_size, spatial_size, spatial_size, votes.shape[-3], votes.shape[-2],
                                         votes.shape[-1]])

        glorot_uniform_initializer = tf.keras.initializers.GlorotUniform()
        beta_v = tf.Variable(lambda: glorot_uniform_initializer(shape=[1, 1, 1, self.o_size], dtype=tf.float32))
        beta_a = tf.Variable(lambda: glorot_uniform_initializer(shape=[1, 1, 1, self.o_size], dtype=tf.float32))

        pose, activation = matrix_capsules_em_routing(votes, inputs_activation, beta_v, beta_a, self.routings)
        pose = tf.reshape(pose, shape=[pose.shape[0], pose.shape[1], pose.shape[2],
                                       pose.shape[3], pose_size, pose_size])

        return pose, activation
