from abc import ABC

import tensorflow as tf


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

        print(pose.shape)
        print(activation.shape)
        return pose, activation


class ConvolutionalCapsule(tf.keras.Model, ABC):
    def __init__(self, shape, strides, routings, name=''):
        super(ConvolutionalCapsule, self).__init__(name)
        self.shape = shape
        self.strides = strides
        self.routings = routings

    def build(self, input_shape):
        pass

    def call(self, inputs, training=None, mask=None):
        pass
