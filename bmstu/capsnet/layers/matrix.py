from abc import ABC

import tensorflow as tf


class PrimaryCapsule2D(tf.keras.Model, ABC):
    def __init__(self, capsules, kernel_size, strides, name=''):
        super(PrimaryCapsule2D, self).__init__(name)
        self.capsules = capsules
        self.kernel_size = kernel_size
        self.strides = strides

        num_filters = capsules * 16
        self.conv2d_pose = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size,
                                                  strides=strides, padding='valid')
        self.conv2d_activation = tf.keras.layers.Conv2D(filters=capsules, kernel_size=kernel_size,
                                                        strides=strides, padding='valid', activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        pose = self.conv2d_pose(inputs)
        activation = self.conv2d_activation(inputs)

        pose = tf.keras.layers.Reshape(target_shape=[inputs.shape[1], inputs.shape[2], self.capsules, 16])(pose)
        activation = tf.keras.layers.Reshape(target_shape=[inputs.shape[1],
                                                           inputs.shape[2], self.capsules, 1])(activation)

        output = tf.keras.layers.Concatenate(axis=4)([pose, activation])
        output = tf.keras.layers.Reshape(target_shape=[inputs.shape[1], inputs.shape[2], -1])(output)

        return output


class ConvolutionalCapsule(tf.keras.Model, ABC):
    def __init__(self, capsules, routings, name=''):
        super(ConvolutionalCapsule, self).__init__(name)
        self.capsules = capsules
        self.routings = routings

    def build(self, input_shape):
        pass

    def call(self, inputs, training=None, mask=None):
        pass
