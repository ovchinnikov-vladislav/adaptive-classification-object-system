import tensorflow as tf
import numpy as np
from tensorflow.keras import activations
from functools import reduce


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
