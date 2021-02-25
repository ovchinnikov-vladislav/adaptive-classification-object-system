import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, activations
from bmstu.capsnet.em_utils import kernel_tile, matrix_capsules_em_routing, \
    compute_votes, create_routing_map, coord_addition, init_beta


class PrimaryCapsule2D(layers.Layer):
    def __init__(self, capsules, kernel_size, strides, padding, pose_shape, **kwargs):
        super(PrimaryCapsule2D, self).__init__(**kwargs)
        self.capsules = capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.pose_shape = pose_shape
        self.batch_size = self.conv2d_pose = self.conv2d_activation = None

    def build(self, input_shape):
        num_filters = self.capsules * self.pose_shape[0] * self.pose_shape[1]
        self.conv2d_pose = tf.keras.layers.Conv2D(filters=num_filters,
                                                  kernel_size=self.kernel_size,
                                                  strides=self.strides,
                                                  padding=self.padding)
        self.conv2d_activation = tf.keras.layers.Conv2D(filters=self.capsules,
                                                        kernel_size=self.kernel_size,
                                                        strides=self.strides,
                                                        padding=self.padding,
                                                        activation=activations.sigmoid)
        self.built = True

    def call(self, inputs, **kwargs):
        self.batch_size = inputs.shape[0]

        pose = self.conv2d_pose(inputs)
        activation = self.conv2d_activation(inputs)

        spatial_size = int(pose.shape[1])
        pose = tf.reshape(pose, shape=[self.batch_size, spatial_size, spatial_size,
                                       self.capsules, self.pose_shape[0] * self.pose_shape[1]])

        activation = tf.reshape(activation, shape=[self.batch_size, spatial_size, spatial_size,
                                                   self.capsules, 1])

        # tf.print('PrimaryCaps2D activation', activation)
        # tf.print('PrimaryCaps2D pose', pose)

        return pose, activation

    def get_config(self):
        return super(PrimaryCapsule2D, self).get_config()


class ConvolutionalCapsule(layers.Layer):
    """Convolutional capsule layer.
        "The routing procedure is used between each adjacent pair of capsule layers.
        For convolutional capsules, each capsule in layer L + 1 sends feedback only to
        capsules within its receptive field in layer L. Therefore each convolutional
        instance of a capsule in layer L receives at most kernel size X kernel size
        feedback from each capsule type in layer L + 1. The instances closer to the
        border of the image receive fewer feedbacks with corner ones receiving only
        one feedback per capsule type in layer L + 1."
        See Hinton et al. "Matrix Capsules with EM Routing" for detailed description
        convolutional capsule layer.
        Author:
          Vladislav Ovchinnikov
        Args:
          kernel_size:
          strides:
          capsules: depth dimension of parent capsules
          routings:
          weights_regularizer:
        """

    def __init__(self, kernel_size, strides, capsules, routings, weights_regularizer, **kwargs):
        super(ConvolutionalCapsule, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.routings = routings
        self.capsules = capsules
        self.weights_regularizer = weights_regularizer
        self.batch_size = self.shape = None
        self.child_space = self.square_child_space = self.child_caps = None
        self.parent_space = self.square_parent_space = self.parent_caps = None
        self.square_kernel_size = int(kernel_size ** 2)
        self.w = self.beta_a = self.beta_v = None

    def build(self, input_shape):
        self.shape = input_shape[0]
        self.child_space = self.shape[1]
        self.square_child_space = int(self.child_space ** 2)
        self.child_caps = self.shape[3]
        self.parent_space = int(np.floor((self.child_space - self.kernel_size) / self.strides + 1))
        self.square_parent_space = int(self.parent_space ** 2)
        self.parent_caps = self.capsules

        self.w = self.add_weight(
            shape=[1, self.square_kernel_size * self.child_caps, self.parent_caps, 4, 4],
            name='w',
            trainable=True,
            regularizer=self.weights_regularizer,
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0))

        self.beta_a, self.beta_v = init_beta(self, self.parent_caps)
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_pose, inputs_activation = inputs
        self.batch_size = inputs_pose.shape[0]

        # Block votes
        pose_tiled, spatial_routing_matrix = kernel_tile(inputs_pose,
                                                         kernel=self.kernel_size,
                                                         stride=self.strides)
        activation_tiled, _ = kernel_tile(inputs_activation,
                                          kernel=self.kernel_size,
                                          stride=self.strides)
        pose_unroll = tf.reshape(pose_tiled,
                                 shape=[self.batch_size * self.square_parent_space,
                                        self.square_kernel_size * self.child_caps, 16])
        activation_unroll = tf.reshape(activation_tiled,
                                       shape=[self.batch_size * self.square_parent_space,
                                              self.square_kernel_size * self.child_caps, 1])
        votes = compute_votes(pose_unroll, self.parent_caps, self.w)

        # Block routing
        pose, activation = matrix_capsules_em_routing(votes=votes,
                                                      activation=activation_unroll,
                                                      spatial_routing_matrix=spatial_routing_matrix,
                                                      batch_size=self.batch_size,
                                                      routings=self.routings,
                                                      final_lambda=0.01,
                                                      beta_a=self.beta_a,
                                                      beta_v=self.beta_v)

        return pose, activation

    def get_config(self):
        return super(ConvolutionalCapsule, self).get_config()


class ClassCapsule(layers.Layer):
    def __init__(self, classes, routings, weights_regularizer, **kwargs):
        super(ClassCapsule, self).__init__(**kwargs)
        self.classes = classes
        self.routings = routings
        self.weights_regularizer = weights_regularizer
        self.shape = self.batch_size = self.child_space = self.child_caps = None
        self.w = self.beta_v = self.beta_a = None

    def build(self, input_shape):
        self.shape = input_shape[0]
        self.child_space = self.shape[1]
        self.child_caps = self.shape[3]

        self.w = self.add_weight(
            shape=[1, self.child_caps, self.classes, 4, 4],
            name='w',
            trainable=True,
            regularizer=self.weights_regularizer,
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0))
        self.beta_a, self.beta_v = init_beta(self, self.classes)
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_pose, inputs_activation = inputs
        self.batch_size = inputs_pose.shape[0]

        # Block votes
        pose = tf.reshape(inputs_pose, shape=[self.batch_size * self.child_space * self.child_space,
                                              self.child_caps, 16])
        activation = tf.reshape(inputs_activation, shape=[self.batch_size * self.child_space * self.child_space,
                                                          self.child_caps, 1])
        votes = compute_votes(pose, self.classes, self.w)

        # Block coord add
        votes = tf.reshape(votes, shape=[self.batch_size, self.child_space, self.child_space,
                                         self.child_caps, self.classes, votes.shape[-1]])
        votes = coord_addition(votes)

        # Block routing
        votes_flat = tf.reshape(
            votes, shape=[self.batch_size, self.child_space * self.child_space * self.child_caps,
                          self.classes, votes.shape[-1]])
        activation_flat = tf.reshape(
            activation, shape=[self.batch_size, self.child_space * self.child_space * self.child_caps, 1])
        spatial_routing_matrix = create_routing_map(child_space=1, k=1, s=1)
        pose, activation = matrix_capsules_em_routing(votes=votes_flat,
                                                      activation=activation_flat,
                                                      spatial_routing_matrix=spatial_routing_matrix,
                                                      batch_size=self.batch_size,
                                                      routings=self.routings,
                                                      final_lambda=0.01,
                                                      beta_a=self.beta_a,
                                                      beta_v=self.beta_v)
        activation = tf.squeeze(activation)
        pose = tf.squeeze(pose)

        return pose, activation

    def get_config(self):
        return super(ClassCapsule, self).get_config()


if __name__ == '__main__':
    input_layer = layers.Input(shape=[28, 28, 1], batch_size=64)
    conv1 = layers.Conv2D(filters=64, kernel_size=5, strides=2,
                          padding='same', activation=activations.relu)(input_layer)
    primaryCaps = PrimaryCapsule2D(capsules=8, kernel_size=1, strides=1,
                                   padding='valid', pose_shape=[4, 4])(conv1)
    convCaps1 = ConvolutionalCapsule(kernel_size=3, strides=2, capsules=16, routings=3,
                                     weights_regularizer=tf.keras.regularizers.L2(0.0000002))(primaryCaps)
    convCaps2 = ConvolutionalCapsule(kernel_size=3, strides=1, capsules=16, routings=3,
                                     weights_regularizer=tf.keras.regularizers.L2(0.0000002))(convCaps1)
    classCaps = ClassCapsule(classes=10, routings=3,
                             weights_regularizer=tf.keras.regularizers.L2(0.0000002))(convCaps2)
    model = tf.keras.Model(input_layer, classCaps)
    model.summary(line_length=200)
