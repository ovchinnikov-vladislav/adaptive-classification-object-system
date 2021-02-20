from keras import layers
import keras.backend as keras_backend
import tensorflow as tf
import numpy as np
from ml.tf1.matrix_capsnet import utl


class PrimaryCaps(layers.Layer):
    def __init__(self, num_capsules, kernel_size, strides, padding, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.pose_conv2d = self.activation_conv2d = None

    def build(self, input_shape):
        self.built = True
        self.pose_conv2d = layers.Conv2D(filters=self.num_capsules * 16,
                                         kernel_size=self.kernel_size,
                                         strides=self.strides,
                                         padding=self.padding)
        self.activation_conv2d = layers.Conv2D(filters=self.num_capsules,
                                               kernel_size=self.kernel_size,
                                               strides=self.strides,
                                               padding=self.padding,
                                               activation='sigmoid')

    def call(self, inputs, **kwargs):
        pose = self.pose_conv2d(inputs)
        print(pose.shape)
        activation = self.activation_conv2d(inputs)
        print(activation.shape)

        spatial_size = int(pose.shape[1])
        pose = layers.Reshape(target_shape=[spatial_size, spatial_size, self.num_capsules, 16])(pose)
        activation = layers.Reshape(target_shape=[spatial_size, spatial_size, self.num_capsules, 1])(activation)

        assert pose.shape[1:] == [spatial_size, spatial_size, self.num_capsules, 16]
        assert activation.shape[1:] == [spatial_size, spatial_size, self.num_capsules, 1]

        print(f'primary_caps pose shape: {pose.shape}')
        print(f'primary_caps activation shape {activation.shape}')

        return layers.Concatenate()([pose, activation])


class ConvCapsules(layers.Layer):
    def __init__(self, kernel_size, strides, num_capsules, routings, batch_size, weights_regularizer, **kwargs):
        super(ConvCapsules, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.batch_size = batch_size
        self.routings = routings
        self.num_capsules = num_capsules
        self.weights_regularizer = weights_regularizer

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        pose = inputs[:, :, :, :, :16]
        activation = inputs[:, :, :, :, 16:]
        # Get shapes
        shape = pose.shape
        child_space = int(shape[1])
        child_space_2 = int(child_space) ** 2
        child_caps = int(shape[3])
        parent_space = int(np.floor((child_space - self.kernel_size) / self.strides + 1))
        parent_space_2 = int(parent_space) ** 2
        parent_caps = self.num_capsules
        kernel_2 = int(self.kernel_size) ** 2

        # Tile poses and activations
        # (64, 7, 7, 8, 16)  -> (64, 5, 5, 9, 8, 16)
        pose_tiled, spatial_routing_matrix = utl.kernel_tile(pose, kernel_size=self.kernel_size, strides=self.strides,
                                                             batch_size=self.batch_size)
        activation_tiled, _ = utl.kernel_tile(activation, kernel_size=self.kernel_size, strides=self.strides,
                                              batch_size=self.batch_size)

        # Check dimensions of spatial_routing_matrix
        assert spatial_routing_matrix.shape == (child_space_2, parent_space_2)

        # Unroll along batch_size and parent_space_2
        # (64, 5, 5, 9, 8, 16) -> (64*5*5, 9*8, 16)
        pose_unroll = tf.reshape(pose_tiled, shape=[self.batch_size * parent_space_2, kernel_2 * child_caps, 16])
        activation_unroll = tf.reshape(activation_tiled,
                                       shape=[self.batch_size * parent_space_2, kernel_2 * child_caps, 1])

        # (64*5*5, 9*8, 16) -> (64*5*5, 9*8, 32, 16)
        votes = utl.compute_votes(pose_unroll, parent_caps, self.weights_regularizer, tag=True)
        print(f'{self.name} votes shape: {votes.shape}')

        # votes (64*5*5, 9*8, 32, 16)
        # activations (64*5*5, 9*8, 1)
        # pose_out: (N, OH, OW, o, 4x4)
        # activation_out: (N, OH, OW, o, 1)
        pose_out, activation_out = utl.em_routing(votes, activation_unroll, self.batch_size, spatial_routing_matrix,
                                                  self.routings, 0.01)

        print(f'{self.name} pose_out shape: {pose_out.shape}')
        print(f'{self.name} activation_out shape: {activation_out.shape}')

        return layers.Concatenate()([activation_out, pose_out])


def ClassCapsules(inputs, num_capsules, batch_size, routings, weights_regularizer):
    pose = inputs[:, :, :, :, :16]
    activation = inputs[:, :, :, :, 16:]
    shape = pose.shape
    child_space = int(shape[1])
    child_caps = int(shape[3])

    # In the class_caps layer, we apply same multiplication to every spatial
    # location, so we unroll along the batch and spatial dimensions
    # (64, 5, 5, 32, 16) -> (64*5*5, 32, 16)
    pose = tf.reshape(pose, shape=[batch_size * child_space * child_space, child_caps, 16])
    activation = tf.reshape(activation, shape=[batch_size * child_space * child_space, child_caps, 1],
                            name="activation")
    # (64*5*5, 32, 16) -> (65*5*5, 32, 5, 16)
    votes = utl.compute_votes(pose, num_capsules, weights_regularizer)

    # (65*5*5, 32, 5, 16)
    assert (votes.shape == [batch_size * child_space * child_space, child_caps, num_capsules, 16])
    print(f'class_caps votes original shape: {votes.shape}')

    # (64*5*5, 32, 5, 16)
    votes = tf.reshape(votes, [batch_size, child_space, child_space, child_caps,
                               num_capsules, votes.shape[-1]])
    votes = utl.coord_addition(votes)

    # Flatten the votes:
    # Combine the 4 x 4 spacial dimensions to appear as one spacial dimension       # with many capsules.
    # [64*5*5, 16, 5, 16] -> [64, 5*5*16, 5, 16]
    votes_flat = tf.reshape(votes, shape=[batch_size, child_space * child_space * child_caps,
                                          num_capsules, votes.shape[-1]])
    activation_flat = tf.reshape(activation, shape=[batch_size, child_space * child_space * child_caps, 1])
    spatial_routing_matrix = utl.create_routing_map(child_space=1, k=1, s=1)

    print(f'class_caps votes in to routing shape: {votes_flat.shape}')
    pose_out, activation_out = utl.em_routing(votes_flat, activation_flat, batch_size, spatial_routing_matrix,
                                              routings, 0.01)

    activation_out = tf.squeeze(activation_out, name="activation_out")
    pose_out = tf.squeeze(pose_out, name="pose_out")

    print(f'class_caps activation shape: {activation_out.shape}')
    print(f'class_caps pose shape: {pose_out.shape}')

    return [activation_out, pose_out]
