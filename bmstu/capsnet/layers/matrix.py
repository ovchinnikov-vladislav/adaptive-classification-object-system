import tensorflow as tf
from tensorflow.keras import layers
from bmstu.capsnet.em_utils import kernel_tile, mat_transform, matrix_capsules_em_routing, coord_addition


class PrimaryCapsule2D(layers.Layer):
    def __init__(self, capsules, kernel_size, strides, padding, pose_shape, **kwargs):
        super(PrimaryCapsule2D, self).__init__(**kwargs)
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

    def call(self, inputs, **kwargs):
        pose = self.conv2d_pose(inputs)
        activation = self.conv2d_activation(inputs)

        pose = tf.reshape(pose, shape=[-1, inputs.shape[-3], inputs.shape[-2],
                                       self.capsules, self.pose_shape[0], self.pose_shape[1]])

        # tf.print('PrimaryCaps2D activation', activation)
        # tf.print('PrimaryCaps2D pose', pose)

        return pose, activation

    def get_config(self):
        return super(PrimaryCapsule2D, self).get_config()


class ConvolutionalCapsule(layers.Layer):
    def __init__(self, shape, strides, routings, **kwargs):
        super(ConvolutionalCapsule, self).__init__(**kwargs)
        self.shape = shape
        self.strides = strides
        self.routings = routings
        self.stride = strides[1]
        self.i_size = shape[-2]
        self.o_size = shape[-1]
        self.batch_size = self.w = self.beta_a = self.beta_v = None

    def build(self, input_shape):
        truncated_normal_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0)
        # (1, 288, 32, 4, 4)
        self.w = tf.Variable(lambda: truncated_normal_initializer(shape=[1, 3 * 3 * self.i_size, self.o_size, 4, 4],
                                                                  dtype=tf.float32), name='w', trainable=True)
        glorot_uniform_initializer = tf.keras.initializers.GlorotUniform()
        self.beta_v = tf.Variable(lambda: glorot_uniform_initializer(shape=[1, 1, 1, self.o_size], dtype=tf.float32),
                                  trainable=True)
        self.beta_a = tf.Variable(lambda: glorot_uniform_initializer(shape=[1, 1, 1, self.o_size], dtype=tf.float32),
                                  trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_pose, inputs_activation = inputs
        batch_size = inputs_pose.shape[0]
        pose_size = inputs_pose.shape[-1]

        inputs_pose = kernel_tile(inputs_pose, 3, self.stride)
        inputs_activation = kernel_tile(inputs_activation, 3, self.stride)

        spatial_size = int(inputs_activation.shape[1])
        inputs_pose = tf.reshape(inputs_pose, shape=[-1, 3 * 3 * self.i_size, 16])
        inputs_activation = tf.reshape(inputs_activation, shape=[-1, spatial_size, spatial_size, 3 * 3 * self.i_size])

        votes = mat_transform(inputs_pose, self.o_size, size=batch_size * spatial_size * spatial_size, w=self.w)
        votes = tf.reshape(votes, shape=[batch_size, spatial_size, spatial_size, votes.shape[-3], votes.shape[-2],
                                         votes.shape[-1]])

        pose, activation = matrix_capsules_em_routing(votes, inputs_activation, self.beta_v, self.beta_a, self.routings)
        pose = tf.reshape(pose, shape=[pose.shape[0], pose.shape[1], pose.shape[2],
                                       pose.shape[3], pose_size, pose_size])

        # tf.print('ConvCaps activation', activation)

        return pose, activation

    def get_config(self):
        return super(ConvolutionalCapsule, self).get_config()


class ClassCapsule(layers.Layer):
    def __init__(self, classes, routings, **kwargs):
        super(ClassCapsule, self).__init__(**kwargs)
        self.classes = classes
        self.routings = routings
        self.w = self.beta_v = self.beta_a = None

    def build(self, input_shape):
        truncated_normal_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0)
        self.w = tf.Variable(lambda: truncated_normal_initializer(shape=[1, input_shape[0][-3], self.classes, 4, 4],
                                                                  dtype=tf.float32), name='w',
                             trainable=True)
        glorot_uniform_initializer = tf.keras.initializers.GlorotUniform()
        self.beta_v = tf.Variable(lambda: glorot_uniform_initializer(shape=[1, self.classes], dtype=tf.float32),
                                  trainable=True)
        self.beta_a = tf.Variable(lambda: glorot_uniform_initializer(shape=[1, self.classes], dtype=tf.float32),
                                  trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_pose, inputs_activation = inputs

        inputs_shape = inputs_pose.shape
        spatial_size = int(inputs_shape[1])
        pose_size = int(inputs_shape[-1])
        i_size = int(inputs_shape[3])
        batch_size = int(inputs_shape[0])

        inputs_pose = tf.reshape(inputs_pose, shape=[batch_size * spatial_size * spatial_size, inputs_shape[-3],
                                                     inputs_shape[-2] * inputs_shape[-2]])

        votes = mat_transform(inputs_pose, self.classes, size=batch_size * spatial_size * spatial_size, w=self.w)

        votes = tf.reshape(votes, shape=[batch_size, spatial_size, spatial_size, i_size,
                                         self.classes, pose_size * pose_size])

        votes = coord_addition(votes, spatial_size, spatial_size)

        votes_shape = votes.shape
        votes = tf.reshape(votes, shape=[batch_size, votes_shape[1] * votes_shape[2] * votes_shape[3],
                                         votes_shape[4], votes_shape[5]])

        inputs_activation = tf.reshape(inputs_activation, shape=[batch_size,
                                                                 votes_shape[1] * votes_shape[2] * votes_shape[3]])

        pose, activation = matrix_capsules_em_routing(votes, inputs_activation, self.beta_v,
                                                      self.beta_a, self.routings)

        pose = tf.reshape(pose, shape=[batch_size, self.classes, pose_size, pose_size])

        # tf.print('ClassCaps activation', activation)

        return pose, activation

    def get_config(self):
        return super(ClassCapsule, self).get_config()
