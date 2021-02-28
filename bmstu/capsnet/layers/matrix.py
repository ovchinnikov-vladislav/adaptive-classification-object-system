import tensorflow as tf
from tensorflow.keras import layers, activations
from bmstu.capsnet.em_utils import kernel_tile, coord_addition, mat_transform, matrix_capsules_em_routing


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
        self.batch_size = tf.shape(inputs)[0]

        pose = self.conv2d_pose(inputs)
        pose = tf.reshape(pose, shape=[-1, inputs.shape[-3], inputs.shape[-2],
                                       self.capsules, self.pose_shape[0], self.pose_shape[1]])

        activation = self.conv2d_activation(inputs)

        # pose (?, 14, 14, 32, 4, 4), activation (?, 14, 14, 32)
        # print('1. primaryCaps {', 'pose', pose.shape, 'activation', activation.shape, '}')

        return pose, activation

    def get_config(self):
        return super(PrimaryCapsule2D, self).get_config()


class ConvolutionalCapsule(layers.Layer):
    def __init__(self, kernel_size, strides, capsules, routings, **kwargs):
        super(ConvolutionalCapsule, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.routings = routings
        self.capsules = capsules
        self.batch_size = None
        self.w = self.beta_a = self.beta_v = None
        self.stride = strides[1]  # 2
        self.i_size = self.o_size = None
        self.pose_size = None

    def build(self, input_shape):
        pose_shape = input_shape[0]
        self.i_size = pose_shape[3]  # 32
        self.o_size = self.capsules  # 32
        self.pose_size = pose_shape[-1]  # 4
        caps_num_i = 3 * 3 * self.i_size  # 288

        self.w = self.add_weight(
            shape=[1, caps_num_i, self.capsules, 4, 4],
            name='w',
            trainable=True,
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0))  # (1, 288, 32, 4, 4)

        # print('2. convCaps { w', self.w.shape, '}')

        self.beta_a = self.add_weight(
            name='beta_a',
            shape=[1, 1, 1, self.o_size],
            trainable=True,
            dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotUniform())
        self.beta_v = self.add_weight(
            name='beta_v',
            trainable=True,
            shape=[1, 1, 1, self.o_size],
            dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotUniform())

        self.built = True

    def call(self, inputs, **kwargs):
        inputs_pose, inputs_activation = inputs
        batch_size = inputs_pose.shape[0]

        inputs_pose = kernel_tile(inputs_pose, 3, self.stride)  # (?, 14, 14, 32, 4, 4) -> (?, 6, 6, 3x3=9, 32x16=512)

        inputs_activation = kernel_tile(inputs_activation, 3, self.stride)  # (?, 14, 14, 32) -> (?, 6, 6, 9, 32)
        # print('2.1. convCaps { inputs_pose', inputs_pose.shape, 'inputs_activation', inputs_activation.shape, '}')

        spatial_size = int(inputs_activation.shape[1])  # 6

        inputs_pose = tf.reshape(inputs_pose, shape=[-1, 3 * 3 * self.i_size, 16])  # (?, 9x32=288, 16)
        inputs_activation = tf.reshape(inputs_activation,
                                       shape=[-1, spatial_size, spatial_size,
                                              3 * 3 * self.i_size])  # (?, 6, 6, 9x32=288)
        # print('2.2. convCaps { inputs_pose', inputs_pose.shape, 'inputs_activation', inputs_activation.shape, '}')

        # Block votes
        votes = mat_transform(inputs_pose, self.o_size,
                              batch_size * spatial_size * spatial_size, self.w)  # (864, 288, 32, 16)

        # print('2.3. convCaps { votes', votes.shape, '}')

        votes_shape = votes.shape
        votes = tf.reshape(votes, shape=[batch_size, spatial_size, spatial_size, votes_shape[-3], votes_shape[-2],
                                         votes_shape[-1]])  # (24, 6, 6, 288, 32, 16)

        # print('2.4. convCaps { votes', votes.shape, '}')

        # Block routing
        # votes (24, 6, 6, 3x3x32=288, 32, 16), inputs_activation (?, 6, 6, 288)
        # pose (24, 6, 6, 32, 16), activation (24, 6, 6, 32)
        pose, activation = matrix_capsules_em_routing(votes, inputs_activation,
                                                      self.beta_v, self.beta_a, self.routings)
        # print('2.5. convCaps { pose', pose.shape, 'activation', activation.shape, '}')

        pose_shape = pose.shape
        pose = tf.reshape(pose, [pose_shape[0], pose_shape[1], pose_shape[2], pose_shape[3],
                                 self.pose_size, self.pose_size])  # (24, 6, 6, 32, 4, 4)

        # print('2.6. convCaps { pose', pose.shape, '}')

        return pose, activation

    def get_config(self):
        return super(ConvolutionalCapsule, self).get_config()


class ClassCapsule(layers.Layer):
    def __init__(self, classes, routings, **kwargs):
        super(ClassCapsule, self).__init__(**kwargs)
        self.classes = classes
        self.routings = routings
        self.pose_shape = self.batch_size = self.spatial_size = self.pose_size = self.i_size = None
        self.w = self.beta_v = self.beta_a = None

    def build(self, input_shape):
        self.pose_shape = input_shape[0]
        self.spatial_size = int(self.pose_shape[1])
        self.pose_size = int(self.pose_shape[-1])
        self.i_size = int(self.pose_shape[3])

        self.w = self.add_weight(
            shape=[1, self.pose_shape[-3], self.classes, 4, 4],
            name='w',
            trainable=True,
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0))  # (1, 32, 10, 4, 4)

        # print('3. classCaps { w', self.w.shape, '}')

        self.beta_a = self.add_weight(
            name='beta_a',
            shape=[1, self.classes],
            dtype=tf.float32,
            trainable=True,
            initializer=tf.keras.initializers.GlorotUniform())
        self.beta_v = self.add_weight(
            name='beta_v',
            shape=[1, self.classes],
            dtype=tf.float32,
            trainable=True,
            initializer=tf.keras.initializers.GlorotUniform())

        self.built = True

    def call(self, inputs, **kwargs):
        inputs_pose, inputs_activation = inputs
        batch_size = inputs_pose.shape[0]

        # (24 * 4 * 4=384, 32, 16)
        inputs_pose = tf.reshape(inputs_pose, shape=[batch_size * self.spatial_size * self.spatial_size,
                                                     self.pose_shape[-3], self.pose_shape[-2] * self.pose_shape[-2]])

        # print('3.1. classCaps { inputs_pose', inputs_pose.shape, 'inputs_activation', inputs_activation.shape, '}')

        # Block votes
        # inputs_poses (384, 32, 16)
        # votes: (384, 32, 10, 16)
        votes = mat_transform(inputs_pose, self.classes, batch_size * self.spatial_size * self.spatial_size, self.w)

        # print('3.2. classCaps { votes', votes.shape, '}')

        # votes (24, 4, 4, 32, 10, 16)
        votes = tf.reshape(votes, shape=[batch_size, self.spatial_size, self.spatial_size, self.i_size,
                                         self.classes, self.pose_size * self.pose_size])

        # print('3.3. classCaps { votes', votes.shape, '}')

        votes = coord_addition(votes, self.spatial_size, self.spatial_size)  # (24, 4, 4, 32, 10, 16)

        # print('3.4. classCaps { votes', votes.shape, '}')

        # Block routing
        # votes (24, 4, 4, 32, 10, 16) -> (24, 512, 10, 16)
        votes_shape = votes.shape
        votes = tf.reshape(votes, shape=[batch_size, votes_shape[1] * votes_shape[2] * votes_shape[3],
                                         votes_shape[4], votes_shape[5]])

        # print('3.5. classCaps { votes', votes.shape, '}')

        # inputs_activations (24, 4, 4, 32) -> (24, 512)
        inputs_activation = tf.reshape(inputs_activation, shape=[batch_size,
                                                                 votes_shape[1] * votes_shape[2] * votes_shape[3]])

        # print('3.6. classCaps { inputs_activation', inputs_activation.shape, '}')

        # votes (24, 512, 10, 16), inputs_activations (24, 512)
        # poses (24, 10, 16), activation (24, 10)
        pose, activation = matrix_capsules_em_routing(votes, inputs_activation, self.beta_v, self.beta_a, self.routings)

        # print('3.7. classCaps { pose', pose.shape, 'activation', activation.shape, '}')

        # poses (24, 10, 16) -> (24, 10, 4, 4)
        pose = tf.reshape(pose, shape=[batch_size, self.classes, self.pose_size, self.pose_size])

        # print('3.8. classCaps { pose', pose.shape, '}')

        return pose, activation

    def get_config(self):
        return super(ClassCapsule, self).get_config()


if __name__ == '__main__':
    input_layer = layers.Input(shape=[28, 28, 1], batch_size=24)
    conv1 = layers.Conv2D(filters=32, kernel_size=5, strides=2,
                          padding='same', activation=activations.relu)(input_layer)
    primaryCaps = PrimaryCapsule2D(capsules=32, kernel_size=1, strides=1,
                                   padding='valid', pose_shape=[4, 4])(conv1)
    convCaps1 = ConvolutionalCapsule(kernel_size=3, capsules=32,
                                     routings=3, strides=[1, 2, 2, 1])(primaryCaps)
    convCaps2 = ConvolutionalCapsule(kernel_size=3, strides=[1, 1, 1, 1],
                                     capsules=32, routings=3)(convCaps1)
    classCaps = ClassCapsule(classes=10, routings=3)(convCaps2)
    model = tf.keras.Model(input_layer, classCaps)
    model.summary(line_length=200)

    import bmstu.utls as utls

    # load data
    (x_train, y_train), (x_test, y_test) = utls.load('mnist')
    # define model

    x_val = x_test[:1000]
    x_test = x_test[1000:]
    y_val = y_test[:1000]
    y_test = y_test[1000:]

    from tensorflow.keras import optimizers

    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=['mse'],
                  metrics='accuracy')

    model.fit(x_train, y_train, batch_size=24, epochs=5,
              validation_data=[x_val, y_val])
