import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, activations
from bmstu.capsnet.em_utils import kernel_tile, mat_transform, em_routing


class PrimaryCapsule2D(layers.Layer):
    def __init__(self, capsules, kernel_size, strides, padding, pose_shape,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(PrimaryCapsule2D, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.capsules = capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.pose_shape = pose_shape
        self.conv2d_pose = layers.Conv2D(filters=capsules * pose_shape[0] * pose_shape[1],
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         padding=padding,
                                         activation=None)
        self.conv2d_activation = layers.Conv2D(filters=capsules,
                                               kernel_size=kernel_size,
                                               strides=strides,
                                               padding=padding,
                                               activation=activations.sigmoid)

    def build(self, input_shape):
        super(PrimaryCapsule2D, self).build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        # inputs (50, 12, 12, 32), capsules 8, kernel_size [1, 1], strides 1, padding 'valid'

        # (50, 12, 12, 8x16=128)
        pose = self.conv2d_pose(inputs)

        # (50, 12, 12, 8)
        activation = self.conv2d_activation(inputs)

        # (50, 12, 12, 8, 16)
        pose = tf.reshape(pose, shape=[-1, inputs.shape[1], inputs.shape[2],
                                       self.capsules, self.pose_shape[0] * self.pose_shape[1]])

        # (50, 12, 12, 8, 1)
        activation = tf.reshape(activation, shape=[-1, inputs.shape[1], inputs.shape[2], self.capsules, 1])

        # (50, 12, 12, 8, 17)
        output = tf.concat([pose, activation], axis=4)

        # (50, 12, 12, 136)
        output = tf.reshape(output, shape=[-1, inputs.shape[1], inputs.shape[2], self.capsules * 17])

        assert output.shape[1:] == [inputs.shape[1], inputs.shape[2], self.capsules * 17]
        tf.get_logger().info(f'primary capsule output shape: {output.shape}')

        return output

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2], self.capsules * 17]

    def get_config(self):
        return super(PrimaryCapsule2D, self).get_config()


class ConvolutionalCapsule(layers.Layer):
    def __init__(self, capsules, kernel, stride, routings,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(ConvolutionalCapsule, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.capsules = capsules
        self.kernel = kernel
        self.stride = stride
        self.routings = routings
        self.w = None
        self.beta_v = None
        self.beta_a = None

    def build(self, input_shape):
        super(ConvolutionalCapsule, self).build(input_shape)

        # (1, 72, 16, 4, 4)
        self.w = self.add_weight(name='w', shape=[1, 3 * 3 * (input_shape[3] // 17), self.capsules, 4, 4],
                                 initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0),
                                 regularizer=tf.keras.regularizers.L2(5e-04),
                                 trainable=True)

        # (16, 16)
        self.beta_v = self.add_weight(
            name='beta_v', shape=[self.capsules, 16], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),  # tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
            trainable=True, regularizer=tf.keras.regularizers.L2(5e-04))

        # (16,)
        self.beta_a = self.add_weight(
            name='beta_a', shape=[self.capsules], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),  # tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
            trainable=True, regularizer=tf.keras.regularizers.L2(5e-04))

        self.built = True

    def call(self, inputs, **kwargs):
        # inputs (50, 12, 12, 136), capsules 16, kernel 3, stride 2, routings 3

        # (50, 12, 12, 8x(16+1)=136) -> (50, 5, 5, 3x3=9, 136)
        output = kernel_tile(inputs, self.kernel, self.stride)
        data_size = output.shape[1]

        # (1250, 9x8=72, 17)
        output = tf.reshape(output, shape=[-1, 3 * 3 * (inputs.shape[3] // 17), 17])

        # (1250, 72, 1)
        activation = tf.reshape(output[:, :, 16], shape=[-1, 3 * 3 * (inputs.shape[3] // 17), 1])

        # block votes
        # (1250, 72, 16, 16)
        votes = mat_transform(output[:, :, :16], self.capsules, self.w)

        # block routing
        # votes (1250, 3x3x8=72, 16, 4x4), activation (1250, 72, 1), capsule 16
        # miu (1250, 1, 16, 16), activation (1250, 16)
        miu, activation, _ = em_routing(votes, activation, self.capsules, self.beta_v, self.beta_a, self.routings)

        # (50, 5, 5, 16, 16)
        pose = tf.reshape(miu, shape=[-1, data_size, data_size, self.capsules, 16])

        # (50, 5, 5, 16, 1)
        activation = tf.reshape(activation, shape=[-1, data_size, data_size, self.capsules, 1])

        # (50, 5, 5, 272)
        output = tf.reshape(tf.concat([pose, activation], axis=4), [-1, data_size, data_size, self.capsules * 17])

        assert output.shape[1:] == [data_size, data_size, self.capsules * 17]
        tf.get_logger().info(f'conv capsule output shape: {output.shape}')

        return output

    def get_config(self):
        return super(ConvolutionalCapsule, self).get_config()


class ClassCapsule(layers.Layer):
    def __init__(self, capsules, routings, coord_add,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(ClassCapsule, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.capsules = capsules
        self.routings = routings
        self.coord_add = coord_add
        self.w = None
        self.beta_v = None
        self.beta_a = None

    def build(self, input_shape):
        super(ClassCapsule, self).build(input_shape)

        # (9, 1, 1, 2)
        self.coord_add = np.reshape(self.coord_add, newshape=[input_shape[1] * input_shape[1], 1, 1, 2])

        # (450, 16, 10, 2)
        self.coord_add = np.tile(self.coord_add, [input_shape[0], input_shape[3] // 17, self.capsules, 1])

        # (1, 16, 16, 4, 4)
        self.w = self.add_weight(name='w', shape=[1, input_shape[3] // 17, self.capsules, 4, 4],
                                 initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0),
                                 regularizer=tf.keras.regularizers.L2(5e-04),
                                 trainable=True)

        # (10, 18)
        self.beta_v = self.add_weight(
            name='beta_v', shape=[self.capsules, 18], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),  # tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
            trainable=True, regularizer=tf.keras.regularizers.L2(5e-04))

        # (10,)
        self.beta_a = self.add_weight(
            name='beta_a', shape=[self.capsules], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),  # tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
            trainable=True, regularizer=tf.keras.regularizers.L2(5e-04))

        self.built = True

    def call(self, inputs, **kwargs):
        capsules_in = inputs.shape[3] // 17
        # (50, 3, 3, 16, 17)
        inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1], inputs.shape[2], capsules_in, 17])
        data_size = inputs.shape[1]

        # (450, 16, 16)
        pose = inputs[:, :, :, :, :16]
        pose = tf.reshape(pose, shape=[-1, capsules_in, 16])

        # (450, 16, 1)
        activation = inputs[:, :, :, :, 16]
        activation = tf.reshape(activation, shape=[-1, capsules_in, 1])

        # block votes
        # (450, 16, 10, 16)
        votes = mat_transform(pose, self.capsules, self.w)

        assert votes.shape[1:] == [capsules_in, self.capsules, 16]

        coord_add_op = tf.constant(self.coord_add, dtype=tf.float32)

        # (450, 16, 10, 18)
        votes = tf.concat([coord_add_op, votes], axis=3)

        # block routing
        # (450, 1, 10, 18), (450, 10)
        miu, activation, test2 = em_routing(votes, activation, self.capsules, self.beta_v, self.beta_a, self.routings)

        # (50, 3, 3, 10)
        outputs = tf.reshape(activation, shape=[-1, data_size, data_size, self.capsules])

        # (50, 10)
        outputs = tf.reshape(tf.nn.avg_pool(outputs, ksize=[1, data_size, data_size, 1],
                                            strides=[1, 1, 1, 1], padding='VALID'), shape=[-1, self.capsules])

        # (50, 1, 1, 180)
        pose = tf.nn.avg_pool(tf.reshape(miu, shape=[-1, data_size, data_size, self.capsules * 18]),
                              ksize=[1, data_size, data_size, 1], strides=[1, 1, 1, 1], padding='VALID')

        # (50, 10, 18)
        pose_out = tf.reshape(pose, shape=[-1, self.capsules, 18])

        return outputs, pose_out

    def get_config(self):
        return super(ClassCapsule, self).get_config()


if __name__ == '__main__':
    coord_add = [[[8., 8.], [12., 8.], [16., 8.]],
                 [[8., 12.], [12., 12.], [16., 12.]],
                 [[8., 16.], [12., 16.], [16., 16.]]]

    coord_add = np.array(coord_add, dtype=np.float32) / 28.

    input_layer = layers.Input([28, 28, 1], batch_size=50)  # (50, 28, 28, 1)
    conv1 = layers.Conv2D(filters=32, kernel_size=[5, 5], strides=2, padding='valid',
                          activation=activations.relu)(input_layer)  # (50, 12, 12, 32)
    primaryCaps = PrimaryCapsule2D(capsules=8, kernel_size=[1, 1], strides=1,
                                   padding='valid', pose_shape=[4, 4])(conv1)
    convCaps1 = ConvolutionalCapsule(capsules=16, kernel=3, stride=2, routings=3)(primaryCaps)
    convCaps2 = ConvolutionalCapsule(capsules=16, kernel=3, stride=1, routings=3)(convCaps1)
    classCaps = ClassCapsule(capsules=10, routings=3, coord_add=coord_add)(convCaps2)
    model = tf.keras.Model(input_layer, classCaps)
    model.summary()
