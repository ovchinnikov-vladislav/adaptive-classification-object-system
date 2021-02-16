import keras.backend as K
from keras import initializers, layers, regularizers
import tensorflow as tf2
import numpy as np

from keras import layers, models, optimizers, regularizers, constraints
import time

tf = tf2.compat.v1

class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):

    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of vectors of capsules
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])


def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return tf.multiply(scale, vectors)


class Conv_Capsule(layers.Layer):

    def __init__(self, kernel_shape, strides, dim_vector,
                 num_routing=3, batchsize=50, name='conv_caps',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer='l2',
                 **kwargs):
        super(Conv_Capsule, self).__init__(**kwargs)
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.batchsize = batchsize
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 5, "The input Tensor should have shape=[None, height, width, input_num_capsule, input_dim_vector]"
        self.caps_num_i = self.kernel_shape[0] * self.kernel_shape[1] * self.kernel_shape[2]
        self.output_cap_channel = self.kernel_shape[3]
        self.input_dim_vector = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(
            shape=[1, self.caps_num_i, self.output_cap_channel, self.input_dim_vector, self.dim_vector],
            initializer=tf.truncated_normal_initializer(mean=0., stddev=1.),
            # regularizer=regularizers.l2(5e-4),
            name=self.name+'W')

        # Coupling coefficient.
        self.bias = self.add_weight(shape=[1, 1, 1, self.caps_num_i, self.output_cap_channel, 1, 1],
                                    initializer=self.bias_initializer,
                                    name=self.name+'bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        inputs_poses = inputs

        i_pose_dim = inputs_poses.shape[-1]

        inputs_poses = self.kernel_tile(inputs_poses)

        spatial_x = inputs_poses.shape[1]
        spatial_y = inputs_poses.shape[2]

        inputs_poses = K.reshape(inputs_poses, shape=[-1, self.kernel_shape[0] * self.kernel_shape[1] * self.kernel_shape[2],
                                                      i_pose_dim])

        votes = self.mat_transform(inputs_poses, size=self.batchsize * spatial_x * spatial_y)

        votes_shape = votes.shape
        votes = K.reshape(votes, shape=[-1, spatial_x, spatial_y, votes_shape[-3],
                                        votes_shape[-2], votes_shape[-1]])

        poses = self.capsules_dynamic_routing(votes)

        return poses

    def compute_output_shape(self, input_shape):
        h = (input_shape[1] - self.kernel_shape[0]) // self.strides[1] + 1
        w = (input_shape[2] - self.kernel_shape[1]) // self.strides[2] + 1
        return tuple([self.batchsize, h, w, self.kernel_shape[-1], self.dim_vector])

    def kernel_tile(self, input):

        input_shape = input.shape
        input = tf.reshape(input, shape=[-1, input_shape[1], input_shape[2],
                                         input_shape[3] * input_shape[4]])

        input_shape = input.shape
        tile_filter = np.zeros(shape=[self.kernel_shape[0], self.kernel_shape[1], input_shape[3],
                                      self.kernel_shape[0] * self.kernel_shape[1]], dtype=np.float32)
        for i in range(self.kernel_shape[0]):
            for j in range(self.kernel_shape[1]):
                tile_filter[i, j, :, i * self.kernel_shape[0] + j] = 1.0

        tile_filter_op = K.constant(tile_filter, dtype=tf.float32)

        output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=self.strides, padding='VALID')

        output_shape = output.shape
        output = K.reshape(output, shape=[-1, output_shape[1], output_shape[2], input_shape[3],
                                          self.kernel_shape[0] * self.kernel_shape[1]])
        output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

        return output

    def mat_transform(self, input, size):

        shape = input.shape
        caps_num_i = shape[1]  # 72
        output = K.reshape(input, shape=[-1, caps_num_i, 1, 1, shape[2]])

        self.W = K.tile(self.W, [size, 1, 1, 1, 1])

        output = K.tile(output, [1, 1, self.kernel_shape[-1], 1, 1])

        votes = tf.matmul(output, self.W)

        votes = K.reshape(votes, [size, caps_num_i, self.kernel_shape[-1], self.dim_vector])

        return votes

    def capsules_dynamic_routing(self, votes):

        shape = votes.shape

        votes = K.expand_dims(votes, axis=5)

        votes_stopped = K.stop_gradient(votes)
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=4)  # dim=4 is the num_capsule dimension

            if i == self.num_routing - 1:
                outputs = squash(tf.reduce_sum(c * votes, 3, keepdims=True))

            else:
                outputs = squash(tf.reduce_sum(c * votes_stopped, 3, keepdims=True))
                self.bias.assign_add(tf.reduce_sum(votes * outputs, -1, keepdims=True))

        return K.reshape(outputs, [-1, shape[1], shape[2], shape[4], shape[5]])


class Class_Capsule(layers.Layer):

    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer='l2',
                 bias_regularizer='l2',
                 **kwargs):
        super(Class_Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 5, "The input Tensor should have shape=[None, height, width, input_num_capsule, input_dim_vector]"
        self.input_num_capsule = input_shape[1] * input_shape[2] * input_shape[3]
        self.input_dim_vector = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
            initializer=self.kernel_initializer,
            name='W')

        # Coupling coefficient.
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):

        shape = inputs.shape
        inputs = K.reshape(inputs, [-1, shape[1] * shape[2] * shape[3], shape[4]])
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))

        inputs_hat_stopped = K.stop_gradient(inputs_hat)

        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=2)

            if i == self.num_routing - 1:
                outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            else:
                outputs = squash(K.sum(c * inputs_hat_stopped, 1, keepdims=True))
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])


def PrimaryCaps(inputs, dim_vector, n_channels, kernel_size, strides, padding):

    output = layers.Conv2D(filters=dim_vector * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    output = layers.BatchNormalization(momentum=0.9, name='primarycap_bn')(output)
    output = layers.Activation('relu', name='primarycap_relu')(output)
    shape = np.shape(output)

    outputs = layers.Reshape(target_shape=[shape[1], shape[2], n_channels, dim_vector], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

    # # test
    # start = time.time()
    # model.load_weights('./result/weights-test.h5')
    # i = 0
    # test_nsamples = 0
    # matrix = np.zeros([args.n_class, args.n_class], dtype=np.float32)
    # while 1:
    #     data = readdata(image_file, label_file, train_nsamples=200, validation_nsamples=100,
    #                     windowsize=args.windowsize, istraining=False, shuffle_number=test_shuffle_number, times=i)
    #     if data == None:
    #         OA, AA_mean, Kappa, AA = cal_results(matrix)
    #         print('-' * 50)
    #         print('OA:', OA)
    #         print('AA:', AA_mean)
    #         print('Kappa:', Kappa)
    #         print('Classwise_acc:', AA)
    #         end = time.time()
    #         print('test time:', end - start)
    #         break
    #     test_nsamples += data[0].shape[0]
    #     matrix = matrix + test(model=model, data=(data[0], data[1]))
    #     i = i + 1


def CapsNet(input_shape, n_class, num_routing):

    x = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', name='conv1')(x)
    conv1 = layers.BatchNormalization(momentum=0.9, name='bn1')(conv1)
    conv1 = layers.Activation('relu', name='conv1_relu')(conv1)

    conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', name='conv2')(conv1)
    conv2 = layers.BatchNormalization(momentum=0.9, name='bn2')(conv2)
    conv2 = layers.Activation('relu', name='conv2_relu')(conv2)

    primarycaps = PrimaryCaps(conv2, dim_vector=8, n_channels=4, kernel_size=4, strides=2, padding='valid')

    Conv_caps1 = Conv_Capsule(kernel_shape=[3, 3, 4, 8], dim_vector=8, strides=[1, 2, 2, 1],
                              num_routing=num_routing, batchsize=100, name='Conv_caps1')(primarycaps)

    digitcaps = Class_Capsule(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(Conv_caps1)

    out_caps = Length(name='out_caps')(digitcaps)

    return models.Model(x, out_caps)


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


if __name__ == "__main__":
    import numpy as np
    import os
    from keras import callbacks
    from keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    # file path of HSI dataset
    image_file = r'E:\KSC.mat'
    label_file = r'E:\KSC_gt.mat'

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_valid = x_valid.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_valid = to_categorical(y_valid.astype('float32'))

    # define model
    model = CapsNet(input_shape=x_train.shape[1:],
                    n_class=len(np.unique(np.argmax(y_train, 1))),
                    num_routing=3)
    model.summary()

    # training
    start = time.time()
    tb = callbacks.TensorBoard(log_dir='./tensorboard-logs',
                               batch_size=100)
    checkpoint = callbacks.ModelCheckpoint('./weights-test.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss],
                  metrics={'out_caps': 'accuracy'})

    model.fit(x_train, y_train, batch_size=100, epochs=20,
              validation_data=[x_valid, y_valid], callbacks=[tb, checkpoint], verbose=2)
    end = time.time()
    print('train time:', end - start)
