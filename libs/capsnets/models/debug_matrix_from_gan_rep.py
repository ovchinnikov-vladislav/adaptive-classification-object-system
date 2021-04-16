from tensorflow.keras import layers, initializers
from tensorflow.keras import backend
import numpy as np
import tensorflow as tf


class PrimaryCapsule2D(layers.Layer):
    def __init__(self, num_capsules, pose_shape, kernel_size, strides, padding='valid', **kwargs):
        super(PrimaryCapsule2D, self).__init__(**kwargs)

        self.num_capsules = num_capsules
        self.pose_shape = pose_shape
        num_filters = num_capsules * pose_shape[0] * pose_shape[1]
        self.conv2d_pose = layers.Conv2D(filters=num_filters,
                                         kernel_size=kernel_size,
                                         kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                         strides=strides,
                                         padding=padding)
        self.conv2d_activation = layers.Conv2D(filters=num_capsules,
                                               kernel_size=kernel_size,
                                               strides=strides,
                                               padding=padding)

        # TODO: требуется? self.batch = layers.BatchNormalization(axis=-1)

    def call(self, inputs_tensor, **kwargs):
        # pose = [batch_size, size0, size1, num_capsules * pose_shape[0] * pose_shape[1]
        pose = self.conv2d_pose(inputs_tensor)
        # TODO: см. выше pose = self.batch(pose)

        # pose = [batch_size, size0, size1, num_capsules, pose_shape[0] * pose_shape[1]
        pose = layers.Reshape(target_shape=(pose.shape[1], pose.shape[2], self.num_capsules,
                                            self.pose_shape[0] * self.pose_shape[1]))(pose)

        # activation = [batch_size, size0, size1, num_capsules]
        activation = self.conv2d_activation(inputs_tensor)
        activation = layers.Activation('sigmoid')(activation)
        activation = layers.Reshape(target_shape=[inputs_tensor.shape[1], inputs_tensor.shape[2],
                                                  self.num_capsules, 1])(activation)

        assert pose.shape[1:] == [pose.shape[1], pose.shape[2], self.num_capsules,
                                  self.pose_shape[0] * self.pose_shape[1]]
        assert activation.shape[1:] == [pose.shape[1], pose.shape[2], self.num_capsules, 1]

        return layers.Concatenate()([pose, activation])

    def get_config(self):
        return super(PrimaryCapsule2D, self).get_config()


class ConvolutionalCapsule(layers.Layer):
    def __init__(self, num_capsules, pose_shape, kernel_size=None, strides=None, routings=3, trainable=True,
                 name=None, **kwargs):
        super(ConvolutionalCapsule, self).__init__(trainable=trainable, name=name, **kwargs)

        self.num_capsules = num_capsules
        self.pose_shape = pose_shape
        self.routings = routings
        self.strides = strides

        # Кейс Convolutional Capsules
        if kernel_size:
            self.kernel_size = kernel_size
            self.spatial_dim = [1, 1]
        # Кейс Class Capsules
        else:
            self.kernel_size = [1, 1]
            self.spatial_dim = []

        self.pose_shape_in = None
        self.num_capsules_in = None
        self.spatial_size_in = None
        self.spatial_size = None
        self.beta_v = None
        self.beta_a = None
        self.w = None
        self.batch_size = None

    def build(self, input_shape):
        super(ConvolutionalCapsule, self).build(input_shape)

        self.pose_shape_in = [int(np.sqrt(input_shape[-1] - 1)), int(np.sqrt(input_shape[-1] - 1))]
        self.num_capsules_in = input_shape[-2]
        self.spatial_size_in = [int(input_shape[1]), int(input_shape[2])]
        self.spatial_size = self.spatial_size_in

        # beta_v: [1, (1, 1) 1, num_capsules, 1], trainable parameter (vector of dim: # capsules in layer L+1)
        self.beta_v = self.add_weight(
            name='beta_v',
            shape=[1, ] + self.spatial_dim + [1, self.num_capsules, 1],
            initializer=initializers.GlorotNormal())

        # beta_a: SHAPE=[1, (1, 1,) 1, num_capsules, 1], trainable parameter (vector of dim: # capsules in layer L+1)
        self.beta_a = self.add_weight(
            name='beta_a',
            shape=[1, ] + self.spatial_dim + [1, self.num_capsules, 1],
            initializer=initializers.GlorotNormal())

        # w: [1, 1, 1, kernel_size[0] * kernel_size[1] * num_capsules_in, num_capsules, pose_shape[0], pose_shape[1]]
        self.w = self.add_weight(
            name='w',
            shape=[1, 1, 1, self.kernel_size[0] * self.kernel_size[1] * self.num_capsules_in,
                   self.num_capsules, self.pose_shape[0], self.pose_shape[1]],
            initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))

        self.built = True

    def call(self, inputs_tensor, **kwargs):
        self.batch_size = inputs_tensor.shape[0]

        # Стадия предварительной обработки
        # pose: [batch_size, size[0], size[1], num_capsules_in, pose_shape[0] * pose_shape[1]], pose matrix
        # activation: [batch_size, size[0], size[1], num_capsules_in], activations
        pose = inputs_tensor[:, :, :, :, :16]
        activation = inputs_tensor[:, :, :, :, 16]

        pose = layers.Reshape(target_shape=[self.spatial_size[0], self.spatial_size[1], self.num_capsules_in,
                                            self.pose_shape_in[0], self.pose_shape_in[1]])(pose)

        # Стадия depthwise conv (Convolutional Capsules)
        # pose: [batch_size, size[0]', size[1]', kernel_size[0] * kernel_size[1] * num_capsules_in,
        #        pose_shape[0] * pose_shape[1]], pose matrix
        # activation: [batch_size, size[0]', size[1]', kernel_size[0] * kernel_size[1] * num_capsules_in], activations
        if len(self.spatial_dim):
            pose, activation = self.depthwise_conv(pose, activation)

        # Стадия Spatial Transformation
        # votes_ij: [batch_size, size[0]', size[1]', kernel_size[0] * kernel_size[1] * num_capsules_in,
        #            num_capsules, pose_shape[0], pose_shape[1]], learn the spatial transformations of the features
        votes_ij = self.spatial_transform(pose)

        # Стадия Coordinate Addition (Class Capsules)
        # votes_ij: [batch_size, size[0]', size[1]', num_capsules_in, num_capsules, pose_shape[0], pose_shape[1]],
        #            new vote matrix with values addition along an axis
        if not len(self.spatial_dim):
            votes_ij = self.coord_addition(votes_ij)

        # Стадия EM routing
        # pose: [batch_size, size[0]', size[1]', num_capsules, pose_shape[0], pose_shape[1]],
        #        pose matrix of the new capsules' layer
        # activation: [batch_size, size[0]', size[1]', num_capsules], activations of the new capsules' layer
        pose, activation = self.em_routing(votes_ij, activation)

        # Convolutional Capsules
        if len(self.spatial_dim):
            pose = backend.reshape(pose, [-1, self.spatial_size[0], self.spatial_size[1], self.num_capsules,
                                          self.pose_shape[0] * self.pose_shape[1]])
            activation = backend.expand_dims(activation, -1)
            activation = layers.Activation('sigmoid')(activation)
            output = layers.Concatenate()([pose, activation])
        # Class Capsules
        else:
            output = layers.Activation('sigmoid')(activation)

        return output

    def depthwise_conv(self, pose, activation):
        def kernel_tile(inputs_tensor, kernel, stride):
            # (?, 14, 14, 32x(16)=512)
            input_shape = inputs_tensor.shape
            size = input_shape[4] * input_shape[5] if len(input_shape) > 5 else 1
            inputs_tensor = layers.Reshape(target_shape=[input_shape[1], input_shape[2],
                                                         input_shape[3] * size])(inputs_tensor)
            tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                          kernel * kernel], dtype=np.float32)
            for i in range(kernel):
                for j in range(kernel):
                    tile_filter[i, j, :, i * kernel + j] = 1.0  # (3, 3, 512, 9)
            # (3, 3, 512, 9)
            tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
            # (?, 6, 6, 4608)
            output = tf.nn.depthwise_conv2d(inputs_tensor, tile_filter_op, strides=[
                1, stride, stride, 1], padding='VALID')
            output_shape = output.get_shape()
            output = layers.Reshape(target_shape=[output_shape[1], output_shape[2], input_shape[3],
                                                  kernel * kernel])(output)
            output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
            return output

        # pose: [batch_size, size[0], size[1], num_capsules_in * pose_shape[0] * pose_shape[1]],
        #        prepare the tensor for a depth_conv
        pose = layers.Reshape(target_shape=[self.spatial_size_in[0], self.spatial_size_in[1],
                                            self.num_capsules_in * self.pose_shape_in[0] * self.pose_shape_in[1]])(pose)

        # pose: [batch_size, size[1], size[2], kernel_size[1] * kernel_size[2],
        #        num_capsules_in * pose_shape[1]*pose_shape[2]],
        #        tiled pose matrix to be multiplied by the transformation matrices to generate the votes
        pose = kernel_tile(pose, kernel=self.kernel_size[0], stride=self.strides)

        # spatial_size: [1], new spatial size of the capsule (after the convolution)
        self.spatial_size = [int(pose.shape[1]), int(pose.shape[2])]

        # pose: [batch_size, size[0]', size[1]', kernel_size[0] * kernel_size[1] * num_capsules_in,
        #        pose_shape[0], pose_shape[1]], reshape the pose matrix back to its standard shape
        pose = layers.Reshape(target_shape=[self.spatial_size[0], self.spatial_size[1],
                                            self.kernel_size[0] * self.kernel_size[1] * self.num_capsules_in,
                                            self.pose_shape_in[0], self.pose_shape_in[1]])(pose)

        # activation: [batch_size, size[0]', size[1]', kernel_size[0]*kernel_size[1], num_capsules_in],
        #              tiled activations
        activation = kernel_tile(activation, kernel=self.kernel_size[0], stride=self.strides)
        # activation: [batch_size, size[0]', size[1]', kernel_size[0] * kernel_size[1] * num_capsules_in],
        #              reshape the activation back to its standard shape
        activation = layers.Reshape(target_shape=[self.spatial_size[0], self.spatial_size[1], self.kernel_size[0] *
                                                  self.kernel_size[1] * self.num_capsules_in])(activation)
        return pose, activation

    def spatial_transform(self, pose):
        # pose: [batch_size, size[0]', size[1]', kernel_size[0] * kernel_size[1] * num_capsules_in,
        #        1, pose_shape[0], pose_shape[1]], expand the tensor with a value equal to the number output caps
        pose = backend.expand_dims(pose, -3)

        # pose: [batch_size, size[0]', size[1]', kernel_size[0] * kernel_size[1] * num_capsules_in,
        #        num_capsules, pose_shape[0], pose_shape[1]],
        #        expand the tensor with a value equal to the number output caps
        pose = backend.tile(pose, [1, 1, 1, 1, self.num_capsules, 1, 1])

        # w: [batch_size, size[0]', size[1]', kernel_size[0] * kernel_size[1] * num_capsules_in,
        #     num_capsules, pose_shape[0], pose_shape[1]], tiled transformation matrices, tile to batch_size
        w = backend.tile(self.w, [-1, self.spatial_size[0], self.spatial_size[1], 1, 1, 1, 1])

        # votes: [batch_size, size[0]', size[1]', kernel_size[0] * kernel_size[1] * num_capsules_in,
        #         num_capsules, pose_shape[0], pose_shape[1]], votes matrices
        votes = backend.batch_dot(pose, w)
        return votes

    def coord_addition(self, votes):
        """
        From the paper: "We therefore share the transformation matrices between different positions of the same capsule
        type and add the scaled coordinate (row, column) of the center of the receptive field of each capsule
        to the first two elements of the right-hand column of its vote matrix."
        """
        # votes: [batch_size, size[0]', size[1]', kernel_size[0] * kernel_size[1] * num_capsules_in,
        #         num_capsules, pose_shape[0] * pose_shape[1]], adapt the shape for computation
        votes = backend.reshape(votes, shape=[-1, self.spatial_size[0], self.spatial_size[1],
                                              self.kernel_size[0] * self.kernel_size[1] * self.num_capsules_in,
                                              self.num_capsules, self.pose_shape_in[0] * self.pose_shape_in[1]])

        # h_values: [1, size[0]', 1, 1, 1], variational axis
        h_values = backend.reshape((tf.range(self.spatial_size[0], dtype=tf.float32) + 0.50) / self.spatial_size[0],
                                   [1, self.spatial_size[0], 1, 1, 1])

        # h_zeros: [1, size[0]', 1, 1, 1], non variational axis
        h_zeros = tf.constant(0.0, shape=[1, self.spatial_size[0], 1, 1, 1], dtype=tf.float32)

        # h_offset: [1, size[0]', 1, 1, pose_shape[0] * pose_shape[1]], new coordinates' offset
        h_offset = tf.stack([h_values, h_zeros] +
                            [h_zeros for _ in range(self.pose_shape_in[0] * self.pose_shape_in[1] - 2)],
                            axis=-1)

        # w_values: [1, 1, size[1]', 1, 1, 1], variational axis
        w_values = tf.reshape((tf.range(self.spatial_size[1], dtype=tf.float32) + 0.50) / self.spatial_size[1],
                              [1, 1, self.spatial_size[1], 1, 1])

        # w_zeros: SHAPE=[1, 1, size[1]', 1, 1], non variational axis
        w_zeros = tf.constant(0.0, shape=[1, 1, self.spatial_size[1], 1, 1], dtype=tf.float32)

        # w_offset: [1, 1, size[1]', 1, pose_shape[0] * pose_shape[1]], new coordinates' offset
        w_offset = tf.stack([w_zeros, w_values] +
                            [w_zeros for _ in range(self.pose_shape_in[0] * self.pose_shape_in[1] - 2)],
                            axis=-1)

        # votes: [batch_size, size[0]', size[1]', num_capsules_in, num_capsules, pose_shape[0] * pose_shape[1]],
        #         votes in the new coordinates
        votes = votes + h_offset + w_offset

        # votes: [batch_size, size[0]', size[1]', num_capsules_in, num_capsules, pose_shape[0], pose_shape[1]],
        #         reshape back to the standard norm
        votes = backend.reshape(votes, shape=[-1, self.spatial_size[0], self.spatial_size[1],
                                              self.kernel_size[0] * self.kernel_size[1] * self.num_capsules_in,
                                              self.num_capsules, self.pose_shape_in[0], self.pose_shape_in[1]])
        return votes

    def em_routing(self, V_ij, a_i):
        def maximization(R_ij, V_ij, a_i, inv_temp):
            # R_ij: SHAPE=[b, s0', s1', k0*k1*I, O, 1] or [b, s0'*s1'*I, O, 1], TYPE= tensor, VALUE=weights assignment according to the activation probabilities CAUTION!!!! maybe reshape it into backend, backend, A, B, 1 .... before multiplication
            R_ij = R_ij * a_i
            # R_ij: SHAPE=[b, (s0', s1',) 1, O, 1] , TYPE= tensor, VALUE=sum over all input capsules i
            R_ij_sum = backend.sum(R_ij, axis=-3, keepdims=True)
            # M_j: SAHPE=[b, (s0', s1',) 1, O, p0*p1], TYPE= tensor, VALUE= mean of capsule j
            M_j = backend.sum(R_ij * V_ij, axis=-3, keepdims=True ) / R_ij_sum
            # stdv_j: SAHPE=[b, (s0', s1',) 1, O, p0*p1], TYPE= tensor, VALUE= standard deviation of capsule j
            stdv_j = backend.sqrt(backend.sum(R_ij_sum * tf.square(V_ij - M_j), axis=-3, keepdims=True) / R_ij_sum)
            # cost_j_h: SHAPE=[b, (s0', s1',) 1, O, p0*p1], TYPE= tensor, VALUE= expected energy of a capsule j
            cost_j_h = (self.beta_v + backend.log(stdv_j + backend.epsilon())) * R_ij_sum
            # cost_j: SHAPE=[b, (s0', s1',) 1, O, 1], TYPE= tensor, VALUE= expected energy
            cost_j = backend.sum(cost_j_h, axis=-1, keepdims=True)
            # cost_j_mean: SHAPE=[b, (s0', s1',) 1, 1, 1], TYPE= tensor, VALUE= mean the expected energy over the output capsules
            cost_j_mean = backend.mean(cost_j, axis=-2, keepdims=True)
            # cost_j_stdv: SHAPE=[b, (s0', s1',) 1, 1, 1], TYPE= tensor, VALUE= mean the expected energy
            cost_j_stdv = backend.sqrt(backend.sum(backend.square(cost_j - cost_j_mean), axis=-2, keepdims=True) / self.num_capsules)
            # a_j_cost: SHAPE=[b, (s0', s1',) 1, O, 1], TYPE= tensor, VALUE= cost of the activation of capsule j
            a_j_cost = self.beta_a + (cost_j_mean - cost_j) / (cost_j_stdv + backend.epsilon())
            # a_j: SHAPE=[b, (s0', s1',) 1, O, 1], TYPE= tensor, VALUE= activation of capsule j
            a_j = tf.sigmoid(inv_temp * a_j_cost)
            # a_j: SHAPE=[b, (s0', s1',) O], TYPE= tensor, VALUE= squeezed activation of capsule j
            a_j = backend.squeeze(backend.squeeze(a_j, axis=-3), axis=-1)
            # M_j: SAHPE=[b, (s0', s1',) O, p0*p1], TYPE= tensor, VALUE=squeezed mean of capsule j (pose matrix)
            M_j = backend.squeeze(M_j, axis=-3)
            # stdv_j: SAHPE=[b, (s0', s1',) O, p0*p1], TYPE= tensor, VALUE=squeezed standard deviation of capsule j
            stdv_j = backend.squeeze(stdv_j, axis=-3)
            return M_j, stdv_j, a_j
        def estimation(M_j, stdv_j, V_ij, a_j):
            # M_j: SAHPE=[b, (s0', s1',) 1, O, p0*p1], TYPE= tensor, VALUE=squeezed mean of capsule j (pose matrix)
            M_j = backend.expand_dims(M_j, -3)
            # a_j: SAHPE=[b, (s0', s1',) 1, O, 1], TYPE= tensor, VALUE=squeezed mean of capsule j (pose matrix)
            a_j = backend.expand_dims(backend.expand_dims(a_j, -2), -1)
            # stdv_j: SAHPE=[b, (s0', s1',) 1, O, p0*p1], TYPE= tensor, VALUE=squeezed mean of capsule j (pose matrix)
            stdv_j = backend.expand_dims(stdv_j, -3)
            # R_ij: SHAPE= [b, s0', s1', k0*k1*I, O, 1] or [b, s0'*s1'*I, O, 1], TYPE= tensor, VALUE= routing matrix
            a_j_p_j  = backend.log(a_j + backend.epsilon()) - backend.sum(backend.square(V_ij - M_j) /(2 * tf.square(stdv_j)), axis=-1, keepdims=True) - backend.sum(tf.math.log(stdv_j + backend.epsilon()), axis=-1, keepdims=True)
            # R_ij: SHAPE= [b, s0', s1', k0*k1*I, O, 1], TYPE= tensor, VALUE= activated routing matrix
            R_ij = tf.nn.softmax(a_j_p_j, axis=len(a_j_p_j.get_shape().as_list())-2)
            return R_ij
        if len(self.spatial_dim):
            # V_ij: SHAPE=[b, s0', s1', k0*k1*I, O, p0*p1], TYPE= tensor, VALUE= adapt the shape for computation
            V_ij = backend.reshape(V_ij, shape=[-1, self.spatial_size[0], self.spatial_size[1], self.kernel_size[0]*self.kernel_size[1]*self.num_capsules_in, self.num_capsules, self.pose_shape[0]*self.pose_shape[1]])
            # ai: SHAPE=[b, s0', s1', k0*k1*I, 1, 1], TYPE= tensor, VALUE= expanded i activations
            a_i = backend.expand_dims(backend.expand_dims(a_i,-1),-1)
            # R_ij: SHAPE=[k0*k1*I, O, 1], TYPE= tensor, VALUE= routing assignment matrix from each input capsule (i) in L to each output capsule (j) in L+1 initilized with uniform distribution
            R_ij = backend.constant(1.0/self.num_capsules, shape=(self.kernel_size[0]*self.kernel_size[0]*self.num_capsules_in, self.num_capsules, 1))
        else:
            # V_ij: SHAPE=[b, s0'*s1'*I, O, p0*p1], TYPE= tensor, VALUE= adapt the shape for computation
            V_ij = backend.reshape(V_ij, shape=[-1, self.spatial_size[0]*self.spatial_size[1]*self.num_capsules_in, self.num_capsules, self.pose_shape[0]*self.pose_shape[1]])
            # a_i: SHAPE=[b, s0'*s1'*I, 1, 1], TYPE= tensor, VALUE= reshape to standard form
            a_i = backend.reshape(a_i, shape=[-1, self.spatial_size[0]*self.spatial_size[1]*self.num_capsules_in, 1, 1])
            # R_ij: SHAPE=[s0'*s1'*I, O, 1], TYPE= tensor, VALUE= routing assignment matrix from each input capsule (i) in L to each output capsule (j) in L+1 initilized with uniform distribution
            R_ij = backend.constant(1.0/self.num_capsules, shape=(self.spatial_size[0]*self.spatial_size[0]*self.num_capsules_in, self.num_capsules, 1))
        for iter in range(self.routings):
            # inv_temp: SHAPE=[1], TYPE= int, VALUE= Lambda: inverse temperature schedule (1, min(routings, 3.0)-1)
            inv_temp = 1.0 + (min(self.routings, 3.0) - 1.0) * iter / max(1.0, self.routings - 1.0)
            # M_j: SAHPE=[b, (s0', s1',) O, p0*p1], TYPE= tensor, VALUE= mean of capsule j
            # stdv_j: SAHPE=[b, (s0', s1',) O, p0*p1], TYPE= tensor, VALUE= standard deviation of capsule j
            # a_j: SHAPE= [b, (s0', s1',) O], TYPE= tensor, VALUE= activation of capsule j
            M_j, stdv_j, a_j = maximization(R_ij, V_ij, a_i, inv_temp=inv_temp)
            # R_ij: SHAPE= [b, (s0', s1',) k0*k1*I, O, 1], TYPE= tensor, VALUE= activated routing matrix
            if iter < self.routings - 1:
                R_ij = estimation(M_j, stdv_j, V_ij, a_j)
        # M_j: SHAPE=[b, (s0', s1',) O, p0, p1], TYPE= tensor, VALUE= reshape back to the standard norm
        M_j = backend.reshape(M_j, shape=[-1, self.spatial_size[0], self.spatial_size[1], self.num_capsules, self.pose_shape[0], self.pose_shape[1]])
        return M_j, a_j

    def compute_output_shape(self, input_shape):
        if len(self.spatial_dim):
            # pose: [batch_size, (size[0]', size[1]',) num_capsule, pose_shape[0], pose_shape[1]],
            #        pose matrix of the new capsules' layer (reshaped back to p0xp1 pose matrix)
            output_sh = [input_shape[0], self.spatial_size[0], self.spatial_size[1], self.num_capsules,
                         self.pose_shape[0] * self.pose_shape[1] + 1]
        else:
            output_sh = [input_shape[0], self.num_capsules]
        return tuple(output_sh)


if __name__ == '__main__':
    from libs import utls

    (x_train, y_train), (x_test, y_test) = utls.load('mnist')

    # inputs = img_shape : Input
    inputs = layers.Input(shape=(28, 28, 1))

    # net = [batch_size, s0, s1, A] : ReLU Conv1
    net = layers.Conv2D(filters=64, kernel_size=[5, 5], strides=2, padding='same', activation='relu',
                        name='ReLU_Conv1')(inputs)  # add batch normalization ??
    # net = [ poses = [?, s0, s1, B, p*p], activations = [?, s0, s1, B] ] : PrimaryCaps
    net = PrimaryCapsule2D(pose_shape=[4, 4], num_capsules=8, kernel_size=[1, 1], strides=1,
                           padding='valid', name='PrimaryCaps')(net)
    # nets = [ poses = [?, s0', s1', C, p*p], activations = [?, s0', s1', C] ] : ConvCaps1
    net = ConvolutionalCapsule(num_capsules=8, pose_shape=[4, 4], kernel_size=[3, 3], strides=2, routings=3,
                               name='ConvCaps1')(net)
    # nets = [ poses (?, s0'', s1'', D, p*p), activations = [?, s0'', s1'', D] ] : ConvCaps2
    net = ConvolutionalCapsule(num_capsules=8, pose_shape=[4, 4], kernel_size=[3, 3], strides=1, routings=3,
                               name='ConvCaps2')(net)
    # output  = [ poses = [?, E, p*p], activations = [?, E] ] : Class Capsules
    net = ConvolutionalCapsule(num_capsules=10, pose_shape=[4, 4], routings=3,
                               name='Class_Capsules')(net)

    model = tf.keras.Model(inputs, net)
    model.summary()
