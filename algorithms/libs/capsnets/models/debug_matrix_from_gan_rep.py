from tensorflow.keras import layers, initializers
from tensorflow.keras import backend
import numpy as np
import tensorflow as tf
from libs import utls

tfv1 = tf.compat.v1


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
        self.pose_shape_in = [int(np.sqrt(input_shape[-1] - 1)), int(np.sqrt(input_shape[-1] - 1))]
        self.num_capsules_in = input_shape[-2]
        self.spatial_size_in = [int(input_shape[1]), int(input_shape[2])]
        self.spatial_size = self.spatial_size_in

        # beta_v: SHAPE=[1, (1, 1,) 1, O, 1], TYPE=tensor, VALUE= trainable parameter (vector of dim: # capsules in layer L+1)
        self.beta_v = self.add_weight(shape=[1, ] + self.spatial_dim + [1, self.num_capsules, 1],
                                      initializer=initializers.glorot_normal(),
                                      name='beta_v')
        # beta_a: SHAPE=[1, (1, 1,) 1, O, 1], TYPE=tensor, VALUE= trainable parameter (vector of dim: # capsules in layer L+1)b, s, s, k, k, 1, B, p, p
        self.beta_a = self.add_weight(shape=[1, ] + self.spatial_dim + [1, self.num_capsules, 1],
                                      initializer=initializers.glorot_normal(),
                                      name='beta_a')
        # W_ij: SHAPE=[1, 1, 1, (k0*k1*)I, O, p0, p1]
        self.W_ij = self.add_weight(
            shape=[1, 1, 1, self.kernel_size[0] * self.kernel_size[1] * self.num_capsules_in, self.num_capsules,
                   self.pose_shape[0], self.pose_shape[1]],
            initializer=initializers.RandomNormal(mean=0.0, stddev=0.05),
            name='W_ij')  # vll. hier die 1 durch size_batch ersetzen
        # run build method with __init__
        self.built = True
        super(ConvolutionalCapsule, self).build(input_shape)

    def call(self, inputs, **kwargs):
        self.batch_size = backend.shape(inputs)[0]
        ################ inputs ################
        # M_i:SHAPE=[b, s0, s1, I, p0*p1], TYPE= tensor, VALUE= pose matrix
        # a_i:SHAPE=[b, s0, s1, I], TYPE= tensor, VALUE= activations
        M_i = inputs[:, :, :, :, :16]
        a_i = inputs[:, :, :, :, 16]

        M_i = backend.reshape(M_i,
                        shape=[-1, self.spatial_size[0], self.spatial_size[1], self.num_capsules_in, self.pose_shape_in[0],
                               self.pose_shape_in[1]])
        ################ depthwise conv ################
        # M_i:SHAPE=[b, s0', s1', (k0*k1*)I, p0*p1], TYPE= tensor, VALUE= pose matrix
        # a_i:SHAPE=[b, s0', s1', (k0*k1*)I], TYPE= tensor, VALUE= activations
        if len(self.spatial_dim): M_i, a_i = self.depthwise_conv(M_i, a_i)
        ################ spatial tansformation ################
        # V_ij: SHAPE=[b, s0', s1', (k0*k1*)I, O, p0, p1], TYPE= tensor, VALUE= learn the spatial transformations of the features
        V_ij = self.spatial_transform(M_i)
        ################ coordinate addition ################
        # V_ij: SHAPE= [b, s0', s1', I, O, p0, p1], TYPE= tensor, VALUE= new vote matrix with values addition along an axis
        if not len(self.spatial_dim): V_ij = self.coord_addition(V_ij)
        ################ EM routing ################
        # M_j: SHAPE=[b, s0', s1', O, p0, p0], TYPE= tensor, VALUE= pose matrix of the new capsules' layer
        # a_j: SHAPE=[b, s0', s1', O], TYPE= tensor, VALUE= activations of the new capsules' layer
        M_j, a_j = self.em_routing(V_ij, a_i)

        if len(self.spatial_dim):
            M_j = backend.reshape(M_j, [-1, self.spatial_size[0], self.spatial_size[1], self.num_capsules,
                                  self.pose_shape[0] * self.pose_shape[1]])
            a_j = backend.expand_dims(a_j, -1)
            a_j = layers.Activation('sigmoid')(a_j)
            net = layers.Concatenate()([M_j, a_j])
        else:
            net = layers.Activation('sigmoid')(a_j)
        return net

    def depthwise_conv(self, M_i, a_i):
        def depthwise_operation (input, kernel, stride):
            # (?, 14, 14, 32x(16)=512)
            input_shape = input.get_shape()
            size = input_shape[4]*input_shape[5] if len(input_shape)>5 else 1
            input = tf.reshape(input, shape=[-1, input_shape[1], input_shape[2], input_shape[3]*size])
            tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                          kernel * kernel], dtype=np.float32)
            for i in range(kernel):
                for j in range(kernel):
                    tile_filter[i, j, :, i * kernel + j] = 1.0 # (3, 3, 512, 9)
            # (3, 3, 512, 9)
            tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
            # (?, 6, 6, 4608)
            output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[
                                            1, stride, stride, 1], padding='VALID')
            output_shape = output.get_shape()
            output = tf.reshape(output, shape=[-1, output_shape[1], output_shape[2], input_shape[3], kernel * kernel])
            output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
            return output
        # M_i: SHAPE=[b, s1, s2, I*p1*p2], TYPE= tensor, VALUE= prepare the tensor for a depthconv
        M_i= backend.reshape(M_i, shape=[self.batch_size, self.spatial_size_in[0], self.spatial_size_in[1], self.num_capsules_in*self.pose_shape_in[0]*self.pose_shape_in[1]])
        # M_i: SHAPE=[b, s1, s2, k1*k2, I*p1*p2], TYPE= tensor, VALUE= tiled pose matrix to be mutiplied by the transformation matrices to generate the votes
        M_i = depthwise_operation(M_i, kernel=self.kernel_size[0], stride=self.strides)
        # spatial_size: SHAPE=[1], TYPE= int, VALUE= new spatial size of the capsule (after the convolution)
        self.spatial_size = [int(M_i.shape[1]), int(M_i.shape[2])]
        # M_i: SHAPE=[b, s0', s1', k1*k2*I, p1, p2], TYPE= tensor, VALUE= reshape the pose matrix back to its standard shape
        M_i= backend.reshape(M_i, shape=[self.batch_size, self.spatial_size[0], self.spatial_size[1], self.kernel_size[0]*self.kernel_size[1]*self.num_capsules_in, self.pose_shape_in[0],self.pose_shape_in[1]])
        # a_i: SHAPE=[b, s1', s2', k1*k2, I], TYPE= tensor, VALUE= tiled activations
        a_i = depthwise_operation(a_i, kernel=self.kernel_size[0], stride=self.strides)
        # a_i: SHAPE=[b, s1', s2', k1*k2*I], TYPE= tensor, VALUE= reshape the activation back to its standard shape
        a_i= backend.reshape(a_i, shape=[self.batch_size, self.spatial_size[0], self.spatial_size[1], self.kernel_size[0]*self.kernel_size[1]*self.num_capsules_in])
        return M_i, a_i

    def spatial_transform(self, M_i):
        # M_i: SHAPE=[b, s0', s1', (k0*k1*)I, 1, p0, p1], TYPE= tensor, VALUE= expand the tensor with a value equal to the number output caps
        M_i = backend.expand_dims(M_i, -3)
        # M_i: SHAPE=[b, s0', s1', (k0*k1*)I, O, p0, p1], TYPE= tensor, VALUE= expand the tensor with a value equal to the number output caps
        M_i = backend.tile(M_i, [1, 1, 1, 1, self.num_capsules, 1, 1])

        # W_ij: SHAPE=[b, s0', s1', (k0*k1*)I, O, p0, p1], VALUE= tiled transformation matrices, tile to batch_size
        W_ij= backend.tile(self.W_ij, [self.batch_size, self.spatial_size[0], self.spatial_size[1], 1, 1, 1, 1])

        # V_ij: SHAPE=[b, s0', s1', (k0*k1*)I, O, p0, p1], TYPE= tensor, VALUE= vote matrices
        V_ij = backend.batch_dot(M_i, W_ij)
        return V_ij
    def coord_addition(self, V_ij):
        """
        From the paper: "We therefore share the transformation matrices between different positions of the same capsule type and
        add the scaled coordinate (row, column) of the center of the receptive field of each capsule to the first
        two elements of the right-hand column of its vote matrix."
        """
        # V_ij: SHAPE=[b, s0', s1', k0*k1*I, O, p0*p1], TYPE= tensor, VALUE= adapt the shape for computation
        V_ij = backend.reshape(V_ij, shape=[self.batch_size, self.spatial_size[0], self.spatial_size[1], self.kernel_size[0]*self.kernel_size[1]*self.num_capsules_in, self.num_capsules, self.pose_shape_in[0]*self.pose_shape_in[1]])
        # H_values: SHAPE=[1, s0', 1, 1, 1], TYPE= tensor, VALUE= variational axis
        H_values = backend.reshape((tf.range(self.spatial_size[0], dtype=tf.float32) + 0.50) / self.spatial_size[0], [1, self.spatial_size[0], 1, 1, 1])
        # H_values: SHAPE=[1, s0', 1, 1, 1], TYPE= tensor, VALUE= non variational axis
        H_zeros = tf.constant(0.0, shape=[1, self.spatial_size[0], 1, 1, 1], dtype=tf.float32)
        # H_values: SHAPE=[1, s0', 1, 1, p0*p1], TYPE= tensor, VALUE= new coordinates' offset
        H_offset = tf.stack([H_values, H_zeros] + [H_zeros for _ in range(self.pose_shape_in[0]*self.pose_shape_in[1]-2)], axis=-1) 
        # W_values: SHAPE=[1, 1, s1', 1, 1, 1], TYPE= tensor, VALUE= variational axis
        W_values = tf.reshape((tf.range(self.spatial_size[1], dtype=tf.float32) + 0.50) / self.spatial_size[1], [1, 1, self.spatial_size[1], 1, 1])
        # H_values: SHAPE=[1, 1, s1', 1, 1], TYPE= tensor, VALUE= non variational axis
        W_zeros = tf.constant(0.0, shape=[1, 1, self.spatial_size[1], 1, 1], dtype=tf.float32)
        # H_values: SHAPE=[1, 1, s1', 1, p0*p1], TYPE= tensor, VALUE= new coordinates' offset
        W_offset = tf.stack([W_zeros, W_values] + [W_zeros for _ in range(self.pose_shape_in[0]*self.pose_shape_in[1]-2)], axis=-1)
        # V_ij: SHAPE=[b, s0', s1', I, O, p0*p1], TYPE= tensor, VALUE= V_ij in the new coordinates
        V_ij = V_ij + H_offset + W_offset
        # V_ij: SHAPE=[b, s0', s1', I, O, p0, p1], TYPE= tensor, VALUE= reshape back to the standard norm
        V_ij = backend.reshape(V_ij, shape=[self.batch_size, self.spatial_size[0], self.spatial_size[1], self.kernel_size[0]*self.kernel_size[1]*self.num_capsules_in, self.num_capsules, self.pose_shape_in[0], self.pose_shape_in[1]])
        return V_ij

    def em_routing(self, V_ij, a_i):
        def maximization(R_ij, V_ij, a_i, inv_temp):
            # R_ij: SHAPE=[b, s0', s1', k0*k1*I, O, 1] or [b, s0'*s1'*I, O, 1], TYPE= tensor, VALUE=weights assignment according to the activation probabilities CAUTION!!!! maybe reshape it into backend, backend, A, B, 1 .... before multiplication
            R_ij = R_ij * a_i
            # R_ij: SHAPE=[b, (s0', s1',) 1, O, 1] , TYPE= tensor, VALUE=sum over all input capsules i
            R_ij_sum = backend.sum(R_ij, axis=-3, keepdims=True)
            # M_j: SAHPE=[b, (s0', s1',) 1, O, p0*p1], TYPE= tensor, VALUE= mean of capsule j
            M_j = backend.sum(R_ij * V_ij, axis=-3, keepdims=True) / R_ij_sum
            # stdv_j: SAHPE=[b, (s0', s1',) 1, O, p0*p1], TYPE= tensor, VALUE= standard deviation of capsule j
            stdv_j = backend.sqrt(backend.sum(R_ij_sum * tf.square(V_ij - M_j), axis=-3, keepdims=True) / R_ij_sum)
            # cost_j_h: SHAPE=[b, (s0', s1',) 1, O, p0*p1], TYPE= tensor, VALUE= expected energy of a capsule j
            cost_j_h = (self.beta_v + backend.log(stdv_j + backend.epsilon())) * R_ij_sum
            # cost_j: SHAPE=[b, (s0', s1',) 1, O, 1], TYPE= tensor, VALUE= expected energy
            cost_j = backend.sum(cost_j_h, axis=-1, keepdims=True)
            # cost_j_mean: SHAPE=[b, (s0', s1',) 1, 1, 1], TYPE= tensor, VALUE= mean the expected energy over the output capsules
            cost_j_mean = backend.mean(cost_j, axis=-2, keepdims=True)
            # cost_j_stdv: SHAPE=[b, (s0', s1',) 1, 1, 1], TYPE= tensor, VALUE= mean the expected energy
            cost_j_stdv = backend.sqrt(
                backend.sum(backend.square(cost_j - cost_j_mean), axis=-2, keepdims=True) / self.num_capsules)
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
            a_j_p_j = backend.log(a_j + backend.epsilon()) - backend.sum(
                backend.square(V_ij - M_j) / (2 * tf.square(stdv_j)), axis=-1, keepdims=True) - backend.sum(
                tf.math.log(stdv_j + backend.epsilon()), axis=-1, keepdims=True)
            # R_ij: SHAPE= [b, s0', s1', k0*k1*I, O, 1], TYPE= tensor, VALUE= activated routing matrix
            R_ij = tf.nn.softmax(a_j_p_j, axis=len(a_j_p_j.get_shape().as_list()) - 2)
            return R_ij

        if len(self.spatial_dim):
            # V_ij: SHAPE=[b, s0', s1', k0*k1*I, O, p0*p1], TYPE= tensor, VALUE= adapt the shape for computation
            V_ij = backend.reshape(V_ij, shape=[-1, self.spatial_size[0], self.spatial_size[1],
                                                self.kernel_size[0] * self.kernel_size[1] * self.num_capsules_in,
                                                self.num_capsules, self.pose_shape[0] * self.pose_shape[1]])
            # ai: SHAPE=[b, s0', s1', k0*k1*I, 1, 1], TYPE= tensor, VALUE= expanded i activations
            a_i = backend.expand_dims(backend.expand_dims(a_i, -1), -1)
            # R_ij: SHAPE=[k0*k1*I, O, 1], TYPE= tensor, VALUE= routing assignment matrix from each input capsule (i) in L to each output capsule (j) in L+1 initilized with uniform distribution
            R_ij = backend.constant(1.0 / self.num_capsules, shape=(
            self.kernel_size[0] * self.kernel_size[0] * self.num_capsules_in, self.num_capsules, 1))
        else:
            # V_ij: SHAPE=[b, s0'*s1'*I, O, p0*p1], TYPE= tensor, VALUE= adapt the shape for computation
            V_ij = backend.reshape(V_ij, shape=[-1, self.spatial_size[0] * self.spatial_size[1] * self.num_capsules_in,
                                                self.num_capsules, self.pose_shape[0] * self.pose_shape[1]])
            # a_i: SHAPE=[b, s0'*s1'*I, 1, 1], TYPE= tensor, VALUE= reshape to standard form
            a_i = backend.reshape(a_i,
                                  shape=[-1, self.spatial_size[0] * self.spatial_size[1] * self.num_capsules_in, 1, 1])
            # R_ij: SHAPE=[s0'*s1'*I, O, 1], TYPE= tensor, VALUE= routing assignment matrix from each input capsule (i) in L to each output capsule (j) in L+1 initilized with uniform distribution
            R_ij = backend.constant(1.0 / self.num_capsules, shape=(
            self.spatial_size[0] * self.spatial_size[0] * self.num_capsules_in, self.num_capsules, 1))
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
        M_j = backend.reshape(M_j, shape=[-1, self.spatial_size[0], self.spatial_size[1], self.num_capsules,
                                          self.pose_shape[0], self.pose_shape[1]])
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


class MatrixForTraining(utls.BaseModelForTraining):
    def __init__(self, name='MatrixCapsuleNet'):
        super(MatrixForTraining, self).__init__(name=name)
        self.batch_size = None
        self.margin = None

    def create(self, input_shape, **kwargs):  # L3_n is the same as output_shape
        if kwargs.get('pose_shape') is None:
            pose_shape = [4, 4]
        else:
            pose_shape = kwargs.get('pose_shape')
        # inputs = img_shape : Input
        inputs = layers.Input(shape=input_shape)
        # net = [batch_size, s0, s1, A] : ReLU Conv1
        net = layers.Conv2D(filters=32, kernel_size=[5, 5], strides=2, padding='valid', activation='relu',
                            name='ReLU_Conv1')(inputs)  # add batch normalization ??
        # net = [ poses = [?, s0, s1, B, p*p], activations = [?, s0, s1, B] ] : PrimaryCaps
        net = PrimaryCapsule2D(pose_shape=pose_shape, num_capsules=8, kernel_size=[1, 1], strides=1,
                               padding='valid', name='PrimaryCaps')(net)
        # nets = [ poses = [?, s0', s1', C, p*p], activations = [?, s0', s1', C] ] : ConvCaps1
        net = ConvolutionalCapsule(num_capsules=16, pose_shape=pose_shape, kernel_size=[3, 3], strides=2, routings=1,
                                   name='ConvCaps1')(net)
        # nets = [ poses (?, s0'', s1'', D, p*p), activations = [?, s0'', s1'', D] ] : ConvCaps2
        net = ConvolutionalCapsule(num_capsules=16, pose_shape=pose_shape, kernel_size=[3, 3], strides=1, routings=1,
                                   name='ConvCaps2')(net)
        # output  = [ poses = [?, E, p*p], activations = [?, E] ] : Class Capsules
        net = ConvolutionalCapsule(num_capsules=10, pose_shape=pose_shape, routings=1,
                                   name='Class_Capsules')(net)

        return tf.keras.Model(inputs, net)

    def compile(self, batch_size, n_samples, optimizer=None, **kwargs):
        self.batch_size = batch_size
        iterations_per_epoch = int(n_samples / batch_size)

        # self.margin = tfv1.train.piecewise_constant(tf.Variable(1, trainable=False, dtype=tf.int32),
        #                                             boundaries=[int(iterations_per_epoch * 10.0 * x / 7) for x in
        #                                                         range(1, 8)],
        #                                             values=[x / 10.0 for x in range(2, 10)])

        self.margin = 2 / 10.0

        if optimizer is None:
            optimizer = self.loss_fn
        super(MatrixForTraining, self).compile(optimizer=optimizer, **kwargs)

    def fit(self, x, y, batch_size, **kwargs):
        super(MatrixForTraining, self).fit(x=x, y=y, batch_size=batch_size, **kwargs)

    def loss_fn(self, y_true, y_pred):
        # y_pred_t = [b, 1] : true predictions
        y_pred_true = tf.reshape(tf.boolean_mask(y_pred, tf.equal(y_true, 1)), shape=(-1, 1))
        # y_pred_i = [b, 9] : false predictions
        y_pred_false = tf.reshape(tf.boolean_mask(y_pred, tf.equal(y_true, 0)),
                                  shape=(-1, 10 - 1)) # TODO: 10 - это количество классов
        # loss = [1] : loss function
        # loss = K.sum(K.square(K.relu(self.margin - (y_pred_true - y_pred_false))))
        loss = tf.math.reduce_sum(tf.square(tf.maximum(0., self.margin - (y_pred_true - y_pred_false))))
        return loss


if __name__ == '__main__':
    from libs import utls

    (x_train, y_train), (x_test, y_test) = utls.load('mnist')

    model = MatrixForTraining()
    model.build(input_shape=(28, 28, 1))
    model.compile(batch_size=1, n_samples=60000, optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=model.loss_fn, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=1, epochs=5, validation_data=(x_test, y_test))
