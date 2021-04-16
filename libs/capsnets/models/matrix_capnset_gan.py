class MCapsNet(Discriminator):
    def __init__(self, batch_size, name='MCapsNet', **kwargs):
        self.batch_size = batch_size
        super(MCapsNet, self).__init__(name=name, **kwargs)

    def build_sequential(self, input_shape, output_shape, L1_n, L2_n, L3_n, L4_n, pose_shape=[4, 4], routing=3,
                         decoder=False):  # L3_n is the same as output_shape
        self.output_shape = output_shape
        # inputs = img_shape : Input
        inputs = layers.Input(shape=input_shape)
        # net = [b, s0, s1, A] : ReLU Conv1
        net = layers.Conv2D(filters=L1_n, kernel_size=[5, 5], strides=2, padding='SAME', activation='relu',
                            name='ReLU_Conv1')(inputs)  # add batch normalization ??
        # net = [ poses = [?, s0, s1, B, p*p], activations = [?, s0, s1, B] ] : PrimaryCaps
        net = self.PrimaryCaps(net, pose_shape=pose_shape, n_caps_out=L2_n, kernel_size=[1, 1], strides=1,
                               padding='VALID', name='PrimaryCaps')
        # nets = [ poses = [?, s0', s1', C, p*p], activations = [?, s0', s1', C] ] : ConvCaps1
        net = CapsLayer(n_caps_out=L3_n, pose_shape=pose_shape, kernel_size=[3, 3], strides=2, routing_iters=routing,
                        name='ConvCaps1')(net)
        # nets = [ poses (?, s0'', s1'', D, p*p), activations = [?, s0'', s1'', D] ] : ConvCaps2
        net = CapsLayer(n_caps_out=L4_n, pose_shape=pose_shape, kernel_size=[3, 3], strides=1, routing_iters=routing,
                        name='ConvCaps2')(net)
        # output  = [ poses = [?, E, p*p], activations = [?, E] ] : Class Capsules
        net = CapsLayer(n_caps_out=np.prod(output_shape), pose_shape=pose_shape, routing_iters=routing,
                        name='Class_Capsules')(net)
        return models.Model(inputs, net)

    def PrimaryCaps(self, inputs, pose_shape, n_caps_out, kernel_size, strides, padding, name):
        # M = [b, s0, s1, I*p0*p1] : generate the pose matrices of the caps
        M = layers.Conv2D(filters=n_caps_out * pose_shape[0] * pose_shape[1], kernel_size=kernel_size, strides=strides,
                          padding=padding)(inputs)
        # M = [b, s0, s1, I, p0, p1] : reshape the pose matrices from 16 scalar values into a 4x4 matrix
        M = layers.Reshape(target_shape=[M.get_shape().as_list()[1], M.get_shape().as_list()[2], n_caps_out,
                                         pose_shape[0] * pose_shape[1]])(M)
        # a = [b, s0, s1, I] : generate the activation for the caps
        a = layers.Conv2D(filters=n_caps_out, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
        a = layers.Activation('sigmoid')(a)
        a = layers.Reshape(
            target_shape=[inputs.get_shape().as_list()[1], inputs.get_shape().as_list()[2], n_caps_out, 1])(a)
        net = layers.Concatenate()([M, a])
        return net

    def compile(self, batch_size, n_samples, optimizer=None, **kwargs):
        self.batch_size = batch_size
        iterations_per_epoch = int(n_samples / batch_size)
        self.margin = tf.train.piecewise_constant(tf.Variable(1, trainable=False, dtype=tf.int32),
                                                  boundaries=[int(iterations_per_epoch * 10.0 * x / 7) for x in
                                                              range(1, 8)],
                                                  values=[x / 10.0 for x in range(2, 10)])
        if optimizer is None: optimizer = self.loss_fn
        super(MCapsNet, self).compile(optimizer=optimizer, **kwargs)

    def fit(self, x, y, batch_size, **kwargs):
        super(MCapsNet, self).fit(x=x, y=y, batch_size=batch_size, **kwargs)

    def loss_fn(self, y_true, y_pred):
        # y_pred_t = [b, 1] : true predictions
        y_pred_true = K.reshape(tf.boolean_mask(y_pred, tf.equal(y_true, 1)), shape=(self.batch_size, 1))
        # y_pred_i = [b, 9] : false predictions
        y_pred_false = K.reshape(tf.boolean_mask(y_pred, tf.equal(y_true, 0)),
                                 shape=(self.batch_size, np.prod(self.output_shape) - 1))
        # loss = [1] : loss function
        # loss = K.sum(K.square(K.relu(self.margin - (y_pred_true - y_pred_false))))
        loss = K.sum(K.square(K.maximum(0., self.margin - (y_pred_true - y_pred_false))))
        return loss


network = MCapsNet(batch_size=train_dict["batch_size"])
network.build(input_shape=disc_dict["inputs_shape"], output_shape=disc_dict["output_shape"],
              L1_n=disc_net_dict["L1_n"],
              L2_n=disc_net_dict["L2_n"],
              L3_n=disc_net_dict["L3_n"],
              L4_n=disc_net_dict["L4_n"],
              routing=disc_net_dict["routings"],
              pose_shape=disc_net_dict["pose_shape"])
network.compile(optimizer=disc_net_dict["optimizer"], batch_size=train_dict["batch_size"],
                n_samples=DATASET['train'].labels.shape[0], loss=network.loss_fn, metrics=['accuracy'])
network.fit(x=DATASET['train'].imgs, y=DATASET['train'].labels, batch_size=train_dict["batch_size"],
            epochs=train_dict["epochs"],
            validation_data=[DATASET['test'].imgs, DATASET['test'].labels],
            logdir=checkpt_dict["logdir"], TensorBoard=False)
