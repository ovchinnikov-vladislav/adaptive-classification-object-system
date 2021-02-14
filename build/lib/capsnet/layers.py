import tensorflow as tf
import tensorflow.keras.backend as keras_backend
from keras.utils.conv_utils import conv_output_length, deconv_length
from tensorflow.keras import initializers, layers
import numpy as np
from capsnet import utils as capsnet_utils
import tf_slim as slim


class Length(layers.Layer):
    """
    Слой для вычисления длины векторов. Этот слой используется для вычисления тензора, который имеет ту же форму (shape)
    c y_true в margin_loss. Использование этого слоя в качестве выходных данных модели позволяет предсказывать метки с
    помощью соотношения `y_pred = np.argmax(model.predict(x), 1)`

    Вход слоя (inputs): shape=[None, num_vectors, dim_vector]
    Выход слоя: shape=[None, num_vectors]
    """

    def __init__(self, num_classes=None, seg=False, **kwargs):
        super(Length, self).__init__(**kwargs)
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        self.seg = seg

    def call(self, inputs, **kwargs):
        if inputs.shape.ndims == 5:
            assert inputs.get_shape()[-2].value == 1, 'Error: Must have num_capsules = 1 going into Length'
            inputs = keras_backend.squeeze(inputs, axis=-2)
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1) + keras_backend.epsilon())

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 5:
            input_shape = input_shape[0:-2] + input_shape[-1:]
        if self.seg and self.num_classes is not None:
            return input_shape[:-1] + (self.num_classes,)
        else:
            return input_shape[:-1]

    def get_config(self):
        base_config = super(Length, self).get_config()
        config = base_config.copy()
        config['num_classes'] = self.num_classes
        config['seg'] = self.seg
        return config


class Mask(layers.Layer):
    """
    Слой маскирования тензора с размером (shape) shape=[None, num_capsule, dim_vector], или по капсуле максимальной
    длины, или по дополнительной входной маске. За исключением капсулы максимальной длины (или указанной капсулы),
    все векторы маркируются нулями. Затем слой выравнивает максируемый тензор.
    Например:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, каждый пример содержит 3 капсулы с dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # Правдивые метки. 8 примеров, 3 класса, one_hot coding
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """

    def __init__(self, resize_masks=False, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self.resize_masks = resize_masks

    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            _inputs, mask = inputs
            height = _inputs.shape[1]
            width = _inputs.shape[2]
            if self.resize_masks:
                mask = tf.compat.v1.image.resize_bicubic(mask, (height.value, width.value))
            mask = keras_backend.expand_dims(mask, -1)
            if _inputs.shape.ndims == 3:
                masked = keras_backend.batch_flatten(mask * _inputs)
            else:
                masked = mask * input
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            if inputs.shape.ndims == 3:
                x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
                # generate the mask which is a one-hot code.
                # mask.shape=[None, n_classes]=[None, num_capsule]
                mask = tf.one_hot(indices=tf.argmax(x, 1), depth=x.shape[1])

                # inputs.shape=[None, num_capsule, dim_capsule]
                # mask.shape=[None, num_capsule]
                # masked.shape=[None, num_capsule * dim_capsule]
                masked = keras_backend.batch_flatten(inputs * tf.expand_dims(mask, -1))
            else:
                masked = inputs

        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


class PrimaryCaps(layers.Layer):
    """
    Слой первичных капсул, который применяет Conv2D `n_channels` раз и соединяет все капсулы
    :param name:
    :param pose_shape:
    :param inputs: 4-х мерный тензор, shape=[None, width, height, channels]
    :param dim_capsule: размерность выходного вектора капсулы
    :param n_channels: количество типов капсул
    :param kernel_size: целое число или кортеж / список из 2 целых чисел,
    определяющих высоту и ширину окна двумерной свертки
    :param strides: целое число или кортеж / список из 2 целых чисел, определяющих шаги свертки по высоте и ширине
    :param padding: `valid` - отсутствие отступов или `same` - равномерное добавление отступов слева / справа
    сверху / снизу. Добавление отступов приводит к тому, что ввод и вывод имеют одинаковую высоту и ширину
    :return: выходом является тензор формы shape=[None, num_capsule, dim_capsule]
    """

    def __init__(self, dim_capsule, kernel_size, strides, padding, n_channels=None, pose_shape=None, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.dim_capsule = dim_capsule
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.n_channels = n_channels
        self.pose_shape = pose_shape
        self.conv2d = self.poses_conv2d = self.activations_conv2d = None
        self.conv2d_output_shape = None

    def build(self, input_shape):
        if self.pose_shape is None and self.n_channels is not None:
            self.conv2d = layers.Conv2D(filters=self.dim_capsule * self.n_channels, kernel_size=self.kernel_size,
                                        strides=self.strides, padding=self.padding)
        elif self.pose_shape is not None and self.n_channels is None:
            self.poses_conv2d = layers.Conv2D(filters=self.dim_capsule * self.pose_shape[0] * self.pose_shape[1],
                                              kernel_size=self.kernel_size, strides=self.strides, padding=self.padding)

            self.activations_conv2d = layers.Conv2D(filters=self.dim_capsule, kernel_size=self.kernel_size,
                                                    strides=self.strides, padding=self.padding, activation='sigmoid')
        else:
            raise Exception(f'Invalid args: n_channels={self.n_channels} and pose_shape={self.pose_shape}')
        self.built = True

    def call(self, inputs, **kwargs):
        if self.pose_shape is None and self.n_channels is not None:
            outputs = self.conv2d(inputs)
            self.conv2d_output_shape = outputs.shape
            outputs = layers.Reshape(target_shape=[-1, self.dim_capsule])(outputs)
            return layers.Lambda(capsnet_utils.squash)(outputs)
        elif self.pose_shape is not None and self.n_channels is None:
            poses = self.poses_conv2d(inputs)

            input_shape = inputs.shape
            poses = tf.reshape(poses, shape=[-1, input_shape[-3], input_shape[-2], self.dim_capsule,
                                             self.pose_shape[0], self.pose_shape[1]])

            activations = self.activations_conv2d(inputs)

            return poses, activations

    def compute_output_shape(self, input_shape):
        return tuple([None, self.conv2d_output_shape[1] * self.conv2d_output_shape[2]
                      * self.conv2d_output_shape[3] / self.dim_capsule, self.dim_capsule])


class TimePrimaryCaps(layers.Layer):
    """
    Слой первичных капсул, который применяет Conv2D `n_channels` раз и соединяет все капсулы
    :param name:
    :param pose_shape:
    :param inputs: 4-х мерный тензор, shape=[None, width, height, channels]
    :param dim_capsule: размерность выходного вектора капсулы
    :param n_channels: количество типов капсул
    :param kernel_size: целое число или кортеж / список из 2 целых чисел,
    определяющих высоту и ширину окна двумерной свертки
    :param strides: целое число или кортеж / список из 2 целых чисел, определяющих шаги свертки по высоте и ширине
    :param padding: `valid` - отсутствие отступов или `same` - равномерное добавление отступов слева / справа
    сверху / снизу. Добавление отступов приводит к тому, что ввод и вывод имеют одинаковую высоту и ширину
    :return: выходом является тензор формы shape=[None, num_capsule, dim_capsule]
    """

    def __init__(self, dim_capsule, kernel_size, strides, padding, n_channels, **kwargs):
        super(TimePrimaryCaps, self).__init__(**kwargs)
        self.dim_capsule = dim_capsule
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.n_channels = n_channels
        self.conv2d = self.poses_conv2d = self.activations_conv2d = None
        self.conv2d_output_shape = None

    def build(self, input_shape):
        self.conv2d = layers.TimeDistributed(layers.Conv2D(filters=self.dim_capsule * self.n_channels,
                                                           kernel_size=self.kernel_size,
                                                           strides=self.strides,
                                                           padding=self.padding))
        self.built = True

    def call(self, inputs, **kwargs):
        outputs = self.conv2d(inputs)
        self.conv2d_output_shape = outputs.shape
        outputs = layers.TimeDistributed(layers.Reshape(target_shape=[-1, self.dim_capsule]))(outputs)
        return layers.TimeDistributed(layers.Lambda(capsnet_utils.squash))(outputs)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.conv2d_output_shape[1] * self.conv2d_output_shape[2]
                      * self.conv2d_output_shape[3] / self.dim_capsule, self.dim_capsule])


class CapsuleLayer(layers.Layer):
    """
    Слой капсул. Данный слой похож на полносвязный (Dense). Полносвязный слой имеет `in_num` входов, каждый из которых
    представляет собой скаляр, являющийся выходом нейрона из предыдущего слоя. Также полносвязный слой имеет `out_num`
    выходных нейронов. CapsuleLayer просто является расширением выхода нейрона от скалярного значения до векторного.

    Вход слоя имеет форму (shape) shape = [None, input_num_capsule, input_dim_capsule]
    Выход слоя имеет форму (shape) shape = [None, num_capsule, dim_capsule]
    Для полносвязного слоя input_dim_capsule = dim_capsule = 1. Т.е. размерность капсулы равна 1.

    :param num_capsule: количество капсул в этом слое
    :param dim_capsule: размер выходных векторов капсул в этом слое
    :param routings: количество итераций алгоритма маршрутизации между капсулами
    """

    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.input_num_capsule = self.input_dim_capsule = self.W = None

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix, from each input capsule to each output capsule, there's a unique weight as in Dense layer.
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # Expand the input in axis=1, tile in that axis to num_capsule, and
        # expands another axis at the end to prepare the multiplication with W.
        #  inputs.shape=[None, input_num_capsule, input_dim_capsule]
        #  inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        #  inputs_tiled.shape=[None, num_capsule, input_num_capsule,
        #                            input_dim_capsule, 1]
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_tiled = tf.expand_dims(inputs_tiled, 4)

        # Compute `W * inputs` by scanning inputs_tiled on dimension 0 (map_fn).
        # - Use matmul (without transposing any element). Note the order!
        # Thus:
        #  x.shape=[num_capsule, input_num_capsule, input_dim_capsule, 1]
        #  W.shape=[num_capsule, input_num_capsule, dim_capsule,input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [dim_capsule, input_dim_capsule] x [input_dim_capsule, 1]->
        #              [dim_capsule, 1].
        #  inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule, 1]

        inputs_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled)

        # Begin: Routing algorithm ----------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        #  b.shape = [None, self.num_capsule, self.input_num_capsule, 1, 1].
        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule,
                            self.input_num_capsule, 1, 1])

        assert self.routings > 0, 'The routings should be > 0.'
        outputs = None
        for i in range(self.routings):
            # Apply softmax to the axis with `num_capsule`
            #  c.shape=[batch_size, num_capsule, input_num_capsule, 1, 1]
            c = tf.nn.softmax(b, axis=1)

            # Compute the weighted sum of all the predicted output vectors.
            #  c.shape =  [batch_size, num_capsule, input_num_capsule, 1, 1]
            #  inputs_hat.shape=[None, num_capsule, input_num_capsule,dim_capsule,1]
            # The function `multiply` will broadcast axis=3 in c to dim_capsule.
            #  outputs.shape=[None, num_capsule, input_num_capsule, dim_capsule, 1]
            # Then sum along the input_num_capsule
            #  outputs.shape=[None, num_capsule, 1, dim_capsule, 1]
            # Then apply squash along the dim_capsule
            outputs = tf.multiply(c, inputs_hat)
            outputs = tf.reduce_sum(outputs, axis=2, keepdims=True)
            outputs = capsnet_utils.squash(outputs, axis=-2)  # [None, 10, 1, 16, 1]

            if i < self.routings - 1:
                # Update the prior b.
                #  outputs.shape =  [None, num_capsule, 1, dim_capsule, 1]
                #  inputs_hat.shape=[None,num_capsule,input_num_capsule,dim_capsule,1]
                # Multiply the outputs with the weighted_inputs (inputs_hat) and add
                # it to the prior b.
                outputs_tiled = tf.tile(outputs, [1, 1, self.input_num_capsule, 1, 1])
                agreement = tf.matmul(inputs_hat, outputs_tiled, transpose_a=True)
                b = tf.add(b, agreement)

        # End: Routing algorithm ------------------------------------------------#
        # Squeeze the outputs to remove useless axis:
        #  From  --> outputs.shape=[None, num_capsule, 1, dim_capsule, 1]
        #  To    --> outputs.shape=[None, num_capsule,    dim_capsule]
        outputs = tf.squeeze(outputs, [2, 4])
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        base_config = super(CapsuleLayer, self).get_config()
        config = base_config.clone()
        config['num_capsule'] = self.num_capsule
        config['dim_capsule'] = self.dim_capsule
        config['routings'] = self.routings
        return config


def ConvCapsuleLayer(inputs, shape, strides, routings, batch_size):
    stride = strides[1]
    i_size = shape[-2]
    o_size = shape[-1]

    inputs_poses, inputs_activations = inputs
    pose_size = inputs_poses.shape[-1]

    inputs_poses = CapsuleUtils.kernel_tile(inputs_poses, 3, stride)
    inputs_activations = CapsuleUtils.kernel_tile(inputs_activations, 3, stride)
    spatial_size = int(inputs_activations.shape[1])

    inputs_poses = tf.reshape(inputs_poses, shape=[-1, 3 * 3 * i_size, 16])
    inputs_activations = tf.reshape(inputs_activations, shape=[-1, spatial_size, spatial_size, 3 * 3 * i_size])

    votes = CapsuleUtils.mat_transform(inputs_poses, o_size, size=batch_size * spatial_size * spatial_size)

    votes_shape = votes.shape
    votes = tf.reshape(votes, shape=[batch_size, spatial_size, spatial_size, votes_shape[-3],
                                     votes_shape[-2], votes_shape[-1]])

    initializer = tf.initializers.GlorotUniform()
    beta_v = tf.Variable(lambda: initializer(shape=[1, 1, 1, o_size], dtype=tf.float32), name='beta_v')
    beta_a = tf.Variable(lambda: initializer(shape=[1, 1, 1, o_size], dtype=tf.float32), name='beta_a')

    poses, activations = CapsuleUtils.matrix_capsules_em_routing(votes, inputs_activations, beta_v, beta_a, routings)
    poses_shape = poses.shape
    poses = tf.reshape(poses, [poses_shape[0], poses_shape[1], poses_shape[2],
                               poses_shape[3], pose_size, pose_size])

    return poses, activations


class ClassCapsuleLayer(layers.Layer):
    def __init__(self, num_classes, batch_size, routings, **kwargs):
        super(ClassCapsuleLayer, self).__init__(**kwargs)
        self.routings = routings
        self.batch_size = batch_size
        self.num_classes = num_classes

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_poses, inputs_activations = inputs
        inputs_shape = inputs_poses.shape
        spatial_size = int(inputs_shape[1])
        pose_size = int(inputs_shape[-1])
        i_size = int(inputs_shape[3])

        inputs_poses = tf.reshape(inputs_poses, shape=[self.batch_size * spatial_size * spatial_size, inputs_shape[-3],
                                                       inputs_shape[-2] * inputs_shape[-2]])

        votes = CapsuleUtils.mat_transform(inputs_poses, self.num_classes,
                                           size=self.batch_size * spatial_size * spatial_size)
        votes = tf.reshape(votes, shape=[self.batch_size, spatial_size, spatial_size,
                                         i_size, self.num_classes, pose_size * pose_size])
        votes = CapsuleUtils.coord_addition(votes, spatial_size, spatial_size)

        initializer = tf.initializers.GlorotUniform()
        beta_v = tf.Variable(lambda: initializer(shape=[1, self.num_classes], dtype=tf.float32), name='beta_v')
        beta_a = tf.Variable(lambda: initializer(shape=[1, self.num_classes], dtype=tf.float32), name='beta_a')

        votes_shape = votes.get_shape()
        votes = tf.reshape(votes, shape=[self.batch_size, votes_shape[1] * votes_shape[2] *
                                         votes_shape[3], votes_shape[4], votes_shape[5]])

        inputs_activations = tf.reshape(inputs_activations, shape=[self.batch_size,
                                                                   votes_shape[1] * votes_shape[2] * votes_shape[3]])
        poses, activations = CapsuleUtils.matrix_capsules_em_routing(votes, inputs_activations,
                                                                     beta_v, beta_a, self.routings)
        poses = tf.reshape(poses, shape=[self.batch_size, self.num_classes, pose_size, pose_size])
        self.pose_size = pose_size

        return poses, activations

    def compute_output_shape(self, input_shape):
        return [(self.batch_size, self.num_classes, self.pose_size, self.pose_size),
                (self.batch_size, self.num_classes)]


# class ConvCapsuleLayer(layers.Layer):
#     def __init__(self, kernel_size, num_capsule, num_atoms, strides=1, padding='same', routings=3,
#                  kernel_initializer='he_normal', **kwargs):
#         super(ConvCapsuleLayer, self).__init__(**kwargs)
#         self.kernel_size = kernel_size
#         self.num_capsule = num_capsule
#         self.num_atoms = num_atoms
#         self.strides = strides
#         self.padding = padding
#         self.routings = routings
#         self.kernel_initializer = initializers.get(kernel_initializer)
#
#     def build(self, input_shape):
#         assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
#                                       " input_num_capsule, input_num_atoms]"
#         self.input_height = input_shape[1]
#         self.input_width = input_shape[2]
#         self.input_num_capsule = input_shape[3]
#         self.input_num_atoms = input_shape[4]
#
#         # Transform matrix
#         self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
#                                         self.input_num_atoms, self.num_capsule * self.num_atoms],
#                                  initializer=self.kernel_initializer,
#                                  name='W')
#
#         self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
#                                  initializer=initializers.constant(0.1),
#                                  name='b')
#
#         self.built = True
#
#     def call(self, input_tensor, training=None):
#         input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
#         input_shape = keras_backend.shape(input_transposed)
#         input_tensor_reshaped = keras_backend.reshape(input_transposed, [
#             input_shape[0] * input_shape[1], self.input_height, self.input_width, self.input_num_atoms])
#         input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_num_atoms))
#
#         conv = keras_backend.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
#                                     padding=self.padding, data_format='channels_last')
#
#         votes_shape = keras_backend.shape(conv)
#         _, conv_height, conv_width, _ = conv.get_shape()
#
#         votes = keras_backend.reshape(conv, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2],
#                                              self.num_capsule, self.num_atoms])
#         votes.set_shape((None, self.input_num_capsule, conv_height.value, conv_width.value,
#                          self.num_capsule, self.num_atoms))
#
#         logit_shape = keras_backend.stack([
#             input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule])
#         biases_replicated = keras_backend.tile(self.b, [conv_height.value, conv_width.value, 1, 1])
#
#         activations = update_routing(
#             votes=votes,
#             biases=biases_replicated,
#             logit_shape=logit_shape,
#             num_dims=6,
#             input_dim=self.input_num_capsule,
#             output_dim=self.num_capsule,
#             num_routing=self.routings)
#
#         return activations
#
#     def compute_output_shape(self, input_shape):
#         space = input_shape[1:-2]
#         new_space = []
#         for i in range(len(space)):
#             new_dim = conv_output_length(
#                 space[i],
#                 self.kernel_size,
#                 padding=self.padding,
#                 stride=self.strides,
#                 dilation=1)
#             new_space.append(new_dim)
#
#         return (input_shape[0],) + tuple(new_space) + (self.num_capsule, self.num_atoms)
#
#     def get_config(self):
#         config = {
#             'kernel_size': self.kernel_size,
#             'num_capsule': self.num_capsule,
#             'num_atoms': self.num_atoms,
#             'strides': self.strides,
#             'padding': self.padding,
#             'routings': self.routings,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer)
#         }
#         base_config = super(ConvCapsuleLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# class DeconvCapsuleLayer(layers.Layer):
#     def __init__(self, kernel_size, num_capsule, num_atoms, scaling=2, upsamp_type='deconv', padding='same', routings=3,
#                  kernel_initializer='he_normal', **kwargs):
#         super(DeconvCapsuleLayer, self).__init__(**kwargs)
#         self.kernel_size = kernel_size
#         self.num_capsule = num_capsule
#         self.num_atoms = num_atoms
#         self.scaling = scaling
#         self.upsamp_type = upsamp_type
#         self.padding = padding
#         self.routings = routings
#         self.kernel_initializer = initializers.get(kernel_initializer)
#
#     def build(self, input_shape):
#         assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
#                                       " input_num_capsule, input_num_atoms]"
#         self.input_height = input_shape[1]
#         self.input_width = input_shape[2]
#         self.input_num_capsule = input_shape[3]
#         self.input_num_atoms = input_shape[4]
#
#         # Transform matrix
#         if self.upsamp_type == 'subpix':
#             self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
#                                             self.input_num_atoms,
#                                             self.num_capsule * self.num_atoms * self.scaling * self.scaling],
#                                      initializer=self.kernel_initializer,
#                                      name='W')
#         elif self.upsamp_type == 'resize':
#             self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
#                                             self.input_num_atoms, self.num_capsule * self.num_atoms],
#                                      initializer=self.kernel_initializer, name='W')
#         elif self.upsamp_type == 'deconv':
#             self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
#                                             self.num_capsule * self.num_atoms, self.input_num_atoms],
#                                      initializer=self.kernel_initializer, name='W')
#         else:
#             raise NotImplementedError('Upsampling must be one of: "deconv", "resize", or "subpix"')
#
#         self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
#                                  initializer=initializers.constant(0.1),
#                                  name='b')
#
#         self.built = True
#
#     def call(self, input_tensor, training=None):
#         input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
#         input_shape = keras_backend.shape(input_transposed)
#         input_tensor_reshaped = keras_backend.reshape(input_transposed, [
#             input_shape[1] * input_shape[0], self.input_height, self.input_width, self.input_num_atoms])
#         input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_num_atoms))
#
#         if self.upsamp_type == 'resize':
#             upsamp = keras_backend.resize_images(input_tensor_reshaped, self.scaling, self.scaling, 'channels_last')
#             outputs = keras_backend.conv2d(upsamp, kernel=self.W, strides=(1, 1), padding=self.padding,
#                                            data_format='channels_last')
#         elif self.upsamp_type == 'subpix':
#             conv = keras_backend.conv2d(input_tensor_reshaped, kernel=self.W, strides=(1, 1), padding='same',
#                                         data_format='channels_last')
#             outputs = tf.nn.depth_to_space(conv, self.scaling)
#         else:
#             batch_size = input_shape[1] * input_shape[0]
#
#             # Infer the dynamic output shape:
#             out_height = deconv_length(self.input_height, self.scaling, self.kernel_size, self.padding)
#             out_width = deconv_length(self.input_width, self.scaling, self.kernel_size, self.padding)
#             output_shape = (batch_size, out_height, out_width, self.num_capsule * self.num_atoms)
#
#             outputs = keras_backend.conv2d_transpose(input_tensor_reshaped, self.W, output_shape,
#                                                      (self.scaling, self.scaling),
#                                                      padding=self.padding, data_format='channels_last')
#
#         votes_shape = keras_backend.shape(outputs)
#         _, conv_height, conv_width, _ = outputs.get_shape()
#
#         votes = keras_backend.reshape(outputs, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2],
#                                     self.num_capsule, self.num_atoms])
#         votes.set_shape((None, self.input_num_capsule, conv_height.value, conv_width.value,
#                          self.num_capsule, self.num_atoms))
#
#         logit_shape = keras_backend.stack([
#             input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule])
#         biases_replicated = keras_backend.tile(self.b, [votes_shape[1], votes_shape[2], 1, 1])
#
#         activations = update_routing(
#             votes=votes,
#             biases=biases_replicated,
#             logit_shape=logit_shape,
#             num_dims=6,
#             input_dim=self.input_num_capsule,
#             output_dim=self.num_capsule,
#             num_routing=self.routings)
#
#         return activations
#
#     def compute_output_shape(self, input_shape):
#         output_shape = list(input_shape)
#
#         output_shape[1] = deconv_length(output_shape[1], self.scaling, self.kernel_size, self.padding)
#         output_shape[2] = deconv_length(output_shape[2], self.scaling, self.kernel_size, self.padding)
#         output_shape[3] = self.num_capsule
#         output_shape[4] = self.num_atoms
#
#         return tuple(output_shape)
#
#     def get_config(self):
#         config = {
#             'kernel_size': self.kernel_size,
#             'num_capsule': self.num_capsule,
#             'num_atoms': self.num_atoms,
#             'scaling': self.scaling,
#             'padding': self.padding,
#             'upsamp_type': self.upsamp_type,
#             'routings': self.routings,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer)
#         }
#         base_config = super(DeconvCapsuleLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# def update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
#                    num_routing):
#     if num_dims == 6:
#         votes_t_shape = [5, 0, 1, 2, 3, 4]
#         r_t_shape = [1, 2, 3, 4, 5, 0]
#     elif num_dims == 4:
#         votes_t_shape = [3, 0, 1, 2]
#         r_t_shape = [1, 2, 3, 0]
#     else:
#         raise NotImplementedError('Not implemented')
#
#     votes_trans = tf.transpose(votes, votes_t_shape)
#     _, _, _, height, width, caps = votes_trans.get_shape()
#
#     def _body(i, logits, activations):
#         """Routing while loop."""
#         # route: [batch, input_dim, output_dim, ...]
#         route = tf.nn.softmax(logits, dim=-1)
#         preactivate_unrolled = route * votes_trans
#         preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
#         preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
#         activation = capsnet_utils.squash(preactivate)
#         activations = activations.write(i, activation)
#         act_3d = keras_backend.expand_dims(activation, 1)
#         tile_shape = np.ones(num_dims, dtype=np.int32)
#         tile_shape[1] = input_dim
#         act_replicated = tf.tile(act_3d, tile_shape)
#         distances = tf.reduce_sum(votes * act_replicated, axis=-1)
#         logits += distances
#         return (i + 1, logits, activations)
#
#     activations = tf.TensorArray(
#         dtype=tf.float32, size=num_routing, clear_after_read=False)
#     logits = tf.fill(logit_shape, 0.0)
#
#     i = tf.constant(0, dtype=tf.int32)
#     _, logits, activations = tf.while_loop(
#         lambda i, logits, activations: i < num_routing,
#         _body,
#         loop_vars=[i, logits, activations],
#         swap_memory=True)
#
#     return keras_backend.cast(activations.read(num_routing - 1), dtype='float32')

class CapsuleUtils:
    @staticmethod
    def matrix_capsules_em_routing(votes, i_activations, beta_v, beta_a, routings):
        votes_shape = votes.shape

        rr = tf.constant(1.0 / votes_shape[-2], shape=votes_shape[-3:-1] + [1], dtype=tf.float32)
        i_activations = i_activations[..., tf.newaxis, tf.newaxis]
        beta_v = beta_v[..., tf.newaxis, :, tf.newaxis]
        beta_a = beta_a[..., tf.newaxis, :, tf.newaxis]

        it_min = 1.0
        it_max = min(routings, 3.0)
        o_mean = o_activations = None
        for it in range(routings):
            inverse_temperature = it_min + (it_max - it_min) * it / max(1.0, routings - 1.0)
            o_mean, o_stdv, o_activations = CapsuleUtils.m_step(rr, votes, i_activations, beta_v, beta_a,
                                                                inverse_temperature=inverse_temperature)

            if it < routings - 1:
                rr = CapsuleUtils.e_step(o_mean, o_stdv, o_activations, votes)

        poses = tf.squeeze(o_mean, axis=-3)
        activations = tf.squeeze(o_activations, axis=[-3, -1])

        return poses, activations

    @staticmethod
    def m_step(rr, votes, i_activations, beta_v, beta_a, inverse_temperature):
        rr_prime = rr * i_activations
        rr_prime_sum = tf.reduce_sum(rr_prime, axis=-3, keepdims=True, name='rr_prime_sum')

        o_mean = tf.reduce_sum(rr_prime * votes, axis=-3, keepdims=True) / rr_prime_sum
        o_stdv = tf.sqrt(tf.reduce_sum(rr_prime * tf.square(votes - o_mean), axis=-3, keepdims=True) / rr_prime_sum)

        o_cost_h = (beta_v + keras_backend.log(o_stdv + keras_backend.epsilon())) * rr_prime_sum

        o_cost = tf.reduce_sum(o_cost_h, axis=-1, keepdims=True)
        o_cost_mean = tf.reduce_mean(o_cost, axis=-2, keepdims=True)
        o_cost_stdv = tf.sqrt(tf.reduce_sum(tf.square(o_cost - o_cost_mean), axis=-2, keepdims=True) /
                              o_cost.get_shape().as_list()[-2])
        o_activations_cost = beta_a + (o_cost_mean - o_cost) / (o_cost_stdv + keras_backend.epsilon())

        o_activations = tf.sigmoid(inverse_temperature * o_activations_cost)

        return o_mean, o_stdv, o_activations

    @staticmethod
    def e_step(o_mean, o_stdv, o_activations, votes):
        o_p_unit0 = - tf.reduce_sum(tf.square(votes - o_mean) / (2 * tf.square(o_stdv)), axis=-1, keepdims=True)
        o_p_unit2 = - tf.reduce_sum(keras_backend.log(o_stdv + keras_backend.epsilon()), axis=-1, keepdims=True)
        o_p = o_p_unit0 + o_p_unit2
        zz = keras_backend.log(o_activations + keras_backend.epsilon()) + o_p
        rr = tf.nn.softmax(zz, axis=len(zz.get_shape().as_list()) - 2)

        return rr

    @staticmethod
    def kernel_tile(inputs, kernel_size, stride):
        inputs_shape = inputs.shape
        size = inputs_shape[4] * inputs_shape[5] if len(inputs_shape) > 5 else 1
        inputs = tf.reshape(inputs, shape=[-1, inputs_shape[1], inputs_shape[2], inputs_shape[3] * size])

        inputs_shape = inputs.shape
        tile_filter = np.zeros(shape=[kernel_size, kernel_size, inputs_shape[3],
                                      kernel_size * kernel_size], dtype=np.float32)

        for i in range(kernel_size):
            for j in range(kernel_size):
                tile_filter[i, j, :, i * kernel_size + j] = 1.0

        tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)

        outputs = tf.nn.depthwise_conv2d(inputs, tile_filter_op,
                                         strides=[1, stride, stride, 1], padding='VALID')
        outputs_shape = outputs.shape
        outputs = tf.reshape(outputs, shape=[-1, outputs_shape[1], outputs_shape[2],
                                             inputs_shape[3], kernel_size * kernel_size])
        outputs = tf.transpose(outputs, perm=[0, 1, 2, 4, 3])

        return outputs

    @staticmethod
    def mat_transform(inputs, outputs_cap_size, size):
        caps_num_i = int(inputs.shape[1])
        outputs = tf.reshape(inputs, shape=[size, caps_num_i, 1, 4, 4])

        initializer = tf.initializers.truncated_normal(mean=0.0, stddev=1.0)
        w = tf.Variable(lambda: initializer(shape=[1, caps_num_i, outputs_cap_size, 4, 4], dtype=tf.float32),
                        name='w')
        w = tf.tile(w, [size, 1, 1, 1, 1])

        outputs = tf.tile(outputs, [1, 1, outputs_cap_size, 1, 1])

        votes = tf.matmul(outputs, w)
        votes = tf.reshape(votes, shape=[size, caps_num_i, outputs_cap_size, 16])

        return votes

    @staticmethod
    def coord_addition(votes, H, W):
        coordinate_offset_hh = tf.reshape((tf.range(H, dtype=tf.float32) + 0.50) / H, [1, H, 1, 1, 1])
        coordinate_offset_h0 = tf.constant(0.0, shape=[1, H, 1, 1, 1], dtype=tf.float32)
        coordinate_offset_h = tf.stack([coordinate_offset_hh, coordinate_offset_h0] +
                                       [coordinate_offset_h0 for _ in range(14)], axis=-1)  # (1, 4, 1, 1, 1, 16)

        coordinate_offset_ww = tf.reshape((tf.range(W, dtype=tf.float32) + 0.50) / W, [1, 1, W, 1, 1])
        coordinate_offset_w0 = tf.constant(0.0, shape=[1, 1, W, 1, 1], dtype=tf.float32)
        coordinate_offset_w = tf.stack([coordinate_offset_w0, coordinate_offset_ww] +
                                       [coordinate_offset_w0 for _ in range(14)], axis=-1)

        votes = votes + coordinate_offset_h + coordinate_offset_w

        return votes
