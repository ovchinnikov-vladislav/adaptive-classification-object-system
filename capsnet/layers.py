import tensorflow as tf
import tensorflow.keras.backend as keras_backend
from tensorflow.keras import initializers, layers
from capsnet import utils as capsnet_utils


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
        return keras_backend.expand_dims(tf.norm(inputs, axis=-1), axis=-1)

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
                x = keras_backend.sqrt(keras_backend.sum(keras_backend.square(inputs), -1))
                # generate the mask which is a one-hot code.
                # mask.shape=[None, n_classes]=[None, num_capsule]
                mask = keras_backend.one_hot(indices=keras_backend.argmax(x, 1), num_classes=x.shape[1])

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
        #  bias.shape = [None, self.num_capsule, self.input_num_capsule, 1, 1].
        bias = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule,
                            self.input_num_capsule, 1, 1])

        assert self.routings > 0, 'The routings should be > 0.'
        outputs = None
        for i in range(self.routings):
            # Apply softmax to the axis with `num_capsule`
            #  c.shape=[batch_size, num_capsule, input_num_capsule, 1, 1]
            c = tf.nn.softmax(bias, axis=1)

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
                bias = tf.add(bias, agreement)

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


def PrimaryCaps(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Слой первичных капсул, который применяет Conv2D `n_channels` раз и соединяет все капсулы
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
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycaps_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycaps_reshape')(output)
    return layers.Lambda(capsnet_utils.squash, name='primarycaps_squash')(outputs)
