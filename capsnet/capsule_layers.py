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
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1) + keras_backend.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
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
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = tf.one_hot(indices=tf.argmax(x, 1), depth=x.shape[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = keras_backend.batch_flatten(inputs * tf.expand_dims(mask, -1))
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
        self.input_num_capsule = None
        self.input_dim_capsule = None
        self.W = None

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Матрица преобразования, от каждой входной капсулы до каждой выходной капсулы имеется свой уникальный вес
        # как в полносвязном (Dense) слое
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule, 1]
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 1), -1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule, 1]
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule, 1]
        # Regard the first two dimensions as `batch` dimension, then
        # matmul(W, x): [..., dim_capsule, input_dim_capsule] x [..., input_dim_capsule, 1] -> [..., dim_capsule, 1].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = tf.squeeze(tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled))

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, 1, self.input_num_capsule].
        b = tf.zeros(shape=[inputs.shape[0], self.num_capsule, 1, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        outputs = None
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, 1, input_num_capsule]
            c = tf.nn.softmax(b, axis=1)

            # c.shape = [batch_size, num_capsule, 1, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [..., 1, input_num_capsule] x [..., input_num_capsule, dim_capsule] -> [..., 1, dim_capsule].
            # outputs.shape=[None, num_capsule, 1, dim_capsule]
            outputs = capsnet_utils.squash(tf.matmul(c, inputs_hat))  # [None, 10, 1, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, 1, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension, then
                # matmal:[..., 1, dim_capsule] x [..., input_num_capsule, dim_capsule]^T -> [..., 1, input_num_capsule].
                # b.shape=[batch_size, num_capsule, 1, input_num_capsule]
                b += tf.matmul(outputs, inputs_hat, transpose_b=True)
        # End: Routing algorithm -----------------------------------------------------------------------#

        return tf.squeeze(outputs)

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
