import keras.backend as keras_backend
import tensorflow as tf
from keras import initializers, layers
from ml.tf1.capsnet import utl


class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return keras_backend.sqrt(keras_backend.sum(keras_backend.square(inputs), -1) + keras_backend.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = keras_backend.sqrt(keras_backend.sum(keras_backend.square(inputs), -1))
            mask = keras_backend.one_hot(indices=keras_backend.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        masked = keras_backend.batch_flatten(inputs * keras_backend.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def PrimaryCaps(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycaps_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycaps_reshape')(output)
    return layers.Lambda(utl.squash, name='primarycaps_squash')(outputs)


class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.input_num_capsule = self.input_dim_capsule = self.W = None
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = keras_backend.expand_dims(inputs, 1)
        inputs_tiled = keras_backend.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = keras_backend.map_fn(lambda x: keras_backend.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)
        b = tf.zeros(shape=[keras_backend.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        outputs = None
        for i in range(self.routings):
            c = tf.nn.softmax(b, dim=1)
            outputs = utl.squash(keras_backend.batch_dot(c, inputs_hat, [2, 2]))
            if i < self.routings - 1:
                b += keras_backend.batch_dot(outputs, inputs_hat, [2, 3])

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

