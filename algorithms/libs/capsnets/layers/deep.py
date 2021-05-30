from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, initializers, regularizers, constraints
from tensorflow.python.keras.utils import conv_utils
from libs.capsnets.utils import own_batch_dot
from tensorflow.keras.backend import epsilon


def squeeze(vectors):
    vectors_q = tf.reduce_sum(tf.square(vectors), axis=-1, keepdims=True)
    return (vectors_q / (1 + vectors_q)) * (vectors / tf.sqrt(vectors_q + epsilon()))


class ConvertToCapsule(layers.Layer):
    def __init__(self, **kwargs):
        super(ConvertToCapsule, self).__init__(**kwargs)
        # self.input_spec = InputSpec(min_ndim=2)

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape.insert(len(output_shape), 1)
        return tuple(output_shape)

    def get_config(self):
        return super(ConvertToCapsule, self).get_config()


class FlattenCapsule(layers.Layer):
    def __init__(self, **kwargs):
        super(FlattenCapsule, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=4)

    def call(self, inputs, **kwargs):
        shape = inputs.shape
        return tf.reshape(inputs, shape=(-1, np.prod(shape[1:-1]), shape[-1]))

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "FlattenCaps" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                                                              'Make sure to pass a complete "input_shape" '
                                                              'or "batch_input_shape" argument to the first '
                                                              'layer in your model.')
        return input_shape[0], np.prod(input_shape[1:-1]), input_shape[-1]


class CapsuleToScalars(layers.Layer):
    def __init__(self, **kwargs):
        super(CapsuleToScalars, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=3)

    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs + epsilon()), axis=-1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]


class Convolutional2DCapsule(layers.Layer):
    def __init__(self, num_capsules, num_neurons, kernel_size=(3, 3), strides=(1, 1), routings=1, b_alphas=None,
                 padding='same', data_format='channels_last',
                 dilation_rate=(1, 1), kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, **kwargs):
        super(Convolutional2DCapsule, self).__init__(**kwargs)
        if b_alphas is None:
            b_alphas = [8, 8, 8]
        rank = 2
        self.num_capsules = num_capsules  # Number of capsules in layer J
        self.num_neurons = num_neurons  # Number of neurons in a capsule in J
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.routings = routings
        self.b_alphas = b_alphas
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = dilation_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = layers.InputSpec(ndim=rank + 3)
        self.h_i = None
        self.w_i = None
        self.ch_i = None
        self.n_i = None
        self.h_j = None
        self.w_j = None
        self.ah_j = None
        self.aw_j = None
        self.w_shape = None
        self.w = None

    def build(self, input_shape):

        self.h_i, self.w_i, self.ch_i, self.n_i = input_shape[1:5]

        self.h_j, self.w_j = [conv_utils.conv_output_length(input_shape[i + 1],
                                                            self.kernel_size[i],
                                                            padding=self.padding,
                                                            stride=self.strides[i],
                                                            dilation=self.dilation_rate[i]) for i in (0, 1)]

        self.ah_j, self.aw_j = [conv_utils.conv_output_length(input_shape[i + 1],
                                                              self.kernel_size[i],
                                                              padding=self.padding,
                                                              stride=1,
                                                              dilation=self.dilation_rate[i]) for i in (0, 1)]

        self.w_shape = self.kernel_size + (self.ch_i, self.n_i,
                                           self.num_capsules, self.num_neurons)

        self.w = self.add_weight(shape=self.w_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        self.built = True

    def call(self, inputs, **kwargs):
        outputs = None
        if self.routings == 1:
            # if there is no routing (and this is so when routings is 1 and all c are equal)
            # then this is a common convolution
            outputs = backend.conv2d(
                tf.reshape(inputs, (-1, self.h_i, self.w_i, self.ch_i * self.n_i)),
                tf.reshape(self.w, self.kernel_size + (self.ch_i * self.n_i, self.num_capsules * self.num_neurons)),
                data_format='channels_last',
                strides=self.strides,
                padding=self.padding,
                dilation_rate=self.dilation_rate)

            outputs = squeeze(tf.reshape(outputs, shape=(-1, self.h_j, self.w_j, self.num_capsules, self.num_neurons)))

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.h_j, self.w_j, self.num_capsules, self.num_neurons

    def get_config(self):
        return super(Convolutional2DCapsule, self).get_config()
