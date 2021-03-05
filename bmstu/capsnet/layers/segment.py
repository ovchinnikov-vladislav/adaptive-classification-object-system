import tensorflow as tf
import numpy as np

from tensorflow.keras import backend
from tensorflow.keras import initializers, layers
from tensorflow.python.keras.utils.conv_utils import conv_output_length, deconv_output_length


class ConvCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, num_capsule, num_atoms, strides=1,
                 padding='same', routings=3, kernel_initializer='he_normal',
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(ConvCapsuleLayer, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.input_height = self.input_width = self.input_num_capsule = self.input_num_atoms = self.W = self.b = None

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width, " \
                                      "input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size, self.input_num_atoms,
                                        self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer, name='W')
        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1), name='b')
        self.built = True

    def call(self, inputs, **kwargs):
        input_transposed = tf.transpose(inputs, [3, 0, 1, 2, 4])
        input_shape = backend.shape(input_transposed)
        input_tensor_reshaped = backend.reshape(input_transposed, [input_shape[0] * input_shape[1], self.input_height,
                                                                   self.input_width, self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_num_atoms))

        conv = backend.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
                              padding=self.padding, data_format='channels_last')

        votes_shape = backend.shape(conv)
        _, conv_height, conv_width, _ = conv.shape

        votes = backend.reshape(conv, [input_shape[1], input_shape[0], votes_shape[1],
                                       votes_shape[2], self.num_capsule, self.num_atoms])

        votes.set_shape((None, self.input_num_capsule, conv_height, conv_width, self.num_capsule, self.num_atoms))

        logit_shape = backend.stack([input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule])
        biases_replicated = backend.tile(self.b, [conv_height, conv_width, 1, 1])

        activations = update_routing(votes=votes, biases=biases_replicated, logit_shape=logit_shape,
                                     num_dims=6, input_dim=self.input_num_capsule,
                                     output_dim=self.num_capsule, num_routing=self.routings)

        return activations

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(space[i], self.kernel_size, padding=self.padding,
                                         stride=self.strides, dilation=1)

        return (input_shape[0],) + tuple(new_space) + (self.num_capsule, self.num_atoms)

    def get_config(self):
        config = super(ConvCapsuleLayer, self).get_config()
        config['kernel_size'] = self.kernel_size
        config['num_capsule'] = self.num_capsule
        config['num_atoms'] = self.num_atoms
        config['strides'] = self.strides
        config['padding'] = self.padding
        config['routings'] = self.routings
        config['kernel_initializer'] = initializers.serialize(self.kernel_initializer)

        return config


class Length(layers.Layer):
    def __init__(self, num_classes, seg=True, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(Length, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        self.seg = seg

    def call(self, inputs, **kwargs):
        if inputs.ndims == 5:
            assert inputs.shape[-2] == 1, 'Error: must have capsules = 1 going into Length'
            inputs = backend.squeeze(inputs, axis=-2)
        return backend.expand_dims(tf.norm(inputs, axis=-1), axis=-1)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 5:
            input_shape = input_shape[0:-2] + input_shape[-1:]
        if self.seg:
            return input_shape[:-1] + (self.num_classes,)
        else:
            return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        config['num_classes'] = self.num_classes
        config['seg'] = self.seg
        return config


class Mask(layers.Layer):
    def __init__(self, resize_masks=False, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(Mask, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.resize_masks = resize_masks

    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            input, mask = inputs
            _, hei, wid, _, _ = input.shape
            if self.resize_masks:
                mask = tf.image.resize(mask, (hei.value, wid.value), method=tf.image.ResizeMethod.BICUBIC)
            mask = backend.expand_dims(mask, -1)
            if input.ndims == 3:
                masked = backend.batch_flatten(mask * input)
            else:
                masked = mask * input
        else:
            if inputs.ndims == 3:
                x = backend.sqrt(backend.sum(backend.square(inputs), -1))
                mask = backend.one_hot(indices=backend.argmax(x, 1), num_classes=x.shape[1])
                masked = backend.batch_flatten(backend.expand_dims(mask, -1) * inputs)
            else:
                masked = inputs

        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            if len(input_shape[0]) == 3:
                return tuple([None, input_shape[0][1] * input_shape[0][2]])
            else:
                return input_shape[0]
        else:
            if len(input_shape) == 3:
                return tuple([None, input_shape[1] * input_shape[2]])
            else:
                return input_shape

    def get_config(self):
        config = super(Mask, self).get_config()
        config['resize_masks'] = self.resize_masks
        return config
