import tensorflow as tf
import numpy as np

from tensorflow.keras import backend
from tensorflow.keras import initializers, layers
from tensorflow.python.keras.utils.conv_utils import conv_output_length, deconv_output_length


class Length(layers.Layer):
    def __init__(self, classes, seg=True, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(Length, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        if classes == 2:
            self.classes = 1
        else:
            self.classes = classes
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
            return input_shape[:-1] + (self.classes,)
        else:
            return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        config['classes'] = self.classes
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
