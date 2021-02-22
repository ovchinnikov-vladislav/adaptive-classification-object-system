import tensorflow as tf


class PrimaryCapsule3D(tf.keras.Model):
    def __init__(self, capsules, dim_capsules, kernel_size, strides, name=''):
        super(PrimaryCapsule3D, self).__init__(name)

    def call(self, inputs, training=None, mask=None):
        # TODO: Написать первичные 3D капсулы
        pass

    def get_config(self):
        return super(PrimaryCapsule3D, self).get_config()
