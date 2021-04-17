import tensorflow as tf
from libs.capsnets.layers.basic import Length, PrimaryCapsule2D, PrimaryCapsule3D, Capsule, Decoder
from tensorflow.keras.layers import Input, Conv2D, Conv3D, BatchNormalization, Activation
from tensorflow.keras import Model
from libs.utls import BaseModelForTraining


class CapsuleNetworkV1(BaseModelForTraining):
    def create(self, input_shape, **kwargs):
        self.is_decoder = True

        num_classes = kwargs.get('num_classes')
        routings = kwargs.get('routings')

        x = inputs = Input(shape=input_shape)

        x = Conv2D(256, (9, 9), padding='valid', activation=tf.nn.relu)(x)
        x = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=9, strides=2, do_reshape=True)(x)
        capsules = Capsule(num_capsules=num_classes, dim_capsules=16, routings=routings)(x)
        output = Length(name='length')(capsules)

        input_decoder = Input(shape=(num_classes,))

        decoder = Decoder(num_classes=num_classes, output_shape=input_shape, name='decoder')

        train_model = Model([inputs, input_decoder],
                            [output, decoder([capsules, input_decoder])])

        eval_model = Model(inputs, [output, decoder(capsules)])

        return train_model, eval_model


class CapsuleNetworkV2(BaseModelForTraining):
    def create(self, input_shape, **kwargs):
        num_classes = kwargs.get('num_classes')
        routings = kwargs.get('routings')

        x = inputs = Input(shape=input_shape)

        x = Conv2D(256, 5, strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        x = PrimaryCapsule2D(num_capsules=9, dim_capsules=8, kernel_size=5, strides=2, padding='same')(x)
        x = PrimaryCapsule2D(num_capsules=4, dim_capsules=4, kernel_size=5, strides=2, do_reshape=True)(x)
        capsules = Capsule(num_capsules=num_classes, dim_capsules=16, routings=routings)(x)
        output = Length(name='length')(capsules)

        input_decoder = Input(shape=(num_classes,))

        decoder = Decoder(num_classes=num_classes, output_shape=input_shape, name='decoder')

        train_model = Model([inputs, input_decoder],
                            [output, decoder([capsules, input_decoder])])

        eval_model = Model(inputs, [output, decoder(capsules)])

        return train_model, eval_model
