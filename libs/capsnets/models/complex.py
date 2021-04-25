from libs.capsnets.layers.basic import Decoder
from libs.capsnets.layers.residual import block_caps, Length
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation)
from tensorflow.keras.models import Model
import tensorflow as tf
from libs.resnets.blocks import residual_block, identity_block, conv_block
from libs.utls import BaseModelForTraining


class CapsuleNetworkWith3Level(BaseModelForTraining):
    def create(self, input_shape, **kwargs):
        self.is_decoder = True

        num_classes = kwargs.get('num_classes')
        routings = kwargs.get('routings')

        inputs = Input(shape=input_shape)

        x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(inputs)
        x, capsules_1 = block_caps(x, routings, num_classes, kernel_size=5, strides=2)

        x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
        x, capsules_2 = block_caps(x, routings, num_classes, kernel_size=5, strides=2)

        x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
        x, capsules_3 = block_caps(x, routings, num_classes, kernel_size=3, strides=1)

        capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

        output = Length(name='length')(capsules)

        input_decoder = Input(shape=(num_classes,))

        decoder = Decoder(name='decoder', num_classes=num_classes, dim=18, output_shape=input_shape)

        train_model = Model([inputs, input_decoder], [output, decoder([capsules, input_decoder])], name=self.name)

        eval_model = Model(inputs, [output, decoder(capsules)], name=self.name)

        return train_model, eval_model


class CapsuleNetworkWith4Level(BaseModelForTraining):
    def create(self, input_shape, **kwargs):
        self.is_decoder = True

        num_classes = kwargs.get('num_classes')
        routings = kwargs.get('routings')

        inputs = Input(shape=input_shape)

        x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(inputs)
        x, capsules_01 = block_caps(x, routings, num_classes, kernel_size=5, strides=2)

        x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
        x, capsules_02 = block_caps(x, routings, num_classes, kernel_size=5, strides=2)

        x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
        x, capsules_11 = block_caps(x, routings, num_classes, kernel_size=3, strides=1)

        x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
        x, capsules_12 = block_caps(x, routings, num_classes, kernel_size=2, strides=1)

        capsules0 = tf.keras.layers.Concatenate()([capsules_01, capsules_02])
        capsules1 = tf.keras.layers.Concatenate()([capsules_11, capsules_12])

        capsules = tf.keras.layers.Concatenate()([capsules0, capsules1])

        output = Length(name='length')(capsules)

        input_decoder = Input(shape=(num_classes,))

        decoder = Decoder(name='decoder', num_classes=num_classes, dim=24, output_shape=input_shape)

        train_model = Model([inputs, input_decoder], [output, decoder([capsules, input_decoder])], name=self.name)

        eval_model = Model(inputs, [output, decoder(capsules)], name=self.name)

        return train_model, eval_model


class ResCapsuleNetworkWith3LevelV1(BaseModelForTraining):
    def create(self, input_shape, **kwargs):
        self.is_decoder = True

        num_classes = kwargs.get('num_classes')
        routings = kwargs.get('routings')

        x = inputs = Input(shape=input_shape)

        num_filters = 256

        x = residual_block(x, downsample=True, filters=num_filters)
        x = residual_block(x, downsample=False, filters=num_filters)
        x = residual_block(x, downsample=False, filters=num_filters)

        x, capsules_1 = block_caps(x, routings, num_classes, kernel_size=3, strides=1)

        num_filters = 64

        x = residual_block(x, downsample=True, filters=num_filters)
        x = residual_block(x, downsample=False, filters=num_filters)
        x = residual_block(x, downsample=False, filters=num_filters)

        x, capsules_2 = block_caps(x, routings, num_classes, kernel_size=3, strides=1)

        num_filters = 32

        x = residual_block(x, downsample=True, filters=num_filters)
        x = residual_block(x, downsample=False, filters=num_filters)
        x = residual_block(x, downsample=False, filters=num_filters)
        x = residual_block(x, downsample=False, filters=num_filters)

        x, capsules_3 = block_caps(x, routings, num_classes, kernel_size=2, strides=1)

        capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

        output = Length(name='length')(capsules)

        input_decoder = Input(shape=(num_classes,))

        decoder = Decoder(name='decoder', num_classes=num_classes, dim=18, output_shape=input_shape)

        train_model = Model([inputs, input_decoder], [output, decoder([capsules, input_decoder])], name=self.name)

        eval_model = Model(inputs, [output, decoder(capsules)], name=self.name)

        return train_model, eval_model


class ResCapsuleNetworkWith3LevelV2(BaseModelForTraining):
    def create(self, input_shape, **kwargs):
        self.is_decoder = True

        num_classes = kwargs.get('num_classes')
        routings = kwargs.get('routings')

        inputs = Input(shape=input_shape)

        x = residual_block(inputs, filters=128, downsample=True)
        x = residual_block(x, filters=128)
        x = residual_block(x, filters=128)
        x, capsules_1 = block_caps(x, routings, num_classes, kernel_size=9, strides=1)

        x = residual_block(x, filters=64, downsample=True)
        x = residual_block(x, filters=64)
        x = residual_block(x, filters=64)
        x, capsules_2 = block_caps(x, routings, num_classes, kernel_size=4, strides=1)

        x = residual_block(x, filters=32, downsample=True)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x, capsules_3 = block_caps(x, routings, num_classes, kernel_size=1, strides=1)

        capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

        output = Length(name='length')(capsules)

        input_decoder = Input(shape=(num_classes,))

        decoder = Decoder(name='decoder', num_classes=num_classes, dim=18, output_shape=input_shape)

        train_model = Model([inputs, input_decoder], [output, decoder([capsules, input_decoder])], name=self.name)

        eval_model = Model(inputs, [output, decoder(capsules)], name=self.name)

        return train_model, eval_model


class ResCapsuleNetworkWith3LevelV3(BaseModelForTraining):
    def create(self, input_shape, **kwargs):
        self.is_decoder = True

        num_classes = kwargs.get('num_classes')
        routings = kwargs.get('routings')

        inputs = Input(shape=input_shape)

        x = residual_block(inputs, filters=32, downsample=True)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x, capsules_1 = block_caps(x, routings, num_classes, kernel_size=5, strides=1)

        x = residual_block(x, filters=32, downsample=True)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x, capsules_2 = block_caps(x, routings, num_classes, kernel_size=5, strides=1, padding='same')

        x = residual_block(x, filters=32, downsample=True)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x = residual_block(x, filters=32)
        x, capsules_3 = block_caps(x, routings, num_classes, kernel_size=3, strides=1)

        capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

        output = Length(name='length')(capsules)

        input_decoder = Input(shape=(num_classes,))

        decoder = Decoder(name='decoder', num_classes=num_classes, dim=18, output_shape=input_shape)

        train_model = Model([inputs, input_decoder], [output, decoder([capsules, input_decoder])], name=self.name)

        eval_model = Model(inputs, [output, decoder(capsules)], name=self.name)

        return train_model, eval_model


class Resnet50WithCapsuleNetworkWith3Level(BaseModelForTraining):
    def create(self, input_shape, **kwargs):
        self.is_decoder = True

        num_classes = kwargs.get('num_classes')
        routings = kwargs.get('routings')

        inputs = Input(shape=input_shape)

        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
        x = BatchNormalization(axis=-1, name='bn_conv1')(x)
        x = Activation('relu')(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        _, capsules_1 = block_caps(x, routings, num_classes, kernel_size=5, strides=1, num_capsule=16,
                                   primary_dim_capsule=32)

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        _, capsules_2 = block_caps(x, routings, num_classes, kernel_size=3, strides=1, num_capsule=16,
                                   primary_dim_capsule=32)

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        _, capsules_3 = block_caps(x, routings, num_classes, kernel_size=2, strides=1, num_capsule=16,
                                   primary_dim_capsule=32)

        capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

        output = Length(name='length')(capsules)

        input_decoder = Input(shape=(num_classes,))

        decoder = Decoder(name='is_decoder', num_classes=num_classes, dim=18, output_shape=input_shape)

        train_model = Model([inputs, input_decoder], [output, decoder([capsules, input_decoder])], name=self.name)

        eval_model = Model(inputs, [output, decoder(capsules)], name=self.name)

        return train_model, eval_model
