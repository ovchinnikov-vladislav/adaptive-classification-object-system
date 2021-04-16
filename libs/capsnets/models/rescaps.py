from libs.capsnets.layers.basic import Decoder
from libs.capsnets.layers.residual import PrimaryCapsule2D, Capsule, bottleneck, res_block_caps, Length
from tensorflow.keras.layers import (Input, Conv2D, Add, BatchNormalization, LeakyReLU,
                                     Reshape, Concatenate, Activation, Dropout)
from tensorflow.keras.models import Model
import tensorflow as tf
from libs import utls
import numpy as np
from tensorflow.keras.optimizers import Adam
from libs.capsnets.losses import margin_loss
from libs.resnets.blocks import relu_bn, residual_block, identity_block, conv_block


def res_caps_v1_net(shape, num_classes, routings):
    x = inputs = Input(shape=shape)
    num_filters = 64

    t = BatchNormalization()(x)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    _, capsules = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=4, strides=1)(t)
    capsules = Capsule(num_capsules=num_classes, dim_capsules=16, routings=routings)(capsules)
    output = Length()(capsules)

    input_decoder = Input(shape=(num_classes,))

    train_model = Model([inputs, input_decoder],
                        [output, Decoder(num_classes=num_classes, output_shape=shape)([capsules, input_decoder])])

    eval_model = Model(inputs, [output, Decoder(num_classes=num_classes, output_shape=shape)(capsules)])

    return train_model, eval_model


def capsnet_3level(shape, num_classes, routings):
    input_capsnet = Input(shape=shape)

    x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(input_capsnet)
    x, capsules_1 = res_block_caps(x, routings, num_classes, kernel_size=5, strides=2)

    x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
    x, capsules_2 = res_block_caps(x, routings, num_classes, kernel_size=5, strides=2)

    x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
    x, capsules_3 = res_block_caps(x, routings, num_classes, kernel_size=3, strides=1)

    capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

    output = Length()(capsules)

    model = Model(input_capsnet, output)

    return model


def res_capsnet_3level(shape, num_classes, routings):
    x = inputs = Input(shape=shape)

    num_filters = 256

    x = residual_block(x, downsample=True, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)

    x, capsules_1 = res_block_caps(x, routings, num_classes, kernel_size=3, strides=1)

    num_filters = 64

    x = residual_block(x, downsample=True, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)

    x, capsules_2 = res_block_caps(x, routings, num_classes, kernel_size=3, strides=1)

    num_filters = 32

    x = residual_block(x, downsample=True, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)

    x, capsules_3 = res_block_caps(x, routings, num_classes, kernel_size=2, strides=1)

    capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

    output = Length()(capsules)

    input_decoder = Input(shape=(num_classes,))

    decoder = Decoder(name='decoder', num_classes=num_classes, dim=18, output_shape=shape)

    train_model = Model([inputs, input_decoder],
                        [output, decoder([capsules, input_decoder])])

    eval_model = Model(inputs, [output, decoder(capsules)])

    return train_model, eval_model


def res_caps_v2_net(shape, num_classes, routings):
    input_capsnet = Input(shape=shape)

    x = residual_block(input_capsnet, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x, capsules_1 = res_block_caps(x, routings, num_classes, kernel_size=9, strides=2)

    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x, capsules_2 = res_block_caps(x, routings, num_classes, kernel_size=6, strides=2)

    x = residual_block(x, filters=32)
    x = residual_block(x, filters=32)
    x = residual_block(x, filters=32)
    x, capsules_3 = res_block_caps(x, routings, num_classes, kernel_size=3, strides=1)

    capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

    output = Length()(capsules)

    model = Model(input_capsnet, output)

    return model


def capsnet_4level(shape, num_classes, routings):
    input_capsnet = Input(shape=shape)

    x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(input_capsnet)
    x, capsules_01 = res_block_caps(x, routings, num_classes, kernel_size=5, strides=2)

    x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
    x, capsules_02 = res_block_caps(x, routings, num_classes, kernel_size=5, strides=2)

    x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
    x, capsules_11 = res_block_caps(x, routings, num_classes, kernel_size=3, strides=1)

    x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
    x, capsules_12 = res_block_caps(x, routings, num_classes, kernel_size=2, strides=1)

    capsules0 = tf.keras.layers.Concatenate()([capsules_01, capsules_02])
    capsules1 = tf.keras.layers.Concatenate()([capsules_11, capsules_12])

    capsules = tf.keras.layers.Concatenate()([capsules0, capsules1])

    output = Length()(capsules)

    train_model = Model(input_capsnet, output)

    return train_model


def residual_primary_caps_block(x, kernel_size=5, downsample=False):
    _, capsules = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=kernel_size, strides=1)(x)
    _, capsules = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=kernel_size, strides=1)(x)

    if downsample:
        _, capsules = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=kernel_size, strides=1)(x)
    out = Add()([x, capsules])
    return out


def res_primary_caps(shape, num_classes, routings):
    input_capsnet = Input(shape=shape)

    capsules = Conv2D(256, (9, 9), padding='valid', activation=tf.nn.relu)(input_capsnet)
    _, capsules = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=9, strides=2)(capsules)
    # x = residual_primary_caps_block(capsules)
    # x = residual_primary_caps_block(x)
    # x = residual_primary_caps_block(x)
    # x = residual_primary_caps_block(x)
    # x = residual_primary_caps_block(x)

    capsules = Capsule(num_capsules=num_classes, dim_capsules=16, routings=routings)(capsules)
    output = Length()(capsules)

    train_model = Model(input_capsnet, output)

    return train_model


def res_capsnet(shape, num_classes, routings):
    x = inputs = Input(shape=shape)

    x = Conv2D(256, (9, 9), padding='valid', activation=tf.nn.relu)(x)
    capsules = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=9, strides=2)(x)
    capsules = Capsule(num_capsules=num_classes, dim_capsules=16, routings=routings)(capsules)
    output = Length(name='length')(capsules)

    input_decoder = Input(shape=(num_classes,))

    decoder = Decoder(name='decoder', num_classes=num_classes, dim=18, output_shape=shape)

    train_model = Model([inputs, input_decoder],
                        [output, decoder([capsules, input_decoder])])

    eval_model = Model(inputs, [output, decoder(capsules)])

    return train_model, eval_model


def res50_caps(shape, num_classes, routings):
    input = Input(shape=shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(input)
    x = BatchNormalization(axis=-1, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    _, capsules = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=2, strides=2)(x)
    capsules = Capsule(num_capsules=num_classes, dim_capsules=16, routings=routings)(capsules)
    output = Length()(capsules)

    train_model = Model(input, output)

    return train_model


def res50_capsnet_3level(shape, num_classes, routings):
    input = Input(shape=shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_regularizer=tf.keras.regularizers.l2(0.001))(input)
    x = BatchNormalization(axis=-1, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    _, capsules_1 = res_block_caps(x, routings, num_classes, primary_dim_capsule=32, num_capsule=16, kernel_size=5, strides=1)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    _, capsules_2 = res_block_caps(x, routings, num_classes, primary_dim_capsule=32, num_capsule=16, kernel_size=3, strides=1)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    _, capsules_3 = res_block_caps(x, routings, num_classes, primary_dim_capsule=32, num_capsule=16, kernel_size=2, strides=1)

    capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

    output = Length(name='length')(capsules)

    input_decoder = Input(shape=(num_classes,))

    decoder = Decoder(name='decoder', num_classes=num_classes, dim=18, output_shape=shape)

    train_model = Model([input, input_decoder],
                        [output, decoder([capsules, input_decoder])])

    eval_model = Model(input, [output, decoder(capsules)])

    return train_model, eval_model


if __name__ == '__main__':
    # load data
    (x_train, y_train), (x_test, y_test) = utls.load('mnist')
    # define model

    model, eval_model = res50_capsnet_3level(shape=x_train.shape[1:], num_classes=len(np.unique(np.argmax(y_train, 1))),
                                             routings=3)
    model.summary(line_length=200)

    # compile the model
    model.compile(optimizer=Adam(lr=0.001),
                  loss=[margin_loss, 'mse'],
                  metrics=['accuracy'])

    model.fit([x_train, y_train], [y_train, x_train], batch_size=100, epochs=25,
              validation_data=([x_test, y_test], [y_test, x_test]))
