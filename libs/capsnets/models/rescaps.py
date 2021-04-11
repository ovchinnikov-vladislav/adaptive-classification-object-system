from libs.capsnets.layers.basic import Decoder
from libs.capsnets.layers.residual import PrimaryCapsule2D, Capsule, bottleneck, res_block_caps, Length
from tensorflow.keras.layers import Input, Conv2D, Add, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf
from libs import utls
import numpy as np
from tensorflow.keras.optimizers import Adam
from libs.capsnets.losses import margin_loss


def residual_block(x, filters=32, kernels=3):
    y = Conv2D(filters, kernels, activation=tf.nn.relu, padding='same')(x)
    y = Conv2D(filters, kernels, activation=tf.nn.relu, padding='same')(y)

    x = Conv2D(filters, 1, padding='same')(x)
    return Add()([y, x])


def res_caps_v1_net(shape, num_classes, routings):
    input_capsnet = Input(shape=shape)

    x = residual_block(input_capsnet)
    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)

    _, capsules = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=9, strides=2)(x)
    capsules = Capsule(num_capsules=num_classes, dim_capsules=16, routings=routings)(capsules)
    output = Length()(capsules)

    # input_decoder = Input(shape=(num_classes,))
    # input_noise_decoder = Input(shape=(num_classes, 16))
    #
    # train_model = Model([input_capsnet, input_decoder],
    #                     [output, Decoder(num_classes=num_classes, output_shape=shape)([capsules, input_decoder])])
    #
    # eval_model = Model(input_capsnet, [output, Decoder(num_classes=num_classes, output_shape=shape)(capsules)])
    #
    # noised_digitcaps = Add()([capsules, input_noise_decoder])
    # manipulate_model = Model([input_capsnet, input_decoder, input_noise_decoder],
    #                          Decoder(num_classes=num_classes, output_shape=shape)([noised_digitcaps,
    #                                                                                input_decoder]))

    #return train_model, eval_model, manipulate_model

    train_model = Model(input_capsnet, output)
    return train_model


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


def res_caps_v3_net(shape, num_classes, routings):
    input_capsnet = Input(shape=shape)

    x = bottleneck(input_capsnet, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='hard_swish')

    x, capsules_1 = res_block_caps(x, routings, num_classes, kernel_size=5, strides=2)

    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='hard_swish')

    x, capsules_2 = res_block_caps(x, routings, num_classes, kernel_size=5, strides=2)

    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')
    x = bottleneck(x, 32, (3, 3), e=1, stride=1, activation='relu')

    x, capsules_3 = res_block_caps(x, routings, num_classes, kernel_size=3, strides=1)

    capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

    output = Length()(capsules)

    model = Model(input_capsnet, output)

    return model


if __name__ == '__main__':
    # load data
    (x_train, y_train), (x_test, y_test) = utls.load('cifar10')
    # define model

    model = res_caps_v2_net(shape=x_train.shape[1:],
                            num_classes=len(np.unique(np.argmax(y_train, 1))),
                            routings=3)

    model.summary()

    # compile the model
    model.compile(optimizer=Adam(lr=0.001),
                  loss=margin_loss,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=100, epochs=25,
              validation_data=(x_test, y_test))
