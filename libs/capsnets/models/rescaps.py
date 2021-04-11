from libs.capsnets.layers.basic import Capsule, PrimaryCapsule2D, Length, Decoder
from tensorflow.keras.layers import Input, Conv2D, Add, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf
from libs import utls
import numpy as np
from tensorflow.keras.optimizers import Adam
from libs.capsnets.losses import margin_loss


def residual_block(x):
    y = Conv2D(32, 3, activation=tf.nn.relu, padding='same')(x)
    y = Conv2D(32, 3, activation=tf.nn.relu, padding='same')(y)

    x = Conv2D(32, 1, padding='same')(x)
    return Add()([y, x])


def res_caps_v1_net(shape, num_classes, routings):
    input_capsnet = Input(shape=shape)

    resnet_1 = residual_block(input_capsnet)
    resnet_1 = residual_block(resnet_1)
    resnet_1 = residual_block(resnet_1)

    conv1 = Conv2D(32, (9, 9), padding='valid', activation=tf.nn.relu)(resnet_1)
    capsules = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=9, strides=2)(conv1)
    capsules = Capsule(num_capsules=num_classes, dim_capsules=16, routings=routings)(capsules)
    output = Length()(capsules)

    input_decoder = Input(shape=(num_classes,))
    input_noise_decoder = Input(shape=(num_classes, 16))

    train_model = Model([input_capsnet, input_decoder],
                        [output, Decoder(num_classes=num_classes, output_shape=shape)([capsules, input_decoder])])

    eval_model = Model(input_capsnet, [output, Decoder(num_classes=num_classes, output_shape=shape)(capsules)])

    noised_digitcaps = Add()([capsules, input_noise_decoder])
    manipulate_model = Model([input_capsnet, input_decoder, input_noise_decoder],
                             Decoder(num_classes=num_classes, output_shape=shape)([noised_digitcaps,
                                                                                   input_decoder]))

    return train_model, eval_model, manipulate_model


def res_block_caps(x, routings, classes):
    capsules = PrimaryCapsule2D(num_capsules=12, dim_capsules=8, kernel_size=9, strides=2)(x)
    capsules = Capsule(num_capsules=classes, dim_capsules=6, routings=routings)(capsules)

    return capsules, x


def res_caps_v2_net(shape, num_classes, routings):
    input_capsnet = Input(shape=shape)

    x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(input_capsnet)
    capsules_1, x = res_block_caps(x, routings, num_classes)

    x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
    capsules_2, x = res_block_caps(x, routings, num_classes)

    x = Conv2D(32, (9, 9), padding='same', activation=tf.nn.relu)(x)
    capsules_3, x = res_block_caps(x, routings, num_classes)

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
                  loss_weights=[1., 0.392],
                  metrics='accuracy')

    model.fit(x_train, y_train, batch_size=100, epochs=25,
              validation_data=(x_test, y_test))
