from libs.capsnets.layers.basic import Decoder
from libs.capsnets.layers.residual import PrimaryCapsule2D, Capsule, bottleneck, res_block_caps, Length
from tensorflow.keras.layers import Input, Conv2D, Add, BatchNormalization, LeakyReLU, Reshape, Concatenate, Activation
from tensorflow.keras.models import Model
import tensorflow as tf
from libs import utls
import numpy as np
from tensorflow.keras.optimizers import Adam
from libs.capsnets.losses import margin_loss


def relu_bn(inputs):
    relu = tf.keras.layers.ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def residual_block(x, filters, kernel_size=3, downsample=False):
    y = Conv2D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding="same")(x)

    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def res_caps_v1_net(shape, num_classes, routings):
    inputs = Input(shape=shape)
    num_filters = 64

    t = BatchNormalization()(inputs)
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

    # return train_model, eval_model, manipulate_model

    train_model = Model(inputs, output)
    return train_model


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
    inputs = Input(shape=shape)

    num_filters = 32

    x = residual_block(inputs, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)

    x, capsules_1 = res_block_caps(x, routings, num_classes, kernel_size=5, strides=2)

    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)

    x, capsules_2 = res_block_caps(x, routings, num_classes, kernel_size=5, strides=2)

    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)
    x = residual_block(x, downsample=False, filters=num_filters)

    x, capsules_3 = res_block_caps(x, routings, num_classes, kernel_size=3, strides=1)

    capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

    output = Length()(capsules)

    model = Model(inputs, output)

    return model


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


def res50_caps(shape, num_classes, routings):
    input = Input(shape=shape)

    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
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


def res50_caspnet_3level(shape, num_classes, routings):
    input = Input(shape=shape)

    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    _, capsules_1 = res_block_caps(x, routings, num_classes, kernel_size=5, strides=1)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    _, capsules_2 = res_block_caps(x, routings, num_classes, kernel_size=3, strides=1)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    _, capsules_3 = res_block_caps(x, routings, num_classes, kernel_size=2, strides=1)

    capsules = tf.keras.layers.Concatenate()([capsules_1, capsules_2, capsules_3])

    output = Length(name='length')(capsules)

    input_decoder = Input(shape=(num_classes,))

    decoder = Decoder(name='decoder', num_classes=num_classes, dim=18, output_shape=shape)

    train_model = Model([input, input_decoder],
                        [output, decoder([capsules, input_decoder])])

    eval_model = Model(input, [output, decoder(capsules)])

    return train_model, eval_model


def resnet50(shape, num_classes):
    input = Input(shape=shape)

    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = tf.keras.layers.ZeroPadding2D((3, 3))(input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

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

    #    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    # if include_top:
    #     x = Flatten(name='flattenx')(x)
    #     x = Dense(classes, activation='softmax', name='fc1000')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)

    # Create model.
    model = Model(input, x, name='resnet50')

    x = model.get_layer('res5a_branch2a').input
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    #    x = Dense(512, activation='relu',name='fc-1')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')(x)
    return Model(inputs=input, outputs=out)


if __name__ == '__main__':
    # load data
    (x_train, y_train), (x_test, y_test) = utls.load('cifar10')
    # define model

    model = res50_caspnet_3level(shape=(224, 224, 3), num_classes=len(np.unique(np.argmax(y_train, 1))), routings=3)
    model.summary()

    # compile the model
    # model.compile(optimizer=Adam(lr=0.001),
    #               loss=margin_loss,
    #               metrics=['accuracy'])
    #
    # model.fit(x_train, y_train, batch_size=100, epochs=25,
    #           validation_data=(x_test, y_test))

    # model = resnet50(shape=(224, 224, 3), num_classes=len(np.unique(np.argmax(y_train, 1))))
    #
    # model.summary()

    # model.compile(loss='binary_crossentropy',
    #               optimizer=tf.keras.optimizers.SGD(lr=1e-3, momentum=0.8),
    #               metrics=['accuracy'])
    #
    # model.fit(x_train, y_train, batch_size=100, epochs=25,
    #           validation_data=(x_test, y_test))
