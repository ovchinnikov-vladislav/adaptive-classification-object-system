from tensorflow.keras.layers import (BatchNormalization, ReLU, Conv2D, Activation,
                                     Add, Dropout, DepthwiseConv2D, Concatenate)
from tensorflow.keras import regularizers
from tensorflow.keras.backend import relu


def relu_bn(inputs):
    bn = BatchNormalization(axis=-1)(inputs)
    relu = ReLU()(bn)
    return relu


def bottleneck(inputs, filters, kernel, e, stride, activation):
    def _relu6(x):
        return relu(x, max_value=6.0)

    def _hard_swish(x):
        return x * relu(x + 3.0, max_value=6.0) / 6.0

    channel_axis = -1
    input_shape = inputs.shape
    t_channel = input_shape[channel_axis] * e

    x = Conv2D(t_channel, (1, 1), padding='same', strides=(1, 1))(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    if activation == 'hard_swish':
        x = Activation(_hard_swish)(x)
    if activation == 'relu':
        x = Activation(_relu6)(x)

    x = DepthwiseConv2D(kernel, strides=(stride, stride), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if activation == 'hard_swish':
        x = Activation(_hard_swish)(x)
    if activation == 'relu':
        x = Activation(_relu6)(x)
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    out = Concatenate(axis=-1)([inputs, x])
    out = Dropout(0.3)(out)

    return out


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
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a',
               kernel_regularizer=regularizers.l2(0.0001))(input_tensor)
    x = relu_bn(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = relu_bn(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    x = Dropout(0.5)(x)

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
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a', kernel_regularizer=regularizers.l2(0.0001))(input_tensor)
    x = relu_bn(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = relu_bn(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    x = Dropout(0.5)(x)

    return x


def residual_block(x, filters, kernel_size=3, downsample=False):
    y = Conv2D(kernel_size=kernel_size,
               kernel_regularizer=regularizers.l2(0.0001),
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

    out = Dropout(0.5)(out)
    return out
