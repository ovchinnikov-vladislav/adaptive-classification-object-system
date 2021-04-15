from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation, Dense,
                                     ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D, Dropout)
from tensorflow.keras.models import Model
from libs.resnets.blocks import conv_block, identity_block


def resnet50(shape, num_classes):
    inputs = Input(shape=shape)

    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

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
    model = Model(inputs, x, name='resnet50')

    x = model.get_layer('res5a_branch2a').input
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    #    x = Dense(512, activation='relu',name='fc-1')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax', name='output_layer')(x)
    return Model(inputs=inputs, outputs=out)
