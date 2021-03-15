from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from bmstu.yolo3.utils import compose
from bmstu.capsnets.layers.basic import PrimaryCapsule2D, Capsule, Length


def DarknetConv2D(filters, kernel_size, strides=(1, 1), use_bias=True):
    return Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  use_bias=use_bias,
                  kernel_regularizer=l2(5e-4),
                  padding='valid' if strides == (2, 2) else 'same')


def DarknetConv2DBNLeakyRelu(filters, kernel_size, strides=(1, 1)):
    return compose(DarknetConv2D(filters=filters, kernel_size=kernel_size,
                                 strides=strides, use_bias=False), BatchNormalization(), LeakyReLU(alpha=0.1))


# class DarknetConv2D(Layer):
#     def __init__(self, filters, kernel_size, strides=(1, 1), use_bias=True, **kwargs):
#         super(DarknetConv2D, self).__init__(**kwargs)
#         self.darknet_conv_2d = Conv2D(filters=filters,
#                                       kernel_size=kernel_size,
#                                       strides=strides,
#                                       use_bias=use_bias,
#                                       kernel_regularizer=l2(5e-4),
#                                       padding='valid' if strides == (2, 2) else 'same')
#
#     def call(self, inputs, **kwargs):
#         return self.darknet_conv_2d(inputs)
#
#     def get_config(self):
#         return self.darknet_conv_2d.get_config()
#
#
# class DarknetConv2DBNLeakyRelu(Layer):
#     def __init__(self, filters, kernel_size, strides=(1, 1), **kwargs):
#         super(DarknetConv2DBNLeakyRelu, self).__init__(**kwargs)
#         self.darknet_conv_2d = DarknetConv2D(filters=filters, kernel_size=kernel_size,
#                                              strides=strides, use_bias=False)
#         self.batch_norm = BatchNormalization()
#         self.leaky_relu = LeakyReLU(alpha=0.1)
#
#     def call(self, inputs, **kwargs):
#         return compose(self.darknet_conv_2d, self.batch_norm, self.leaky_relu)(inputs)
#
#     def get_config(self):
#         return self.darknet_conv_2d.get_config()


def residual_block_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2DBNLeakyRelu(filters=num_filters, kernel_size=(3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(DarknetConv2DBNLeakyRelu(filters=num_filters // 2, kernel_size=(1, 1)),
                    DarknetConv2DBNLeakyRelu(filters=num_filters, kernel_size=(3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    x = DarknetConv2DBNLeakyRelu(filters=32, kernel_size=(3, 3))(x)
    x = residual_block_body(x, 64, 1)
    x = residual_block_body(x, 128, 2)
    x = residual_block_body(x, 256, 8)
    x = residual_block_body(x, 512, 8)
    x = residual_block_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    x = compose(DarknetConv2DBNLeakyRelu(filters=num_filters, kernel_size=(1, 1)),
                DarknetConv2DBNLeakyRelu(filters=num_filters * 2, kernel_size=(3, 3)),
                DarknetConv2DBNLeakyRelu(filters=num_filters, kernel_size=(1, 1)),
                DarknetConv2DBNLeakyRelu(filters=num_filters * 2, kernel_size=(3, 3)),
                DarknetConv2DBNLeakyRelu(filters=num_filters, kernel_size=(1, 1)))(x)

    # TODO: y - заменить на CapsuleNets (обычную, а также попробовать остаточную)
    # y = compose(DarknetConv2DBNLeakyRelu(filters=num_filters * 2, kernel_size=(3, 3)),
    #             DarknetConv2D(filters=out_filters, kernel_size=(1, 1)))(x)

    y = PrimaryCapsule2D(num_capsules=num_filters // 2, dim_capsules=2, kernel_size=9, strides=2)(x)
    y = Capsule(num_capsules=out_filters, dim_capsules=16, routings=1)(y)
    y = Length()(y)

    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(DarknetConv2DBNLeakyRelu(filters=256, kernel_size=(1, 1)),
                UpSampling2D(2))(x)
    # add 18
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(DarknetConv2DBNLeakyRelu(filters=128, kernel_size=(1, 1)),
                UpSampling2D(2))(x)
    # add 10
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    model = Model(inputs, [y1, y2, y3])
    model.summary()

    return model

