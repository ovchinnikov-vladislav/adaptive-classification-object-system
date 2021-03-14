import tensorflow as tf
from bmstu.capsnets.layers.basic import Length, PrimaryCapsule2D, Capsule, Decoder
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras import Model


def caps_net(shape, num_classes, routings):
    input_capsnet = Input(shape=shape)

    capsules = Conv2D(256, (9, 9), padding='valid', activation=tf.nn.relu)(input_capsnet)
    capsules = PrimaryCapsule2D(num_capsules=32, dim_capsules=8, kernel_size=9, strides=2)(capsules)
    capsules = Capsule(num_capsules=num_classes, dim_capsules=16, routings=routings)(capsules)
    output = Length()(capsules)

    input_decoder = Input(shape=(num_classes,))
    input_noise_decoder = Input(shape=(num_classes, 16))

    train_model = Model([input_capsnet, input_decoder],
                        [output, Decoder(num_classes=num_classes,
                                         output_shape=shape)([capsules, input_decoder])])

    eval_model = Model(input_capsnet,
                       [output, Decoder(num_classes=num_classes,
                                        output_shape=shape)(capsules)])

    noised_digitcaps = tf.keras.layers.Add()([capsules, input_noise_decoder])
    manipulate_model = tf.keras.models.Model([input_capsnet, input_decoder, input_noise_decoder],
                                             Decoder(num_classes=num_classes, output_shape=shape)([noised_digitcaps,
                                                                                                   input_decoder]))

    return train_model, eval_model, manipulate_model
