import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Input, Conv2D, Flatten, Dropout, Dense,
                                     MaxPooling2D, Layer, BatchNormalization, Lambda)
from tensorflow.keras.models import Model
from . import residual_net


class Logits(Layer):
    def __init__(self, num_classes, feature_dim, **kwargs):
        super(Logits, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.w = None
        self.scale = None

    def build(self, input_shape):
        self.w = self.add_weight(shape=[self.feature_dim, self.num_classes],
                                 dtype=tf.float32,
                                 initializer=TruncatedNormal(stddev=0.1),
                                 trainable=True)
        self.scale = self.add_weight(shape=(),
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.),
                                     regularizer=l2(1e-1))
        self.scale = tf.nn.softplus(self.scale)

        self.built = True

    def call(self, inputs, **kwargs):
        weights_normed = tf.nn.l2_normalize(self.w, axis=0)

        return self.scale * tf.matmul(inputs, weights_normed)

    def get_config(self):
        return super(Logits, self).get_config()


def create_network(inputs_shape, num_classes=None, add_logits=True, weight_decay=1e-8):
    nonlinearity = tf.nn.elu
    conv_weight_init = TruncatedNormal(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = l2(weight_decay)
    fc_weight_init = TruncatedNormal(stddev=1e-3)
    fc_bias_init = tf.zeros_initializer()
    fc_regularizer = l2(weight_decay)

    network = inputs = Input(shape=inputs_shape)
    network = Conv2D(32, (3, 3), strides=1, padding='same',
                     kernel_initializer=conv_weight_init,
                     bias_initializer=conv_bias_init,
                     kernel_regularizer=conv_regularizer)(network)
    network = BatchNormalization()(network)
    network = nonlinearity(network)

    network = Conv2D(32, (3, 3), strides=1, padding='same',
                     kernel_initializer=conv_weight_init,
                     bias_initializer=conv_bias_init,
                     kernel_regularizer=conv_regularizer)(network)
    network = BatchNormalization()(network)
    network = nonlinearity(network)

    network = MaxPooling2D((3, 3), (2, 2), padding='same')(network)

    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=False, is_first=True)
    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=False)

    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=True)
    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=False)

    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=True)
    network = residual_net.residual_block(network, nonlinearity, conv_weight_init, conv_bias_init,
                                          conv_regularizer, increase_dim=False)

    feature_dim = network.shape[-1]
    network = Flatten()(network)

    network = Dropout(0.6)(network)
    network = Dense(feature_dim, kernel_regularizer=fc_regularizer,
                    kernel_initializer=fc_weight_init,
                    bias_initializer=fc_bias_init)(network)
    network = BatchNormalization()(network)
    network = nonlinearity(network)

    features = network

    features = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), name='features_output')(features)

    if add_logits:
        logits = Logits(num_classes, feature_dim, name='logits_output')(features)
        return Model(inputs, [features, logits])
    else:
        return Model(inputs, features)


def preprocess(image, is_training=False, input_is_bgr=False):
    if input_is_bgr:
        image = image[:, :, ::-1]  # BGR to RGB
    image = tf.divide(tf.cast(image, tf.float32), 255.0)
    if is_training:
        image = tf.image.random_flip_left_right(image)
    return image
