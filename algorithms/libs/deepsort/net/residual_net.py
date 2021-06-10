from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout
import tensorflow as tf


def create_link(incoming, network_builder, nonlinearity='elu',
                weights_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                regularizer=None, is_first=False):
    if is_first:
        network = incoming
    else:
        network = BatchNormalization()(incoming)
        if nonlinearity == 'elu':
            network = tf.keras.layers.ELU()(incoming)

    pre_block_network = incoming
    post_block_network = network_builder(network)

    incoming_dim = pre_block_network.shape[-1]
    outgoing_dim = post_block_network.shape[-1]

    if incoming_dim != outgoing_dim:
        assert outgoing_dim == 2 * incoming_dim, "%d != %d" % (outgoing_dim, 2 * incoming_dim)

        projection = Conv2D(outgoing_dim, 1, 2, padding='same', activation=None,
                            kernel_initializer=weights_initializer, bias_initializer=None,
                            kernel_regularizer=regularizer)(incoming)
        network = tf.keras.layers.Add()([projection, post_block_network])
    else:
        network = tf.keras.layers.Add()([incoming, post_block_network])

    return network


def create_inner_block(incoming, nonlinearity='elu',
                       weights_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                       bias_initializer=tf.zeros_initializer(), regularizer=None, increase_dim=False):
    n = incoming.shape[-1]
    stride = 1
    if increase_dim:
        n *= 2
        stride = 2

    incoming = Conv2D(n, (3, 3), stride, activation=None, padding='same',
                      kernel_initializer=weights_initializer,
                      bias_initializer=bias_initializer,
                      kernel_regularizer=regularizer)(incoming)
    incoming = BatchNormalization()(incoming)
    if nonlinearity == 'elu':
        incoming = tf.keras.layers.ELU()(incoming)

    incoming = Dropout(0.6)(incoming)

    incoming = Conv2D(n, [3, 3], 1, activation=None, padding='same',
                      kernel_initializer=weights_initializer,
                      bias_initializer=bias_initializer,
                      kernel_regularizer=regularizer)(incoming)

    return incoming


def residual_block(incoming, nonlinearity='elu',
                   weights_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
                   bias_initializer=tf.zeros_initializer(), regularizer=None, increase_dim=False, is_first=False):
    def network_builder(x):
        return create_inner_block(x, nonlinearity, weights_initializer, bias_initializer, regularizer, increase_dim)

    return create_link(incoming, network_builder, nonlinearity, weights_initializer, regularizer, is_first)
