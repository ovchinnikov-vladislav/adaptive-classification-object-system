import tensorflow as tf
from tensorflow.keras import activations
from bmstu.capsnet.utls import squash
import numpy as np


class GammaCapsule(tf.keras.layers.Layer):
    def __init__(self, capsules, dim_capsules, routings, **kwargs):
        super(GammaCapsule, self).__init__(**kwargs)
        self.capsules = capsules
        self.dim_capsules = dim_capsules
        self.routings = routings
        self.W = self.bias = None

    def build(self, input_shape):
        w_init = tf.keras.initializers.TruncatedNormal(stddev=0.1)
        self.W = tf.Variable(initial_value=w_init(
            shape=(1, self.capsules, input_shape[1], self.dim_capsules, input_shape[2]),
            dtype='float32'), trainable=True)

        bias_init = tf.constant_initializer(0.1)
        self.bias = tf.Variable(initial_value=bias_init(
            shape=(1, self.capsules, self.dim_capsules),
            dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        """
            param: u - (batch_size, in_caps, in_dim)
        """
        batch_size = inputs.shape[0]
        u_norm = tf.norm(inputs, axis=-1)  # (batch_size, in_caps)

        # Reshape u into (batch_size, out_caps, in_caps, out_dim, in_dim)
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.expand_dims(inputs, 3)
        inputs = tf.tile(inputs, [1, self.capsules, 1, 1, 1])
        inputs = tf.tile(inputs, [1, 1, 1, self.dim_capsules, 1])

        # Duplicate transformation matrix for each batch
        w = tf.tile(self.W, [batch_size, 1, 1, 1, 1])

        # Dotwise product between u and w to get all votes
        # shape = (batch_size, out_caps, in_caps, out_dim)
        u_hat = tf.reduce_sum(inputs * w, axis=-1)

        # Ensure that ||u_hat|| <= ||v_i||
        u_hat_norm = tf.norm(u_hat, axis=-1, keepdims=True)
        u_norm = tf.expand_dims(u_norm, axis=1)
        u_norm = tf.expand_dims(u_norm, axis=3)
        u_norm = tf.tile(u_norm, [1, self.capsules, 1, self.dim_capsules])
        new_u_hat_norm = tf.math.minimum(u_hat_norm, u_norm)
        u_hat = u_hat / u_hat_norm * new_u_hat_norm

        # Scaled-distance-agreement routing
        bias = tf.tile(self.bias, [batch_size, 1, 1])
        b_ij = tf.zeros(shape=[batch_size, self.capsules, inputs.shape[2], 1])

        v_j = c_ij = None
        for r in range(self.routings):
            c_ij = tf.nn.softmax(b_ij, axis=1)
            c_ij_tiled = tf.tile(c_ij, [1, 1, 1, self.dim_capsules])
            s_j = tf.reduce_sum(c_ij_tiled * u_hat, axis=2) + bias
            v_j = squash(s_j)

            if r < self.routings - 1:
                v_j = tf.expand_dims(v_j, 2)
                v_j = tf.tile(v_j, [1, 1, inputs.shape[2], 1])  # (batch_size, out_caps, in_caps, out_dim)

                # Calculate scale factor t
                p_p = 0.9
                d = tf.norm(v_j - u_hat, axis=-1, keepdims=True)
                d_o = tf.reduce_mean(tf.reduce_mean(d))
                d_p = d_o * 0.5
                t = tf.constant(np.log(p_p * (self.capsules - 1)) - np.log(1 - p_p), dtype=tf.float32) \
                    / (d_p - d_o + 1e-12)
                t = tf.expand_dims(t, axis=-1)

                # Calc log prior using inverse distances
                b_ij = t * d

        return v_j, c_ij

    def get_config(self):
        return super(GammaCapsule, self).get_config()


class GammaDecoder(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(GammaDecoder, self).__init__(**kwargs)
        self.dim = dim
        self.decoder = None

    def build(self, input_shape):
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.Flatten())
        self.decoder.add(tf.keras.layers.Dense(512, activation=activations.relu))
        self.decoder.add(tf.keras.layers.Dense(1024, activation=activations.relu))
        self.decoder.add(tf.keras.layers.Dense(self.dim * self.dim, activation=activations.sigmoid))

    def call(self, inputs, **kwargs):
        return self.decoder(inputs)

    def get_config(self):
        return super(GammaDecoder, self).get_config()

