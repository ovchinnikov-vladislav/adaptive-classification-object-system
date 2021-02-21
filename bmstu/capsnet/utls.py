import tensorflow as tf
import numpy as np
from tensorflow.keras.backend import epsilon


# v = ((||sj||^2) / (1 + ||sj||^2)) * (sj / ||sj||)
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)

    additional_squashing = s_squared_norm / (1 + s_squared_norm)
    unit_scaling = vectors / tf.sqrt(s_squared_norm + epsilon())

    return additional_squashing * unit_scaling


def attention_em_routing(votes_v1, votes_v2, votes_f, activations_i_f, beta_v_v, beta_a_v, beta_v_f, beta_a_f,
                         routings):
    # TODO: написать роутинг для attention
    pass
