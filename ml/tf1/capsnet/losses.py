import tensorflow as tf
import keras.backend as keras_backend


def margin_loss(y_true, y_pred):
    L = y_true * keras_backend.square(keras_backend.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * keras_backend.square(keras_backend.maximum(0., y_pred - 0.1))

    return keras_backend.mean(keras_backend.sum(L, 1))
