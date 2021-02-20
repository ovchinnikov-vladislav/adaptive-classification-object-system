import keras.backend as keras_backend


def squash(vectors, axis=-1):
    s_squared_norm = keras_backend.sum(keras_backend.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / keras_backend.sqrt(s_squared_norm + keras_backend.epsilon())
    return scale * vectors
