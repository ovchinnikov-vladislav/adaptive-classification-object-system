import tensorflow.keras.backend as keras_backend
import tensorflow as tf


def squash(vectors, axis=-1):
    """
    Функция нелинейной активации, используемая в капсулах. Она приводит длину большого вектора к значению близкому 1,
    а малый вектор к значению до 0.
    :param vectors: некоторые векторы для сжатия (операции squash), N-разерный тензор
    :param axis: ось для сжатия (операции squash)
    :return: на входе тензор той же формы (shape) что и входные векторы
    """

    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + keras_backend.epsilon())
    return scale * vectors
