from tensorflow.keras.layers import Lambda
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from libs.capsnets.layers.diverse import Capsule, PrimaryCapsule2D, Length, bottleneck
from tensorflow.keras.layers import Concatenate
from libs.capsnets.utls import squash


def CapsNet(input_shape, num_classes, routings):
    x = layers.Input(shape=input_shape)
    l_1 = bottleneck(x, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_1 = Concatenate(axis=-1)([x, l_1])

    l_2 = bottleneck(m_1, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_2 = Concatenate(axis=-1)([m_1, l_2])

    l_3 = bottleneck(m_2, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_3 = Concatenate(axis=-1)([m_2, l_3])

    l_4 = bottleneck(m_3, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_4 = Concatenate(axis=-1)([m_3, l_4])

    l_5 = bottleneck(m_4, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_5 = Concatenate(axis=-1)([m_4, l_5])

    l_6 = bottleneck(m_5, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_6 = Concatenate(axis=-1)([m_5, l_6])

    l_7 = bottleneck(m_6, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m_7 = Concatenate(axis=-1)([m_6, l_7])

    l_8 = bottleneck(m_7, 32, (3, 3), e=1, s=1, squeeze=False, nl='HS')
    m_8 = Concatenate(axis=-1)([m_7, l_8])

    input1, primarycaps1, a1 = PrimaryCapsule2D(dim_capsules=12, num_capsules=10, kernel_size=5, strides=(2, 2))(m_8)

    # level 1

    l2_1 = bottleneck(input1, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_1 = Concatenate(axis=-1)([input1, l2_1])

    l2_2 = bottleneck(m2_1, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_2 = Concatenate(axis=-1)([m2_1, l2_2])

    l2_3 = bottleneck(m2_2, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_3 = Concatenate(axis=-1)([m2_2, l2_3])

    l2_4 = bottleneck(m2_3, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_4 = Concatenate(axis=-1)([m2_3, l2_4])

    l2_5 = bottleneck(m2_4, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_5 = Concatenate(axis=-1)([m2_4, l2_5])

    l2_6 = bottleneck(m2_5, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_6 = Concatenate(axis=-1)([m2_5, l2_6])

    l2_7 = bottleneck(m2_6, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m2_7 = Concatenate(axis=-1)([m2_6, l2_7])

    l2_8 = bottleneck(m2_7, 32, (3, 3), e=1, s=1, squeeze=False, nl='HS')
    m2_8 = Concatenate(axis=-1)([m2_7, l2_8])

    input2, primarycaps2, a2 = PrimaryCapsule2D(dim_capsules=12, num_capsules=10, kernel_size=5, strides=(2, 2))(m2_8)

    # level 2

    l3_1 = bottleneck(input2, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_1 = Concatenate(axis=-1)([input2, l3_1])

    l3_2 = bottleneck(m3_1, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_2 = Concatenate(axis=-1)([m3_1, l3_2])

    l3_3 = bottleneck(m3_2, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_3 = Concatenate(axis=-1)([m3_2, l3_3])

    l3_4 = bottleneck(m3_3, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_4 = Concatenate(axis=-1)([m3_3, l3_4])

    l3_5 = bottleneck(m3_4, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_5 = Concatenate(axis=-1)([m3_4, l3_5])

    l3_6 = bottleneck(m3_5, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_6 = Concatenate(axis=-1)([m3_5, l3_6])

    l3_7 = bottleneck(m3_6, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_7 = Concatenate(axis=-1)([m3_6, l3_7])

    l3_8 = bottleneck(m3_7, 32, (3, 3), e=1, s=1, squeeze=False, nl='RE')
    m3_8 = Concatenate(axis=-1)([m3_7, l3_8])

    input3, primarycaps3, a3 = PrimaryCapsule2D(dim_capsules=12, num_capsules=10, kernel_size=3, strides=(1, 1))(m3_8)

    # level 3

    primarycaps1 = layers.Reshape(target_shape=(-1, 12), name='primarycaps11')(primarycaps1)
    primarycaps2 = layers.Reshape(target_shape=(-1, 12), name='primarycaps21')(primarycaps2)
    primarycaps3 = layers.Reshape(target_shape=(-1, 12), name='primarycaps31')(primarycaps3)

    digitcaps2 = Capsule(num_capsule=num_classes, dim_capsule=6, routings=routings, name='digitcaps2')(primarycaps1)
    digitcaps3 = Capsule(num_capsule=num_classes, dim_capsule=6, routings=routings, name='digitcaps3')(primarycaps2)
    digitcaps4 = Capsule(num_capsule=num_classes, dim_capsule=6, routings=routings, name='digitcaps4')(primarycaps3)

    digitcaps2 = layers.Reshape(target_shape=(-1, 6), name='digitcaps21')(digitcaps2)
    a1 = K.tile(a1, [1, 10, 6])
    weight_1 = Lambda(lambda x: x * a1)
    digitcaps2 = weight_1(digitcaps2)

    digitcaps3 = layers.Reshape(target_shape=(-1, 6), name='digitcaps31')(digitcaps3)
    a2 = K.tile(a2, [1, 10, 6])
    weight_2 = Lambda(lambda x: x * a2)
    digitcaps3 = weight_2(digitcaps3)

    digitcaps4 = layers.Reshape(target_shape=(-1, 6), name='digitcaps41')(digitcaps4)
    a3 = K.tile(a3, [1, 10, 6])
    weight_3 = Lambda(lambda x: x * a3)
    digitcaps4 = weight_3(digitcaps4)

    digitcaps = Concatenate(axis=-1)([digitcaps2, digitcaps3])
    digitcaps = Concatenate(axis=-1)([digitcaps, digitcaps4])

    digitcaps = layers.Lambda(squash)(digitcaps)
    out_caps = Length(name='capsnet')(digitcaps)
    y = layers.Input(shape=(num_classes,))
    train_model = models.Model([x, y], out_caps)
    eval_model = models.Model(x, out_caps)

    return train_model, eval_model


if __name__ == '__main__':
    model, eval_model = CapsNet(input_shape=(32, 32, 3), num_classes=10, routings=3)
    model.summary()
